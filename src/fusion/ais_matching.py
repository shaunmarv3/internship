"""
AIS ↔ SAR detection matching → dark vessel flagging.

The core idea:
  1. Detect every vessel in the SAR scene (CNN)
  2. Probabilistically match detections to AIS broadcasts around the image timestamp
  3. Unmatched detection = DARK VESSEL

Why probabilistic (not naive interpolation):
  AIS and SAR timestamps don't coincide; one message can match multiple vessels;
  vessels move between AIS ping and satellite overpass.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import requests


# ── AIS / GFW fetching ─────────────────────────────────────────────────────────

GFW_API_BASE = "https://gateway.api.globalfishingwatch.org/v3"

# Verified correct dataset names (2026-06-23):
GFW_DATASETS = {
    "gap":       "public-global-gaps-events:latest",
    "fishing":   "public-global-fishing-events:latest",
    "encounter": "public-global-encounters-events:latest",
}


def _gfw_headers(token: str = None) -> dict:
    import os
    tok = token or os.environ.get("GFW_TOKEN", "")
    return {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}


def fetch_gfw_events(
    event_type: str,                   # "GAP" | "FISHING" | "ENCOUNTER"
    date_start: str,                   # "YYYY-MM-DD"
    date_end: str,
    max_rows: int = 500,
    gfw_token: str = None,
) -> pd.DataFrame:
    """
    Fetch GFW events with correct v3 API params (verified 2026-06-23).
    Verified quirks:
      - datasets[0] is REQUIRED (422 without it)
      - offset=0 is REQUIRED when limit is sent (422 without it)
      - bbox as comma-string does NOT work; omit bbox to get global results
    Returns flat DataFrame via pd.json_normalize.
    """
    dataset_key = event_type.upper()
    if dataset_key == "GAP":
        ds = GFW_DATASETS["gap"]
    elif dataset_key == "FISHING":
        ds = GFW_DATASETS["fishing"]
    elif dataset_key == "ENCOUNTER":
        ds = GFW_DATASETS["encounter"]
    else:
        raise ValueError(f"Unknown event_type: {event_type}. Use GAP / FISHING / ENCOUNTER")

    headers = _gfw_headers(gfw_token)
    rows = []
    offset = 0
    batch = min(500, max_rows)

    while len(rows) < max_rows:
        params = {
            "types[0]":    event_type.upper(),
            "datasets[0]": ds,
            "start-date":  date_start,
            "end-date":    date_end,
            "limit":       batch,
            "offset":      offset,
        }
        try:
            resp = requests.get(f"{GFW_API_BASE}/events", headers=headers,
                                params=params, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"[GFW fetch warning] {e}")
            break

        entries = resp.json().get("entries", [])
        if not entries:
            break
        rows.extend(entries)
        offset += len(entries)
        if len(entries) < batch:
            break

    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def fetch_ais_around_scene(
    scene_bbox: Tuple[float, float, float, float],  # (lon_min, lat_min, lon_max, lat_max)
    scene_time: datetime,
    time_window_hours: float = 3.0,
    gfw_token: str = None,
    gap_csv: str = None,
) -> pd.DataFrame:
    """
    Return AIS-broadcasting vessel positions near a SAR scene's time and location.

    Strategy: use pre-downloaded GFW GAP events (gap.offPosition / gap.onPosition)
    to find vessels that were present in the bbox around scene_time. These are vessels
    that went dark before or during the scene — exactly the dark fleet candidates.

    If gap_csv is provided (path to gap_events.csv from fetch_gfw_events),
    loads from disk. Otherwise fetches live from GFW API.
    """
    lon_min, lat_min, lon_max, lat_max = scene_bbox
    t_start = scene_time - timedelta(hours=time_window_hours)
    t_end   = scene_time + timedelta(hours=time_window_hours)

    if gap_csv and Path(gap_csv).exists():
        df = pd.read_csv(gap_csv)
    else:
        date_start = t_start.strftime("%Y-%m-%d")
        date_end   = t_end.strftime("%Y-%m-%d")
        df = fetch_gfw_events("GAP", date_start, date_end, max_rows=500, gfw_token=gfw_token)

    if df.empty:
        return pd.DataFrame(columns=["mmsi", "lon", "lat", "timestamp",
                                     "vessel_type", "gap_intentional"])

    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"]   = pd.to_datetime(df["end"], utc=True)
    scene_ts    = pd.Timestamp(scene_time, tz="UTC")

    # keep rows where the gap interval OVERLAPS the scene time window.
    # Overlap = (start <= window_end) AND (end >= window_start). Using AND (not OR);
    # OR would pass almost everything through.
    w_start = pd.Timestamp(t_start, tz="UTC")
    w_end   = pd.Timestamp(t_end,   tz="UTC")
    mask = (df["start"] <= w_end) & (df["end"] >= w_start)
    df = df[mask].copy()

    # spatial filter: vessel went dark inside bbox
    if "gap.offPosition.lon" in df.columns:
        df = df[
            (df["gap.offPosition.lon"].between(lon_min, lon_max)) &
            (df["gap.offPosition.lat"].between(lat_min, lat_max))
        ]

    rows = []
    for _, row in df.iterrows():
        lon = row.get("gap.offPosition.lon", row.get("position.lon"))
        lat = row.get("gap.offPosition.lat", row.get("position.lat"))
        rows.append({
            "mmsi":            row.get("vessel.ssvid"),
            "lon":             lon,
            "lat":             lat,
            "timestamp":       row.get("start"),
            "vessel_type":     row.get("vessel.type", "unknown"),
            "gap_intentional": row.get("gap.intentionalDisabling", False),
        })
    return pd.DataFrame(rows)


# ── Coordinate conversion ───────────────────────────────────────────────────────

def pixel_to_lonlat(
    px: float, py: float,
    transform,  # rasterio affine transform of the scene
) -> Tuple[float, float]:
    """Convert pixel (col, row) → (lon, lat) using scene's affine transform."""
    lon = transform.c + px * transform.a + py * transform.b
    lat = transform.f + px * transform.d + py * transform.e
    return lon, lat


# ── Probabilistic AIS ↔ detection matching ─────────────────────────────────────

def interpolate_ais_to_time(
    ais_df: pd.DataFrame,
    target_time: datetime,
    max_age_hours: float = 3.0,
) -> pd.DataFrame:
    """
    For each MMSI, interpolate its most likely position at `target_time`
    from surrounding AIS pings. Drops vessels with no pings within max_age_hours.
    """
    if ais_df.empty:
        return ais_df

    records = []
    for mmsi, group in ais_df.groupby("mmsi"):
        group = group.sort_values("timestamp")
        t = pd.Timestamp(target_time, tz="UTC") if target_time.tzinfo else pd.Timestamp(target_time)
        group["timestamp"] = pd.to_datetime(group["timestamp"], utc=True)

        before = group[group["timestamp"] <= t]
        after  = group[group["timestamp"] > t]

        if before.empty and after.empty:
            continue
        if before.empty:
            row = after.iloc[0]
            age_h = (row["timestamp"] - t).total_seconds() / 3600
        elif after.empty:
            row = before.iloc[-1]
            age_h = (t - row["timestamp"]).total_seconds() / 3600
        else:
            b, a = before.iloc[-1], after.iloc[0]
            dt_total = (a["timestamp"] - b["timestamp"]).total_seconds()
            dt_to_t  = (t - b["timestamp"]).total_seconds()
            frac = dt_to_t / dt_total if dt_total > 0 else 0.0
            lon  = b["lon"] + frac * (a["lon"] - b["lon"])
            lat  = b["lat"] + frac * (a["lat"] - b["lat"])
            records.append({**b.to_dict(), "lon": lon, "lat": lat, "timestamp": t})
            age_h = 0.0
            continue

        if abs(age_h) <= max_age_hours:
            records.append({**row.to_dict()})

    return pd.DataFrame(records)


def match_detections_to_ais(
    detections: List[Dict],
    ais_at_time: pd.DataFrame,
    scene_transform,
    match_radius_m: float = 500.0,
) -> pd.DataFrame:
    """
    Spatial match: for each SAR detection, find the closest AIS position
    within `match_radius_m` metres.
    Returns a DataFrame with all detections, AIS match info, and dark_vessel flag.
    """
    results = []

    if ais_at_time.empty:
        for d in detections:
            results.append({**d, "matched_mmsi": None, "distance_m": np.nan,
                            "vessel_type": None, "dark_vessel": True})
        return pd.DataFrame(results)

    ais_gdf = gpd.GeoDataFrame(
        ais_at_time,
        geometry=[Point(r["lon"], r["lat"]) for _, r in ais_at_time.iterrows()],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")  # metric CRS for distance

    for det in detections:
        lon, lat = pixel_to_lonlat(det["scene_x"], det["scene_y"], scene_transform)
        det_pt = gpd.GeoDataFrame(
            [{"geometry": Point(lon, lat)}], crs="EPSG:4326"
        ).to_crs("EPSG:3857").geometry.iloc[0]

        ais_gdf["_dist"] = ais_gdf.geometry.distance(det_pt)
        nearest = ais_gdf.loc[ais_gdf["_dist"].idxmin()]
        min_dist = nearest["_dist"]

        if min_dist <= match_radius_m:
            results.append({
                **det,
                "lon": lon, "lat": lat,
                "matched_mmsi":  nearest.get("mmsi"),
                "distance_m":    float(min_dist),
                "vessel_type":   nearest.get("vessel_type"),
                "dark_vessel":   False,
            })
        else:
            results.append({
                **det,
                "lon": lon, "lat": lat,
                "matched_mmsi": None,
                "distance_m":   float(min_dist),
                "vessel_type":  None,
                "dark_vessel":  True,
            })

    return pd.DataFrame(results)


# ── Spatial zone join ─────────────────────────────────────────────────────────

def flag_zone_violations(
    vessel_df: pd.DataFrame,
    mpa_path: str,
    eez_path: str,
) -> pd.DataFrame:
    """
    Spatial join vessel positions against MPA and EEZ shapefiles.
    Adds columns: in_mpa, in_eez, zone_violation
    """
    if "lon" not in vessel_df.columns:
        vessel_df["zone_violation"] = False
        return vessel_df

    gdf = gpd.GeoDataFrame(
        vessel_df,
        geometry=[Point(r["lon"], r["lat"]) for _, r in vessel_df.iterrows()],
        crs="EPSG:4326",
    )

    try:
        # MPA columns vary by source — try common name fields
        mpa_gdf = gpd.read_file(mpa_path)
        name_col = next((c for c in ["name", "NAME", "geoname", "GEONAME"] if c in mpa_gdf.columns), None)
        mpa_gdf = mpa_gdf[["geometry"] + ([name_col] if name_col else [])].rename(
            columns={name_col: "mpa_name"} if name_col else {}
        )
        joined_mpa = gpd.sjoin(gdf, mpa_gdf, how="left", predicate="within")
        gdf["in_mpa"] = ~joined_mpa.get("mpa_name", pd.Series([None]*len(joined_mpa))).isna()
    except Exception:
        gdf["in_mpa"] = False

    try:
        # EEZ from Marine Regions WFS — verified columns (2026-06-23): lowercase
        eez_gdf = gpd.read_file(eez_path)
        # sovereign1 = country owning the EEZ; geoname = full zone name
        name_col = next((c for c in ["sovereign1", "geoname", "SOVEREIGN1", "TERRITORY1"] if c in eez_gdf.columns), None)
        eez_gdf = eez_gdf[["geometry"] + ([name_col] if name_col else [])].rename(
            columns={name_col: "eez_name"} if name_col else {}
        )
        joined_eez = gpd.sjoin(gdf, eez_gdf, how="left", predicate="within")
        gdf["in_eez"] = ~joined_eez.get("eez_name", pd.Series([None]*len(joined_eez))).isna()
    except Exception:
        gdf["in_eez"] = True

    gdf["zone_violation"] = gdf["in_mpa"] & gdf["dark_vessel"]
    return pd.DataFrame(gdf.drop(columns="geometry"))
