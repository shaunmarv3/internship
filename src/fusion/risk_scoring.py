"""
Maritime Risk Assessment — Module 6.
Fuses all module outputs into Security + Environmental risk scores.
The fusion (spill ↔ dark vessel linkage) is the headline contribution.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import Dict, List, Optional


# ── Spill ↔ dark vessel linkage ─────────────────────────────────────────────

def link_spills_to_dark_vessels(
    spill_polygons: gpd.GeoDataFrame,       # from oil segmentation, CRS=4326
    vessel_df: pd.DataFrame,                # from AIS matching, has lon/lat/dark_vessel
    search_radius_km: float = 160.0,
) -> pd.DataFrame:
    """
    For each oil slick polygon, find all dark vessels within search_radius_km.
    ~90% of spills occur within 160 km of shore — nearby dark vessel = candidate.
    Returns a linkage DataFrame: spill_id, vessel_mmsi/lon/lat, distance_km.
    """
    dark = vessel_df[vessel_df["dark_vessel"] == True].copy()
    if dark.empty or spill_polygons.empty:
        return pd.DataFrame(columns=["spill_id", "vessel_lon", "vessel_lat",
                                      "distance_km", "candidate_responsible"])

    vessels_gdf = gpd.GeoDataFrame(
        dark, geometry=[Point(r["lon"], r["lat"]) for _, r in dark.iterrows()],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    spills_m = spill_polygons.to_crs("EPSG:3857")
    radius_m = search_radius_km * 1000

    links = []
    for sid, spill_row in spills_m.iterrows():
        centroid = spill_row.geometry.centroid
        vessels_gdf["_dist"] = vessels_gdf.geometry.distance(centroid)
        nearby = vessels_gdf[vessels_gdf["_dist"] <= radius_m]
        for _, v in nearby.iterrows():
            links.append({
                "spill_id":             sid,
                "vessel_lon":           v.get("lon"),
                "vessel_lat":           v.get("lat"),
                "matched_mmsi":         v.get("matched_mmsi"),
                "distance_km":          round(v["_dist"] / 1000, 2),
                "candidate_responsible": True,
            })
    return pd.DataFrame(links)


# ── Risk scoring ──────────────────────────────────────────────────────────────

def compute_security_risk(vessel_row: pd.Series) -> float:
    """
    Score 0–100: how risky is this vessel from a security perspective.
    Higher = darker / more suspicious.
    """
    score = 0.0

    # Dark vessel (no AIS) is the strongest single signal
    if vessel_row.get("dark_vessel", False):
        score += 40

    # Inside an MPA while dark = very suspicious
    if vessel_row.get("in_mpa", False) and vessel_row.get("dark_vessel", False):
        score += 25

    # Fishing vessel classified as such by Module 2
    if vessel_row.get("suspected_illegal_fishing", False):
        score += 20

    # AIS anomaly (speed/position jump — populated upstream)
    if vessel_row.get("ais_anomaly", False):
        score += 15

    return min(score, 100.0)


def compute_environmental_risk(
    spill_row: pd.Series,
    drift_hours: float = 24.0,
    near_coast: bool = False,
    near_mpa: bool = False,
) -> float:
    """
    Score 0–100: how threatening is this spill to the environment.
    """
    score = 0.0

    severity = spill_row.get("severity", 0.5)   # 0–1 from segmentation (area-based)
    score += severity * 40

    if near_coast:
        score += 25
    if near_mpa:
        score += 20

    # shorter drift-to-shore time = worse
    if drift_hours < 12:
        score += 15
    elif drift_hours < 48:
        score += 10

    return min(score, 100.0)


def risk_label(score: float) -> str:
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    return "LOW"


# ── Aggregate risk for all vessels ─────────────────────────────────────────────

def build_risk_table(
    vessel_df: pd.DataFrame,
    spill_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Assembles the final risk table used by the dashboard.
    One row per vessel, with security_risk, env_risk, risk_label.
    """
    df = vessel_df.copy()
    df["security_risk"] = df.apply(compute_security_risk, axis=1)

    if spill_df is not None and not spill_df.empty:
        dark_spill = set(spill_df["matched_mmsi"].dropna())
        df["near_active_spill"] = df.get("matched_mmsi", pd.Series()).isin(dark_spill)
    else:
        df["near_active_spill"] = False

    df["env_risk"] = df.apply(
        lambda r: compute_environmental_risk(r, near_coast=r.get("in_eez", False)),
        axis=1,
    )
    df["risk_label"] = df["security_risk"].apply(risk_label)
    return df[["lon", "lat", "dark_vessel", "matched_mmsi", "vessel_type",
               "in_mpa", "in_eez", "security_risk", "env_risk", "risk_label"]].copy()
