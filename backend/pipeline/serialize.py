"""Serialize pipeline outputs into the backend case-study asset contract (spec section 6).

Pure functions only (stdlib + json) so they are unit-testable without torch/GEE/geopandas.
The orchestrator (run_scene.py) converts pandas/GeoDataFrames to plain records and calls these.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

Record = dict[str, Any]
FeatureCollection = dict[str, Any]

LAYER_NAMES = ["ships", "dark_vessels", "oil", "ais_tracks", "zones"]


def _feature(geometry: dict, properties: dict) -> dict:
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def _fc(features: list[dict]) -> FeatureCollection:
    return {"type": "FeatureCollection", "features": features}


def _clean(props: dict) -> dict:
    """Drop None values so the contract stays tidy."""
    return {k: v for k, v in props.items() if v is not None}


def points_fc(records: Iterable[Record], prop_keys: list[str]) -> FeatureCollection:
    """Point FeatureCollection from records that carry 'lon'/'lat'."""
    feats = []
    for r in records:
        if r.get("lon") is None or r.get("lat") is None:
            continue
        feats.append(
            _feature(
                {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
                _clean({k: r.get(k) for k in prop_keys}),
            )
        )
    return _fc(feats)


def _vessel_feature(r: Record) -> dict | None:
    """Build a GeoJSON Feature for one vessel.

    Emits a Polygon (axis-aligned bounding box) when 'bbox_lonlat' is present —
    so the dashboard can draw green/red ship outlines instead of dots.
    Falls back to a Point for legacy records that have no bbox.
    """
    lon, lat = r.get("lon"), r.get("lat")
    if lon is None or lat is None:
        return None
    props = _clean({
        "matched":    r.get("matched"),
        "mmsi":       r.get("mmsi"),
        "type":       r.get("type"),
        "confidence": r.get("confidence"),
    })
    bbox = r.get("bbox_lonlat")
    if bbox:
        geom: dict = {"type": "Polygon", "coordinates": [bbox]}
    else:
        geom = {"type": "Point", "coordinates": [float(lon), float(lat)]}
    return _feature(geom, props)


def vessel_layers(vessels: Iterable[Record]) -> tuple[FeatureCollection, FeatureCollection]:
    """Split vessel records into (ships, dark_vessels) by the 'dark_vessel' flag.

    Each record: lon, lat, dark_vessel(bool), matched_mmsi, vessel_type, conf,
    and optionally bbox_lonlat (5-point ring [[lon,lat],...]) for box display.
    """
    ships: list[Record] = []
    dark: list[Record] = []
    for r in vessels:
        norm = {
            "lon": r.get("lon"),
            "lat": r.get("lat"),
            "matched": not bool(r.get("dark_vessel", False)),
            "mmsi": r.get("matched_mmsi"),
            "type": r.get("vessel_type"),
            "confidence": r.get("conf", r.get("confidence")),
            "bbox_lonlat": r.get("bbox_lonlat"),
        }
        (dark if r.get("dark_vessel") else ships).append(norm)

    def _fc_vessels(records: list[Record]) -> FeatureCollection:
        return _fc([f for r in records if (f := _vessel_feature(r)) is not None])

    return _fc_vessels(ships), _fc_vessels(dark)


def polygons_fc(polys: Iterable[Record]) -> FeatureCollection:
    """Polygon FeatureCollection. Each record: 'coordinates' (GeoJSON rings) + props."""
    feats = []
    for p in polys:
        coords = p.get("coordinates")
        if not coords:
            continue
        props = _clean({k: v for k, v in p.items() if k != "coordinates"})
        feats.append(_feature({"type": "Polygon", "coordinates": coords}, props))
    return _fc(feats)


def lines_fc(lines: Iterable[Record]) -> FeatureCollection:
    """LineString FeatureCollection. Each record: 'coordinates' ([[lon,lat],...]) + props."""
    feats = []
    for ln in lines:
        coords = ln.get("coordinates")
        if not coords:
            continue
        props = _clean({k: v for k, v in ln.items() if k != "coordinates"})
        feats.append(_feature({"type": "LineString", "coordinates": coords}, props))
    return _fc(feats)


def build_layers(
    vessels: Iterable[Record],
    oil: Iterable[Record],
    ais_tracks: Iterable[Record],
    zones: Iterable[Record],
) -> dict[str, FeatureCollection]:
    ships, dark = vessel_layers(vessels)
    return {
        "ships": ships,
        "dark_vessels": dark,
        "oil": polygons_fc(oil),
        "ais_tracks": lines_fc(ais_tracks),
        "zones": polygons_fc(zones),
    }


def write_case_study(
    out_dir: str | Path,
    meta: dict,
    layers: dict[str, FeatureCollection],
    metrics: dict,
    explain: dict,
) -> Path:
    """Write the 4 JSON assets + overlays/ dir; returns the case-study directory."""
    d = Path(out_dir)
    (d / "overlays").mkdir(parents=True, exist_ok=True)
    missing = set(LAYER_NAMES) - set(layers)
    if missing:
        raise ValueError(f"layers missing required keys: {sorted(missing)}")
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (d / "layers.geojson").write_text(json.dumps(layers, indent=2), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (d / "explain.json").write_text(json.dumps(explain, indent=2), encoding="utf-8")
    return d
