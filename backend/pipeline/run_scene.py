"""End-to-end scene pipeline: Sentinel-1 -> case-study assets for the dashboard.

Build-time orchestrator (design spec section 4.1). Reuses the verified src/ modules:
  preprocess (sar_preprocess) -> YOLO ships (detection) -> SegFormer oil (segment+geo)
  -> GFW AIS match + dark flag (fusion.ais_matching) -> fusion + risk (fusion.risk_scoring)
  -> Grad-CAM (explain.gradcam) -> serialize (pipeline.serialize).

Runs on a GPU box (Colab/Lightning) with the HF checkpoints + a GFW token. It is NOT
executed in the dev environment. Output lands in backend/data/case_studies/<id>/ and is
served as-is by backend/app.py.

Example:
  python -m backend.pipeline.run_scene \
      --scene-tif data/gee_scenes/gulf_2024-03-12.tif \
      --id gulf-2024-03-12 --title "Gulf of Mexico 2024-03-12" \
      --acquired 2024-03-12T06:14:00Z \
      --yolo checkpoints/vessel/best.pt \
      --segformer checkpoints/oil/best_segformer.pt --backbone b4 \
      --gap-csv data/gfw_data/gap_events.csv \
      --eez data/shapefiles/world_eez.gpkg
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# allow `python -m backend.pipeline.run_scene` to import src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _parse_acquired(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    import pandas as pd

    from src.data.sar_preprocess import preprocess_sar_tif, gee_to_chips
    from src.models.detection import SARVesselDetector
    from src.fusion.ais_matching import (
        fetch_ais_around_scene, interpolate_ais_to_time,
        match_detections_to_ais, flag_zone_violations,
    )
    from src.fusion.risk_scoring import link_spills_to_dark_vessels, build_risk_table
    from backend.pipeline.segment import segment_scene, best_oil_chip
    from backend.pipeline.geo import mask_to_polygons
    from backend.pipeline import serialize

    acquired = _parse_acquired(args.acquired)

    # 1. preprocess -> chips (carry transform/crs)
    if args.scene_tif:
        chips = preprocess_sar_tif(args.scene_tif, chip_size=args.chip_size)
    else:
        chips = gee_to_chips(args.bbox, args.start, args.end, chip_size=args.chip_size)
    if not chips:
        raise SystemExit("No chips produced — scene was empty after land-mask/skip.")
    transform = chips[0]["transform"]
    chip_size = chips[0]["chip_size"]
    H = max(c["y"] for c in chips) + chip_size
    W = max(c["x"] for c in chips) + chip_size

    # bbox from the scene transform corners (lon/lat)
    lon0, lat0 = transform.c, transform.f
    lon1 = transform.c + W * transform.a + H * transform.b
    lat1 = transform.f + W * transform.d + H * transform.e
    bbox = (min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))

    # 2. ships (YOLO)
    detector = SARVesselDetector(model_path=args.yolo, conf_thresh=args.conf)
    detections = detector.detect_scene(chips, W, H)

    # 3. AIS at scene time -> match -> dark flag
    ais = fetch_ais_around_scene(bbox, acquired, gfw_token=args.gfw_token, gap_csv=args.gap_csv)
    ais_at_time = interpolate_ais_to_time(ais, acquired)
    vessels = match_detections_to_ais(detections, ais_at_time, transform, match_radius_m=args.match_radius)

    # 4. zone violations (illegal-fishing rule lives here)
    if args.eez:
        vessels = flag_zone_violations(vessels, mpa_path=args.mpa or args.eez, eez_path=args.eez)

    # 5. oil (SegFormer) -> polygons
    scene_mask = segment_scene(chips, args.segformer, backbone=args.backbone)
    oil_polys = mask_to_polygons(scene_mask, transform, scale_m=args.scale)

    # 6. fusion + risk
    vessel_df = pd.DataFrame(vessels)
    oil_gdf = _polys_to_gdf(oil_polys)
    links = link_spills_to_dark_vessels(oil_gdf, vessel_df, search_radius_km=args.fusion_radius_km)
    risk = build_risk_table(vessel_df) if not vessel_df.empty else pd.DataFrame()

    security = float(risk["security_risk"].max() / 100.0) if not risk.empty else 0.0
    environmental = _env_risk(oil_polys)

    # 7. Grad-CAM exhibit (oil model)
    overlays_dir = Path(args.out) / args.id / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    gradcam_oil = _gradcam_oil(chips, scene_mask, chip_size, args, overlays_dir)

    # 8. serialize
    vessels_records = vessel_df.to_dict("records") if not vessel_df.empty else []
    ais_tracks = _ais_tracks(ais_at_time)
    fusion_records = _fusion_links(links, oil_polys)
    layers = serialize.build_layers(
        vessels=vessels_records, oil=oil_polys, ais_tracks=ais_tracks,
        zones=[], fusion_links=fusion_records,
    )
    meta = {
        "id": args.id, "title": args.title, "sensor": "Sentinel-1",
        "acquired_utc": args.acquired, "bbox": [round(b, 4) for b in bbox],
        "summary": f"{len(oil_polys)} slick(s), {int((vessel_df.get('dark_vessel', pd.Series()) == True).sum()) if not vessel_df.empty else 0} dark vessel(s).",
    }
    metrics = {
        "oil_iou": args.oil_iou, "yolo_map50": args.yolo_map50,
        "dark_count": int((vessel_df.get("dark_vessel", pd.Series()) == True).sum()) if not vessel_df.empty else 0,
        "slick_count": len(oil_polys),
        "security_risk": round(security, 3), "environmental_risk": round(environmental, 3),
    }
    explain = {
        "gradcam_oil_png": f"{args.id}/overlays/gradcam_oil.png" if gradcam_oil else None,
        "gradcam_ship_png": None,
        "risk_factors": _risk_factors(vessel_df, oil_polys),
    }
    out = serialize.write_case_study(Path(args.out) / args.id, meta, layers, metrics, explain)
    print(f"Wrote case study -> {out}")
    return out


def _polys_to_gdf(oil_polys):
    import geopandas as gpd
    from shapely.geometry import Polygon
    if not oil_polys:
        return gpd.GeoDataFrame(columns=["geometry", "severity"], crs="EPSG:4326")
    geoms, sev = [], []
    for p in oil_polys:
        geoms.append(Polygon(p["coordinates"][0]))
        sev.append(min(p["area_km2"] / 50.0, 1.0))  # crude severity from area
    return gpd.GeoDataFrame({"severity": sev}, geometry=geoms, crs="EPSG:4326")


def _env_risk(oil_polys) -> float:
    if not oil_polys:
        return 0.0
    total = sum(p["area_km2"] for p in oil_polys)
    return min(total / 50.0, 1.0)


def _ais_tracks(ais_at_time):
    if ais_at_time is None or ais_at_time.empty:
        return []
    tracks = []
    for mmsi, g in ais_at_time.groupby("mmsi"):
        pts = [[float(r["lon"]), float(r["lat"])] for _, r in g.iterrows()
               if r.get("lon") is not None and r.get("lat") is not None]
        if len(pts) >= 2:
            tracks.append({"coordinates": pts, "mmsi": str(mmsi)})
    return tracks


def _fusion_links(links, oil_polys):
    if links is None or links.empty or not oil_polys:
        return []
    out = []
    for _, r in links.iterrows():
        sid = int(r["spill_id"]) if r.get("spill_id") is not None else 0
        if sid >= len(oil_polys):
            sid = 0
        ring = oil_polys[sid]["coordinates"][0]
        cx = sum(p[0] for p in ring) / len(ring)
        cy = sum(p[1] for p in ring) / len(ring)
        out.append({
            "coordinates": [[cx, cy], [float(r["vessel_lon"]), float(r["vessel_lat"])]],
            "reason": f"dark vessel {r.get('distance_km')} km from slick",
        })
    return out


def _gradcam_oil(chips, scene_mask, chip_size, args, overlays_dir):
    try:
        import cv2
        import torch
        from src.models.segmentation import build_segmentation_model
        from src.explain.gradcam import generate_gradcam
        from backend.pipeline.segment import _chip_tensor

        path = best_oil_chip(chips, scene_mask, chip_size)
        if not path:
            return None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_segmentation_model(model_type="segformer", num_classes=2, backbone=args.backbone)
        state = torch.load(args.segformer, map_location=device)
        model.load_state_dict(state.get("model_state", state))
        model.to(device).eval()
        rgb = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        generate_gradcam(
            model, _chip_tensor(path).to(device), rgb, target_class=1,
            model_type="segformer", save_path=str(overlays_dir / "gradcam_oil.png"),
        )
        return True
    except Exception as e:  # gradcam is an exhibit, never fail the run
        print(f"[gradcam warning] {e}")
        return None


def _risk_factors(vessel_df, oil_polys):
    import pandas as pd
    dark = int((vessel_df.get("dark_vessel", pd.Series()) == True).sum()) if not vessel_df.empty else 0
    in_mpa = int((vessel_df.get("in_mpa", pd.Series()) == True).sum()) if not vessel_df.empty else 0
    factors = [
        ("AIS signal absence", 0.4 if dark else 0.05),
        ("Restricted-area presence", 0.3 if in_mpa else 0.05),
        ("Proximity to slick", 0.2 if oil_polys and dark else 0.05),
        ("Slick size", min(_env_risk(oil_polys), 0.3)),
    ]
    total = sum(w for _, w in factors) or 1.0
    return [{"label": k, "weight": round(w / total, 3)} for k, w in factors]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentinel-1 scene -> dashboard case study")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene-tif", help="Local Sentinel-1 GeoTIFF")
    src.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="GEE bbox")
    p.add_argument("--start"); p.add_argument("--end")  # for GEE
    p.add_argument("--id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--acquired", required=True, help="ISO UTC, e.g. 2024-03-12T06:14:00Z")
    p.add_argument("--yolo", required=True, help="YOLO ship checkpoint")
    p.add_argument("--segformer", required=True, help="SegFormer oil checkpoint")
    p.add_argument("--backbone", default="b4")
    p.add_argument("--gap-csv", default=None, help="GFW gap_events.csv (else live API)")
    p.add_argument("--gfw-token", default=None)
    p.add_argument("--eez", default=None, help="EEZ gpkg/shapefile for zone join")
    p.add_argument("--mpa", default=None, help="MPA shapefile (defaults to --eez)")
    p.add_argument("--out", default="backend/data/case_studies")
    p.add_argument("--chip-size", type=int, default=512)
    p.add_argument("--scale", type=float, default=10.0, help="metres/pixel for area")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--match-radius", type=float, default=500.0)
    p.add_argument("--fusion-radius-km", type=float, default=160.0)
    p.add_argument("--oil-iou", type=float, default=None, help="reported test metric")
    p.add_argument("--yolo-map50", type=float, default=None, help="reported test metric")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
