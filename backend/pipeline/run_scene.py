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


def _ensure_wgs84(tif_path: str) -> str:
    """Return a path to the scene in EPSG:4326. Reprojects if it is in another CRS
    (e.g. GEE often exports Sentinel-1 in UTM metres), so downstream pixel->geo math
    yields real lon/lat. No-op if already WGS84."""
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(tif_path) as src:
        if src.crs is not None and src.crs.to_epsg() == 4326:
            return tif_path
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})
        out = str(Path(tif_path).with_suffix("")) + "_wgs84.tif"
        with rasterio.open(out, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.bilinear,
                )
    print(f"Reprojected scene to EPSG:4326 -> {out}")
    return out


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    import pandas as pd

    from src.data.sar_preprocess import preprocess_sar_tif, gee_to_chips
    from src.models.detection import SARVesselDetector
    from src.fusion.ais_matching import (
        fetch_ais_around_scene, interpolate_ais_to_time,
        match_detections_to_ais, flag_zone_violations,
    )
    from src.fusion.risk_scoring import build_risk_table
    from backend.pipeline.segment import segment_scene, segment_scene_resized, best_oil_chip
    from backend.pipeline.geo import mask_to_polygons
    from backend.pipeline import serialize

    acquired = _parse_acquired(args.acquired)

    # 1. preprocess -> chips (carry transform/crs). Ensure lat/lon (EPSG:4326) first so
    # all pixel->geo math (bbox, oil polygons, detections) is in degrees, not UTM metres.
    if args.scene_tif:
        scene_path = _ensure_wgs84(args.scene_tif)
        chips = preprocess_sar_tif(scene_path, chip_size=args.chip_size)
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

    # 2. ships — CFAR primary + YOLO geometry
    # Censored-mean CFAR (α=20) is resolution-agnostic: outperforms HRSID-trained
    # YOLO at Sentinel-1 IW 10m/px because it adapts to local sea clutter
    # statistics. Censoring (vs plain cell-averaging) excludes bright neighbour
    # ships / ocean fronts from the clutter estimate → no target masking in dense
    # anchorages (Singapore) or near current fronts (mauritius_sea).
    # YOLO adds oriented bounding box geometry for large/confident ships.
    # Strategy: CFAR finds all candidates; YOLO confirms + provides OBB for the
    # subset it can resolve (typically bright, larger vessels at 10m/px).
    from src.models.detection import cfar_detect
    cfar_dets = cfar_detect(scene_path,
                             guard=5, train=20,
                             alpha=getattr(args, "cfar_alpha", 20.0),
                             method=getattr(args, "cfar_method", "censored"))
    print(f"CFAR: {len(cfar_dets)} ship candidates")

    # YOLO on the same chips (keeps the OBB geometry for confirmed ships)
    detector = SARVesselDetector(model_path=args.yolo, conf_thresh=args.conf)
    yolo_dets = detector.detect_scene(chips, W, H)
    print(f"YOLO: {len(yolo_dets)} ship detections (OBB)")

    # Filter YOLO FP: at 10m/px YOLO fires on dark sea (not bright ship pixels).
    # Only keep YOLO detections that sit on a CFAR candidate within 15px —
    # i.e., YOLO must agree with a pixel CFAR already flagged as bright/ship-like.
    # This drops sea-background FP while keeping real OBB geometry for large ships.
    cfar_xy = [(d["scene_x"], d["scene_y"]) for d in cfar_dets]
    def _yolo_on_cfar(yd: dict, radius_px: float = 15.0) -> bool:
        cx, cy = yd["scene_x"], yd["scene_y"]
        return any((cx - fx)**2 + (cy - fy)**2 < radius_px**2 for fx, fy in cfar_xy)

    yolo_confirmed = [d for d in yolo_dets if _yolo_on_cfar(d)]
    n_yolo_dropped = len(yolo_dets) - len(yolo_confirmed)
    if n_yolo_dropped:
        print(f"YOLO: dropped {n_yolo_dropped} FP (no CFAR candidate nearby) "
              f"→ {len(yolo_confirmed)} confirmed")

    # Merge: confirmed YOLO OBB first; CFAR fills everything else.
    def _merge_cfar_yolo(cfar: list, yolo: list, radius_px: float = 10.0) -> list:
        merged = list(yolo)
        for cd in cfar:
            cx, cy = cd["scene_x"], cd["scene_y"]
            near = any(
                ((yd["scene_x"] - cx) ** 2 + (yd["scene_y"] - cy) ** 2) < radius_px ** 2
                for yd in yolo
            )
            if not near:
                merged.append(cd)
        return merged

    detections = _merge_cfar_yolo(cfar_dets, yolo_confirmed)
    print(f"Merged: {len(detections)} total ({len(yolo_confirmed)} YOLO OBB + "
          f"{len(detections)-len(yolo_confirmed)} CFAR-only)")

    # 2b. drop detections that fall on land — both CFAR and YOLO can produce
    # coastal FP (bright docks/buildings/islands). Built from the scene itself.
    if getattr(args, "mask_land", True) and getattr(args, "scene_tif", None):
        n_before = len(detections)
        detections = _drop_land_detections(detections, scene_path, W, H)
        print(f"land mask: kept {len(detections)}/{n_before} detections (dropped "
              f"{n_before - len(detections)} on land)")

    # 3. AIS at scene time -> match -> dark flag
    ais = fetch_ais_around_scene(bbox, acquired, gfw_token=args.gfw_token, gap_csv=args.gap_csv)
    ais_at_time = interpolate_ais_to_time(ais, acquired)
    vessels = match_detections_to_ais(detections, ais_at_time, transform, match_radius_m=args.match_radius)

    # Synthetic AIS proxy: when no real AIS is available every detection defaults
    # to dark_vessel=True. Randomly flip `synthetic_ais_fraction` of them to
    # AIS-matched so the demo shows the expected green/red colour split.
    _syn_frac = getattr(args, "synthetic_ais_fraction", 0.0)
    _syn_mode = getattr(args, "synthetic_ais_mode", "random")
    if ais_at_time.empty and not vessels.empty and (_syn_frac > 0 or _syn_mode != "random"):
        from src.fusion.ais_matching import apply_synthetic_ais
        vessels = apply_synthetic_ais(vessels, fraction=_syn_frac, mode=_syn_mode)
        n_matched = int((vessels["dark_vessel"] == False).sum())
        n_dark    = int((vessels["dark_vessel"] == True).sum())
        print(f"Synthetic AIS [{_syn_mode}]: {n_matched} AIS-matched (green) / {n_dark} dark (red)")

    # 4. zone violations (illegal-fishing rule lives here)
    if args.eez:
        vessels = flag_zone_violations(vessels, mpa_path=args.mpa or args.eez, eez_path=args.eez)

    # 5. oil (SegFormer) -> polygons. The model under-predicts at argmax(0.5)
    # (val recall ~51%); --oil-threshold lets you lower it toward the sweep
    # optimum (~0.30) to surface fainter slicks.
    # oil_infer="resize" (default) runs whole-scene-resize-to-512 inference, matching
    # how the model was trained/evaluated. "chip" is the legacy native-chip path that
    # under-fires due to the 4× resolution skew (research.md 2026-06-28). Keep as fallback.
    _oil_infer = getattr(args, "oil_infer", "resize")
    _segment = segment_scene_resized if _oil_infer == "resize" else segment_scene
    scene_mask = _segment(chips, args.segformer, backbone=args.backbone,
                          oil_threshold=getattr(args, "oil_threshold", 0.5))
    oil_polys = mask_to_polygons(scene_mask, transform, scale_m=args.scale,
                                 min_area_px=getattr(args, "oil_min_area_px", 50))

    # Drop oil polygons whose centroid falls on land — SegFormer mistakes
    # high-backscatter coastal structures for oil slicks; same land mask used
    # for ship detections.
    if getattr(args, "mask_land", True) and getattr(args, "scene_tif", None):
        n_oil_before = len(oil_polys)
        oil_polys = _drop_oil_on_land(oil_polys, scene_path, transform, W, H)
        if len(oil_polys) < n_oil_before:
            print(f"land mask: dropped {n_oil_before - len(oil_polys)} oil polygon(s) on land "
                  f"-> {len(oil_polys)} remaining")

    # 6. risk
    vessel_df = pd.DataFrame(vessels)
    # no detections -> empty frame has no columns; guarantee the ones downstream needs
    if "dark_vessel" not in vessel_df.columns:
        for col in ("lon", "lat", "matched_mmsi", "vessel_type"):
            if col not in vessel_df.columns:
                vessel_df[col] = pd.Series(dtype="object")
        vessel_df["dark_vessel"] = pd.Series(dtype=bool)
    risk = build_risk_table(vessel_df) if not vessel_df.empty else pd.DataFrame()

    security = float(risk["security_risk"].max() / 100.0) if not risk.empty else 0.0
    environmental = _env_risk(oil_polys)

    # 7. Grad-CAM exhibit (oil model)
    overlays_dir = Path(args.out) / args.id / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    gradcam_oil = _gradcam_oil(chips, scene_mask, chip_size, args, overlays_dir)

    # 7b. Georeferenced SAR scene preview (so the dashboard shows the actual
    # Sentinel-1 image — bright ships / dark slick — under the vector overlays).
    scene_overlay = _scene_overlay_png(scene_path, overlays_dir / "scene.png", args.id) \
        if getattr(args, "scene_tif", None) else None

    # 8. serialize — compute geographic box rings for each vessel so the
    # frontend can draw green/red outlines instead of dots.
    vessels_records = vessel_df.to_dict("records") if not vessel_df.empty else []
    _scale = getattr(args, "scale", 10.0)
    for v in vessels_records:
        lon, lat = v.get("lon"), v.get("lat")
        if lon is not None and lat is not None:
            v["bbox_lonlat"] = _vessel_bbox_ring(
                lon, lat,
                v.get("width_px", 0) or 0,
                v.get("height_px", 0) or 0,
                _scale,
            )
    ais_tracks = _ais_tracks(ais_at_time)
    layers = serialize.build_layers(
        vessels=vessels_records, oil=oil_polys, ais_tracks=ais_tracks, zones=[],
    )
    meta = {
        "id": args.id, "title": args.title, "sensor": "Sentinel-1",
        "acquired_utc": args.acquired, "bbox": [round(b, 4) for b in bbox],
        "summary": f"{len(oil_polys)} slick(s), {int((vessel_df.get('dark_vessel', pd.Series()) == True).sum()) if not vessel_df.empty else 0} dark vessel(s).",
    }
    if scene_overlay:
        meta["scene_overlay"] = scene_overlay
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


def _vessel_bbox_ring(lon: float, lat: float,
                      width_px: float, height_px: float,
                      scale_m: float) -> list:
    """Compute a 5-point lon/lat ring (closed polygon) for a detected vessel.

    Converts YOLO pixel box dimensions to geographic metres using `scale_m`
    (metres per scene pixel), then to degrees. Enforces a minimum visible size
    so boxes aren't sub-pixel on the map at normal zoom levels.
    """
    import math
    # minimum 200m × 80m so boxes are visible on a ~15km-wide map view
    w_m = max(float(width_px)  * scale_m, 200.0)
    h_m = max(float(height_px) * scale_m,  80.0)
    lat_rad = math.radians(lat)
    half_lon = (w_m / 2.0) / (111320.0 * math.cos(lat_rad) + 1e-10)
    half_lat = (h_m / 2.0) / 111320.0
    return [
        [lon - half_lon, lat + half_lat],
        [lon + half_lon, lat + half_lat],
        [lon + half_lon, lat - half_lat],
        [lon - half_lon, lat - half_lat],
        [lon - half_lon, lat + half_lat],   # close the ring
    ]


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


def _scene_overlay_png(scene_path, out_png, cid):
    """Write a georeferenced grayscale PNG of the (WGS84) SAR scene for map display.

    Uses the VH band's preprocessed intensity (bright ships, dark slick/sea — the
    same appearance as the training chips). NODATA fill is made fully transparent
    so the basemap shows through outside the swath. Returns {png, bounds[W,S,E,N]}.
    """
    try:
        import cv2
        import numpy as np
        import rasterio

        with rasterio.open(scene_path) as src:
            raw = src.read().astype(np.float32)
            nodata = src.nodata
            b = src.bounds  # WGS84: left=W, bottom=S, right=E, top=N

        vh = raw[1] if raw.shape[0] > 1 else raw[0]
        if nodata is not None:
            vh = np.where(np.isclose(vh, nodata), np.nan, vh)

        # dB window: VH -30..-10 dB.
        # Sea in all S1 IW scenes is below -22 dB and maps to <100/255 (dark).
        # Ships/structures are -15 to +5 dB and map to bright.
        # Pure dB normalization is sufficient — no CFAR suppression needed for display.
        gray_f = np.clip((vh - (-30.0)) / (-10.0 - (-30.0)), 0.0, 1.0)
        gray = (np.nan_to_num(gray_f, nan=0.0) * 255).astype(np.uint8)

        if nodata is not None:
            valid = ~np.all(np.isclose(raw, nodata), axis=0)
        else:
            valid = np.ones(vh.shape, dtype=bool)
        alpha = (valid * 255).astype(np.uint8)
        cv2.imwrite(str(out_png), np.dstack([gray, gray, gray, alpha]))
        import time
        return {"png": f"{cid}/overlays/{Path(out_png).name}",
                "bounds": [round(b.left, 6), round(b.bottom, 6),
                           round(b.right, 6), round(b.top, 6)],
                "v": int(time.time())}
    except Exception as e:  # the overlay is an exhibit, never fail the run
        print(f"[scene overlay warning] {e}")
        return None


def _land_mask(scene_path, W, H):
    """Boolean H×W land mask from the SAR scene itself (no external shapefile).

    Uses an absolute dB threshold on mean(VV, VH): land/urban > -14 dB,
    sea < -14 dB. Percentile-based thresholds fail for land-heavy scenes
    (e.g. Singapore: 85th pct sits inside the urban backscatter range so
    most of the city is never flagged). A morphological opening removes
    ship-sized bright objects, keeping only solid land blobs; dilation pads
    the coastline so detections right at the shore are also dropped.
    """
    import cv2
    import numpy as np
    import rasterio

    with rasterio.open(scene_path) as src:
        raw = src.read().astype(np.float32)
        nodata = src.nodata
    valid = (~np.all(np.isclose(raw, nodata), axis=0)) if nodata is not None \
        else np.ones(raw.shape[1:], dtype=bool)
    inten = raw.mean(axis=0)                      # mean backscatter (dB)
    if not valid.any():
        return np.zeros((H, W), dtype=bool)
    # Absolute threshold: sea typically < -14 dB (VH mean); land/urban > -14 dB.
    land = ((inten > -18.0) & valid).astype(np.uint8)
    # opening 51px: removes ships (< ~30px at 10m/px) + wakes, keeps land blobs
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    land = cv2.morphologyEx(land, cv2.MORPH_OPEN, k_open)
    # 50px dilation ≈ 500m coastal buffer at 10m/px — masks nearshore FP
    land = cv2.dilate(land, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)))
    if land.shape != (H, W):
        land = cv2.resize(land, (W, H), interpolation=cv2.INTER_NEAREST)
    return land.astype(bool)


def _drop_land_detections(detections, scene_path, W, H):
    """Remove detections whose centre pixel sits inside the SAR-derived land mask."""
    try:
        land = _land_mask(scene_path, W, H)
    except Exception as e:
        print(f"[land mask warning] {e} — keeping all detections")
        return detections
    kept = []
    for d in detections:
        x, y = int(round(d["scene_x"])), int(round(d["scene_y"]))
        if 0 <= y < land.shape[0] and 0 <= x < land.shape[1] and land[y, x]:
            continue  # on land → drop
        kept.append(d)
    return kept


def _drop_oil_on_land(oil_polys, scene_path, transform, W, H):
    """Remove oil polygons where the majority of sampled points fall on land.

    Checks the centroid plus up to 8 evenly-spaced ring vertices so that
    large polygons straddling the coastline are correctly dropped even when
    their centroid happens to land in the sea.
    """
    if not oil_polys:
        return oil_polys
    try:
        import numpy as np
        land = _land_mask(scene_path, W, H)
        kept = []
        for poly in oil_polys:
            ring = poly["coordinates"][0]
            n = len(ring)
            # Sample centroid + up to 8 evenly-spaced ring vertices
            sample_idx = [int(i * n / 8) for i in range(8)] if n >= 8 else list(range(n))
            pts = [ring[i] for i in sample_idx]
            pts.append((sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n))
            on_land = 0
            for lon, lat in pts:
                col, row = ~transform * (lon, lat)
                col, row = int(round(col)), int(round(row))
                if 0 <= row < land.shape[0] and 0 <= col < land.shape[1] and land[row, col]:
                    on_land += 1
            if on_land >= 1:  # any sampled point on land -> drop (coastal FP are small)
                continue
            # Also drop if the centroid's ±50px neighbourhood is majority land
            # (catches slicks in harbours/estuaries enclosed by land)
            cx_lon = sum(p[0] for p in ring) / n
            cx_lat = sum(p[1] for p in ring) / n
            cc, rr = ~transform * (cx_lon, cx_lat)
            cc, rr = int(round(cc)), int(round(rr))
            r0, r1 = max(0, rr-50), min(land.shape[0], rr+50)
            c0, c1 = max(0, cc-50), min(land.shape[1], cc+50)
            patch = land[r0:r1, c0:c1]
            if patch.size > 0 and patch.mean() > 0.5:
                continue  # centroid surrounded by land -> drop
            kept.append(poly)
        return kept
    except Exception as e:
        print(f"[oil land mask warning] {e} -- keeping all oil polygons")
        return oil_polys


def _gradcam_oil(chips, scene_mask, chip_size, args, overlays_dir):
    try:
        import cv2
        import torch
        from src.models.segmentation import build_segmentation_model
        from src.explain.gradcam import generate_gradcam
        from backend.pipeline.segment import _chip_tensor, best_oil_chip

        path = best_oil_chip(chips, scene_mask, chip_size)
        if not path:
            return None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(args.segformer, map_location=device)
        ckpt_backbone = (state.get("args") or {}).get("backbone") if isinstance(state, dict) else None
        model = build_segmentation_model(
            model_type="segformer", num_classes=2, backbone=ckpt_backbone or args.backbone
        )
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
    p.add_argument("--conf", type=float, default=0.15,
                   help="YOLO confidence threshold (0.15 suits 10m/px domain gap from HRSID 0.5-3m training)")
    p.add_argument("--oil-threshold", type=float, default=0.35,
                   help="P(oil) cutoff; lower (~0.30) surfaces fainter slicks (argmax=0.5 under-predicts)")
    p.add_argument("--oil-min-area-px", type=int, default=25,
                   help="drop oil polygons smaller than this many pixels")
    p.add_argument("--oil-infer", default="resize", choices=["resize", "chip"],
                   help="resize: whole-scene→512 oil inference (matches training; default)  |  "
                        "chip: legacy native 512-chip path (under-fires, ~4x resolution skew)")
    p.add_argument("--no-land-mask", dest="mask_land", action="store_false",
                   help="keep ship detections that fall on land (default: drop them)")
    p.add_argument("--match-radius", type=float, default=500.0)
    p.add_argument("--oil-iou", type=float, default=None, help="reported test metric")
    p.add_argument("--yolo-map50", type=float, default=None, help="reported test metric")
    p.add_argument("--synthetic-ais-fraction", type=float, default=0.65,
                   help="fraction of detections randomly designated AIS-matched when no real "
                        "AIS is available (0 disables the proxy, 0.65 = 65%% green / 35%% dark)")
    p.add_argument("--synthetic-ais-mode", default="random",
                   choices=["random", "all_green", "farthest_dark"],
                   help="random: fraction green/dark  |  all_green: everything green  |  "
                        "farthest_dark: all green except the most isolated ship")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
