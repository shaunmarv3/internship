"""Maritime Intelligence API -- serves precomputed case-study assets."""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# Load .env from the project root (two levels up from this file: backend/app.py)
# so developers can set GFW_TOKEN, HF_TOKEN, etc. without touching shell config.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — fall back to shell environment

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.schemas import CaseStudyDetail, CaseStudySummary

DATA_DIR = Path(__file__).parent / "data" / "case_studies"
UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
PNG_DEMO_DIR = Path(__file__).parent / "data" / "png_demos"

app = FastAPI(title="Maritime Intelligence API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/assets/{path:path}")
def serve_asset(path: str):
    """Serve case-study static files (PNGs, GeoJSON, etc.) with no-cache headers
    so the browser always fetches the latest scene overlay after reprocessing."""
    file_path = DATA_DIR / path
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"asset not found: {path}")
    return FileResponse(str(file_path), headers={"Cache-Control": "no-store"})


@app.get("/png-assets/{path:path}")
def serve_png_asset(path: str):
    """Serve annotated PNG-demo images (YOLO-only exhibit, no case study)."""
    file_path = PNG_DEMO_DIR / path
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"png asset not found: {path}")
    return FileResponse(str(file_path), headers={"Cache-Control": "no-store"})


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/case-studies", response_model=list[CaseStudySummary])
def list_case_studies() -> list[dict]:
    out: list[dict] = []
    if not DATA_DIR.exists():
        return out
    for d in sorted(DATA_DIR.iterdir()):
        meta = d / "meta.json"
        if d.is_dir() and meta.exists():
            out.append(_load_json(meta))
    return out


@app.get("/case-study/{cid}", response_model=CaseStudyDetail)
def get_case_study(cid: str) -> dict:
    d = DATA_DIR / cid
    if not (d.is_dir() and (d / "meta.json").exists()):
        raise HTTPException(status_code=404, detail=f"case study '{cid}' not found")
    return {
        "meta": _load_json(d / "meta.json"),
        "layers": _load_json(d / "layers.geojson"),
        "metrics": _load_json(d / "metrics.json"),
        "explain": _load_json(d / "explain.json"),
    }


# checkpoint + aux-data paths (override via env vars)
YOLO_CKPT = os.environ.get("MARITIME_YOLO", "checkpoints/vessel/hrsid_yolo11m_obb/best.pt")
SEGFORMER_CKPT = os.environ.get("MARITIME_SEGFORMER", "checkpoints/oil/best_segformer.pt")


def _read_scene_acquired(tif_path: Path) -> str | None:
    """Try to read the Sentinel-1 acquisition time from GEE TIFF metadata.

    GEE embeds 'system:time_start' (epoch ms) in the GDAL tags. Falls back to
    None so the caller can substitute datetime.now().
    """
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            tags = src.tags() or {}
        ts_ms = tags.get("system:time_start") or tags.get("TIFFTAG_DATETIME")
        if ts_ms and str(ts_ms).isdigit():
            dt = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        # TIFFTAG_DATETIME is "YYYY:MM:DD HH:MM:SS"
        if ts_ms and ":" in str(ts_ms):
            from datetime import datetime as _dt
            dt = _dt.strptime(str(ts_ms), "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass
    return None


def _pipeline_namespace(**kw) -> SimpleNamespace:
    cid = kw.get("id", "") or ""

    # Per-scene AIS display behaviour:
    #   Singapore / Mumbai  → all ships AIS-matched (all green, busy commercial ports)
    #   Hormuz / Malacca    → all green except the one most-isolated ship (single dark vessel)
    #   everything else     → random 65/35 split
    if any(x in cid for x in ("singapore", "mumbai")):
        syn_fraction, syn_mode = 1.0, "all_green"
    elif any(x in cid for x in ("hormuz", "malacca")):
        syn_fraction, syn_mode = 0.65, "farthest_dark"
    else:
        syn_fraction, syn_mode = 0.65, "random"

    base = dict(
        scene_tif=None, bbox=None, start=None, end=None,
        id=None, title=None, acquired=None,
        yolo=YOLO_CKPT, segformer=SEGFORMER_CKPT, backbone="b4",
        gap_csv=os.environ.get("MARITIME_GFW_CSV"), gfw_token=os.environ.get("GFW_TOKEN"),
        eez=os.environ.get("MARITIME_EEZ"), mpa=None,
        out=str(DATA_DIR), chip_size=512, scale=10.0, conf=0.15,
        match_radius=500.0, oil_iou=None, yolo_map50=None,
        oil_threshold=0.35, oil_min_area_px=25, mask_land=True,
        oil_infer="resize",
        synthetic_ais_fraction=syn_fraction, synthetic_ais_mode=syn_mode,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@app.post("/process")
def process_scene(
    file: UploadFile = File(...),
    id: str = Form(None),
    title: str = Form(None),
    acquired: str = Form(None),
):
    """Accept a Sentinel-1 GeoTIFF upload, run the full pipeline, return the new case-study id.

    Heavy inference (YOLO + SegFormer) runs here, so the backend host needs the model deps
    + the downloaded checkpoints. Sync def → FastAPI runs it in a threadpool.
    """
    stem = Path(file.filename or "scene").stem
    cid = id or re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-").lower() or "scene"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    tif_path = UPLOAD_DIR / f"{cid}.tif"
    with open(tif_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    acquired = acquired or _read_scene_acquired(tif_path) or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        from backend.pipeline.run_scene import run as run_pipeline
        run_pipeline(_pipeline_namespace(
            scene_tif=str(tif_path), id=cid, title=title or cid, acquired=acquired,
        ))
    except Exception as e:  # surface a clean message to the UI, full trace to the console
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"pipeline failed: {type(e).__name__}: {e}")
    return {"id": cid}


@app.post("/detect-png")
def detect_png(file: UploadFile = File(...)):
    """YOLO-only demo on a raw HRSID PNG — no CFAR, no SegFormer, no geo.

    HRSID images are the model's own training domain (0.5-3 m/px amplitude), so this
    exhibits YOLO standing alone. A PNG has no georeferencing, so the result is drawn
    directly onto the image (pixel space) and shown instead of the map.
    """
    stem = Path(file.filename or "hrsid").stem
    cid = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-").lower() or "hrsid"
    out_dir = PNG_DEMO_DIR / cid
    out_dir.mkdir(parents=True, exist_ok=True)
    in_png = out_dir / "input.png"
    with open(in_png, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from backend.pipeline.png_demo import run_yolo_png
        info = run_yolo_png(str(in_png), str(out_dir / "annotated.png"), yolo_ckpt=YOLO_CKPT)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"YOLO demo failed: {type(e).__name__}: {e}")

    import time
    return {
        "id": cid,
        "filename": file.filename or f"{cid}.png",
        "image": f"{cid}/annotated.png",
        "ship_count": info["ship_count"],
        "width": info["width"],
        "height": info["height"],
        "confidences": info["confidences"],
        "v": int(time.time()),
    }
