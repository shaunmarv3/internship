"""
SAR preprocessing pipeline — Sentinel-1 GRD → model-ready chips.

Two entry points:
  1. GEE live pull:  gee_to_chips(bbox, date_start, date_end) → chips
  2. Local TIF:      preprocess_sar_tif(path) → chips

Preprocessing steps (applied to every band):
  1. Format detection: dB (values < -5) vs linear ([0,1] or [0,65535])
  2. dB → linear power: 10^(x/10)
  3. Lee speckle filter (7×7 window)
  4. Percentile-clip + normalize to [0,1]
  5. Chip into 512×512 tiles with 64px overlap
  6. Skip near-empty tiles (mean < 1%)
  7. Save chips as 3-channel PNG (model-ready)

IMPORTANT — pixel format (UNVERIFIED until Lightning.ai day 1):
  Run: rasterio.open(img).read().min()
  If < -5  → dB format  → step 2 converts
  If >= 0  → linear     → step 2 skipped
"""

import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ── Format detection ───────────────────────────────────────────────────────────

def detect_pixel_format(band: np.ndarray) -> str:
    """Return 'db' or 'linear'. dB Sentinel-1 GRD values are typically in [-30, +5]."""
    return "db" if float(band.min()) < -5.0 else "linear"


def db_to_linear(band: np.ndarray) -> np.ndarray:
    """dB backscatter → linear power: 10^(x/10)."""
    return np.power(10.0, band / 10.0).astype(np.float32)


# ── Speckle filter ─────────────────────────────────────────────────────────────

def lee_filter(band: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Lee speckle filter for SAR (multiplicative noise model).
    Reduces salt-and-pepper speckle while preserving bright ship returns.
    Input must be linear scale.
    """
    pad = window // 2
    img = band.astype(np.float32)
    padded = np.pad(img, pad, mode="reflect")

    mean    = cv2.blur(padded, (window, window))[pad:-pad, pad:-pad]
    sq_mean = cv2.blur(padded ** 2, (window, window))[pad:-pad, pad:-pad]
    var     = sq_mean - mean ** 2

    noise_var = float(np.mean(var))
    if noise_var < 1e-10:
        return img

    weight   = np.maximum(var - noise_var, 0) / (var + 1e-10)
    filtered = mean + weight * (img - mean)
    return filtered.astype(np.float32)


# ── Normalisation ──────────────────────────────────────────────────────────────

def normalize_band(band: np.ndarray, clip_pct: float = 99.5) -> np.ndarray:
    """
    Clip at clip_pct percentile (removes bright ship/infra hotspots that
    would compress the sea background dynamic range), then scale to [0, 1].
    """
    hi = float(np.percentile(band, clip_pct))
    if hi <= 0:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip(band / hi, 0.0, 1.0).astype(np.float32)


# ── Core band preprocessing ────────────────────────────────────────────────────

def preprocess_sar_bands(raw: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Full preprocessing for C×H×W float32 SAR array.
    Returns C×H×W float32 in [0,1].
    """
    out = np.zeros_like(raw, dtype=np.float32)
    for c in range(raw.shape[0]):
        band = raw[c].astype(np.float32)
        fmt  = detect_pixel_format(band)
        if fmt == "db":
            band = db_to_linear(band)
            if verbose:
                print(f"  band {c}: dB → converted to linear")
        else:
            if verbose:
                print(f"  band {c}: already linear")
        band  = lee_filter(band)
        out[c] = normalize_band(band)
    return out


# ── Chip a preprocessed scene ─────────────────────────────────────────────────

def chip_preprocessed(
    data: np.ndarray,           # C×H×W, float32 [0,1]
    scene_path: str,
    out_dir: str,
    chip_size: int = 512,
    overlap: int = 64,
    min_signal: float = 0.01,
    transform=None,
    crs=None,
) -> list:
    """Chip a preprocessed C×H×W array into PNG tiles. Returns chip list."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, H, W = data.shape
    stride   = chip_size - overlap
    chips    = []
    idx      = 0

    for y in range(0, H - chip_size + 1, stride):
        for x in range(0, W - chip_size + 1, stride):
            chip = data[:, y:y+chip_size, x:x+chip_size]
            if chip.mean() < min_signal:
                continue

            # → 3-channel PNG for YOLO / SegFormer
            if chip.shape[0] == 1:
                rgb = np.repeat(chip, 3, axis=0)
            elif chip.shape[0] == 2:
                rgb = np.concatenate([chip, chip[:1]], axis=0)
            else:
                rgb = chip[:3]

            rgb_u8    = (rgb * 255).clip(0, 255).astype(np.uint8)
            chip_path = out_dir / f"chip_{idx:05d}.png"
            cv2.imwrite(str(chip_path), rgb_u8.transpose(1, 2, 0)[:, :, ::-1])
            chips.append({
                "path":      str(chip_path),
                "x": x, "y": y,
                "scene":     scene_path,
                "chip_size": chip_size,
                "transform": transform,
                "crs":       crs,
            })
            idx += 1

    return chips


# ── Local TIF entry point ──────────────────────────────────────────────────────

def preprocess_sar_tif(
    tif_path: str,
    out_dir: Optional[str] = None,
    chip_size: int = 512,
    overlap: int = 64,
    min_signal: float = 0.01,
    verbose: bool = True,
) -> list:
    """
    Load a Sentinel-1 GeoTIFF → preprocess → chip → return chip list.

    Usage:
        chips = preprocess_sar_tif("data/gee_scenes/scene.tif")
        # then: detector.detect_scene(chips, ...)
        # or:   segmenter(chip["path"]) for each chip
    """
    tif_path = str(tif_path)
    if out_dir is None:
        out_dir = str(Path(tif_path).with_suffix("")) + "_chips"

    with rasterio.open(tif_path) as src:
        raw       = src.read().astype(np.float32)
        transform = src.transform
        crs       = src.crs

    if verbose:
        print(f"Scene: {Path(tif_path).name}  shape={raw.shape}  "
              f"range=[{raw.min():.2f}, {raw.max():.2f}]  "
              f"format={'dB' if raw.min() < -5 else 'linear'}")

    processed = preprocess_sar_bands(raw, verbose=verbose)
    chips = chip_preprocessed(processed, tif_path, out_dir,
                               chip_size, overlap, min_signal, transform, crs)

    if verbose:
        print(f"→ {len(chips)} chips saved to {out_dir}")
    return chips


# ── GEE live pull ──────────────────────────────────────────────────────────────

def fetch_gee_scene(
    bbox: list,
    date_start: str,
    date_end: str,
    out_tif: str,
    scale_m: int = 10,
) -> str:
    """
    Pull the most recent Sentinel-1 GRD IW scene from Google Earth Engine
    for the given bbox and date range, export as GeoTIFF.

    bbox: [west, south, east, north] in decimal degrees
    date_start / date_end: 'YYYY-MM-DD'
    scale_m: pixel size in metres (10 = native Sentinel-1 IW resolution)

    Requires: pip install earthengine-api geemap
              ee.Authenticate() + ee.Initialize() called before use.
    """
    try:
        import ee
        import geemap
    except ImportError:
        raise ImportError("Run: pip install earthengine-api geemap")

    west, south, east, north = bbox
    roi = ee.Geometry.Rectangle([west, south, east, north])

    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .sort("system:time_start", False)
    )
    if col.size().getInfo() == 0:
        raise ValueError(f"No Sentinel-1 IW scenes found for bbox={bbox} "
                         f"between {date_start} and {date_end}")

    image     = col.first()
    date_acq  = image.date().format("YYYY-MM-dd").getInfo()
    band_names = image.bandNames().getInfo()
    bands     = ["VV", "VH"] if "VH" in band_names else ["VV"]
    image     = image.select(bands).clip(roi)

    print(f"GEE: scene dated {date_acq}, bands={bands}")
    Path(out_tif).parent.mkdir(parents=True, exist_ok=True)
    geemap.ee_export_image(image, filename=out_tif, scale=scale_m, region=roi,
                           file_per_band=False)
    print(f"GEE export → {out_tif}")
    return out_tif


# ── End-to-end: GEE → chips ───────────────────────────────────────────────────

def gee_to_chips(
    bbox: list,
    date_start: str,
    date_end: str,
    work_dir: str = "data/gee_scenes",
    chip_size: int = 512,
    scale_m: int = 10,
) -> list:
    """
    Full pipeline: GEE pull → preprocess → chip.
    Returns chip list ready for detect_scene() or segmentation model.

    Usage:
        import ee
        ee.Authenticate()
        ee.Initialize(project="your-gee-project")

        chips = gee_to_chips(
            bbox=[12.0, 42.0, 16.0, 45.0],   # Adriatic
            date_start="2024-01-01",
            date_end="2024-01-31",
        )
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out_tif = str(Path(work_dir) / "scene.tif")
    fetch_gee_scene(bbox, date_start, date_end, out_tif, scale_m)
    return preprocess_sar_tif(out_tif, chip_size=chip_size)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="SAR preprocessing: TIF → chips")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("chip", help="Preprocess a local GeoTIFF")
    c.add_argument("tif")
    c.add_argument("--out",     default=None)
    c.add_argument("--size",    type=int, default=512)
    c.add_argument("--overlap", type=int, default=64)

    g = sub.add_parser("gee", help="Pull from GEE and chip")
    g.add_argument("--bbox",   nargs=4, type=float, metavar=("W","S","E","N"), required=True)
    g.add_argument("--start",  required=True)
    g.add_argument("--end",    required=True)
    g.add_argument("--outdir", default="data/gee_scenes")
    g.add_argument("--size",   type=int, default=512)
    g.add_argument("--scale",  type=int, default=10)

    args = p.parse_args()

    if args.cmd == "chip":
        chips = preprocess_sar_tif(args.tif, args.out, args.size, args.overlap)
        print(f"Done: {len(chips)} chips")
    elif args.cmd == "gee":
        chips = gee_to_chips(args.bbox, args.start, args.end,
                              args.outdir, args.size, args.scale)
        print(f"Done: {len(chips)} chips")
    else:
        p.print_help()
