#!/usr/bin/env python
"""
Download the HRSID vessel-detection SAR dataset into data/vessels/hrsid.

HRSID (High-Resolution SAR Images Dataset) ships in COCO format:
  - *.jpg            : 800x800 SAR image chips
  - train2017.json   : COCO annotations (single class, 0 = ship)
  - test2017.json    : COCO annotations for the test split

Usage:
    python download_vessel_data.py            # download + extract (skips if present)
    python download_vessel_data.py --force    # re-download even if it exists

After it finishes, browse every image with:
    python data/vessels/view_hrsid.py            # annotated overview of busiest scenes
    python data/vessels/view_hrsid.py --gallery  # build an HTML gallery of ALL images
"""
import argparse
import sys
import zipfile
from pathlib import Path

GDRIVE_FILE_ID = "1NY3ovgc-woDlNoQdyqzRB3t9McOBH5Ms"
ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "vessels" / "hrsid"
ZIP_PATH = ROOT / "data" / "vessels" / "hrsid.zip"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if data exists")
    ap.add_argument("--keep-zip", action="store_true", help="keep hrsid.zip after extraction")
    args = ap.parse_args()

    try:
        import gdown
    except ImportError:
        print("gdown not installed. Run:  pip install gdown", file=sys.stderr)
        return 1

    DEST.parent.mkdir(parents=True, exist_ok=True)

    existing_jpgs = list(DEST.rglob("*.jpg"))
    if existing_jpgs and not args.force:
        print(f"HRSID already present: {len(existing_jpgs)} images under {DEST}")
        print("Use --force to re-download.")
        return 0

    # 1. Download the zip from Google Drive
    if not ZIP_PATH.exists() or args.force:
        print(f"Downloading HRSID (~1.5 GB) from Google Drive -> {ZIP_PATH}")
        gdown.download(id=GDRIVE_FILE_ID, output=str(ZIP_PATH), quiet=False)
    else:
        print(f"Found existing zip: {ZIP_PATH}")

    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size == 0:
        print("Download failed: zip missing or empty.", file=sys.stderr)
        return 1

    # 2. Extract
    print(f"Extracting -> {DEST}")
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DEST)

    jpgs = list(DEST.rglob("*.jpg"))
    jsons = list(DEST.rglob("*.json"))
    print(f"Done. {len(jpgs)} images, {len(jsons)} annotation files under {DEST}")
    for j in jsons:
        print(f"  annotation: {j.relative_to(ROOT)}")

    # 3. Clean up the zip unless asked to keep it
    if not args.keep_zip:
        ZIP_PATH.unlink(missing_ok=True)
        print(f"Removed {ZIP_PATH.name} (use --keep-zip to retain it).")

    if not jpgs:
        print("WARNING: no .jpg images found after extraction.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
