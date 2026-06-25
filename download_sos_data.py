#!/usr/bin/env python
"""
Download the Refined Deep-SAR Oil Spill (SOS) dataset from Zenodo.
Zenodo record: https://zenodo.org/records/15298010

Dataset: 8,070 SAR image patches (256×256) from ALOS PALSAR (Gulf of Mexico)
and Sentinel-1A (Persian Gulf). Enhanced version with ~38% training masks and
~50% validation masks manually corrected vs. the original Zhu et al. 2021 release.

Layout after extraction:
    data/sos/
      images/   ← grayscale PNG/JPEG SAR intensity chips (already preprocessed)
      masks/    ← binary PNG masks (0=background, 255=oil)

Usage:
    python download_sos_data.py            # download + extract
    python download_sos_data.py --force    # re-download even if present
"""
import argparse
import sys
import zipfile
from pathlib import Path

ZENODO_RECORD = "15298010"
FILES = {
    "images": {
        "url": f"https://zenodo.org/records/{ZENODO_RECORD}/files/images.zip",
        "md5": "e5272875611e8b5a2a0d49972224c842",
        "size_gb": 1.1,
    },
    "masks": {
        "url": f"https://zenodo.org/records/{ZENODO_RECORD}/files/masks.zip",
        "md5": "09c8a279603c8d7b73c6dc64f91e1106",
        "size_gb": 0.03,
    },
}

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "sos"


def _download(url: str, out: Path, desc: str):
    try:
        import requests
    except ImportError:
        print("pip install requests", file=sys.stderr)
        return False
    print(f"Downloading {desc} → {out.name}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    mb = done / 1e6
                    print(f"\r  {pct:.1f}%  {mb:.0f} MB", end="", flush=True)
        print()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="Re-download even if data exists")
    ap.add_argument("--dest",  default=str(DEST), help=f"Output directory (default: {DEST})")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    img_dir  = dest / "images"
    mask_dir = dest / "masks"

    existing = list(img_dir.glob("*")) if img_dir.exists() else []
    if existing and not args.force:
        print(f"SOS already present: {len(existing)} files under {dest}")
        print("Use --force to re-download.")
        return 0

    for key, meta in FILES.items():
        zip_path = dest / f"{key}.zip"
        sub_dir  = dest / key

        if not zip_path.exists() or args.force:
            ok = _download(meta["url"], zip_path, f"{key}.zip (~{meta['size_gb']:.1f} GB)")
            if not ok or not zip_path.exists():
                print(f"Download failed for {key}.zip", file=sys.stderr)
                return 1
        else:
            print(f"Found {zip_path.name} — skipping download")

        print(f"Extracting {zip_path.name} → {sub_dir}")
        sub_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.infolist()
            for m in members:
                parts = Path(m.filename).parts
                # Skip macOS resource forks: __MACOSX/ directory and ._* files
                if any(p.startswith("__MACOSX") or p.startswith("._") for p in parts):
                    continue
                if m.filename.endswith("/"):
                    continue
                # Preserve train/val subdirs; strip the leading "images"/"masks" component
                # so "images/train/palsar_1.png" → sub_dir/train/palsar_1.png
                rel_parts = parts[1:] if len(parts) > 1 else parts
                target = sub_dir / Path(*rel_parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(m))
        zip_path.unlink(missing_ok=True)

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    imgs  = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and not p.name.startswith("._")]
    masks = [p for p in mask_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and not p.name.startswith("._")]
    print(f"\nDone. {len(imgs)} images, {len(masks)} masks under {dest}")

    # Quick stem-pair check
    img_stems  = {p.stem for p in imgs}
    mask_stems = {p.stem for p in masks}
    matched = img_stems & mask_stems
    print(f"Paired (matching stems): {len(matched)} / {len(imgs)} images")
    if len(matched) < len(imgs) * 0.9:
        print("WARNING: less than 90% of images have a matching mask — check filenames.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
