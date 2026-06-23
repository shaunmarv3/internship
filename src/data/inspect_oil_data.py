#!/usr/bin/env python
"""
Pre-flight inspector for the Zenodo oil dataset — RUN THIS BEFORE TRAINING.

It answers the one question the plan flags as unverified: are the SAR images in
**dB** or **linear** scale? It also checks mask validity, image/mask pairing, band
count and NaNs — the things that silently wreck M3 if wrong.

You do NOT need the full 40 GB to run this. Download a handful of real .tif images
into <data_root>/images/ (+ matching masks in <data_root>/masks/) and point at it:

    python src/data/inspect_oil_data.py --data_root data/oil --n 8

Needs only rasterio + numpy (no torch), so you can run it before the heavy install.
Diagnostic only — never writes anything, always exits 0.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio


def stat_line(name, arr):
    return (f"  {name:<10} min={arr.min():10.3f}  max={arr.max():10.3f}  "
            f"mean={arr.mean():9.3f}  p1={np.percentile(arr,1):9.3f}  "
            f"p99={np.percentile(arr,99):9.3f}")


def classify(minv):
    return "dB" if minv < -5.0 else "linear"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/oil",
                    help="folder with images/ and masks/ subdirs (convenience default)")
    ap.add_argument("--images_dir", default=None,
                    help="explicit image folder (overrides <data_root>/images) — "
                         "use this to point at each Zenodo part separately")
    ap.add_argument("--masks_dir", default=None,
                    help="explicit mask folder (overrides <data_root>/masks; optional)")
    ap.add_argument("--label", default=None, help="tag for the printout (e.g. 'Part I')")
    ap.add_argument("--n", type=int, default=8, help="how many samples to inspect")
    args = ap.parse_args()

    root = Path(args.data_root)
    img_dir = Path(args.images_dir) if args.images_dir else root / "images"
    msk_dir = Path(args.masks_dir) if args.masks_dir else root / "masks"
    print("=" * 70)
    print(f"  OIL DATA PRE-FLIGHT — {args.label or img_dir}")
    print("=" * 70)

    if not img_dir.exists():
        print(f"!! No images/ dir at {img_dir}. Download a few .tif images there first.")
        sys.exit(0)

    images = sorted(img_dir.glob("*.tif")) + sorted(img_dir.glob("*.tiff"))
    masks_all = {p.stem: p for p in list(msk_dir.glob("*")) } if msk_dir.exists() else {}
    print(f"images found : {len(images)}")
    print(f"masks found  : {len(masks_all)}")
    if not images:
        print("!! No .tif images found — nothing to inspect.")
        sys.exit(0)

    # ── pairing check ──
    paired = [p for p in images if p.stem in masks_all]
    print(f"paired (img has matching mask by stem): {len(paired)}/{len(images)}")
    if len(paired) < len(images):
        missing = [p.name for p in images if p.stem not in masks_all][:5]
        print(f"  e.g. images with NO mask: {missing}")

    sample = (paired or images)[: args.n]
    print(f"\nInspecting {len(sample)} sample(s):\n" + "-" * 70)

    fmts, band_counts, has_nan = [], set(), False
    oil_fracs, mask_dtypes, mask_uniques = [], set(), set()

    for p in sample:
        with rasterio.open(p) as src:
            d = src.read().astype("float32")        # C×H×W
            nod = src.nodata
        band_counts.add(d.shape[0])
        if np.isnan(d).any():
            has_nan = True
        fmt = classify(float(d.min()))
        fmts.append(fmt)
        print(f"IMG {p.name}  shape={tuple(d.shape)}  nodata={nod}  -> {fmt.upper()}")
        print(stat_line("band0", d[0]))

        # matching mask
        mp = masks_all.get(p.stem)
        if mp is not None:
            with rasterio.open(mp) as ms:
                m = ms.read(1)
            u = np.unique(m)
            mask_dtypes.add(str(m.dtype))
            mask_uniques.update(u.tolist()[:10])
            frac = float((m > 0).mean()) * 100
            oil_fracs.append(frac)
            print(f"MASK {mp.name}  dtype={m.dtype}  unique={u.tolist()[:6]}  oil={frac:.2f}%")
        print("-" * 70)

    # ── verdict ──
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    uniq_fmt = set(fmts)
    print(f"image bands seen : {sorted(band_counts)}")
    print(f"NaNs present     : {has_nan}")
    if oil_fracs:
        print(f"mask dtypes      : {sorted(mask_dtypes)}")
        print(f"mask values seen : {sorted(mask_uniques)}")
        print(f"oil pixel %      : min={min(oil_fracs):.2f}  mean={np.mean(oil_fracs):.2f}  max={max(oil_fracs):.2f}")

    if len(uniq_fmt) > 1:
        print("\n⚠️  MIXED formats across samples (both dB and linear) — inspect more files; "
              "the dataset may not be uniform.")
    elif uniq_fmt == {"dB"}:
        print("\n➜ Format = dB. loaders.py currently does per-image MIN-MAX only (no dB->linear),")
        print("  while sar_preprocess.py (live/GEE) DOES convert dB->linear. That's a train/serve")
        print("  mismatch. Recommend: add a dB->linear step in loaders so train matches inference.")
        print("  Tell me 'it's dB' and I'll wire that in before you train.")
    else:
        print("\n➜ Format = linear. loaders.py min-max normalization is appropriate as-is.")
        print("  No dB conversion needed for training. Safe to proceed.")

    bad_mask = mask_uniques and not set(mask_uniques).issubset({0, 1})
    if bad_mask:
        print("\n⚠️  Mask values are not strictly {0,1} — check label encoding before training.")
    print("=" * 70)


if __name__ == "__main__":
    main()
