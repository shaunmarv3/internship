#!/usr/bin/env python
"""Pull ONE Sentinel-1 GRD scene from Google Earth Engine to a local GeoTIFF.

The output .tif is what you drag-drop into the dashboard (or pass to
backend/pipeline/run_scene.py --scene-tif). Both the oil model and the ship
model run on this single georeferenced scene.

Setup (one-time, free):
    pip install earthengine-api geemap
    earthengine authenticate            # opens a browser

Usage (pick a known spill so you have ground truth — e.g. Mauritius 2020):
    python pull_scene.py --bbox 57.4 -20.6 57.9 -20.3 \
        --start 2020-08-06 --end 2020-08-16 \
        --out data/scenes/mauritius.tif --project <your-gee-project>
"""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", default="data/scenes/scene.tif")
    p.add_argument("--scale", type=int, default=10, help="metres/pixel (10 = native IW)")
    p.add_argument("--project", default=None, help="your Google Earth Engine cloud project id")
    p.add_argument("--list", action="store_true",
                   help="don't download — just list S1 scenes in the window with bbox coverage %%, "
                        "so you can pick a date whose swath actually covers the target")
    args = p.parse_args()

    import ee
    try:
        ee.Initialize(project=args.project) if args.project else ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=args.project) if args.project else ee.Initialize()

    from src.data.sar_preprocess import fetch_gee_scene, list_gee_scenes

    if args.list:
        rows = list_gee_scenes(args.bbox, args.start, args.end)
        if not rows:
            raise SystemExit(f"No Sentinel-1 IW scenes for bbox={args.bbox} in "
                             f"[{args.start}, {args.end}). Widen the window (EE end is exclusive; "
                             "S1 revisits a spot only every ~6-12 days).")
        print(f"\n{len(rows)} S1 IW scene(s) over bbox={args.bbox}  (coverage = % of bbox in swath)")
        print(f"{'date':<12}{'orbit':<12}{'coverage':>9}")
        for r in rows:
            print(f"{r['date']:<12}{str(r.get('orbit','?')):<12}{r.get('cov',0)*100:>8.0f}%")
        best = rows[0]
        print(f"\nBest: {best['date']} ({best.get('cov',0)*100:.0f}% coverage). Re-run WITHOUT --list "
              f"using --start {best['date']} --end <next day> to download it.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fetch_gee_scene(args.bbox, args.start, args.end, args.out, args.scale)

    out = Path(args.out)
    if not out.exists() or out.stat().st_size < 1024:
        raise SystemExit(
            "\nDownload failed — the scene exceeded GEE's 50 MB direct-download cap.\n"
            f"Re-run with a coarser --scale (you used {args.scale}; try {args.scale * 2} or 30)\n"
            "or a smaller --bbox."
        )
    print(f"\nSaved scene -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print("Next: drag it into the dashboard, or run")
    print(f"  python -m backend.pipeline.run_scene --scene-tif {args.out} --id myscene "
          f"--title 'My scene' --acquired {args.start}T00:00:00Z "
          f"--yolo checkpoints/vessel/hrsid_yolo11m_obb/best.pt "
          f"--segformer checkpoints/oil/best_segformer.pt")


if __name__ == "__main__":
    main()
