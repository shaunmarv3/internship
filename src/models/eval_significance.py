"""
Statistical significance evaluation for the oil-segmentation benchmark.

Two modes:

1. `infer` — load ONE checkpoint, run the held-out Part III set (data/oil_test),
   and dump PER-IMAGE confusion counts (tp/fp/fn for the oil class, plus
   background counts for mIoU) at a fixed threshold. CPU is fine (inference
   only, ~15-30 min per model).

       python src/models/eval_significance.py infer \
           --checkpoint checkpoints/oil/best_segformer.pt \
           --out checkpoints/oil/per_image_segformer_b5.json

2. `stats` — given two per-image JSONs (model A vs model B), compute:
   - pooled OilIoU / mIoU per model with bootstrap 95% CIs (resampling images),
   - paired Wilcoxon signed-rank test on per-image oil IoU, restricted to the
     images whose ground truth contains oil (per-image IoU is undefined on
     oil-free scenes; those scenes still influence the pooled/bootstrap figures).

       python src/models/eval_significance.py stats \
           --a checkpoints/oil/per_image_segformer_b5.json \
           --b checkpoints/oil/per_image_deeplab.json

No training happens anywhere in this script.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ── mode 1: per-image inference dump ─────────────────────────────────────────

def run_infer(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.data.loaders import OilSpillDataset, get_oil_transforms
    from src.models.segmentation import build_segmentation_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved = ckpt.get("args", {})
    model_tag = ckpt.get("model_tag", saved.get("model", "segformer"))
    backbone = saved.get("backbone", args.backbone)

    model = build_segmentation_model(model_tag, num_classes=2, backbone=backbone).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {args.checkpoint}  model_tag={model_tag}  backbone={backbone}")

    ds = OilSpillDataset(args.test_data, split="all", dataset_type="zenodo",
                         transform=get_oil_transforms(args.img_size, "val"),
                         img_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    print(f"Test set: {len(ds)} samples  threshold={args.threshold}")

    rows = []
    idx = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="inference"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)[:, 1].cpu()          # B×H×W
            preds = (probs > args.threshold).long()
            for b in range(preds.shape[0]):
                p, g = preds[b], masks[b].long()
                rows.append({
                    "index": idx,
                    "oil_tp": int(((p == 1) & (g == 1)).sum()),
                    "oil_fp": int(((p == 1) & (g == 0)).sum()),
                    "oil_fn": int(((p == 0) & (g == 1)).sum()),
                    "bg_tp":  int(((p == 0) & (g == 0)).sum()),
                    "gt_oil_px": int((g == 1).sum()),
                })
                idx += 1

    payload = {"checkpoint": str(args.checkpoint), "model_tag": model_tag,
               "backbone": backbone, "threshold": args.threshold,
               "test_data": args.test_data, "per_image": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    pooled = _pooled_metrics(rows)
    print(f"Pooled OilIoU={pooled['oil_iou']:.4f}  mIoU={pooled['miou']:.4f}")
    print(f"Saved -> {args.out}")


# ── metric helpers ────────────────────────────────────────────────────────────

def _pooled_metrics(rows):
    """Pixel-pooled OilIoU and mIoU over a list of per-image rows."""
    tp = sum(r["oil_tp"] for r in rows)
    fp = sum(r["oil_fp"] for r in rows)
    fn = sum(r["oil_fn"] for r in rows)
    bg_tp = sum(r["bg_tp"] for r in rows)
    oil_iou = tp / (tp + fp + fn + 1e-9)
    # background IoU: bg_tp / (bg_tp + bg_fp + bg_fn) where bg_fp = oil_fn, bg_fn = oil_fp
    bg_iou = bg_tp / (bg_tp + fn + fp + 1e-9)
    return {"oil_iou": oil_iou, "miou": (oil_iou + bg_iou) / 2}


def _per_image_iou(rows):
    """Per-image oil IoU, only for images whose GT contains oil."""
    out = {}
    for r in rows:
        if r["gt_oil_px"] > 0:
            out[r["index"]] = r["oil_tp"] / (r["oil_tp"] + r["oil_fp"] + r["oil_fn"] + 1e-9)
    return out


def _bootstrap_ci(rows, n_boot=10000, seed=0):
    """95% CI of the pooled OilIoU under image-level resampling."""
    rng = np.random.default_rng(seed)
    rows = list(rows)
    n = len(rows)
    vals = []
    for _ in range(n_boot):
        sample = [rows[i] for i in rng.integers(0, n, n)]
        vals.append(_pooled_metrics(sample)["oil_iou"])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ── mode 2: paired statistics ────────────────────────────────────────────────

def run_stats(args):
    from scipy.stats import wilcoxon

    a = json.load(open(args.a, encoding="utf-8"))
    b = json.load(open(args.b, encoding="utf-8"))
    name_a = a.get("model_tag", "A") + "-" + str(a.get("backbone", ""))
    name_b = b.get("model_tag", "B") + "-" + str(b.get("backbone", ""))

    for name, d in ((name_a, a), (name_b, b)):
        pooled = _pooled_metrics(d["per_image"])
        lo, hi = _bootstrap_ci(d["per_image"], n_boot=args.n_boot)
        print(f"{name:>24}: pooled OilIoU={pooled['oil_iou']:.4f} "
              f"(95% bootstrap CI [{lo:.4f}, {hi:.4f}])  mIoU={pooled['miou']:.4f}")

    iou_a = _per_image_iou(a["per_image"])
    iou_b = _per_image_iou(b["per_image"])
    common = sorted(set(iou_a) & set(iou_b))
    xs = np.array([iou_a[i] for i in common])
    ys = np.array([iou_b[i] for i in common])
    stat, p = wilcoxon(xs, ys)
    print(f"\nPaired Wilcoxon signed-rank over {len(common)} oil-bearing images:")
    print(f"  median per-image IoU: {name_a}={np.median(xs):.4f}  {name_b}={np.median(ys):.4f}")
    print(f"  W={stat:.1f}  p={p:.2e}  "
          f"({'significant' if p < 0.05 else 'not significant'} at alpha=0.05)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("infer", help="Dump per-image confusion counts for one checkpoint")
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--out", required=True, help="Output per-image JSON path")
    i.add_argument("--test_data", default="data/oil_test")
    i.add_argument("--backbone", default="b4")
    i.add_argument("--img_size", type=int, default=512)
    i.add_argument("--batch_size", type=int, default=2)
    i.add_argument("--num_workers", type=int, default=2)
    i.add_argument("--threshold", type=float, default=0.5)

    s = sub.add_parser("stats", help="Bootstrap CIs + paired Wilcoxon between two dumps")
    s.add_argument("--a", required=True, help="per-image JSON of model A")
    s.add_argument("--b", required=True, help="per-image JSON of model B")
    s.add_argument("--n_boot", type=int, default=10000)

    args = ap.parse_args()
    run_infer(args) if args.cmd == "infer" else run_stats(args)


if __name__ == "__main__":
    main()
