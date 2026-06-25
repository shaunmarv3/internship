"""
Post-training threshold sweep for oil segmentation.

Loads a SegFormer checkpoint and evaluates it on Part III (data/oil_test) across a
range of oil-class probability thresholds. Identifies the threshold that maximises
OilIoU on the held-out test set without any retraining.

Background: the model was trained to output argmax (implicit threshold = 0.5).
Look-alike samples (Part III p3l_*) tend to produce oil-class probabilities of
0.3–0.6 — these are false positives that hurt test OilIoU. Raising the threshold
above 0.5 trades recall for precision and can recover several IoU points.

Usage:
    python src/models/eval_threshold.py \\
        --checkpoint checkpoints/oil/best_segformer.pt \\
        --test_data  data/oil_test

    # sweep finer grid around a promising region:
    python src/models/eval_threshold.py --thresholds 0.55 0.60 0.65 0.70 0.75
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loaders import OilSpillDataset, get_oil_transforms
from src.models.segmentation import build_segmentation_model
from src.models.hf_utils import setup_logger


def _oil_iou_at_threshold(probs_oil: torch.Tensor, masks: torch.Tensor, t: float):
    """All tensors on CPU. Returns (iou, precision, recall)."""
    pred = (probs_oil > t).long()
    tp = ((pred == 1) & (masks == 1)).sum().float()
    fp = ((pred == 1) & (masks == 0)).sum().float()
    fn = ((pred == 0) & (masks == 1)).sum().float()
    iou  = tp / (tp + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    return iou.item(), prec.item(), rec.item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/oil/best_segformer.pt",
                    help="Path to the .pt checkpoint saved by train_segmentation.py")
    ap.add_argument("--test_data",  default="data/oil_test",
                    help="Root of the held-out Part III dataset (images/ + masks/ sub-dirs)")
    ap.add_argument("--backbone",   default="b4",
                    help="MiT backbone used in the checkpoint (b2/b4/b5). "
                         "Auto-detected from checkpoint args if present.")
    ap.add_argument("--img_size",   type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.30, 0.35, 0.40, 0.45, 0.50,
                             0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
                    help="Oil-class probability thresholds to evaluate")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: <checkpoint_dir>/threshold_sweep.json)")
    args = ap.parse_args()

    logger = setup_logger("threshold-sweep")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ──
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return
    ckpt = torch.load(str(ckpt_path), map_location=device)

    saved_args = ckpt.get("args", {})
    backbone = saved_args.get("backbone", args.backbone)
    model_tag = ckpt.get("model_tag", "segformer")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"  model_tag={model_tag}  backbone={backbone}  "
                f"val_OilIoU={ckpt.get('oil_iou', '?'):.4f}  epoch={ckpt.get('epoch', '?')}")

    model = build_segmentation_model("segformer", num_classes=2, backbone=backbone).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Test dataset (Part III) ──
    ds = OilSpillDataset(
        args.test_data, split="all", dataset_type="zenodo",
        transform=get_oil_transforms(args.img_size, "val"),
        img_size=args.img_size,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    logger.info(f"Test set: {len(ds)} samples")

    # ── Collect oil-class probabilities for all pixels ──
    all_probs: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Inference on test set"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)[:, 1].cpu()  # B×H×W, oil-class prob
            all_probs.append(probs)
            all_masks.append(masks.cpu())

    all_probs = torch.cat(all_probs)   # N×H×W
    all_masks = torch.cat(all_masks)   # N×H×W

    oil_px = (all_masks == 1).sum().item()
    total_px = all_masks.numel()
    logger.info(f"Oil pixels: {oil_px:,} / {total_px:,} ({100*oil_px/total_px:.2f}%)")

    # ── Threshold sweep ──
    header = f"\n{'Threshold':>10}  {'OilIoU':>8}  {'Precision':>10}  {'Recall':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    sweep = []
    best_iou, best_t = 0.0, 0.5
    for t in sorted(args.thresholds):
        iou, prec, rec = _oil_iou_at_threshold(all_probs, all_masks, t)
        marker = "  ← default" if abs(t - 0.5) < 1e-9 else ""
        logger.info(f"{t:>10.2f}  {iou:>8.4f}  {prec:>10.4f}  {rec:>8.4f}{marker}")
        sweep.append({"threshold": t, "oil_iou": round(iou, 4),
                      "precision": round(prec, 4), "recall": round(rec, 4)})
        if iou > best_iou:
            best_iou, best_t = iou, t

    logger.info(f"\nBest threshold: {best_t:.2f}  →  OilIoU = {best_iou:.4f}  "
                f"(vs default 0.5: {next(r['oil_iou'] for r in sweep if r['threshold']==0.5):.4f})")

    out_path = Path(args.out) if args.out else ckpt_path.parent / "threshold_sweep.json"
    payload = {
        "checkpoint":       str(ckpt_path),
        "test_data":        args.test_data,
        "best_threshold":   best_t,
        "best_oil_iou":     round(best_iou, 4),
        "sweep":            sweep,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
