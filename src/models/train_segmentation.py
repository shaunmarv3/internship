"""
Oil spill segmentation — three-model benchmark on Lightning.ai H100.

Run all three in sequence (each saves its own checkpoint):

  # 1. Baseline
  python src/models/train_segmentation.py --model deeplabv3+ --epochs 50

  # 2. Comparison
  python src/models/train_segmentation.py --model segformer --epochs 50

  # 3. SOTA (install OILSAM2 first — see src/models/segmentation.py)
  python src/models/train_segmentation.py --model oilsam2 --epochs 50 --batch_size 4

Dataset layout expected (combined 2570 samples):
  data/oil/images/*.tif    ← Part I oil images
  data/oil/masks/*.tif     ← Part I oil masks  [0,1]
  data/oil/images/*.tif    ← Part II look-alike images  (merged into same folder)
  data/oil/masks/*.tif     ← Part II look-alike masks   [0] hard negatives
  data/oil/images/*.tif    ← Part II no-oil images
  data/oil/masks/*.tif     ← Part II no-oil masks       [0]

On Lightning H100: batch_size=16 for deeplabv3+/segformer, batch_size=4 for oilsam2.
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loaders import (
    get_oil_dataloaders, OilSpillDataset, get_oil_transforms,
    compute_class_weights, OIL_CLASSES_5, OIL_CLASSES_BINARY,
)
from src.models.segmentation import build_segmentation_model, DiceFocalLoss, SegmentationMetrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     default="data/oil")
    p.add_argument("--dataset_type",  default="krestenitis", choices=["krestenitis", "zenodo"])
    p.add_argument("--model",         default="deeplabv3+",  choices=["oilsam2", "segformer", "deeplabv3+", "unet"])
    p.add_argument("--backbone",      default="b2")
    p.add_argument("--img_size",      type=int, default=512)
    p.add_argument("--epochs",        type=int, default=50)
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--lr",            type=float, default=6e-5)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--wandb_project", default="maritime-oil-spill")
    p.add_argument("--no_wandb",      action="store_true")
    return p.parse_args()


def train_epoch(model, loader, optimizer, criterion, scaler, device, metrics):
    model.train()
    metrics.reset()
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        metrics.update(logits.float(), masks)

    return total_loss / len(loader), metrics.compute()


@torch.no_grad()
def val_epoch(model, loader, criterion, device, metrics):
    model.eval()
    metrics.reset()
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="val  ", leave=False):
        images, masks = images.to(device), masks.to(device)
        with autocast():
            logits = model(images)
            loss   = criterion(logits, masks)
        total_loss += loss.item()
        metrics.update(logits.float(), masks)

    return total_loss / len(loader), metrics.compute()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Dataloaders ──
    train_loader, val_loader = get_oil_dataloaders(
        root=args.data_root,
        dataset_type=args.dataset_type,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = 5 if args.dataset_type == "krestenitis" else 2
    class_names = list(OIL_CLASSES_5.values()) if num_classes == 5 else list(OIL_CLASSES_BINARY.values())

    # ── Class weights (critical for 1.2% oil imbalance) ──
    raw_train_ds = OilSpillDataset(
        args.data_root, split="train", dataset_type=args.dataset_type,
        transform=None, img_size=args.img_size,
    )
    weights = compute_class_weights(raw_train_ds, num_classes).to(device)

    # ── Model + loss + optim ──
    model = build_segmentation_model(
        model_type=args.model,
        num_classes=num_classes,
        backbone=args.backbone,
    ).to(device)

    criterion = DiceFocalLoss(num_classes=num_classes, class_weights=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()
    metrics   = SegmentationMetrics(num_classes, class_names)

    # ── W&B ──
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # ── Training loop ──
    best_oil_iou = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_m = train_epoch(model, train_loader, optimizer, criterion, scaler, device, metrics)
        val_loss,   val_m   = val_epoch(model, val_loader, criterion, device, metrics)
        scheduler.step()

        oil_iou = val_m.get("IoU_oil_spill", val_m.get("IoU_1", 0.0))
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
            f"mIoU: {val_m['mIoU']:.4f} | Oil IoU: {oil_iou:.4f}"
        )

        log = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_m}
        history.append(log)
        if not args.no_wandb:
            wandb.log(log)

        # save best checkpoint
        if oil_iou > best_oil_iou:
            best_oil_iou = oil_iou
            ckpt_path = Path(args.checkpoint_dir) / f"best_{args.model}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "oil_iou": oil_iou,
                "miou": val_m["mIoU"],
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (Oil IoU: {oil_iou:.4f}) → {ckpt_path}")

    print(f"\nTraining complete. Best Oil IoU: {best_oil_iou:.4f} | Baseline target: 0.54")
    with open(Path(args.checkpoint_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
