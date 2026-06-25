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
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loaders import (
    get_oil_dataloaders, OilSpillDataset, get_oil_transforms,
    compute_class_weights, OIL_CLASSES_5, OIL_CLASSES_BINARY,
)
from src.models.segmentation import build_segmentation_model, DiceFocalLoss, SegmentationMetrics
from src.models.hf_utils import (
    setup_logger, log_banner, gpu_info, fmt_eta, push_to_hub, add_hf_args,
    init_wandb, add_wandb_args,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     default="data/oil")
    p.add_argument("--dataset_type",  default="krestenitis", choices=["krestenitis", "zenodo"])
    p.add_argument("--model",         default="deeplabv3+",  choices=["oilsam2", "segformer", "deeplabv3+", "unet"])
    p.add_argument("--backbone",      default="b4",
                   help="MiT backbone for SegFormer: b2 (25M), b4 (64M), b5 (82M). "
                        "b4 was the best in the 3-model benchmark; b5 may add 1-2%% OilIoU.")
    p.add_argument("--img_size",      type=int, default=512)
    p.add_argument("--epochs",        type=int, default=50)
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--lr",            type=float, default=6e-5)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--upsample_lookalike", type=float, default=2.0,
                   help="Weight multiplier for look-alike training samples (p2l_*+p2n_* stems). "
                        "2.0 = hard negatives appear ~2x per epoch → cuts FPs on Part III. "
                        "Set to 1.0 to disable. Only active for --dataset_type zenodo.")
    p.add_argument("--sos_root", default=None,
                   help="Path to Refined Deep-SAR SOS dataset root (data/sos). "
                        "If set, SOS samples are concatenated into the Zenodo training set "
                        "for cross-domain generalisation (PALSAR + Sentinel-1A + Zenodo). "
                        "Val set stays pure Zenodo for a clean benchmark.")
    p.add_argument("--checkpoint_dir", default="checkpoints/oil")
    add_wandb_args(p, default_project="maritime-oil-spill")
    add_hf_args(p)
    return p.parse_args()


def train_epoch(model, loader, optimizer, criterion, scaler, device, metrics):
    model.train()
    metrics.reset()
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
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
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, masks)
        total_loss += loss.item()
        metrics.update(logits.float(), masks)

    return total_loss / len(loader), metrics.compute()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("oil-train", logfile=str(ckpt_dir / f"train_{args.model}.log"))

    log_banner(logger, f"M3 OIL SEGMENTATION — model='{args.model}'", {
        "dataset_type":       args.dataset_type,
        "data_root":          args.data_root,
        "img_size":           args.img_size,
        "epochs":             args.epochs,
        "batch_size":         args.batch_size,
        "lr":                 args.lr,
        "upsample_lookalike": args.upsample_lookalike,
        "sos_root":           args.sos_root or "none",
        "checkpoint_dir":     str(ckpt_dir),
        **gpu_info(),
    })

    # ── Dataloaders ──
    logger.info("Building dataloaders (train/val are a disjoint deterministic split)...")
    train_loader, val_loader = get_oil_dataloaders(
        root=args.data_root,
        dataset_type=args.dataset_type,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        upsample_lookalike=args.upsample_lookalike,
        sos_root=args.sos_root,
    )
    num_classes = 5 if args.dataset_type == "krestenitis" else 2
    class_names = list(OIL_CLASSES_5.values()) if num_classes == 5 else list(OIL_CLASSES_BINARY.values())
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | "
                f"classes: {class_names}")

    # ── Class weights (critical for ~1.8% oil imbalance) ──
    logger.info("Scanning training masks for class weights (one-time)...")
    raw_train_ds = OilSpillDataset(
        args.data_root, split="train", dataset_type=args.dataset_type,
        transform=None, img_size=args.img_size,
    )
    weights = compute_class_weights(raw_train_ds, num_classes).to(device)
    logger.info(f"Class weights: {weights.detach().cpu().numpy().round(3).tolist()}")

    # ── Model + loss + optim ──
    model = build_segmentation_model(
        model_type=args.model,
        num_classes=num_classes,
        backbone=args.backbone,
    ).to(device)

    # Honest naming: if oilsam2 silently fell back to SegFormer-b4, label it as such
    # so checkpoints and the metrics table don't claim results OilSAM2 didn't produce.
    model_tag = args.model
    if args.model == "oilsam2" and getattr(model, "is_fallback", False):
        model_tag = "oilsam2_fallback_segformer_b4"
        logger.warning("OilSAM2 code unavailable — this run is SegFormer-b4. "
                       f"Checkpoints/metrics tagged '{model_tag}'.")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model '{model_tag}' built — {n_params:.1f}M params")

    criterion = DiceFocalLoss(num_classes=num_classes, class_weights=weights).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler(device.type, enabled=(device.type == "cuda"))
    metrics   = SegmentationMetrics(num_classes, class_names)

    # ── W&B ──
    run = init_wandb(args.wandb_project, model_tag,
                     config={**vars(args), "model_tag": model_tag},
                     entity=args.wandb_entity, enabled=not args.no_wandb, logger=logger)

    # ── Training loop ──
    best_oil_iou = 0.0
    best_ckpt = ckpt_dir / f"best_{model_tag}.pt"
    history = []
    t0 = time.time()
    logger.info(f"Starting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        train_loss, train_m = train_epoch(model, train_loader, optimizer, criterion, scaler, device, metrics)
        val_loss,   val_m   = val_epoch(model, val_loader, criterion, device, metrics)
        scheduler.step()

        oil_iou = val_m.get("IoU_oil_spill", val_m.get("IoU_1", 0.0))
        ep_time = time.time() - ep_t0
        eta = ep_time * (args.epochs - epoch)
        cur_lr = optimizer.param_groups[0]["lr"]
        is_best = oil_iou > best_oil_iou

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | lr {cur_lr:.2e} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"mIoU {val_m['mIoU']:.4f} | OilIoU {oil_iou:.4f}"
            f"{'  ← best' if is_best else ''} | {ep_time:.0f}s/ep | ETA {fmt_eta(eta)}"
        )

        log = {"epoch": epoch, "lr": cur_lr, "train_loss": train_loss,
               "val_loss": val_loss, "epoch_sec": ep_time, **val_m}
        history.append(log)
        if run is not None:
            run.log(log)

        # save best checkpoint (on oil-class IoU)
        if is_best:
            best_oil_iou = oil_iou
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "oil_iou": oil_iou,
                "miou": val_m["mIoU"],
                "model_tag": model_tag,
                "args": vars(args),
            }, best_ckpt)
            logger.info(f"  ✓ Saved best checkpoint (OilIoU {oil_iou:.4f}) → {best_ckpt}")

    # ── Persist history + run summary ──
    hist_path = ckpt_dir / f"history_{model_tag}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    summary = {"model_tag": model_tag, "best_oil_iou": round(best_oil_iou, 4),
               "epochs": args.epochs, "total_seconds": round(time.time() - t0, 1),
               "best_checkpoint": str(best_ckpt), "baseline_target": 0.54}
    with open(ckpt_dir / f"summary_{model_tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log_banner(logger, "TRAINING COMPLETE", {
        "best Oil IoU":  f"{best_oil_iou:.4f}  (baseline target 0.54)",
        "total time":    fmt_eta(time.time() - t0),
        "checkpoint":    str(best_ckpt),
        "history":       str(hist_path),
    })
    if run is not None:
        run.log({"best_oil_iou": best_oil_iou})
        run.finish()

    # ── Push to Hugging Face Hub (oil/ subfolder) ──
    if args.push_hf:
        logger.info(f"Pushing oil artifacts to HF repo '{args.hf_repo}' (subfolder: oil)...")
        for p in [best_ckpt, hist_path, ckpt_dir / f"summary_{model_tag}.json"]:
            push_to_hub(str(p), path_in_repo="oil", repo_id=args.hf_repo,
                        token=args.hf_token, private=not args.hf_public,
                        commit_message=f"oil/{model_tag}: OilIoU={best_oil_iou:.4f}",
                        logger=logger)


if __name__ == "__main__":
    main()
