"""
Convert HRSID COCO annotations -> YOLO OBB format, then train YOLO11m-OBB.

Usage:
    python src/train/hrsid_to_yolo.py           # convert + train
    python src/train/hrsid_to_yolo.py --convert-only   # just convert labels
    python src/train/hrsid_to_yolo.py --train-only     # skip conversion
"""

import argparse, json, shutil, sys
from pathlib import Path

import cv2
import numpy as np

HRSID_ROOT  = Path("data/vessels/hrsid/HRSID_JPG")
IMGS_SRC    = HRSID_ROOT / "JPEGImages"
ANN_TRAIN   = HRSID_ROOT / "annotations/train2017.json"
ANN_TEST    = HRSID_ROOT / "annotations/test2017.json"

OUT_ROOT    = Path("data/vessels/hrsid_yolo")
TRAIN_IMGS  = OUT_ROOT / "images/train"
TRAIN_LBLS  = OUT_ROOT / "labels/train"
VAL_IMGS    = OUT_ROOT / "images/val"
VAL_LBLS    = OUT_ROOT / "labels/val"
YAML_PATH   = OUT_ROOT / "dataset.yaml"


# ── COCO segmentation polygon -> YOLO OBB 4-corner format ─────────────────────

def seg_to_obb(seg_pts: list, img_w: int, img_h: int) -> str | None:
    """
    Fit minimum bounding rectangle to HRSID segmentation polygon.
    Returns YOLO OBB label string: '0 x1 y1 x2 y2 x3 y3 x4 y4' (normalized).
    Points follow the min-area-rect convention (already sorted by cv2.boxPoints).
    """
    pts = np.array(seg_pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    rect  = cv2.minAreaRect(pts)          # (center, (w,h), angle)
    box   = cv2.boxPoints(rect)           # 4 corners in pixel coords

    # normalize to [0,1]
    box[:, 0] /= img_w
    box[:, 1] /= img_h
    box = np.clip(box, 0.0, 1.0)

    coords = " ".join(f"{v:.6f}" for v in box.flatten())
    return f"0 {coords}"


def convert_split(ann_json: Path, out_imgs: Path, out_lbls: Path, desc: str):
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    with open(ann_json, encoding="utf-8") as f:
        data = json.load(f)

    # build image_id -> {file_name, width, height}
    id2img = {img["id"]: img for img in data["images"]}

    # group annotations by image_id
    from collections import defaultdict
    ann_by_img: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    ok = skip = 0
    for img_id, img_meta in id2img.items():
        fname = img_meta["file_name"]
        src   = IMGS_SRC / fname
        if not src.exists():
            skip += 1
            continue

        W, H = img_meta["width"], img_meta["height"]

        # write label file
        lines = []
        for ann in ann_by_img.get(img_id, []):
            seg = ann.get("segmentation")
            if not seg:
                continue
            label = seg_to_obb(seg[0], W, H)
            if label:
                lines.append(label)

        # skip negative images (no ships) to keep dataset balanced
        # keep a 10% sample of negatives for robustness
        if not lines and (img_id % 10 != 0):
            skip += 1
            continue

        stem = Path(fname).stem
        shutil.copy2(src, out_imgs / fname)
        (out_lbls / f"{stem}.txt").write_text("\n".join(lines))
        ok += 1

    print(f"  {desc}: {ok} images written, {skip} skipped")


def write_yaml():
    YAML_PATH.write_text(f"""# HRSID -> YOLO OBB dataset
path: {OUT_ROOT.resolve().as_posix()}
train: images/train
val:   images/val

nc: 1
names: ['ship']
""")
    print(f"Dataset YAML: {YAML_PATH}")


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    from ultralytics import YOLO

    # start from the HRSID-trained checkpoint (fine-tune, not scratch)
    ckpt = Path("checkpoints/vessel/hrsid_yolo11m_obb/best.pt")
    if not ckpt.exists():
        print("Checkpoint not found, starting from pretrained yolo11m-obb.pt")
        ckpt = "yolo11m-obb.pt"

    model = YOLO(str(ckpt))

    # RTX 3050 4GB — tuned to avoid OOM
    model.train(
        data=str(YAML_PATH),
        epochs=50,
        imgsz=640,           # 640 not 800 — saves ~40% VRAM
        batch=2,             # batch=2 for 4GB; if still OOM set to 1
        device=0,
        half=True,           # fp16 mandatory for 4GB
        lr0=5e-5,
        lrf=0.01,
        warmup_epochs=3,
        patience=15,
        workers=2,           # fewer workers = less RAM
        cache=False,         # don't cache dataset in RAM

        # augmentation for 10m/px domain adaptation
        scale=0.9,           # ships shrink to 5-15px during training
        fliplr=0.5,
        flipud=0.25,
        mosaic=0.0,          # mosaic OFF — loads 4 images at once, kills 4GB GPUs
        degrees=45.0,
        translate=0.2,
        hsv_v=0.3,

        project="checkpoints/vessel",
        name="hrsid_obb_10m",
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )
    print("\nTraining done. Best checkpoint:")
    print("  checkpoints/vessel/hrsid_obb_10m/weights/best.pt")
    print("\nUpdate .env:")
    print("  MARITIME_YOLO=checkpoints/vessel/hrsid_obb_10m/weights/best.pt")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert-only", action="store_true")
    ap.add_argument("--train-only",   action="store_true")
    args = ap.parse_args()

    if not args.train_only:
        print("Converting HRSID COCO -> YOLO OBB...")
        convert_split(ANN_TRAIN, TRAIN_IMGS, TRAIN_LBLS, "train")
        convert_split(ANN_TEST,  VAL_IMGS,   VAL_LBLS,   "val  ")
        write_yaml()
        print(f"Conversion done -> {OUT_ROOT}")

    if not args.convert_only:
        print("\nStarting training...")
        train()
