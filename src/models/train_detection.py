"""
SAR Vessel Detection — 3-model benchmark training script.
Models: YOLOv8m (baseline) → YOLOv11m-OBB (oriented) → RT-DETR-L (transformer)
Dataset: HRSID — 4,042 train / 1,962 val, 800×800 JPG, single class: ship

Run order on Lightning.ai H100:
  python src/models/train_detection.py --model yolov8m
  python src/models/train_detection.py --model yolo11m-obb   (needs OBB labels)
  python src/models/train_detection.py --model rtdetr-l      --batch 8

OBB label generation (run once before yolo11m-obb):
  python src/models/train_detection.py --convert_obb --coco_json /path/to/train2017.json
"""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ── OBB conversion ─────────────────────────────────────────────────────────────

def polygon_to_obb(polygon: list) -> tuple:
    """
    Fit minimum enclosing rotated rectangle to a COCO polygon.
    Returns (cx, cy, w, h, angle_deg) in pixel coords.
    polygon: flat list [x1,y1,x2,y2,...] from COCO segmentation
    """
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(pts)   # ((cx,cy), (w,h), angle)
    (cx, cy), (w, h), angle = rect
    # OpenCV angle convention: normalise so w >= h
    if w < h:
        w, h = h, w
        angle += 90
    angle = angle % 180
    return cx, cy, w, h, angle


def convert_coco_to_obb(coco_json: str, img_dir: str, out_label_dir: str):
    """
    Convert HRSID COCO polygon annotations → YOLO OBB format.
    OBB label format per line: class cx cy w h angle  (all normalised [0,1] except angle in degrees)
    Ultralytics OBB format: class x1 y1 x2 y2 x3 y3 x4 y4 (normalised corner points)
    """
    out_label_dir = Path(out_label_dir)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json) as f:
        coco = json.load(f)

    img_map = {img['id']: img for img in coco['images']}

    from collections import defaultdict
    ann_map = defaultdict(list)
    for ann in coco['annotations']:
        if ann.get('segmentation'):
            ann_map[ann['image_id']].append(ann)

    written = 0
    for img_id, img_info in img_map.items():
        W, H = img_info['width'], img_info['height']
        label_path = out_label_dir / (Path(img_info['file_name']).stem + '.txt')
        lines = []
        for ann in ann_map[img_id]:
            for seg in ann['segmentation']:
                if len(seg) < 6:
                    continue
                cx, cy, w, h, angle = polygon_to_obb(seg)
                # convert to Ultralytics OBB 4-corner format (normalised)
                rect = ((cx, cy), (w, h), angle)
                corners = cv2.boxPoints(rect)   # 4×2 pixel coords
                corners_norm = corners / np.array([W, H])
                line = "0 " + " ".join(f"{v:.6f}" for v in corners_norm.flatten())
                lines.append(line)

        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))
        written += 1

    print(f"OBB labels written: {written} files → {out_label_dir}")


# ── Training ───────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "yolov8m": {
        "weights":  "yolov8m.pt",
        "task":     "detect",
        "imgsz":    800,
        "batch":    16,
        "epochs":   50,
        "name":     "hrsid_yolov8m",
    },
    "yolo11m-obb": {
        "weights":  "yolo11m-obb.pt",
        "task":     "obb",
        "imgsz":    800,
        "batch":    16,
        "epochs":   50,
        "name":     "hrsid_yolo11m_obb",
    },
    "rtdetr-l": {
        "weights":  "rtdetr-l.pt",
        "task":     "detect",
        "imgsz":    800,
        "batch":    8,
        "epochs":   50,
        "name":     "hrsid_rtdetr_l",
    },
}


def train(model_key: str, data_yaml: str, project: str = "checkpoints/vessel"):
    cfg = MODEL_CONFIGS[model_key]
    model = YOLO(cfg["weights"])
    results = model.train(
        data=data_yaml,
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        project=project,
        name=cfg["name"],
        device=0,
        patience=15,
        save=True,
        plots=True,
        verbose=True,
    )
    print(f"\n[{model_key}] Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    return results


# ── Inference (scene-level) ────────────────────────────────────────────────────

def detect_vessels(model_path: str, chips: list, conf: float = 0.25) -> list:
    """
    Run detection on chipped tiles, return scene-level detections.
    Reuses chip_sar_scene() from detection.py for chipping.
    """
    model = YOLO(model_path)
    detections = []
    for chip in chips:
        results = model(chip["path"], conf=conf, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            ox, oy = chip["x"], chip["y"]
            for (x1, y1, x2, y2), c in zip(boxes, confs):
                detections.append({
                    "scene_x":    ox + (x1 + x2) / 2,
                    "scene_y":    oy + (y1 + y2) / 2,
                    "width_px":   x2 - x1,
                    "height_px":  y2 - y1,
                    "conf":       float(c),
                    "class_name": "vessel",
                })
    return detections


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="yolov8m",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--data",        default="data/vessels/HRSID_yolo/data.yaml")
    parser.add_argument("--project",     default="checkpoints/vessel")
    parser.add_argument("--convert_obb", action="store_true",
                        help="Convert COCO polygons → OBB labels (run once before yolo11m-obb)")
    parser.add_argument("--coco_json",   default="",
                        help="Path to COCO JSON for OBB conversion")
    parser.add_argument("--obb_out",     default="data/vessels/HRSID_obb/labels/train")
    args = parser.parse_args()

    if args.convert_obb:
        if not args.coco_json:
            raise ValueError("--coco_json required with --convert_obb")
        convert_coco_to_obb(
            coco_json=args.coco_json,
            img_dir="",
            out_label_dir=args.obb_out,
        )
    else:
        train(args.model, args.data, args.project)
