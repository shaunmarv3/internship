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
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.hf_utils import (
    setup_logger, log_banner, gpu_info, fmt_eta, push_to_hub, add_hf_args, add_wandb_args,
)


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


def train(model_key: str, data_yaml: str, project: str = "checkpoints/vessel",
          push_hf: bool = False, hf_repo: str = None, hf_token: str = None,
          hf_public: bool = False, no_wandb: bool = False,
          wandb_project: str = "maritime-vessel", wandb_entity: str = None,
          epochs: int = None, imgsz: int = None, batch: int = None,
          device: str = "0"):
    cfg = dict(MODEL_CONFIGS[model_key])          # copy so overrides don't mutate the global
    if epochs is not None: cfg["epochs"] = epochs   # overrides (e.g. for smoke tests)
    if imgsz  is not None: cfg["imgsz"]  = imgsz
    if batch  is not None: cfg["batch"]  = batch
    logger = setup_logger("vessel-train", logfile=str(Path(project) / f"{cfg['name']}.log"))

    log_banner(logger, f"M1 VESSEL DETECTION — {model_key}  (task={cfg['task']})", {
        "weights":  cfg["weights"],
        "data":     data_yaml,
        "imgsz":    cfg["imgsz"],
        "batch":    cfg["batch"],
        "epochs":   cfg["epochs"],
        "project":  project,
        "run_name": cfg["name"],
        "device":   device,
        "wandb":    "off" if no_wandb else wandb_project,
        **gpu_info(),
    })

    # Ultralytics has a native W&B integration — enabling it auto-logs the loss/mAP
    # curves, PR/confusion plots and the best model. Project/entity via env vars.
    if not no_wandb:
        try:
            import wandb  # noqa: F401
            from ultralytics import settings as ul_settings
            ul_settings.update({"wandb": True})
            os.environ.setdefault("WANDB_PROJECT", wandb_project)
            if wandb_entity:
                os.environ.setdefault("WANDB_ENTITY", wandb_entity)
            logger.info(f"W&B enabled for Ultralytics (project={wandb_project}). "
                        "Auth: `wandb login` or WANDB_API_KEY.")
        except Exception as e:
            logger.warning(f"Could not enable W&B for Ultralytics: {e}")
    else:
        try:
            from ultralytics import settings as ul_settings
            ul_settings.update({"wandb": False})
        except Exception:
            pass

    t0 = time.time()
    model = YOLO(cfg["weights"])
    results = model.train(
        data=data_yaml,
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        project=project,
        name=cfg["name"],
        device=device,           # "0" single-GPU, "0,1" multi-GPU DDP (e.g. Kaggle 2xT4)
        patience=15,
        save=True,
        plots=True,
        verbose=True,
    )

    # Ultralytics writes weights to <project>/<name>/weights/{best,last}.pt
    rd = results.results_dict
    map50    = rd.get("metrics/mAP50(B)", rd.get("metrics/mAP50", "N/A"))
    map5095  = rd.get("metrics/mAP50-95(B)", rd.get("metrics/mAP50-95", "N/A"))
    prec     = rd.get("metrics/precision(B)", rd.get("metrics/precision", "N/A"))
    rec      = rd.get("metrics/recall(B)", rd.get("metrics/recall", "N/A"))
    save_dir = Path(getattr(model.trainer, "save_dir", Path(project) / cfg["name"]))
    best_pt  = Path(getattr(model.trainer, "best", save_dir / "weights" / "best.pt"))

    def f(x): return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
    log_banner(logger, f"DONE — {model_key}", {
        "mAP50":      f(map50),
        "mAP50-95":   f(map5095),
        "precision":  f(prec),
        "recall":     f(rec),
        "total time": fmt_eta(time.time() - t0),
        "best.pt":    str(best_pt),
        "run dir":    str(save_dir),
    })

    if push_hf:
        logger.info(f"Pushing vessel run to HF '{hf_repo}' (subfolder: vessel/{cfg['name']})...")
        # push best weights + the results CSV/plots folder for reproducibility
        push_to_hub(str(best_pt), path_in_repo=f"vessel/{cfg['name']}", repo_id=hf_repo,
                    token=hf_token, private=not hf_public,
                    commit_message=f"vessel/{cfg['name']}: mAP50={f(map50)}", logger=logger)
        results_csv = save_dir / "results.csv"
        if results_csv.exists():
            push_to_hub(str(results_csv), path_in_repo=f"vessel/{cfg['name']}", repo_id=hf_repo,
                        token=hf_token, private=not hf_public,
                        commit_message=f"vessel/{cfg['name']}: training curve", logger=logger)
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
    parser.add_argument("--epochs", type=int, default=None, help="override cfg epochs")
    parser.add_argument("--imgsz",  type=int, default=None, help="override cfg imgsz")
    parser.add_argument("--batch",  type=int, default=None, help="override cfg batch")
    parser.add_argument("--device", default="0",
                        help="GPU id(s): '0' single, '0,1' multi-GPU DDP (e.g. Kaggle 2xT4)")
    add_wandb_args(parser, default_project="maritime-vessel")
    add_hf_args(parser)
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
        train(args.model, args.data, args.project,
              push_hf=args.push_hf, hf_repo=args.hf_repo,
              hf_token=args.hf_token, hf_public=args.hf_public,
              no_wandb=args.no_wandb, wandb_project=args.wandb_project,
              wandb_entity=args.wandb_entity,
              epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
              device=args.device)
