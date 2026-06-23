"""
SAR vessel detection — YOLOv8/v11 on chipped tiles.
Handles: tile chipping from full-scene SAR, training, inference + dark-vessel output.
"""

import numpy as np
from pathlib import Path
import torch
import rasterio
from rasterio.windows import Window
from ultralytics import YOLO


# ── Tile chipping ──────────────────────────────────────────────────────────────

def chip_sar_scene(
    scene_path: str,
    out_dir: str,
    chip_size: int = 640,
    overlap: int = 64,
    min_brightness: float = 0.01,
):
    """
    Chip a full-scene Sentinel-1 GRD GeoTIFF into fixed-size tiles.
    Skips dark tiles (ocean-only, no vessels) to reduce dataset imbalance.
    Returns list of chip file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chips = []

    with rasterio.open(scene_path) as src:
        W, H = src.width, src.height
        stride = chip_size - overlap
        chip_idx = 0

        for y in range(0, H - chip_size + 1, stride):
            for x in range(0, W - chip_size + 1, stride):
                window = Window(x, y, chip_size, chip_size)
                data = src.read(window=window).astype(np.float32)   # C×H×W

                # normalize
                for c in range(data.shape[0]):
                    b = data[c]
                    vmin, vmax = b.min(), b.max()
                    if vmax > vmin:
                        data[c] = (b - vmin) / (vmax - vmin)

                # skip near-empty tiles
                if data.mean() < min_brightness:
                    continue

                # save as 3-channel PNG (replicate bands if needed)
                if data.shape[0] == 1:
                    rgb = np.repeat(data, 3, axis=0)
                elif data.shape[0] == 2:
                    rgb = np.concatenate([data, data[:1]], axis=0)
                else:
                    rgb = data[:3]

                rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
                chip_path = out_dir / f"chip_{chip_idx:05d}.png"
                import cv2
                cv2.imwrite(str(chip_path), rgb_uint8.transpose(1, 2, 0)[:, :, ::-1])
                chips.append({"path": str(chip_path), "x": x, "y": y,
                               "scene": scene_path, "chip_size": chip_size})
                chip_idx += 1

    print(f"Chipped {scene_path} → {chip_idx} tiles in {out_dir}")
    return chips


# ── YOLO detector ──────────────────────────────────────────────────────────────

class SARVesselDetector:
    """
    YOLOv8/v11 wrapper for SAR vessel detection.
    Two classes: 0=vessel, 1=fixed_infrastructure
    """

    CLASS_NAMES = ["vessel", "fixed_infrastructure"]

    def __init__(self, model_path: str = "yolov8m.pt", conf_thresh: float = 0.25):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        img_size: int = 640,
        batch: int = 16,
        project: str = "checkpoints",
        name: str = "sar_vessel",
    ):
        """
        data_yaml: YOLO-format dataset config pointing to chipped tiles + labels.
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch,
            project=project,
            name=name,
            device=0 if torch.cuda.is_available() else "cpu",
            patience=15,
            save=True,
            plots=True,
        )
        return results

    def detect_scene(
        self,
        chips: list,
        scene_width: int,
        scene_height: int,
    ) -> list:
        """
        Run inference on all chips from one scene.
        Returns list of dicts: {x, y, w, h, conf, cls} in scene-level pixel coords.
        """
        detections = []
        for chip in chips:
            results = self.model(chip["path"], conf=self.conf_thresh, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy()   # chip-level coords
                confs = r.boxes.conf.cpu().numpy()
                clss  = r.boxes.cls.cpu().numpy().astype(int)
                ox, oy = chip["x"], chip["y"]
                for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
                    detections.append({
                        "scene_x": ox + (x1 + x2) / 2,
                        "scene_y": oy + (y1 + y2) / 2,
                        "width_px":  x2 - x1,
                        "height_px": y2 - y1,
                        "conf": float(conf),
                        "class": int(cls),
                        "class_name": self.CLASS_NAMES[cls] if cls < len(self.CLASS_NAMES) else "unknown",
                    })
        # deduplicate overlapping chips with NMS
        detections = _scene_nms(detections, iou_thresh=0.5)
        return detections


def _scene_nms(detections: list, iou_thresh: float = 0.5) -> list:
    """Simple greedy NMS on scene-level detection list."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [
            d for d in detections
            if _iou_center(best, d) < iou_thresh
        ]
    return keep


def _iou_center(a: dict, b: dict) -> float:
    ax1, ay1 = a["scene_x"] - a["width_px"]/2, a["scene_y"] - a["height_px"]/2
    ax2, ay2 = a["scene_x"] + a["width_px"]/2, a["scene_y"] + a["height_px"]/2
    bx1, by1 = b["scene_x"] - b["width_px"]/2, b["scene_y"] - b["height_px"]/2
    bx2, by2 = b["scene_x"] + b["width_px"]/2, b["scene_y"] + b["height_px"]/2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (union + 1e-6)
