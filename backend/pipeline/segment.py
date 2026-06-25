"""SegFormer oil-segmentation inference over a chipped scene.

Runs on GPU with the trained M3 checkpoint. Reassembles per-chip predictions into a
full-scene oil mask, which geo.mask_to_polygons() then vectorises.

NOTE: not executed in the dev environment (needs torch + the HF checkpoint). It reuses
the verified src.models.segmentation factory so the architecture matches training.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _chip_tensor(png_path: str):
    """Load a preprocessed chip PNG -> 1×3×H×W normalised float tensor."""
    import torch

    bgr = cv2.imread(png_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def segment_scene(
    chips: list[dict],
    checkpoint: str,
    backbone: str = "b4",
    num_classes: int = 2,
    oil_class: int = 1,
    device: str | None = None,
) -> np.ndarray:
    """
    Run SegFormer on every chip and stitch into a full-scene oil mask.

    chips: list of {path, x, y, chip_size} (from sar_preprocess.chip_preprocessed).
    Returns: H×W uint8 scene mask (oil_class where oil predicted, else 0).
    """
    import torch

    from src.models.segmentation import build_segmentation_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmentation_model(model_type="segformer", num_classes=num_classes, backbone=backbone)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state", state))
    model.to(device).eval()

    chip_size = chips[0]["chip_size"]
    H = max(c["y"] for c in chips) + chip_size
    W = max(c["x"] for c in chips) + chip_size
    scene = np.zeros((H, W), dtype=np.uint8)

    with torch.no_grad():
        for c in chips:
            t = _chip_tensor(c["path"]).to(device)
            logits = model(t)  # 1×num_classes×h×w
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)  # h×w
            if pred.shape != (chip_size, chip_size):
                pred = cv2.resize(pred, (chip_size, chip_size), interpolation=cv2.INTER_NEAREST)
            y, x = c["y"], c["x"]
            # OR-merge overlaps: keep oil if any chip predicts oil
            region = scene[y:y + chip_size, x:x + chip_size]
            oil_here = (pred == oil_class).astype(np.uint8) * oil_class
            scene[y:y + chip_size, x:x + chip_size] = np.maximum(region, oil_here)

    return scene


def best_oil_chip(chips: list[dict], scene_mask: np.ndarray, chip_size: int, oil_class: int = 1) -> str | None:
    """Return the chip path covering the most oil pixels (for the Grad-CAM exhibit)."""
    best_path, best_count = None, 0
    for c in chips:
        y, x = c["y"], c["x"]
        region = scene_mask[y:y + chip_size, x:x + chip_size]
        count = int((region == oil_class).sum())
        if count > best_count:
            best_count, best_path = count, c["path"]
    return best_path
