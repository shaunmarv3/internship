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
    oil_threshold: float = 0.5,
) -> np.ndarray:
    """
    Run SegFormer on every chip and stitch into a full-scene oil mask.

    chips: list of {path, x, y, chip_size} (from sar_preprocess.chip_preprocessed).
    oil_threshold: a pixel is oil if P(oil) > threshold. argmax (0.5) makes the
        model UNDER-predict oil (val sweep: recall only ~51% at 0.5, optimal ~0.30);
        lower it to surface subtle slicks at the cost of more look-alike false alarms.
    Returns: H×W uint8 scene mask (oil_class where oil predicted, else 0).
    """
    import torch

    from src.models.segmentation import build_segmentation_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device)
    # prefer the backbone the checkpoint was trained with (stored in its args)
    ckpt_backbone = (state.get("args") or {}).get("backbone") if isinstance(state, dict) else None
    model = build_segmentation_model(
        model_type="segformer", num_classes=num_classes, backbone=ckpt_backbone or backbone
    )
    model.load_state_dict(state.get("model_state", state))
    model.to(device).eval()

    chip_size = chips[0]["chip_size"]
    H = max(c["y"] for c in chips) + chip_size
    W = max(c["x"] for c in chips) + chip_size
    scene = np.zeros((H, W), dtype=np.uint8)

    with torch.no_grad():
        for c in chips:
            t = _chip_tensor(c["path"]).to(device)
            out = model(t)  # 1×num_classes×h×w (tensor or HF output)
            logits = out.logits if hasattr(out, "logits") else out
            prob = torch.softmax(logits, dim=1)[0, oil_class]  # h×w P(oil)
            pred = (prob > oil_threshold).cpu().numpy().astype(np.uint8)  # h×w
            if pred.shape != (chip_size, chip_size):
                pred = cv2.resize(pred, (chip_size, chip_size), interpolation=cv2.INTER_NEAREST)
            y, x = c["y"], c["x"]
            # OR-merge overlaps: keep oil if any chip predicts oil
            region = scene[y:y + chip_size, x:x + chip_size]
            oil_here = pred * oil_class
            scene[y:y + chip_size, x:x + chip_size] = np.maximum(region, oil_here)

    return scene


def segment_scene_resized(
    chips: list[dict],
    checkpoint: str,
    backbone: str = "b4",
    num_classes: int = 2,
    oil_class: int = 1,
    device: str | None = None,
    oil_threshold: float = 0.5,
    infer_size: int = 512,
    max_native: int = 2600,
) -> np.ndarray:
    """Whole-scene-resize oil inference — matches how the model was TRAINED & EVALUATED.

    The SegFormer oil model was trained and scored on whole ~2048² Zenodo scenes
    DOWNSAMPLED to 512² (loaders.get_oil_transforms → A.Resize(512); eval_threshold.py).
    segment_scene() instead runs native-resolution 512 CHIPS — a ~4× zoom-in the model
    never saw, so P(oil) never crosses threshold and oil comes back empty (2026-06-28
    finding — see research.md "Oil inference resolution skew"). This path reproduces the
    training field of view:
        raw scene → preprocess_sar_bands → [b0,VH,VH] → resize 512 → model → upscale.

    Returns an (H, W) uint8 scene mask on the same grid as chips' transform — a drop-in
    replacement for segment_scene(). Scenes larger than `max_native` px are processed in
    ~max_native tiles (each resized to 512) so a huge swath isn't shrunk into mush.
    """
    import torch
    import rasterio

    from src.data.sar_preprocess import preprocess_sar_bands
    from src.models.segmentation import build_segmentation_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device)
    ckpt_backbone = (state.get("args") or {}).get("backbone") if isinstance(state, dict) else None
    model = build_segmentation_model(
        model_type="segformer", num_classes=num_classes, backbone=ckpt_backbone or backbone
    )
    model.load_state_dict(state.get("model_state", state))
    model.to(device).eval()

    chip_size = chips[0]["chip_size"]
    out_H = max(c["y"] for c in chips) + chip_size
    out_W = max(c["x"] for c in chips) + chip_size

    # Reconstruct the model input the SAME way training did (loaders._load_image):
    # whole raw scene → dB→linear + Lee + percentile-norm → duplicate the strong band.
    with rasterio.open(chips[0]["scene"]) as src:
        raw = src.read().astype(np.float32)
        nodata = src.nodata
    proc = preprocess_sar_bands(raw, nodata=nodata)        # C×Hn×Wn float32 [0,1]
    img = np.transpose(proc, (1, 2, 0))                    # Hn×Wn×C
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 2:
        img = np.concatenate([img, img[:, :, 1:2]], axis=2)  # [b0, VH, VH] = training order
    Hn, Wn = img.shape[:2]

    def _infer(tile: np.ndarray) -> np.ndarray:
        small = cv2.resize(tile, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
        norm = (small - IMAGENET_MEAN) / IMAGENET_STD       # same as _chip_tensor / A.Normalize
        t = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out = model(t)
            logits = out.logits if hasattr(out, "logits") else out
            return torch.softmax(logits, dim=1)[0, oil_class].cpu().numpy()  # infer_size²

    prob = np.zeros((Hn, Wn), dtype=np.float32)
    if max(Hn, Wn) <= max_native:
        p = _infer(img)
        prob = cv2.resize(p, (Wn, Hn), interpolation=cv2.INTER_LINEAR)
    else:
        step = max_native
        for y in range(0, Hn, step):
            for x in range(0, Wn, step):
                ys, xs = slice(y, min(y + step, Hn)), slice(x, min(x + step, Wn))
                p = _infer(img[ys, xs])
                p = cv2.resize(p, (xs.stop - xs.start, ys.stop - ys.start),
                               interpolation=cv2.INTER_LINEAR)
                prob[ys, xs] = np.maximum(prob[ys, xs], p)

    mask = (prob > oil_threshold).astype(np.uint8) * oil_class
    if (Hn, Wn) != (out_H, out_W):
        mask = cv2.resize(mask, (out_W, out_H), interpolation=cv2.INTER_NEAREST)
    return mask


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
