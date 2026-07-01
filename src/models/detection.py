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

                # dB-window normalization — matches GEE getThumbURL display and HRSID
                # appearance (dark sea, bright ships). Per-chip min-max stretches
                # every chip to 0-255 including pure-sea chips → sea looks gray →
                # YOLO gets wrong input distribution and scores near-zero confidence.
                # VH (band index 1): -30..-10 dB  |  VV / anything else: -25..0 dB
                DB_WINDOWS = [(-25.0, 0.0), (-30.0, -10.0)]
                for c in range(data.shape[0]):
                    lo, hi = DB_WINDOWS[min(c, len(DB_WINDOWS) - 1)]
                    data[c] = np.clip((data[c] - lo) / (hi - lo), 0.0, 1.0)

                # skip near-empty tiles
                if data.mean() < min_brightness:
                    continue

                # save as 3-channel PNG (replicate bands if needed)
                if data.shape[0] == 1:
                    rgb = np.repeat(data, 3, axis=0)
                elif data.shape[0] == 2:
                    # [VV, VH, VH] — duplicate VH (cleanest ship-vs-sea contrast,
                    # near-black sea) rather than VV. Ships are pol-robust so this is
                    # immaterial for detection, but keeps both pipelines consistent.
                    rgb = np.concatenate([data, data[1:2]], axis=0)
                else:
                    rgb = data[:3]

                rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
                chip_path = out_dir / f"chip_{chip_idx:05d}.png"
                import cv2
                cv2.imwrite(str(chip_path), rgb_uint8.transpose(1, 2, 0)[:, :, ::-1])
                chips.append({"path": str(chip_path), "x": x, "y": y,
                               "scene": scene_path, "chip_size": chip_size})
                chip_idx += 1

    print(f"Chipped {scene_path} -> {chip_idx} tiles in {out_dir}")
    return chips


# ── YOLO detector ──────────────────────────────────────────────────────────────

class SARVesselDetector:
    """
    YOLOv8/v11 wrapper for SAR vessel detection.
    HRSID is single-class (0=ship). CLASS_NAMES matches the trained model;
    any out-of-range index falls back to "vessel".
    """

    CLASS_NAMES = ["ship"]

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
        infer_imgsz: int = 1024,
    ) -> list:
        """
        Run inference on all chips from one scene.
        Returns list of dicts: {x, y, w, h, conf, cls} in scene-level pixel coords.

        infer_imgsz: YOLO input size during inference. Default 1024 upsamples the 512px
        chip 2× before the network sees it — makes ~10px ships at 10m/px appear as ~20px,
        reducing the training/inference resolution gap (HRSID 0.5-3m vs GEE 10m).
        """
        detections = []
        for chip in chips:
            results = self.model(chip["path"], conf=self.conf_thresh,
                                 imgsz=infer_imgsz, verbose=False)
            ox, oy = chip["x"], chip["y"]
            for r in results:
                # Oriented-box model (YOLO11m-OBB): use rotated-box centre + size.
                obb = getattr(r, "obb", None)
                if obb is not None and obb.xywhr is not None and len(obb.xywhr):
                    xywhr = obb.xywhr.cpu().numpy()    # cx, cy, w, h, angle (chip coords)
                    confs = obb.conf.cpu().numpy()
                    clss  = obb.cls.cpu().numpy().astype(int)
                    for (cx, cy, w, h, _ang), conf, cls in zip(xywhr, confs, clss):
                        detections.append({
                            "scene_x": ox + float(cx),
                            "scene_y": oy + float(cy),
                            "width_px":  float(w),
                            "height_px": float(h),
                            "conf": float(conf),
                            "class": int(cls),
                            "class_name": self.CLASS_NAMES[cls] if cls < len(self.CLASS_NAMES) else "vessel",
                        })
                    continue

                # Horizontal-box model (YOLOv8m / YOLO26m).
                if r.boxes is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy()   # chip-level coords
                confs = r.boxes.conf.cpu().numpy()
                clss  = r.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
                    detections.append({
                        "scene_x": ox + (x1 + x2) / 2,
                        "scene_y": oy + (y1 + y2) / 2,
                        "width_px":  x2 - x1,
                        "height_px": y2 - y1,
                        "conf": float(conf),
                        "class": int(cls),
                        "class_name": self.CLASS_NAMES[cls] if cls < len(self.CLASS_NAMES) else "vessel",
                    })
        # deduplicate overlapping chips with NMS
        detections = _scene_nms(detections, iou_thresh=0.5)
        return detections


def cfar_detect(
    scene_path: str,
    guard: int = 5,
    train: int = 20,
    alpha: float = 20.0,
    method: str = "censored",
    censor_sigma: float = 3.0,
) -> list:
    """
    Censored-mean CFAR (CMLD) ship detector — masking-robust by default.
    Resolution-agnostic — adapts to local sea clutter at any pixel scale.
    At Sentinel-1 IW 10m/px outperforms HRSID-trained YOLO in recall.

    method="censored" (default): a generalization of OS-CFAR. The plain
        Cell-Averaging clutter estimate is corrupted in DENSE scenes — a second
        ship (or a bright ocean front/wake) sitting inside the training ring
        inflates the local mean, so the threshold `alpha*clutter` rises and a
        nearby dimmer ship falls below it and is MISSED (the classic CA-CFAR
        multiple-target masking failure — observed on Singapore anchorage and
        the mauritius_sea current front). Fix: estimate the ring mean+std, CENSOR
        cells brighter than mean + censor_sigma*std (interfering targets / fronts),
        then recompute the clutter mean over the surviving pure-clutter cells only.
        Done with masked box filters → same speed budget as CA-CFAR, no new deps.
    method="ca": legacy plain Cell-Averaging CFAR (kept as a fallback).

    Returns same format as SARVesselDetector.detect_scene() so it's a
    drop-in: list of {scene_x, scene_y, width_px, height_px, conf, class}.
    """
    import scipy.ndimage as ndi

    with rasterio.open(scene_path) as src:
        H, W = src.height, src.width
        if src.count >= 2:
            # For dual-pol TIFs (VV=band1, VH=band2): pick VH — darker sea gives
            # higher ship-to-clutter ratio. Identify it as the band with lower mean.
            bands = [src.read(i + 1).astype(np.float32) for i in range(min(src.count, 2))]
            valid = [np.nanmean(np.where(b < -50, np.nan, b)) for b in bands]
            db = bands[int(np.argmin(valid))]   # lowest mean dB = VH
        else:
            db = src.read(1).astype(np.float32)

    db  = np.where(db < -50, np.nan, db)
    lin = np.nan_to_num(10 ** (db / 10.0), 0.0)

    total_w, guard_w = train * 2 + 1, guard * 2 + 1
    ring_cells = total_w**2 - guard_w**2

    # Sum / mean over the training ANNULUS (total box minus the guard box).
    def _ring_sum(img):
        tot = ndi.uniform_filter(img, total_w) * total_w**2   # sum over total box
        grd = ndi.uniform_filter(img, guard_w) * guard_w**2   # sum over guard box
        return tot - grd

    def _ring_mean(img):
        return _ring_sum(img) / ring_cells

    ca_mean = np.maximum(_ring_mean(lin), 1e-10)

    if method == "censored":
        # ring std from E[x^2] - E[x]^2
        ring_sq  = np.maximum(_ring_mean(lin * lin), 0.0)
        ring_std = np.sqrt(np.maximum(ring_sq - ca_mean**2, 0.0))
        # keep = pure-clutter cells (exclude bright interfering targets / fronts)
        keep     = (lin <= ca_mean + censor_sigma * ring_std).astype(np.float32)
        kept_sum = _ring_sum(lin * keep)
        kept_cnt = _ring_sum(keep)
        clutter  = np.maximum(kept_sum / np.maximum(kept_cnt, 1.0), 1e-10)
    else:  # "ca" — legacy plain cell-averaging
        clutter = ca_mean

    det = (lin > alpha * clutter).astype(np.uint8)

    import cv2

    # Water mask: suppress land detections at source.
    # SAR VH sea backscatter < -14 dB; land/urban > -14 dB across all scene types.
    # Opening (61px ≈ 610m) removes ship-sized bright spots from the land candidate,
    # keeping only solid land blobs. Dilation (30px ≈ 300m) adds a coastal buffer.
    # Without this, CFAR fires freely on buildings/roads (buildings ARE 20× brighter
    # than adjacent structures), producing hundreds of land FP in urban scenes.
    _land_cand = np.where(np.isnan(db), 0, (db >= -18.0).astype(np.uint8))
    _k61 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    _k10 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    _land = cv2.dilate(cv2.morphologyEx(_land_cand, cv2.MORPH_OPEN, _k61), _k10)
    water = (_land == 0) & (~np.isnan(db))
    det = (det & water.astype(np.uint8))

    # Filter on UNDILATED blob size — dilation inflates every pixel to ~63px²
    # making post-dilation area filtering useless. Pre-dilation: speckle=1px
    # isolated, real ships=cluster of multiple pixels. min_det_px=14 empirically
    # tuned on Mumbai 10m/px scene (37m minimum ship size).
    min_det_px = 14
    _, det_lbl, det_stats, _ = cv2.connectedComponentsWithStats(det)
    det_filtered = np.zeros_like(det)
    for i in range(1, len(det_stats)):
        if det_stats[i, cv2.CC_STAT_AREA] >= min_det_px:
            det_filtered[det_lbl == i] = 1

    # Merge kernel: small on purpose. A 9px (≈90m) dilation fused adjacent ships
    # in dense anchorages (Singapore) into ONE connected component → undercount.
    # 5px (≈50m) still bridges the speckle gaps within a single ship's return
    # while keeping vessels ~50m+ apart as separate detections.
    k5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mrg = cv2.dilate(det_filtered, k5)
    _, _, stats, cents = cv2.connectedComponentsWithStats(mrg)

    detections = []
    for i, (cx, cy) in enumerate(cents[1:], 1):
        cx, cy = int(cx), int(cy)
        if not (0 <= cy < H and 0 <= cx < W):
            continue
        # confidence proxy: peak SCR in the blob (capped at 1.0)
        conf = min(float(lin[cy, cx] / (clutter[cy, cx] + 1e-10)) / alpha, 1.0)
        sz   = max(int(stats[i, cv2.CC_STAT_AREA] ** 0.5), 3)
        detections.append({
            "scene_x":   float(cx),
            "scene_y":   float(cy),
            "width_px":  float(sz),
            "height_px": float(sz),
            "conf":      conf,
            "class":     0,
            "class_name": "ship",
            "detector":  "cfar",
        })
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
