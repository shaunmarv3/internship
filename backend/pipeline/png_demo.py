"""YOLO-only PNG demo path — run the HRSID-trained detector on a raw amplitude image.

This is the "how does our YOLO perform on its own training-domain data" exhibit. It is
DELIBERATELY separate from run_scene.py:

  * No CFAR      — CFAR is the resolution-agnostic detector we bolt on for the 10 m/px
                   Sentinel-1 domain gap; on native HRSID (0.5-3 m/px) YOLO is in-domain
                   and stands on its own.
  * No SegFormer — HRSID chips contain no oil, so oil segmentation is meaningless here.
  * No geo       — a PNG carries no georeferencing, so there is nothing to place on a map;
                   detections stay in pixel space and are drawn straight onto the image.

Preprocessing: NONE. The model was trained on the HRSID JPEGImages as-is (8-bit
grayscale amplitude), so we feed the PNG straight in — matching training exactly
(src/train/hrsid_to_yolo.py: raw JPEGs -> YOLO11m-OBB at imgsz=640).
"""
from __future__ import annotations

from pathlib import Path


def run_yolo_png(
    png_path: str,
    out_png: str,
    yolo_ckpt: str,
    conf: float = 0.25,
    imgsz: int = 1024,
) -> dict:
    """Run YOLO on a single native-resolution SAR image, draw the boxes, save an
    annotated PNG. Returns {ship_count, width, height, confidences}.

    conf=0.25 (vs 0.15 in the scene pipeline): HRSID is in-domain, so the model is
    confident — no need to lower the threshold to claw back the resolution gap.
    """
    import cv2
    import numpy as np
    from src.models.detection import SARVesselDetector

    img = cv2.imread(png_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {png_path}")
    H, W = img.shape[:2]

    detector = SARVesselDetector(model_path=yolo_ckpt, conf_thresh=conf)
    results = detector.model(png_path, conf=conf, imgsz=imgsz, verbose=False)

    green = (80, 220, 80)  # BGR
    confidences: list[float] = []

    for r in results:
        obb = getattr(r, "obb", None)
        if obb is not None and getattr(obb, "xyxyxyxy", None) is not None and len(obb.xyxyxyxy):
            corners = obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)  # (N, 4, 2) pixel coords
            confs = obb.conf.cpu().numpy()
            for quad, cf in zip(corners, confs):
                pts = quad.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], isClosed=True, color=green, thickness=2, lineType=cv2.LINE_AA)
                confidences.append(float(cf))
            continue
        # horizontal-box fallback (YOLOv8/26 non-OBB checkpoints)
        if getattr(r, "boxes", None) is None:
            continue
        for (x1, y1, x2, y2), cf in zip(
            r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        ):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), green, 2)
            confidences.append(float(cf))

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_png, img)
    return {
        "ship_count": len(confidences),
        "width": W,
        "height": H,
        "confidences": [round(c, 3) for c in confidences],
    }
