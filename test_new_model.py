from ultralytics import YOLO
import rasterio, numpy as np, cv2, scipy.ndimage as ndi, matplotlib.pyplot as plt
from pathlib import Path

CKPT = "runs/obb/checkpoints/vessel/hrsid_obb_10m-3/weights/best.pt"

SCENES = [
    "backend/data/uploads/s1_malacca_2024-03-01_2024-04-01.tif",
    "backend/data/uploads/s1_hormuz_2024-01-01_2024-06-01.tif",
    "backend/data/uploads/s1_singapore_2024-03-01_2024-04-01.tif",
]

model = YOLO(CKPT)

for TIF in SCENES:
    name = Path(TIF).stem
    print(f"\n{'='*60}")
    print(f"Scene: {name}")

    with rasterio.open(TIF) as src:
        raw = src.read().astype(np.float32)
        nodata = src.nodata

    db = raw[1] if raw.shape[0] > 1 else raw[0]   # VH band
    if nodata is not None:
        db = np.where(np.isclose(db, nodata), np.nan, db)

    H, W = db.shape
    print(f"  Size: {H}x{W}px")

    # dB window normalization
    gray = np.clip((db - (-30)) / (-10 - (-30)) * 255, 0, 255)
    gray = np.nan_to_num(gray, nan=0.0).astype(np.uint8)

    # ── CFAR ──────────────────────────────────────────────────────────────
    db_clean = np.where((db < -50) | np.isnan(db), np.nan, db)
    lin = np.nan_to_num(10 ** (np.nan_to_num(db_clean, nan=-50.0) / 10.0), nan=0.0)

    valid = ~np.isnan(db_clean)
    t90 = np.nanpercentile(db_clean[valid], 90)
    land = cv2.dilate(
        cv2.morphologyEx((db_clean > t90).astype(np.uint8), cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)))
    water = (land == 0) & valid

    tw, gw = 41, 11
    tm = ndi.uniform_filter(lin * water, tw)
    gm = ndi.uniform_filter(lin * water, gw)
    clutter = np.maximum((tm * tw**2 - gm * gw**2) / (tw**2 - gw**2), 1e-10)
    det = ((lin > 20.0 * clutter) & water).astype(np.uint8)

    MIN_DET_PX = 14
    _, det_lbl, det_stats, _ = cv2.connectedComponentsWithStats(det)
    det_filtered = np.zeros_like(det)
    for i in range(1, len(det_stats)):
        if det_stats[i, cv2.CC_STAT_AREA] >= MIN_DET_PX:
            det_filtered[det_lbl == i] = 1

    mrg = cv2.dilate(det_filtered, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    _, _, stats, cents = cv2.connectedComponentsWithStats(mrg)

    # Raw dB stretch for display — open-ocean sea is already naturally dark
    # (CFAR binary masking creates harsh white dots on black for low-clutter scenes)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cfar_ships = []
    for i, (cx, cy) in enumerate(cents[1:], 1):
        cx, cy = int(cx), int(cy)
        if not (0 <= cy < H and 0 <= cx < W and water[cy, cx]): continue
        sz = max(int(stats[i, cv2.CC_STAT_AREA] ** 0.5), 3)
        cfar_ships.append((cx, cy, sz))

    print(f"  CFAR ships: {len(cfar_ships)}")
    for cx, cy, sz in cfar_ships:
        cv2.rectangle(canvas, (cx-sz, cy-sz), (cx+sz, cy+sz), (0, 200, 0), 1)

    # ── YOLO ──────────────────────────────────────────────────────────────
    yolo_ships = []
    chip_size, stride = 512, 256

    for y0 in range(0, max(H - chip_size + 1, 1), stride):
        for x0 in range(0, max(W - chip_size + 1, 1), stride):
            y1, x1 = min(y0 + chip_size, H), min(x0 + chip_size, W)
            chip = gray[y0:y1, x0:x1]
            if chip.shape[0] < chip_size or chip.shape[1] < chip_size:
                pad = np.zeros((chip_size, chip_size), dtype=np.uint8)
                pad[:chip.shape[0], :chip.shape[1]] = chip
                chip = pad
            res = model(np.stack([chip, chip, chip], 2), conf=0.10, imgsz=640, verbose=False)
            for r in res:
                obb = getattr(r, "obb", None)
                if obb is None or obb.xywhr is None or not len(obb.xywhr): continue
                for (cx, cy, w, h, ang), conf in zip(obb.xywhr.cpu().numpy(), obb.conf.cpu().numpy()):
                    sx, sy = x0 + int(cx), y0 + int(cy)
                    if sx >= W or sy >= H: continue
                    if mrg[sy, sx] == 0: continue
                    yolo_ships.append((sx, sy, w, h, ang, conf))
                    box = cv2.boxPoints(((float(sx), float(sy)), (float(w), float(h)),
                                         float(np.degrees(ang)))).astype(int)
                    cv2.drawContours(canvas, [box], 0, (0, 140, 255), 2)

    print(f"  YOLO ships: {len(yolo_ships)}")

    cfar_only = sum(
        1 for (cx, cy, _) in cfar_ships
        if not any((cx - sx)**2 + (cy - sy)**2 < 15**2 for sx, sy, *_ in yolo_ships)
    )
    total = len(yolo_ships) + cfar_only
    print(f"  Merged total: {total}  ({len(yolo_ships)} YOLO OBB + {cfar_only} CFAR-only)")

    out_png = f"{name}_merged.png"
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(f"{name}\nCFAR {len(cfar_ships)} (green) + YOLO {len(yolo_ships)} OBB (orange) = {total} total")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_png}")
