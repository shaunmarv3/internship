# Maritime Security Intelligence — Explainable AI on Sentinel-1 SAR

An end-to-end, explainable system that turns a **single Sentinel-1 SAR scene** into an
operational maritime-security picture: it detects **ships**, flags **dark (non-broadcasting)
vessels**, segments **oil spills**, scores **security and environmental risk**, and explains
every decision — served through a FastAPI backend and a Next.js / MapLibre dashboard.

> One sensor, one scene. Because ships and oil are extracted from the *same* georeferenced
> Sentinel-1 image, they are automatically co-registered and time-aligned. The contribution
> is the **integrated pipeline**, not six disconnected models.

---

## What it does

| Stage | Task | Approach |
|-------|------|----------|
| **Ship detection** | Find every vessel in the scene | **Censored-mean CFAR** (training-free, resolution-agnostic) as the primary detector + **YOLO11m-OBB** (HRSID-trained) for oriented-box geometry; YOLO is kept only where it agrees with a CFAR cluster |
| **Oil-spill segmentation** | Pixel-level slick vs. sea | **SegFormer** (benchmarked against DeepLabV3+ / U-Net), whole-scene-resize inference matching training |
| **Dark-vessel flagging** | Detections with no AIS match | AIS cross-check at the scene acquisition timestamp |
| **Risk + fusion** | Security / environmental scores | **Rule-based** zone-violation check + spill↔nearby-vessel linkage |
| **Explainability** | Justify decisions | **Grad-CAM** (oil + ship) + **rule-contribution bars** |

---

## Datasets

| Dataset | Used for | Notes |
|---------|----------|-------|
| **HRSID** (Wei et al., *IEEE Access* 2020) | Ship detection | 5,604 chips, 16,951 instances, 800×800; 3,642 train / 1,962 test (5,922 test instances) |
| **Trujillo-Acatitla et al. 2024** (Zenodo [8346860](https://zenodo.org/records/8346860) / [8253899](https://zenodo.org/records/8253899) / [13761290](https://zenodo.org/records/13761290)) | Oil segmentation | Sentinel-1 C-band VV/VH, Sigma0 in dB, 2048×2048. Parts I+II train/val (≈85/15), **Part III held out as a look-alike-heavy stress set** |

> Live inference runs on **Sentinel-1 C-band IW GRD** scenes (VV/VH, ~10 m/px) pulled from
> Google Earth Engine (`COPERNICUS/S1_GRD`).

---

## Headline results

**Ship detection — HRSID test split** (50 epochs, 2× NVIDIA T4)

| Model | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|-------|-----------|--------|-----|--------|-----------|
| YOLOv8m (2023, horizontal) | 0.913 | 0.822 | 0.865 | 0.910 | 0.669 |
| YOLO26m (2026, horizontal) | **0.926** | 0.802 | 0.860 | 0.908 | 0.671 |
| **YOLO11m-OBB** (2024, oriented) | 0.920 | **0.873** | **0.896** | **0.938** | **0.688** |

Oriented-box *geometry* beats model recency: the 2026 YOLO26m is level with YOLOv8m, while
the OBB head holds a visible margin throughout training. A scale-augmented fine-tune
(mAP@50 0.910) trades a little benchmark accuracy for robustness at 10 m/px.

**Medium-resolution detection.** On a dense Sentinel-1 anchorage scene, censored-mean CFAR
recovers **257** vessels where the HRSID-trained YOLO alone is inconsistent (0–165) — CFAR
sidesteps the HRSID→Sentinel-1 resolution gap entirely.

**Oil segmentation** (50 epochs, NVIDIA H100)

| Model | Params | val OilIoU | stress-set OilIoU | stress-set mIoU |
|-------|--------|------------|-------------------|-----------------|
| U-Net | 32.6 M | 0.750 | 0.369 | 0.673 |
| DeepLabV3+ | 26.7 M | 0.773 | 0.356 | 0.667 |
| SegFormer-b4 | 64.0 M | **0.799** | 0.481 | 0.731 |
| SegFormer-b5 | 82.0 M | 0.795 | **0.484** | **0.733** |

SegFormer leads on validation OilIoU ≈ **0.80**. Part III is a deliberately adversarial
partition (⅓ look-alikes) used as a **cross-distribution robustness analysis**: pooled
OilIoU drops to 0.484 (95% bootstrap CI [0.372, 0.612]) while the *median* oil-bearing
scene still reaches 0.767 — the aggregate is pulled down by a hard look-alike tail, not by
uniform failure. Per-scene IoU on Gulf of Mexico / Java Sea / Mediterranean slicks ranges
0.79–0.93, and a Bay of Biscay look-alike yields zero false-positive oil pixels. The
architecture ordering is significant (paired Wilcoxon over 150 oil-bearing scenes:
b5 vs DeepLabV3+ *p* = 9.5e-12, vs U-Net *p* = 3.3e-04).

---

## Architecture

```
Sentinel-1 GRD (.tif / GEE pull)
   └─ SAR preprocessing
        ├─ oil branch:  dB→linear → 7×7 Lee → percentile-norm → [VV,VH,VH] → resize 512
        └─ ship branch: dB-window normalize (VV [-25,0], VH [-30,-10] dB), no Lee
   ├─ SHIP: censored-mean CFAR (primary) + YOLO11m-OBB (confirm/geometry) → land mask
   ├─ OIL:  SegFormer whole-scene-resize → threshold τ∈[0.30,0.35] → lon/lat polygons + area (km²)
   ├─ DARK VESSELS: AIS match at scene acquisition time
   ├─ RISK: rule-based zone violation + security / environmental scores
   └─ EXPLAIN: Grad-CAM (oil/ship) + rule-contribution bars
          │
   FastAPI backend  ──GeoJSON layers + metrics + overlays──►  Next.js / MapLibre dashboard
```

Two design decisions carry most of the accuracy:

- **Censored-mean CFAR (CMLD)** instead of cell-averaging CFAR. CA-CFAR suffers target
  masking in dense anchorages — a bright neighbour inflates the local clutter estimate and
  a dimmer ship falls below threshold. Censoring the brightest training cells fixes it
  (synthetic test: CA threshold 1.135 → miss; censored 0.502 → detect) at the same
  box-filter speed.
- **Whole-scene-resize oil inference**, matching training. Running the SegFormer on native
  512×512 chips of a 2048² scene is a 4× resolution skew and drove oil predictions to
  literally 0.00% on scenes with obvious slicks.

---

## Tech stack

**ML / CV:** PyTorch · HuggingFace Transformers (SegFormer) · Ultralytics YOLO ·
segmentation-models-pytorch · OpenCV / SciPy (CFAR) · albumentations
**Explainability:** pytorch-grad-cam
**Geospatial:** rasterio · geopandas · shapely · pyproj · Google Earth Engine / geemap
**Backend:** FastAPI · Pydantic
**Frontend:** Next.js (App Router, TS) · Tailwind · MapLibre GL (CARTO dark basemap, no token)
**Training / tracking:** Kaggle 2×T4 (ships) · Lightning H100 (oil) · Weights & Biases

---

## Repository layout

```
.
├─ src/
│  ├─ data/       sar_preprocess.py (dB→linear, Lee, dB-window, GEE pull), loaders.py
│  ├─ models/     detection.py (CFAR + YOLO), segmentation.py (SegFormer/DeepLab/UNet),
│  │              train_segmentation.py, eval_threshold.py, eval_significance.py
│  ├─ fusion/     ais_matching.py (match + zone join), risk_scoring.py
│  └─ explain/    gradcam.py
├─ backend/       app.py (FastAPI), pipeline/{run_scene, segment, geo, serialize}.py
├─ frontend/      Next.js dashboard (upload → scene metadata + oil extent → map)
├─ notebooks/     kaggle_ship_detection.ipynb (HRSID training)
├─ res/          training-curve exports for ships and oil
└─ runs/         Ultralytics OBB run artefacts (curves, confusion matrices, args)
```

---

## Quickstart

```bash
# 1. Backend (serves precomputed case studies; runs the pipeline on upload)
pip install -r requirements.txt
uvicorn backend.app:app --port 8000 --app-dir .

# 2. Frontend
cd frontend && npm install && npm run dev        # → localhost:3000

# 3. Process a scene: drag a Sentinel-1 GeoTIFF into the dashboard, or
python -m backend.pipeline.run_scene --scene-tif scene.tif --id demo \
       --title "Demo" --acquired 2024-03-01T00:00:00Z \
       --yolo checkpoints/vessel/hrsid_yolo11m_obb/best.pt \
       --segformer checkpoints/oil/best_segformer.pt
```

There is also a **YOLO-only PNG demo path** (second upload box) that runs the trained
HRSID detector on a plain PNG chip, with no CFAR gating, land mask, oil branch, or
georeferencing — it shows the detector's own behaviour on its native domain.

Checkpoints live on the private HF repo `shaunmarvell/maritime-security-intelligence`
(`vessel/` and `oil/` subfolders).

---

## Scope and limitations

- **No oil volume.** SAR yields slick **area / extent**, never thickness or volume.
- **Oil model = SegFormer.** The "OilSAM2" SOTA has no released code and silently falls
  back to SegFormer; results are reported as SegFormer throughout.
- **Cross-region generalisation is the open problem.** A threshold sweep (τ 0.30–0.80)
  moves stress-set OilIoU only +0.006, so the gap lies in the training distribution, not
  the operating point. Cross-region training data is the principled fix.
- **Ship resolution gap.** HRSID is finer-resolution than Sentinel-1 IW (~10 m/px); CFAR
  bridges the gap at inference, and xView3-SAR fine-tuning is the long-term fix.
- **AIS coverage.** Free historical AIS (Global Fishing Watch) exposes only gap/absence
  events, lags 72–96 h, and covers fishing vessels only. An operational deployment
  connects a position feed such as NOAA MarineCadastre or the Danish Maritime Authority
  open feed; the matching pipeline is unchanged.
