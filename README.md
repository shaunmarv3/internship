# Maritime Security Intelligence — Explainable AI on Sentinel-1 SAR

An end-to-end, explainable system that turns a **single Sentinel-1 SAR scene** into an
operational maritime-security picture: it detects **ships**, flags **dark (non-broadcasting)
vessels**, segments **oil spills**, scores **security and environmental risk**, and explains
every decision — served through a FastAPI backend and a Next.js / MapLibre dashboard.

> One sensor, one scene. Because ships and oil are extracted from the *same* georeferenced
> Sentinel-1 image, they are automatically co-registered and time-aligned. The contribution
> is the **integrated, honest pipeline**, not six disconnected models.

---

## What it does

| Stage | Task | Approach |
|-------|------|----------|
| **Ship detection** | Find every vessel in the scene | **Censored-mean CFAR** (training-free, resolution-agnostic) as the primary detector + **YOLO11m-OBB** (HRSID-trained) for oriented-box geometry; YOLO is kept only where it agrees with a CFAR hit |
| **Oil-spill segmentation** | Pixel-level slick vs. sea | **SegFormer** (benchmarked against DeepLabV3+ / U-Net), whole-scene-resize inference matching training |
| **Dark-vessel flagging** | Detections with no AIS match | AIS cross-check at the scene timestamp; a **synthetic AIS proxy** stands in because free historical AIS is unusable (see notes) |
| **Risk + fusion** | Security / environmental scores | **Rule-based** zone-violation check + spill↔nearby-vessel linkage |
| **Explainability** | Justify decisions | **Grad-CAM** (oil + ship) + **rule-contribution bars** |

> Deliberately **not** included: XGBoost/illegal-fishing classifier, oil-drift forecasting,
> and SHAP — these were dropped in favour of the rule-based zones + Grad-CAM above.

---

## Datasets

| Dataset | Used for | Notes |
|---------|----------|-------|
| **HRSID** (Wei et al., *IEEE Access* 2020) | Ship detection | 5,604 chips, 16,951 instances, 800×800; 3,642 train / 1,962 test + 400 negatives |
| **Trujillo-Acatitla et al. 2024** (Zenodo [8346860](https://zenodo.org/records/8346860) / [8253899](https://zenodo.org/records/8253899) / [13761290](https://zenodo.org/records/13761290)) | Oil segmentation | Sentinel-1 C-band VV/VH, Sigma0 in dB, 2048×2048. Parts I+II train/val, **Part III held out as test** |

> Live inference runs on **Sentinel-1 C-band IW GRD** scenes (VV/VH, ~10 m/px) pulled from
> Google Earth Engine (`COPERNICUS/S1_GRD`).

---

## Headline results

- **Ship detection (HRSID test):** YOLO11m-OBB **mAP@50 = 0.938** (best of YOLOv8m 0.910 /
  YOLO26m 0.908 / YOLO11m-OBB 0.938). Finding: oriented-box *geometry* beats model recency.
  A scale-augmented fine-tune (mAP@50 0.910) adds robustness at 10 m/px.
- **Medium-resolution detection:** on a dense Sentinel-1 anchorage scene, censored-mean CFAR
  recovers **257** vessels where the HRSID-trained YOLO alone is inconsistent (0–165).
- **Oil segmentation:** SegFormer validation OilIoU **≈ 0.79**, honest **held-out Part III
  test OilIoU ≈ 0.48** — the validation→test gap (driven by look-alikes) is reported openly.

---

## Architecture

```
Sentinel-1 GRD (.tif / GEE pull)
   └─ SAR preprocessing
        ├─ oil branch:  dB→linear → 7×7 Lee → percentile-norm → [VV,VH,VH] → resize 512
        └─ ship branch: dB-window normalize (VV [-25,0], VH [-30,-10] dB), no Lee
   ├─ SHIP: censored-mean CFAR (primary) + YOLO11m-OBB (confirm/geometry) → land mask
   ├─ OIL:  SegFormer whole-scene-resize → threshold → lon/lat polygons + area (km²)
   ├─ DARK VESSELS: AIS match at scene time (synthetic proxy when real AIS absent)
   ├─ RISK: rule-based zone violation + security / environmental scores
   └─ EXPLAIN: Grad-CAM (oil/ship) + rule-contribution bars
          │
   FastAPI backend  ──GeoJSON layers + metrics + overlays──►  Next.js / MapLibre dashboard
```

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
│  │              train_segmentation.py, eval_threshold.py
│  ├─ fusion/     ais_matching.py (match + synthetic proxy + zone join), risk_scoring.py
│  └─ explain/    gradcam.py
├─ backend/       app.py (FastAPI), pipeline/{run_scene, segment, geo, serialize}.py
├─ frontend/      Next.js dashboard (upload → scene metadata + oil extent → map)
├─ paper/         maritime_paper.tex (research write-up) + references/
└─ research.md    running lab notebook (source of truth for decisions/results)
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

Checkpoints live on the private HF repo `shaunmarvell/maritime-security-intelligence`
(`vessel/` and `oil/` subfolders).

---

## Honesty notes

- **No oil volume.** SAR yields slick **area / extent**, never thickness or volume.
- **Oil model = SegFormer.** The "OilSAM2" SOTA had no released code and silently falls back
  to SegFormer; results are reported as SegFormer.
- **AIS is a proxy.** Free historical AIS (Global Fishing Watch) exposes only gap/absence
  events, lags 72–96 h, and covers fishing vessels only, so it cannot do same-scene presence
  matching. The dark/normal split is produced by a transparent synthetic-AIS proxy; a paid
  all-vessel feed is future work.
- **Report the held-out number.** Oil accuracy is reported on the disjoint Part III test
  set (~0.48 OilIoU), not the optimistic validation figure (~0.79).
- **Ship resolution gap.** HRSID is finer-resolution than Sentinel-1 IW (~10 m/px); CFAR
  bridges the gap at inference, and xView3-SAR fine-tuning is the principled long-term fix.

See [`research.md`](./research.md) for the full decision/result log and
[`paper/maritime_paper.tex`](./paper/maritime_paper.tex) for the write-up.
