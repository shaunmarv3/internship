# 🛰️ Maritime Security Intelligence — Explainable Multi-Modal AI

An end-to-end, explainable AI framework that fuses **satellite SAR imagery**, **AIS vessel
tracks**, and **oceanographic data** to detect **dark vessels**, flag **illegal fishing**,
segment **oil spills**, predict **spill drift**, and roll it all into an interpretable
**maritime risk dashboard**.

> Built on Sentinel-1 SAR. The headline contribution is the **fusion layer** — linking a
> detected spill to nearby *dark* (non-broadcasting) vessels as candidate responsible parties,
> with explainable risk scoring — not six disconnected models.

---

## 🎯 What it does

| Module | Task | Approach |
|--------|------|----------|
| **1. Dark Fleet Monitoring** | Detect vessels in SAR, match to AIS, flag the unmatched as *dark* | YOLOv8/v11 on chipped SAR tiles + probabilistic AIS matching |
| **2. Illegal Fishing Detection** | Classify fishing behavior, flag activity inside restricted zones | XGBoost (+ BiGRU) on AIS trajectories + geospatial zone join |
| **3. Oil Spill Detection** | Pixel-level segmentation of slicks vs look-alikes | SegFormer (SOTA) vs DeepLabv3+/U-Net baseline |
| **4. Oil Spill Drift Prediction** | Forecast spill trajectory & impact zones | OpenDrift / OpenOil (physics) + optional LSTM |
| **5. Explainable AI** | Justify every decision | SHAP (tabular) + Grad-CAM (imagery) |
| **6. Risk Assessment + Fusion** | Security & environmental risk scores; spill↔vessel linkage | Weighted scoring + spatial fusion → Streamlit dashboard |

---

## 🗂️ Datasets

| Need | Dataset | Access |
|------|---------|--------|
| Oil masks (5-class) | **Krestenitis / M4D** (~400 MB) | request: [m4d.iti.gr](https://m4d.iti.gr/oil-spill-detection-dataset/) |
| Oil masks (instant) | **Zenodo 8346860** (binary, 47 GB) | [direct download](https://zenodo.org/records/8346860) |
| SAR vessels | **SARFishSample** → GRD subset | [HuggingFace](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFish) |
| SAR vessels (benchmark) | **xView3-SAR** | [iuu.xview.us](https://iuu.xview.us/) |
| AIS tracks + labels | **Global Fishing Watch** API, `marinecadastre.gov` | free token |
| Zones | **EEZ** (marineregions.org), **MPA/WDPA** (protectedplanet.net) | free |
| Ocean currents/wind | **Copernicus Marine** + **ERA5** | free API |

> ⚠️ Never clone the full xView3/SARFish (TB-scale). Pull GRD-only subsets:
> `snapshot_download(repo_id="ConnorLuckettDSTG/SARFish", repo_type="dataset", allow_patterns=["*GRD*"])`

---

## 🧱 Architecture

```
Sentinel-1 SAR ─┐
AIS tracks ──────┼─► Preprocessing ─► Feature Extraction ─► [M1 Dark Fleet]
Oceanographic ──┤                                          [M2 Illegal Fishing]
Zone shapefiles ─┘                                          [M3 Oil Spill Seg]
                                                                  │
                                                            [M4 Drift Forecast]
                                                                  │
                                                  [M5 Explainability: SHAP + Grad-CAM]
                                                                  │
                                    [M6 Fusion → Security + Environmental Risk Scores]
                                                                  │
                                              📊 Integrated Streamlit Dashboard
```

---

## ⚙️ Tech Stack

**ML/CV:** PyTorch · segmentation-models-pytorch · HuggingFace Transformers (SegFormer) ·
Ultralytics YOLO · XGBoost · scikit-learn · albumentations
**Explainability:** SHAP · pytorch-grad-cam
**Geospatial:** rasterio · geopandas · shapely · pyproj
**Drift:** OpenDrift (OpenOil)
**Data feeds:** GFW client · earthengine-api/geemap · copernicusmarine · cdsapi
**App:** Streamlit · Folium / kepler.gl · plotly
**Compute:** local RTX 3050 (dev) · Colab Pro (training) · Lightning.ai (storage/long jobs)
**Tracking:** Weights & Biases / TensorBoard

---

## 📦 Project Structure

```
.
├─ PROJECT_PLAN.md          # full plan: models, datasets, phasing, justifications
├─ README.md
├─ requirements.txt
├─ data/                    # gitignored — datasets land here
│  ├─ oil/                  # Krestenitis / Zenodo
│  └─ vessels/              # SARFishSample → GRD subset
├─ notebooks/
│  ├─ 01_dark_fleet.ipynb
│  ├─ 02_illegal_fishing.ipynb
│  ├─ 03_oil_spill_seg.ipynb
│  ├─ 04_oil_drift.ipynb
│  └─ 05_06_xai_risk.ipynb
├─ src/
│  ├─ data/  models/  fusion/  explain/  viz/
└─ app/
   └─ dashboard.py          # Streamlit entrypoint
```

---

## 🛣️ Build Phases (floor-first)

- **Phase 0** — Setup & data: request M4D, pull Zenodo, clone SARFishSample, GFW token.
- **Phase 1** — 🟢 *Safe floor:* Oil segmentation (M3). Baseline → fix 1.2% class imbalance (Dice+Focal) → SegFormer. Beat ~0.54 oil IoU.
- **Phase 2** — Dark fleet (M1): chip tiles → YOLOv8 → AIS match → flag dark.
- **Phase 3** — Illegal fishing (M2): XGBoost/BiGRU + zone join.
- **Phase 4** — Drift (M4): OpenOil + Copernicus/ERA5.
- **Phase 5** — XAI + Risk + Fusion (M5/M6).
- **Phase 6** — Streamlit dashboard integrating all outputs.

> Each phase is independently demo-able — the project degrades gracefully if time runs out.

---

## 🚀 Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate      # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2. Get the unblocking dataset (oil, instant)
#    → download Zenodo record 8346860 into data/oil/
#    → or request Krestenitis 5-class at m4d.iti.gr

# 3. Learn the vessel format (8 GB sample)
git lfs install
git clone https://huggingface.co/datasets/ConnorLuckettDSTG/SARFishSample data/vessels/sample

# 4. Start with Phase 1
jupyter notebook notebooks/03_oil_spill_seg.ipynb

# 5. Run the dashboard (once modules produce outputs)
streamlit run app/dashboard.py
```

---

## 📊 Evaluation

- **Segmentation (M3):** per-class IoU/Dice, oil-class IoU vs 0.54 baseline, mask overlays.
- **Detection (M1):** F1/mAP, dark-vessel flags vs AIS overlay.
- **Fishing (M2):** Precision/Recall/F1/ROC-AUC, flagged tracks inside MPA polygons.
- **Drift (M4):** plume plausibility vs current/wind fields.
- **XAI (M5):** SHAP factor breakdown + Grad-CAM saliency on slicks/vessels.

---

## 📝 Notes

- Deviates from the original spec's TensorFlow/Keras → **PyTorch** (SOTA models are PyTorch-native).
- Optical (Sentinel-2/Landsat) is intentionally **out of scope** — SAR covers all tasks here.
- See [`PROJECT_PLAN.md`](./PROJECT_PLAN.md) for full model justifications and research citations.
