● Let me lay out the entire system end-to-end, nothing skipped.

---

The Big Picture

You're building one integrated system with 6 modules. The "original contribution" is the fusion layer — linking an oil spill to nearby dark vessels as the
probable polluter. Everything else feeds into that.

---

Module 1 — Dark Fleet Monitoring

What it does: Finds ships that deliberately turned off their AIS tracker to hide.

Logic:

1. Detect EVERY vessel visible in a SAR image (ships = bright metal blobs)
2. Match each detection to AIS broadcasts (is there a vessel that should be here according to radio?)
3. Unmatched detection = dark vessel (present in SAR, absent from AIS = hiding)

Training dataset: HRSID

- 5,604 SAR images, 16,951 ship labels, 800×800 JPG
- 4,042 train (3,642 ships + 400 pure-background negatives) / 1,962 val
- Single class: ship
- Format: COCO JSON → converted to YOLO format (normalized cx, cy, w, h per label)
- OBB variant: COCO polygon → cv2.minAreaRect() → 4 corner points (for YOLOv11m-OBB)

3-model benchmark:

| Model | Why | Parameters | Expected mAP50 |
|---|---|---|---|
| YOLOv8m | Baseline, used in ~70% of SAR detection papers | imgsz=800, batch=16, epochs=50 | ~88-90% |
| YOLOv11m-OBB | Ships are diagonal in SAR — horizontal box wastes 40% area on speckle noise; OBB fits the ship tightly | imgsz=800, batch=16, epochs=50, OBB labels | ~91-93% |
| RT-DETR-L | Transformer global attention — sees ship in context of surrounding sea pattern; no NMS needed | imgsz=800, batch=8 (larger model), epochs=50 | ~92-94% |

Why batch=8 for RT-DETR: Transformer self-attention is O(n²) in memory. Half the batch vs YOLO.

AIS data source: GFW GAP events — gap.intentionalDisabling (True/False) is ground truth for whether a vessel deliberately disabled AIS. 5,000 rows
downloaded. gap.offPosition = where it went dark, gap.onPosition = where it reappeared.

---

Module 2 — Illegal Fishing Detection

What it does: Flags vessels doing illegal fishing (wrong zone, no authorization, suspicious behavior).

Training dataset: GFW /events API — FISHING type

- Dataset string: public-global-fishing-events:latest
- 11.7 million total events; we downloaded 5,000
- Target label: fishing.potentialRisk (True/False, pre-labeled by GFW — no manual labeling needed)

Features fed to XGBoost (all direct GFW columns, no raw AIS engineering):

| Feature | Why it matters |
|---|---|
| fishing.averageSpeedKnots | Fishing = slow (2–4 kn); transiting = fast (10+ kn) |
| fishing.totalDistanceKm | Long track = sustained fishing effort |
| fishing.averageDurationHours | Long events = deep offshore operations |
| distances.startDistanceFromPortKm | Far from port = less oversight |
| distances.startDistanceFromShoreKm | Deep offshore = harder to patrol |
| regions.mpa → binary | Fishing INSIDE Marine Protected Area = illegal |
| regions.highSeas → binary | High seas = weaker jurisdiction |
| fishing.vesselPublicAuthorizationStatus | unmatched = no authorization on record |
| vessel.flag → top-20 encoded | Some flags are high-risk (flags of convenience) |

Model: XGBoost

- n_estimators=500, max_depth=6, lr=0.05
- scale_pos_weight = ratio of negatives to positives (handles imbalance automatically)
- eval_metric="aucpr" — area under precision-recall curve, better than AUC for imbalanced data
- early_stopping_rounds=30 — stops when val metric stops improving

Why XGBoost and not deep learning: XGBoost pairs natively with SHAP. The SHAP plot directly shows "this vessel was flagged because: AIS-absence 35%,
restricted-area 25%, movement-pattern 20%" — which is exactly what the spec asks for in Module 5.

Deep upgrade: BiGRU on raw AIS position sequences (time-series), but XGBoost on GFW event features is the primary and will work well.

---

Module 3 — Oil Spill Segmentation (The Safe Floor)

What it does: Pixel-level segmentation of SAR images → which pixels are oil vs sea.

Why it's hard: Oil pixels = ~1.76% of all pixels. Plain cross-entropy → model predicts "all sea" (99.8% accurate but 0% useful). This is the most critical
thing to get right.

Training dataset: Zenodo 8346860

- Part I: 1,200 oil images (40.7 GB) + binary masks (oil=1, bg=0)
- Part II: 685 look-alike images + 685 no-oil images (hard negatives — masks are all-zero)
- Part III: 450 test images (held-out, never trained on)
- Image size: 2,048×2,048 px GeoTIFF, Sentinel-1 VV polarization
- Masks verified: uint8, unique values [0,1], nodata=None

The look-alikes matter: Low wind (<3 m/s), algae blooms, rain cells all look dark in SAR just like oil. Part II forces the model to learn "dark patch ≠
always oil."

Loss function: DiceFocalLoss (mandatory)

- Dice loss: maximizes intersection-over-union between predicted and true mask → works even when oil is 1.76% of pixels
- Focal loss: multiplies each pixel's loss by (1 - p)^γ → down-weights the easy "predict sea" pixels, forces model to learn the hard oil pixels
- Combined: loss = Dice + λ·Focal
- Without this: model gets 99% accuracy predicting all sea. With this: it's forced to actually find oil.

3-model benchmark:

| Model | Architecture | Pretrained on | Batch | mIoU target |
|---|---|---|---|---|
| DeepLabv3+ ResNet-50 | CNN encoder + ASPP atrous convolution | ImageNet | 16 | ~65% |
| SegFormer MiT-b2 | Hierarchical transformer + MLP decoder | ImageNet-1k | 16 | ~67% |
| OilSAM2 | SAM2 (Meta) + hierarchical memory bank + scale-adaptive fusion | SA-V video + M4D oil SAR | 4 | ~72.6% |

Why batch=4 for OilSAM2: SAM2 has a memory bank mechanism that holds multiple frame embeddings. Much larger GPU footprint than SegFormer.

Why this model progression tells a story: CNN → transformer → foundation model fine-tune. Shows you evaluated three paradigms, not just picked one.

Training setup:

- Optimizer: AdamW, lr=6e-5, weight_decay=1e-4
- Scheduler: CosineAnnealingLR (starts at lr, decays to 0 over 50 epochs)
- Mixed precision: GradScaler (FP16 forward, FP32 gradients — halves VRAM)
- Best checkpoint: saved when oil-class IoU improves
- W&B logging: one run per model

---

Module 4 — Oil Drift Prediction

What it does: Given a detected oil polygon, predicts where it'll drift over the next 24–72 hours.

Engine: OpenDrift/OpenOil — NOT a trained ML model

- Physics-based Lagrangian particle tracking
- Releases thousands of virtual particles at the spill location
- Each particle moves according to: ocean surface current + wind drag + Stokes drift + turbulent diffusion
- Output: probability density map of where oil will be at T+6h, T+24h, T+72h

Data inputs (live APIs):

- Ocean currents: Copernicus Marine Service (copernicusmarine Python client)
- Wind: ERA5 reanalysis (cdsapi)

Why physics over LSTM: LSTM trained on historical spills extrapolates poorly to unusual current/wind conditions. OpenOil is the operational tool used by
Norwegian Met Institute and EMSA for real spill response. It needs zero training data.

LSTM comparison: Kept as a "physics vs ML" exhibit — shows intellectual honesty.

---

Module 5 — Explainable AI

Two tools, used on different model types:

| Tool | Used on | What it shows |
|---|---|---|
| SHAP TreeExplainer | XGBoost (M2 fishing, M6 risk scorer) | Which features drove the prediction — "distance from port contributed +0.35 to risk score" |
| Grad-CAM | SegFormer/CNN (M1 detection, M3 oil) | Heatmap overlay on SAR image — red = pixels model focused on for the decision |

Why not SHAP on the CV models: SHAP over 512×512×3 = 786,432 features. Computationally insane and the output is per-pixel noise. Grad-CAM gives you a
clean visual saliency map in seconds.

---

Module 6 — Fusion + Risk Scoring (The Original Contribution)

What it does: Combines all module outputs into one risk assessment.

Fusion logic:

- Oil spill polygon detected in SAR (M3)
- Dark vessel positions within 160 km of the spill (M1)
- → Candidate responsible party (illegal bilge dump use case)
- 160 km chosen because ~90% of oil slicks occur within 160 km of shore, and a vessel 160 km away at a typical cruising speed of 12 knots could have been
  at the spill location within the past few hours

Risk scores:

- Security Risk Score = f(dark vessel count nearby, gap.intentionalDisabling, zone violation)
- Environmental Risk Score = f(spill area km², drift toward coast, MPA overlap)

---

Training Data vs Live (GEE) Data — The Critical Difference

This is what you were really asking. Here's the full picture:

TRAINING:
Zenodo TIF (2048×2048, specific Sentinel-1 scene, already a SAR chip)
→ rasterio.read() → normalize [0,1] → albumentations augment → 512×512 crop → model

HRSID JPG (800×800, already visual-range, pre-processed by dataset creators)
→ YOLO reads directly → internal augmentation

LIVE (REAL WORLD):
GEE pulls a FULL Sentinel-1 swath (25,000 × 17,000 px, ~500MB)
→ sar_preprocess.py:
Step 1: Read VV band (+ VH if available)
Step 2: Detect format — if min pixel value < -5 → dB scale, else linear
Step 3: If dB → convert to linear: pixel = 10^(pixel/10)
(dB range is typically -30 to +5 for Sentinel-1 ocean scenes)
Step 4: Lee speckle filter (7×7 window) - Speckle is multiplicative noise that makes SAR look grainy - Lee filter smooths it while preserving bright ship returns
Step 5: Percentile-clip at 99.5th % then normalize to [0,1] - Without clipping, one bright ship hotspot compresses the whole range
Step 6: Chip into 512×512 tiles, 64px overlap - 64px overlap prevents missing detections at chip boundaries
Step 7: Skip tiles where mean < 1% (empty ocean, no signal of interest)
Step 8: Save each chip as 3-channel PNG (replicate VV to RGB if single-band)
→ model(chip) for each chip
→ reassemble scene-level detections with offset (x, y) per chip

Why the preprocessing gap exists:

- HRSID is already JPG — the dataset creators already did the preprocessing. The model learned from visually "clean" images.
- Zenodo images may or may not be in dB (UNVERIFIED — must check on day 1 of training).
- GEE exports Sentinel-1 in whatever format the archive stores it (typically dB for GRD products).
- If you feed raw dB values (-25 to +5) to a model trained on [0,1] linear values → the model sees completely wrong pixel distributions → random
  predictions.

The sar_preprocess.py file bridges this gap — it makes live GEE data look like training data before it hits the model.

---

End-to-End Data Flow Summary

TRAIN TIME:
┌─────────────┐ ┌──────────────┐ ┌─────────┐ ┌──────────┐
│ Zenodo TIFs │───►│ OilSpillDS │───►│ DiceFoc │───►│ checkpoint│
│ (2048×2048) │ │ normalize │ │ Loss │ │ best.pt │
└─────────────┘ │ crop 512×512 │ └─────────┘ └──────────┘
│ augment │
└──────────────┘

┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐
│ HRSID │───►│ YOLO fmt │───►│ YOLO │───►│best.pt │
│ (800×800)│ │ OBB labels │ │ train │ │ │
└──────────┘ └────────────┘ └──────────┘ └──────────┘

┌─────────────────────┐ ┌──────────────────┐ ┌──────────┐
│ GFW fishing_5k.csv │───►│ load_gfw_features│───►│ XGBoost │───►│xgb.json│
│ potentialRisk label │ │ 12 features │ │ + SHAP │ └────────┘
└─────────────────────┘ └──────────────────┘ └──────────┘

INFERENCE TIME (live dashboard):
User picks area + date
│
▼
GEE pulls Sentinel-1 swath
│
▼
sar_preprocess.py
[dB→linear] → [Lee filter] → [normalize] → [chip 512×512]
│ │
▼ ▼
Oil model (best.pt) Vessel model (best.pt)
pixel masks detections + coords
│ │
▼ ▼
M4: OpenOil drift M1: match to GFW GAP events
forecast polygon → dark vessel flags
│ │
└──────────────┬───────────────┘
▼
M6: Fusion (oil polygon + dark vessel within 160km)
→ Risk scores
│
▼
Streamlit dashboard
(all 7 layers on Folium map)

---

What's Pending Before This Runs

1. Verify Zenodo image format (day 1 of training): rasterio.open(img).read().min() — negative = dB, need conversion; positive = linear, skip conversion
2. Download 40.7 GB Zenodo Part I images
3. Install OilSAM2 + SAM2 checkpoint
4. Train all 6 models in order (DeepLabv3+ → SegFormer → OilSAM2 → YOLOv8m → YOLOv11m-OBB → RT-DETR-L)
5. Train XGBoost on GFW fishing CSV
