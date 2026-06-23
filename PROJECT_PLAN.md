# PROJECT_PLAN.md — Source of Knowledge

## Explainable Multi-Modal AI Framework for Maritime Security Intelligence

### Dark Fleet Monitoring · Illegal Fishing Detection · Oil Spill Risk Assessment

> **This is the single source of truth for the project.** Every decision, dataset, model
> choice, justification, and hard-won lesson lives here. If something contradicts this file,
> this file wins until explicitly updated.

---

## 0. TL;DR — the 60-second version

- **Goal:** one integrated, _explainable_ system over Sentinel-1 SAR + AIS + ocean data that
  finds dark vessels, flags illegal fishing, segments oil spills, predicts spill drift, and
  scores maritime risk — with a Streamlit dashboard on top.
- **The novelty (what makes it ONE system, not six notebooks):** the **fusion layer** —
  link a detected oil slick to nearby _dark_ (non-broadcasting) vessels as candidate
  responsible parties, with explainable risk scoring.
- **Data is the only real blocker, and it's solved.** Two downloads (oil masks + SAR vessel
  sample) = the entire CV foundation. AIS/ocean/zones are live pulls, not big downloads.
- **Framework:** PyTorch (not the doc's TF/Keras — justified below).
- **Models:** modern/SOTA choices replacing the doc's baselines, all evidence-backed below.
- **Strategy:** floor-first phasing — Phase 1 (oil segmentation) alone is a passing grade;
  everything after is upside.

---

## 1. Project objective (from the spec)

Develop an **Explainable Multi-Modal AI framework** that integrates satellite imagery, AIS
data, oceanographic information, and maritime environmental data to identify dark vessels,
detect illegal fishing, assess oil spill risks, and support maritime security and
environmental protection.

**Source spec:** `Intern Maritime Security Intelligence.docx` (in this folder).

---

## 2. Problem statement

Maritime regions face rising **illegal fishing**, **unauthorized vessel operations**, **AIS
spoofing/disabling**, and **marine pollution**. Many vessels intentionally disable or
manipulate AIS to avoid detection while fishing illegally, smuggling, or operating in
restricted zones. Oil discharges (accidental and deliberate bilge-dumping) threaten
ecosystems, fisheries, and coastal communities. Traditional monitoring uses separate
surveillance systems and manual inspection — hard to scale. This project unifies AI,
satellite intelligence, maritime analytics, and **explainable** decision-making into one
framework.

**Key real-world facts grounding the project:**

- AIS data alone misses ~**90%** of SAR-detected fishing vessels inside marine protected areas
  (Raynor et al., _Science_, 2025 — an ecology/policy paper, **not** a methods paper).
- IUU (Illegal, Unreported, Unregulated) fishing is **>20%** of global catch.
- The detection methodology lives in **Paolo et al., _Nature_ 2024** ("Satellite mapping
  reveals extensive industrial activity at sea") — open code + figshare data.
- ~**90%** of oil slicks occur within **160 km of shore** → ships and shorelines are key
  context for attributing a spill's source. This is _why_ fusion makes sense.

---

## 3. Scope & confirmed decisions

| Decision             | Choice                      | Rationale                                                                                                                                                                           |
| -------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Scope**            | All 6 modules, **phased**   | A working core (oil + dark-vessel + dashboard) is a guaranteed floor; M2/M4/M5/M6 layer on as upside. Not all-or-nothing.                                                           |
| **Framework**        | **PyTorch**                 | Doc says TF/Keras, but SOTA models (SegFormer, YOLOv8/11, segmentation-models-pytorch) are PyTorch-native and the user already knows PyTorch. User authorized choosing best models. |
| **Deliverable**      | **Streamlit dashboard** app | Matches the doc's "Integrated Maritime Intelligence Dashboard"; demos well.                                                                                                         |
| **Area of interest** | **Dataset-native regions**  | Mediterranean for oil (Krestenitis), global for vessels. Simplest — no extra Indian-EEZ data pulls.                                                                                 |

---

## 4. Datasets — final, verified (the part that was the actual blocker)

### 4.1 The mental model

> **Zenodo binary trains the oil model. HRSID trains the vessel model. GFW API provides
> pre-labeled fishing + dark-vessel events. Sentinel-1 + ocean data let you run it live.**

You need **2 labeled CV datasets** (oil + vessels) and **1 rich API** (GFW for AIS events).
That's the whole data story. Krestenitis is dropped (gated + Zenodo binary is sufficient).

### 4.2 Train-on datasets

| Dataset               | Module     | What it is                                                                                                                                  | Access                                                                                                                                                                     | Status                                                                      |
| --------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Krestenitis / M4D** | M3 oil     | 5-class masks (sea, oil, look-alike, ship, land), 1002 train / 110 test, ~400 MB, 650×1250 px, Sentinel-1 VV, 10 m                          | request at [m4d.iti.gr](https://m4d.iti.gr/oil-spill-detection-dataset/)                                                                                                   | **❌ DROPPED — gated + Zenodo binary is sufficient. Not worth the wait.**   |
| **Zenodo 8346860**    | M3 oil     | Binary oil masks (oil=1/bg=0), 1200 imgs 2048×2048, 47 GB, CC-BY                                                                            | [direct download](https://zenodo.org/records/8346860)                                                                                                                      | **✅ PRIMARY OIL DATASET — masks verified, images via Colab 100 GB disk**   |
| **HRSID**             | M1 vessels | 5,604 SAR images, 16,951 ship instances, 800×800 px, **COCO JSON** (bbox + instance outline), 0.5/1/3 m res, inshore 18.4% / offshore 81.6% | JPG+labels: https://drive.google.com/file/d/1NY3ovgc-woDlNoQdyqzRB3t9McOBH5Ms/view · 400 negatives: https://drive.google.com/file/d/1U0Sj1SHoq-2VjXXUKwpXae6rBI3YjyDP/view | **✅ VERIFIED + CONVERTED (2026-06-22). See §4.5 for full verified facts.** |
| **LS-SSDD-v1.0**      | M1 vessels | 15 Sentinel-1 scenes → 9,000 pre-cut 800×800 sub-images, SAR-expert labels via AIS + Google Earth                                           | Portal link in README is broken — routes to unrelated OSDataset2.0 (SAR-optical matching), not LS-SSDD                                                                     | **❌ DROPPED — portal link dead/rotated. HRSID alone is sufficient.**       |
| **SARFishSample**     | M1 vessels | 1 scene (GRD+SLC), 8.2 GB unzipped, xView3-derived labels                                                                                   | [HuggingFace](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFishSample) + token, `git lfs`                                                                          | **⚠️ DROPPED — fishing-specific labels, not general vessel detection**      |
| **SARFish (full)**    | M1 vessels | 753 scenes, **6.5 TB** — pull **GRD-only subset**                                                                                           | [HuggingFace](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFish)                                                                                                   | subset only                                                                 |
| **xView3-SAR**        | M1 vessels | Official benchmark splits, ~1000 scenes avg 29,400×24,400 px                                                                                | [iuu.xview.us](https://iuu.xview.us/) signup                                                                                                                               | optional, leaderboard compare                                               |

### 4.3 Live feeds (no big download)

| Feed                              | Module    | Source                                                                                    | Status                       |
| --------------------------------- | --------- | ----------------------------------------------------------------------------------------- | ---------------------------- |
| **GFW fishing events**            | M2        | `GET /v3/events` · dataset: `public-global-fishing-events:latest`                        | ✅ 5,000 rows saved           |
| **GFW AIS gap events**            | M1        | `GET /v3/events` · dataset: `public-global-gaps-events:latest`                           | ✅ 5,000 rows saved           |
| **GFW encounter events**          | M1, M6    | `GET /v3/events` · dataset: `public-global-encounters-events:latest`                     | ✅ 2,000 rows saved           |
| **EEZ boundaries**                | M2        | Marine Regions WFS SHAPE-ZIP · `typeName=MarineRegions:eez`                              | ✅ 285 zones, world_eez.gpkg  |
| **High seas boundary**            | M2        | Marine Regions WFS SHAPE-ZIP · `typeName=MarineRegions:high_seas`                        | ✅ 1 polygon, high_seas.gpkg  |
| **Marine Protected Areas (WDPA)** | M2        | **Not needed as shapefile** — GFW events carry `regions.mpa` + `regions.mpaNoTake` MRGID IDs | ✅ embedded in GFW CSV    |
| **Ocean currents**                | M4        | Copernicus Marine Service                                                                 | pending Phase 4               |
| **Wind (speed/direction)**        | M4        | ERA5 (`cdsapi`) / NOAA                                                                   | pending Phase 4               |
| **Sentinel-1 on demand**          | live demo | Google Earth Engine (`earthengine-api` + `geemap`) or ASF/Copernicus                     | pending Phase 6               |

### 4.4 Skip / out of scope

- **Sentinel-2 / Landsat optical** — doc lists it, but SAR does everything here. Don't burn time.
- **SLC imagery** in SARFish — complex-valued raw radar; detection models don't use it. **GRD only.**

### 4.5 ✅ Verified facts from Colab inspection (2026-06-22)

**Zenodo mask format (actually checked):**

- Mask dtype: `uint8`
- Unique values: `[0, 1]` — confirmed binary
- nodata: `None` — no sentinel/padding value to handle
- Oil pixel % in sample mask: **2.23%** (consistent with median 1.76% across all 1200)
- `loaders.py` mask loading (`src.read(1).astype(np.int64)`) is correct for this format

**Zenodo actual folder structure on disk:**

```
/content/
  masks/Mask_oil/           ← 1200 oil masks (Part I) ✅ downloaded
  nooil_masks/Mask_no_oil/  ← 685 no-oil masks (Part II) ✅ downloaded
  lookalike_masks/Mask_lookalike/ ← 685 look-alike masks (Part II) ✅ downloaded
  dataset/images/           ← EMPTY — SAR images not yet downloaded
```

**SAR image pixel format: ⚠️ UNVERIFIED**

- We have masks but not the actual SAR image TIFFs yet
- Images are the 40.7 GB Part I download (pending on Lightning.ai)
- **Must verify on day 1 of Lightning training:** run `rasterio.open(img).read()` and check:
  - If values in `[-30, 0]` → raw dB → need `10^(x/10)` conversion before inference
  - If values in `[0, 1]` or `[0, 65535]` → already linear → skip conversion
- Do NOT assume format until verified. The GEE preprocessing pipeline depends on this.

**Model pretraining (verified from architecture):**

- All 3 models use **transfer learning, not training from scratch**
- DeepLabv3+: ImageNet pretrained (SMP loads automatically)
- SegFormer MiT-b2: ImageNet-1k pretrained (HuggingFace loads automatically)
- OilSAM2: SAM2 pretrained by Meta on SA-V + already fine-tuned on M4D oil SAR data — strongest prior
- Why this matters: works with ~1200 training images instead of millions; ImageNet edge/texture features transfer well even to grayscale SAR

**SAR oil detection physics (why VV polarization, why dark patches):**

- Sentinel-1 emits C-band microwaves (5.4 GHz), measures backscatter (σ°)
- Wind creates capillary waves (1–10 cm ripples) → rough surface → high σ° → bright
- Oil creates surface tension film → dampens capillary waves → smooth surface → low σ° → **dark patch**
- Look-alikes are also dark: low wind (<3 m/s), algae bloom, rain cells, upwelling zones
- Traditional systems (EMSA CleanSeaNet): CFAR finds dark patches → human analyst cross-checks AIS + wind + optical
- Our DL model: replaces the human confirmation step by learning the difference from 2570 labeled examples
- The model learns pixel intensity patterns + shape — it has no knowledge of dB/VV/physics explicitly

**Ship dataset decision (2026-06-22):**

- Dropped SARFishSample (fishing-specific, not general vessel detection, 8 GB for 1 scene)
- Dropped SAR-Ship-Dataset CAESAR (256px chips too small, Baidu Drive, Gaofen-3 dominated)
- Dropped SSDD (only ~1200 images, outdated)
- Dropped LS-SSDD (radars.ac.cn portal link broken — routes to unrelated OSDataset2.0)
- **Primary + ONLY: HRSID** — fully sufficient alone

**HRSID verified facts (Colab inspection 2026-06-22):**

- Download: 0.57 GB compressed JPG + 0.39 GB negatives (both Google Drive links live)
- Structure: `HRSID_JPG/JPEGImages/` (5,604 images) + `annotations/train2017.json` + `test2017.json`
- Extra: `inshore_offshore/` with separate offshore.json (4,573 imgs, 8,745 anns) and inshore.json
- Train split: **3,642 images** (from train2017.json)
- Val split: **1,962 images** (from test2017.json)
- 400 negatives: `.png` files in `pure background/` subfolder (NOT .jpg, NOT root level)
- After adding negatives: **4,042 train images total** (3,642 ship + 400 negatives)
- 1 stem collision fixed: `P0128_600_1400_4800_5600.png` renamed to `P0128_600_1400_4800_5600_neg.png`
- Classes: 1 (`ship`) — single category
- Annotations: COCO bbox `[x, y, w, h]` (pixel, top-left) + full polygon segmentation outline
- SAR physics confirmed in image: ships = bright white blobs on dark sea; Doppler trail visible on moving ships
- COCO→YOLO conversion done: `data.yaml` at `/content/HRSID_yolo/data.yaml`
- YOLO format: normalized `[cx, cy, w, h]` in `labels/train/` and `labels/val/`
- Ready for `yolo train data=/content/HRSID_yolo/data.yaml model=yolov8m.pt`

**Vessel detection 3-model benchmark (decided 2026-06-22, research-backed):**

| #   | Model        | mAP50 target | Architecture                     | Why                                                                    |
| --- | ------------ | ------------ | -------------------------------- | ---------------------------------------------------------------------- |
| 1   | YOLOv8m      | ~88-90%      | CNN, horizontal boxes            | Baseline — used in ~70% of SAR papers                                  |
| 2   | YOLOv11m-OBB | ~91-93%      | CNN, oriented bounding boxes     | Tight rotated boxes for diagonal ships; uses HRSID polygon annotations |
| 3   | RT-DETR-L    | ~92-94%      | Transformer, anchor-free, no NMS | Global attention captures ship-sea context; paradigm shift             |

- All three models are in Ultralytics — same data.yaml, near-identical training commands
- Story: CNN horizontal → CNN oriented → Transformer (clean progression, matches oil benchmark approach)
- Why OBB matters: ships are elongated at arbitrary angles; horizontal box wastes ~40% of area on speckle noise background
- OBB source: fit minimum enclosing rotated rectangle to HRSID polygon annotations (need conversion script)
- Research: SMEP-DETR achieves 93.2% mAP on HRSID (March 2025) but research code only; RT-DETR-L is the practical SOTA
- AC-YOLO (YOLO11-based, 2026) +1.5% AP over YOLO11 baseline but requires custom implementation
- Training on Lightning.ai H100 (same as oil models)

**Sanity check result (5 epochs, Colab T4, 2026-06-22):**

- mAP50: **84.7%** after 5 epochs — pipeline confirmed clean
- mAP50-95: 57.4%, Precision: 87.8%, Recall: 76.1%
- Dataset scan: 4042 train (400 backgrounds correct), 1962 val (1 background), 0 corrupt
- Time: ~4 min/epoch on T4 → **~40 sec/epoch on H100 → 50 epochs ≈ 35 min**
- All 3 models estimated ~2 hours total on H100

Run order on Lightning.ai H100:

```bash
# 1. Baseline
yolo train model=yolov8m.pt     data=data.yaml epochs=50 imgsz=800 batch=16 name=hrsid_yolov8m

# 2. OBB (generate OBB labels first — see train_detection.py --convert_obb)
yolo train model=yolo11m-obb.pt data=data_obb.yaml epochs=50 imgsz=800 batch=16 name=hrsid_yolo11m_obb

# 3. Transformer
yolo train model=rtdetr-l.pt    data=data.yaml epochs=50 imgsz=800 batch=8  name=hrsid_rtdetr_l
```

### 4.6 ✅ GFW API verified facts (2026-06-23)

**Token:** registered as app name "fishery", created 2026-06-23. JWT format (eyJ...), length ~790 chars.
Auth header: `Authorization: Bearer {TOKEN}`.
Base URL: `https://gateway.api.globalfishingwatch.org/v3`

**Critical API quirks (learned the hard way):**
- `datasets[0]` param is REQUIRED on every `/events` call — 422 without it
- `offset=0` is REQUIRED whenever `limit` is sent — 422 without it
- bbox as comma-string (`"-6,30,36,46"`) does NOT work — use `bbox[0]` through `bbox[3]`
- Event type `GAP` works; `AIS_OFF` does NOT exist as a type string

**Verified dataset names:**

| Event type  | Dataset string                                | Total available | Our download |
| ----------- | --------------------------------------------- | --------------- | ------------ |
| `FISHING`   | `public-global-fishing-events:latest`         | **11,793,636**  | 5,000 rows   |
| `GAP`       | `public-global-gaps-events:latest`            | **562,615**     | 5,000 rows   |
| `ENCOUNTER` | `public-global-encounters-events:latest`      | confirmed 200   | 2,000 rows   |

**Saved files:** `/content/gfw_data/fishing_events_5k.csv`, `gap_events.csv`, `encounter_events.csv`

**Fishing events columns (for Module 2 XGBoost):**
```
start, end, id, type, boundingBox,
position.lat, position.lon,
regions.mpa, regions.eez, regions.rfmo, regions.fao, regions.majorFao,
regions.eez12Nm, regions.highSeas, regions.mpaNoTakePartial, regions.mpaNoTake,
distances.startDistanceFromShoreKm, distances.endDistanceFromShoreKm,
distances.startDistanceFromPortKm, distances.endDistanceFromPortKm,
vessel.id, vessel.name, vessel.ssvid, vessel.flag, vessel.type,
vessel.publicAuthorizations, vessel.nextPort,
fishing.totalDistanceKm, fishing.averageSpeedKnots, fishing.averageDurationHours,
fishing.potentialRisk,                    ← TARGET LABEL for M2 (True/False)
fishing.vesselPublicAuthorizationStatus   ← publicly_authorized / partially_matched / unmatched
```

**GAP events columns (for Module 1 dark fleet):**
```
start, end, id, type, boundingBox, position.lat, position.lon,
regions.mpa, regions.eez, regions.rfmo, regions.fao, regions.majorFao,
regions.eez12Nm, regions.highSeas, regions.mpaNoTakePartial, regions.mpaNoTake,
distances.startDistanceFromShoreKm, distances.endDistanceFromShoreKm,
distances.startDistanceFromPortKm, distances.endDistanceFromPortKm,
vessel.id, vessel.name, vessel.ssvid, vessel.flag, vessel.type, vessel.nextPort,
gap.intentionalDisabling,    ← TARGET LABEL for M1 dark fleet (True/False)
gap.distanceKm, gap.durationHours, gap.impliedSpeedKnots,
gap.positions12HoursBeforeSat, gap.positionsPerDaySatReception,
gap.offPosition.lat, gap.offPosition.lon,   ← where vessel went dark
gap.onPosition.lat, gap.onPosition.lon      ← where vessel reappeared
```

**Impact on module design:**
- M2: no need to engineer "is it fishing?" from raw AIS positions. `fishing.potentialRisk` is
  already the label. XGBoost features = speed, distance from port/shore, MPA zone, duration,
  authorization status, flag state.
- M1: `gap.intentionalDisabling` is GFW's assessment of deliberate AIS-off. Use as ground truth
  for dark-vessel classification. `gap.offPosition` / `gap.onPosition` are the spatial anchors
  for the fusion layer.
- M6: Encounter events (ship-to-ship rendezvous at sea) are direct suspicious-transfer flags.

**EEZ download (2026-06-23, ✅ DONE):**
- GeoJSON WFS fails with `OGR_GEOJSON_MAX_OBJ_SIZE` error — do NOT use outputFormat=application/json
- Correct method: `outputFormat=SHAPE-ZIP` → wget → unzip → `gpd.read_file(shp)`
- Saved: `/content/shapefiles/world_eez.gpkg` — **285 zones**
- Verified columns (all lowercase): `mrgid`, `geoname`, `sovereign1`, `territory1`, `iso_ter1`,
  `iso_sov1`, `area_km2`, `mrgid_eez`, `pol_type`, `geometry`
- Key columns for fusion: `sovereign1` (country), `geoname` (full zone name), `area_km2`
- `ais_matching.py` updated to use `sovereign1` / `geoname` (not old TERRITORY1/GEONAME)

### 4.7 Zenodo on Colab (100 GB disk strategy)

Krestenitis is dropped. Everything runs on Colab (100 GB disk), not Lightning.ai.

**Download order (fits in 100 GB):**
1. Part I images: 40.7 GB + masks ~2 GB → train all 3 oil models → **~43 GB used**
2. Evaluate on Part III test images: 9.86 GB → **~53 GB used**
3. Optionally clear Part I images → download Part II (45.9 GB) for hard-negative fine-tuning

Part II (look-alike + no-oil images) is optional — only needed if models overfit. Skip until
training results show overfitting.

### 4.8 ⚠️ Hard-won dataset lessons (do not repeat these mistakes)

1. **Binary classification ≠ segmentation.** Kaggle "oil spill" sets like
   `harikrishnacs/...` and `vighneshanand/...` are _whole-image oil/no-oil labels with no
   masks_. They kill the fusion layer (no slick polygon to spatially join). **Heuristic:** if
   it ships `/oil` and `/no_oil` folders → classification, skip. If `images/` + `masks/` (or
   `labels/`) pairs → segmentation, usable.
2. **Never `git clone` the full xView3/SARFish** — it's TB-scale and drags SLC. Use selective
   download (see §4.6).
3. **The hackweek repo** (`oceanhackweek/ohw23_proj_oil`) is a prototype — segmentation is
   marked "(future)", uses COSMO-SkyMed not Sentinel-1. Only its reference list is useful.
4. **xView3 the competition is closed** (winners announced 31 Jan 2022). The _data_ stays
   open. We use it as a benchmark dataset, no registration/submission. The signup form still
   looks like a contest but it's just an account gate.

### 4.6 Selective GRD-only download (the right way to get vessel data)

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ConnorLuckettDSTG/SARFish",
    repo_type="dataset",
    allow_patterns=["*GRD*"],     # skip all SLC
    # narrow further to ~20–40 scenes, not all 753
)
```

> Labels route through the DIU/xView3 side (small CSVs) even when imagery comes from HF.

---

## 5. Per-module model choices (evidence-backed, replacing the doc's baselines)

> The doc lists _baseline_ models. The user authorized choosing the best models per use case.
> Each choice below is justified with research found during planning.

### Module 1 — Dark Fleet Monitoring (SAR vessel detection + AIS matching)

- **Doc baseline:** YOLO / Faster R-CNN / CNN.
- **Chosen — THREE-MODEL BENCHMARK** (decided 2026-06-22, same approach as oil spill):
  **YOLOv8m** (baseline) → **YOLOv11m-OBB** (oriented boxes) → **RT-DETR-L** (transformer SOTA).
  All Ultralytics, all trained on HRSID 800×800 chips.
- **Why:** xView3 winners used heavy custom encoder-decoders (1st place = CircleNet encoder +
  U-Net decoder, objectness/length/classification heads, **63 h** train, full-scene inference
  in <15 min on a V100 w/ 60 GB RAM) — out of scope for our compute. A modern single-stage
  detector on chipped tiles is the right effort/payoff, and recent work adapts YOLOv8
  specifically for SAR vessel detection.
- **Training data (decided 2026-06-22):**
  - **Primary: HRSID** — 5,604 SAR images, 16,951 instances, COCO JSON format.
    Google Drive links VERIFIED LIVE (2026-06-22). Download: JPG version + 400 pure-background
    negative images (hard negatives for vessel detection, same concept as oil look-alike masks).
    Has BOTH bounding boxes AND ship instance outlines — outlines can later be used for ship
    size/type estimation in M2 (fishing vs cargo vs tanker differ in size).
    ⚠️ COCO format: Ultralytics YOLOv8 reads COCO JSON directly via data.yaml — no conversion script needed.
  - **Add-on: LS-SSDD** — 9,000 Sentinel-1 sub-images pre-cut to 800×800, AIS-assisted
    labeling (ships without AIS still annotated). Pure Sentinel-1 = matches production sensor.
    ⚠️ Download is through radars.ac.cn (Chinese govt portal) — may require registration or
    be inaccessible from outside China. Do NOT block M1 training waiting for this; HRSID alone
    is sufficient.
  - **Dropped:** SARFishSample (fishing-specific labels, not general vessel detection),
    SAR-Ship-Dataset (256px too small, Baidu Drive, Gaofen-3 dominated), SSDD (too small ~1200 imgs).
- **Dark-vessel logic (the actual intellectual content):**
  1. Detect every vessel in the SAR scene (CNN; classic alternative is CFAR — Constant False
     Alarm Rate thresholding on sea clutter).
  2. **Probabilistically match** detections to AIS broadcasts — timestamps don't coincide and
     one message can match multiple vessels, so use a probabilistic model over AIS records
     before/after image time, **not** naive interpolation.
  3. **Unmatched detection = dark vessel.** Detect everything, subtract what's broadcasting,
     the residual is running dark.
- **Why it's hard CV:** annotated boxes are ~0.005% of pixels (a few bright pixels in a huge
  scene); most vessels <40 m (tiny); SAR speckle + coastline artifacts + rocks/windmills
  mimic ships.
- **Lean fallback:** use a pretrained xView3 winning model for detection, spend the saved time
  on matching + fusion.

### Module 2 — Illegal Fishing Detection (AIS trajectories + zones)

- **Doc baseline:** RandomForest / XGBoost / LSTM.
- **Chosen (primary):** **XGBoost on engineered trajectory features** (speed, heading
  variance, turning angle, time-in-zone, distance-to-port, loitering metrics).
- **Chosen (deep upgrade):** **Bidirectional GRU/LSTM** on raw AIS sequences.
- **Why:** XGBoost is accurate _and_ pairs natively with **SHAP** (Module 5) → directly
  produces the doc's feature-importance plot. Research: **BiGRU ≈ 89.7%** on fishing-vessel
  trajectory classification (best among RNNs); transformers also strong but heavier.
- **Zone logic:** geopandas spatial join of vessel positions vs MPA/EEZ polygons → fishing
  behavior _inside a restricted zone_ = suspected illegal. Learned behavior model + rules layer.
- **Training data (verified 2026-06-23):** GFW `/events` API, `FISHING` type. 11.7M events
  available. `fishing.potentialRisk` (True/False) = target label, pre-computed by GFW.
  Features ready to use: `fishing.averageSpeedKnots`, `fishing.totalDistanceKm`,
  `distances.startDistanceFromPortKm`, `distances.startDistanceFromShoreKm`,
  `regions.mpa`, `regions.highSeas`, `fishing.vesselPublicAuthorizationStatus`, `vessel.flag`.
  **No raw AIS engineering needed** — GFW events are already labeled fishing episodes.

### Module 3 — Oil Spill Detection (SAR semantic segmentation) 🟢 SAFE FLOOR

**THREE-MODEL BENCHMARK** (decided 2026-06-22, all trained on Lightning.ai H100):

| #   | Model                | mIoU target | Train time (H100) | Role             |
| --- | -------------------- | ----------- | ----------------- | ---------------- |
| 1   | DeepLabv3+ ResNet-50 | ~65%        | ~45 min           | Baseline         |
| 2   | SegFormer MiT-b2     | ~67%        | ~60 min           | Mid comparison   |
| 3   | **OilSAM2**          | **~72%+**   | ~90 min           | **SOTA primary** |

- **OilSAM2** (arxiv 2603.10231, March 2026) — current SOTA for SAR oil spill segmentation.
  Extends Meta's SAM2 with: (1) hierarchical memory bank at texture/structure/semantic levels,
  (2) scale-adaptive fusion module, (3) structure-semantic consistent memory updates.
  Achieves **72.62% mIoU on M4D**, beats all CNN + transformer + SAM baselines.
  Code: https://github.com/Chenshuaiyu1120/OILSAM2 ← clone this into src/models/

- **🔑 THE critical lever — class imbalance (confirmed from data):**
  - Zenodo dataset: median oil = **1.76%**, mean 2.98%, min 0.12%, max 57.37%
  - ALL 1200 masks have oil (100% hit rate — no wasted samples)
  - Part II look-alikes: masks ALL-ZERO → hard negatives, force model to learn oil vs imposters
  - Use **Dice + Focal loss** — plain CrossEntropy → model predicts "all sea"

- **Dataset for training (all three parts, all on Lightning storage):**
  - Part I: 1200 oil images (40.7 GB) + binary masks ✅ masks downloaded
  - Part II: 685 look-alike + 685 no-oil images (45.9 GB) + masks ✅ masks downloaded
  - Part III: 450 test images (9.86 GB) — held-out evaluation only
  - Total training: 2570 samples (binary [0,1])

- **How masks were made (EMSA CleanSeaNet):**
  Sentinel-1 → CFAR auto-detector flags dark patches → human analyst cross-checks:
  AIS (ship nearby in ±3h?), wind (<3 m/s = look-alike risk), optical (Sentinel-2 if available).
  Polygon drawn manually. Human-expert labels backed by 3 independent sources.

### Module 4 — Oil Spill Drift Prediction

- **Doc baseline:** LSTM Temporal / XGBoost.
- **Chosen (primary):** **OpenDrift / OpenOil** — physics-based Lagrangian particle tracking,
  driven by Copernicus currents + ERA5 wind.
- **Chosen (comparison exhibit):** small **LSTM**, framed as "physics vs ML".
- **Why:** research shows physics models handle extreme wind/current regimes far better than
  pure LSTM (which produces erratic boundary evolution); OpenOil needs **no training data** and
  is the operational gold standard (Norwegian Met Institute). Pure-ML drift on limited data
  underperforms — so don't make LSTM load-bearing.

### Module 5 — Explainable AI

- **Doc baseline:** SHAP / LIME.
- **Chosen:** **SHAP** for tabular models (M2 fishing classifier, M6 risk scorer) — native fit
  with XGBoost, renders the doc's factor breakdown (e.g. AIS-absence 35%, restricted-area 25%,
  movement-pattern 20%, currents 12%, wind 8%). **Grad-CAM** (`pytorch-grad-cam`) for the CV
  models (M1, M3) — visual saliency over SAR is the correct explanation; SHAP over dense pixels
  is impractical. LIME optional.

### Module 6 — Maritime Risk Assessment + Fusion (the original contribution)

- **Chosen:** rule-weighted scoring + a small **gradient-boosted** model over fused features →
  **Security Risk Score** (dark-vessel proximity, AIS anomaly, zone violation) and
  **Environmental Risk Score** (spill severity, drift toward coast/MPA). Outputs high-priority
  surveillance zones.
- **Fusion (headline novelty):** geopandas/shapely co-location — fresh slick polygon + nearby
  dark-vessel points within X km → candidate responsible party (illegal bilge-dumping use
  case). This is what makes it one system.

---

## 6. System architecture

```
Sentinel-1 SAR ─┐
AIS tracks ──────┼─► Preprocessing ─► Feature Extraction ─► [M1 Dark Fleet Detection]
Oceanographic ──┤                                          [M2 Illegal Fishing Classifier]
Zone shapefiles ─┘                                          [M3 Oil Spill Segmentation]
                                                                      │
                                                            [M4 Drift Forecast (OpenOil)]
                                                                      │
                                          [M5 Explainability: SHAP (tabular) + Grad-CAM (CV)]
                                                                      │
                            [M6 Fusion + Risk Scoring → Security & Environmental Risk]
                                                                      │
                                          📊 Integrated Maritime Intelligence Dashboard
                                                          (Streamlit + Folium/kepler)
```

---

## 7. Technical stack (PyTorch-centric)

| Layer            | Tools                                                                                                                                              |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| CV               | PyTorch, torchvision, segmentation-models-pytorch, HuggingFace Transformers (SegFormer), Ultralytics YOLO, albumentations                          |
| Tabular/sequence | XGBoost, scikit-learn, PyTorch (BiGRU)                                                                                                             |
| Explainability   | shap, pytorch-grad-cam, lime (optional)                                                                                                            |
| Geospatial       | rasterio (SAR GeoTIFFs), geopandas + shapely (spatial joins / fusion), pyproj                                                                      |
| Drift            | opendrift (OpenOil)                                                                                                                                |
| Data feeds       | GFW Python client, earthengine-api + geemap, copernicusmarine, cdsapi                                                                              |
| App / viz        | Streamlit, Folium / kepler.gl, plotly                                                                                                              |
| Compute          | local RTX 3050 (dev/debug) · **Colab Pro 100 GB disk (ALL training — oil + vessel + fishing models)** · Lightning.ai H100 only if Colab is insufficient |
| Tracking         | Weights & Biases or TensorBoard                                                                                                                    |

> **Compute reality:** with Colab Pro + Lightning.ai, GPU is _not_ the bottleneck — **data
> logistics is**. Colab/Lightning give compute, not free terabytes. Keep oil data tiny
> (lives anywhere) and vessels as GRD-only subsets. Mount Drive / use Lightning storage so
> checkpoints survive disconnects.

---

## 8. Repo structure

```
D:\larplarplarpsahur\
├─ PROJECT_PLAN.md          # this file — source of truth
├─ README.md
├─ requirements.txt
├─ data/                    # gitignored — datasets land here
│  ├─ oil/                  # Zenodo 8346860 (images + masks)
│  └─ vessels/              # HRSID + LS-SSDD (chipped tiles)
├─ notebooks/
│  ├─ 01_dark_fleet.ipynb
│  ├─ 02_illegal_fishing.ipynb
│  ├─ 03_oil_spill_seg.ipynb
│  ├─ 04_oil_drift.ipynb
│  └─ 05_06_xai_risk.ipynb
├─ src/
│  ├─ data/                 # loaders, chipping, AIS parsing
│  ├─ models/               # segmentation, detection, classifiers
│  ├─ fusion/               # AIS matching, spatial co-location
│  ├─ explain/              # SHAP + Grad-CAM
│  └─ viz/                  # map + plot helpers
└─ app/
   └─ dashboard.py          # Streamlit entrypoint
```

---

## 9. Phased execution (floor-first)

| Phase    | Module(s)          | Deliverable                                         | Notes                                                                                                                                                          |
| -------- | ------------------ | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0**    | Setup              | Repo skeleton, data access                          | Zenodo masks ✅; HRSID downloaded + YOLO-converted ✅; pipeline sanity-checked (84.7% mAP50/5ep) ✅; GFW token ✅; GFW data (fishing/gap/encounter) ✅; EEZ ⏳; OBB conversion pending |
| **1** 🟢 | M3 oil             | Trained segmenter beating ~0.54 oil IoU             | **Safe floor — passes alone.** Baseline → Dice+Focal → SegFormer                                                                                               |
| **2**    | M1 dark fleet      | Detector + AIS matching + dark flags + map          | Chip tiles → YOLOv8 (or pretrained)                                                                                                                            |
| **3**    | M2 illegal fishing | XGBoost/BiGRU + zone join → suspected-illegal flags | GFW labeled tracks                                                                                                                                             |
| **4**    | M4 drift           | OpenOil drift paths + impact zones                  | Copernicus + ERA5; optional LSTM compare                                                                                                                       |
| **5**    | M5 + M6            | SHAP + Grad-CAM; fusion; risk scores                | The original contribution                                                                                                                                      |
| **6**    | App                | Streamlit dashboard integrating all outputs         | All 7 result views from the spec                                                                                                                               |

> Each phase is independently demo-able; the project degrades gracefully if time runs out.

**Result views to produce (from spec):** Dark-fleet map · Illegal-fishing risk map · Oil-spill
detection map · Drift-forecast map · Risk dashboard (low/med/high) · SHAP feature-importance
plot · Model performance metrics table (Accuracy/Precision/Recall/F1/ROC-AUC).

---

## 10. Current build state (as of 2026-06-22) — what already exists

> Read this section first in any new conversation so you don't rebuild what's done.

### Files already written and their purpose

| File                                  | Status  | What it does                                                                                                                                                                                                                                                                                        |
| ------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/models/segmentation.py`          | ✅ Done | Three-model factory: `SMPSegModel` (DeepLabv3+/UNet via SMP), `SegFormerModel` (HuggingFace MiT-b2), `OilSAM2Model` (tries OILSAM2 import, falls back to SegFormer-b4). `DiceFocalLoss` (critical for imbalance). `SegmentationMetrics` (per-class IoU/Dice). `build_segmentation_model()` factory. |
| `src/models/train_segmentation.py`    | ✅ Done | Full training loop: mixed precision (GradScaler), CosineAnnealingLR, W&B logging, best checkpoint saving. Cmd: `python src/models/train_segmentation.py --model deeplabv3+ --epochs 50`                                                                                                             |
| `src/data/loaders.py`                 | ✅ Done | `OilSpillDataset` (handles Krestenitis 5-class + Zenodo binary), `color_mask_to_index()`, `get_oil_transforms()` (SAR speckle-aware albumentations), `get_oil_dataloaders()`, `compute_class_weights()`                                                                                             |
| `src/models/detection.py`             | ✅ Done | `chip_sar_scene()` (chips large SAR GeoTIFFs into 640px tiles), `SARVesselDetector` (YOLOv8 wrapper), `_scene_nms()` (greedy NMS)                                                                                                                                                                   |
| `src/fusion/ais_matching.py`          | ✅ Done | `fetch_ais_around_scene()` (GFW API), `interpolate_ais_to_time()`, `match_detections_to_ais()` (SAR det → AIS match, unmatched = dark), `flag_zone_violations()` (geopandas sjoin vs MPA/EEZ)                                                                                                       |
| `src/fusion/risk_scoring.py`          | ✅ Done | `link_spills_to_dark_vessels()` (oil polygon + dark vessel within 160km), `compute_security_risk()`, `compute_environmental_risk()`, `build_risk_table()`                                                                                                                                           |
| `src/explain/gradcam.py`              | ✅ Done | `generate_gradcam()` — Grad-CAM overlay for SegFormer/SMP, target_class=1 (oil)                                                                                                                                                                                                                     |
| `src/explain/shap_explain.py`         | ✅ Done | `explain_xgboost()` (SHAP TreeExplainer + bar plot), `get_top_factors()`                                                                                                                                                                                                                            |
| `src/viz/maps.py`                     | ✅ Done | 5 Folium map builders: `dark_fleet_map`, `illegal_fishing_map`, `oil_spill_map`, `drift_forecast_map`, `risk_dashboard_map`                                                                                                                                                                         |
| `app/dashboard.py`                    | ✅ Done | Full 7-tab Streamlit dashboard. Works NOW with demo data (no model needed). Tabs: Dark Fleet, Illegal Fishing, Oil Spill, Drift Forecast, Risk Dashboard, Explainability, Model Metrics. Run: `streamlit run app/dashboard.py`                                                                      |
| `src/models/train_detection.py`       | ✅ Done | 3-model vessel detection benchmark: YOLOv8m / YOLOv11m-OBB / RT-DETR-L + OBB conversion from HRSID polygons                                                                                                                                                                                         |
| `src/models/train_illegal_fishing.py` | ✅ Done | XGBoost + BiGRU training for M2                                                                                                                                                                                                                                                                     |
| `src/models/drift_prediction.py`      | ✅ Done | OpenOil wrapper for M4                                                                                                                                                                                                                                                                              |
| `src/data/sar_preprocess.py`          | ✅ Done | GEE live pipeline: format detect (dB vs linear) → Lee filter → normalize → chip 512×512. Two entry points: `preprocess_sar_tif(path)` for local TIF, `gee_to_chips(bbox, dates)` for live GEE pull. CLI: `python src/data/sar_preprocess.py chip scene.tif` |
| `requirements.txt`                    | ✅ Done | All dependencies including `sam2>=1.0`, `streamlit-folium>=0.20.0`                                                                                                                                                                                                                                  |

### What is NOT done yet (pending)

**Phase 0 complete as of 2026-06-23 ✅ — all prep done, ready for training:**

Done this session:
- ✅ EEZ shapefile — 285 zones, `/content/shapefiles/world_eez.gpkg`
- ✅ High seas shapefile — 1 polygon, `/content/shapefiles/high_seas.gpkg`
- ✅ MPA — embedded in GFW events (`regions.mpa`, `regions.mpaNoTake` columns)
- ✅ HRSID OBB labels — 4,042 train / 1,962 val converted from COCO polygons via cv2.minAreaRect
- ✅ OBB data.yaml — `/content/HRSID_obb/data.yaml` with 400 negatives included
- ✅ `src/data/sar_preprocess.py` — GEE live pipeline written (dB detection + Lee filter + normalize + chip)
- ✅ `src/fusion/ais_matching.py` — fixed GFW API calls (correct dataset names, offset=0 required)
- ✅ `src/models/train_illegal_fishing.py` — `load_gfw_fishing_features()` added for GFW CSV format

**Next Colab session (the big training run):**
- [ ] Download Zenodo Part I SAR images (40.7 GB) from zenodo.org/records/8346860
- [ ] Verify pixel format: `rasterio.open(first_img).read().min()` — if < -5 → dB, else linear
- [ ] Install OilSAM2: `git clone https://github.com/Chenshuaiyu1120/OILSAM2 && pip install -e OILSAM2`
- [ ] Download SAM2 checkpoint: `wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`
- [ ] Train oil models: DeepLabv3+ → SegFormer → OilSAM2
- [ ] Train vessel models: YOLOv8m → YOLOv11m-OBB (use /content/HRSID_obb/data.yaml) → RT-DETR-L
- [ ] Train M2 XGBoost: `python src/models/train_illegal_fishing.py --csv /content/gfw_data/fishing_events_5k.csv`

### Known bugs / fixes applied

- **starlette version conflict:** `streamlit>=1.35` requires `starlette>=0.46.0`. If you see
  `ImportError: cannot import name 'DEFAULT_EXCLUDED_CONTENT_TYPES' from starlette.middleware.gzip`
  → run `pip install "starlette>=0.46.0"`. fastapi/gradio will show conflict warnings but are
  unused in this project — ignore them.
- **OilSAM2 graceful fallback:** `OilSAM2Model` catches `ImportError` and falls back to
  SegFormer-b4 so the training loop never crashes on a machine without OILSAM2 installed.

### Training run order on Lightning.ai H100

```bash
# Step 1 — verify image format first
python -c "import rasterio, numpy as np; src=rasterio.open('data/oil/images/FIRST.tif'); d=src.read().astype(np.float32); print(d.min(), d.max())"

# Step 2 — baseline
python src/models/train_segmentation.py --model deeplabv3+ --epochs 50 --batch_size 16 --dataset_type zenodo

# Step 3 — transformer comparison
python src/models/train_segmentation.py --model segformer --epochs 50 --batch_size 16 --dataset_type zenodo

# Step 4 — SOTA (install OILSAM2 first)
python src/models/train_segmentation.py --model oilsam2 --epochs 50 --batch_size 4 --dataset_type zenodo
```

---

## 11. Verification — how we know each piece works

- **M3:** per-class IoU/Dice on held-out test split; confusion matrix; oil IoU vs 0.54
  baseline; predicted-mask overlays on SAR.
- **M1:** detection F1/mAP on chipped tiles; visual dark-vessel flags vs AIS overlay.
- **M2:** Precision/Recall/F1 + ROC-AUC; spot-check flagged tracks inside MPA polygons.
- **M4:** OpenOil plumes plausible vs current/wind fields; (optional) LSTM RMSE.
- **M5:** SHAP summary renders the factor breakdown; Grad-CAM highlights actual slick/vessel pixels.
- **M6/app:** end-to-end — load SAR scene → detect spill + vessels → match AIS → flag dark →
  score risk → render all layers in Streamlit without errors.

---

## 11. Risks & mitigations

| Risk                                             | Mitigation                                                                       |
| ------------------------------------------------ | -------------------------------------------------------------------------------- |
| Data logistics (TB-scale) is the real bottleneck | GRD-only subset via `snapshot_download(allow_patterns=["*GRD*"])`; oil data tiny |
| M4D gate is slow                                 | Zenodo binary set unblocks M3 same day; swap to 5-class on approval              |
| Class imbalance silently kills M3                | Dice+Focal + weighting + augmentation from the start, not plain CE               |
| Vessel training overruns                         | Pretrained xView3 winner as de-risked fallback                                   |
| Over-scoping all 6 modules                       | Phasing guarantees a floor; M4–M6 are explicit upside                            |
| Binary-classification dataset trap               | Verify image+mask pairs before committing to any oil dataset                     |
| SAR image format unknown                         | Verify pixel range on day 1 (dB vs linear) before any GEE pipeline work          |
| HRSID Google Drive links dead                    | Check README before downloading; LS-SSDD is GitHub-direct as fallback            |
| starlette version conflict                       | `pip install "starlette>=0.46.0"` fixes it; fastapi/gradio warnings are benign   |

---

## 12. Key references

- **xView3-SAR paper** — _Detecting Dark Fishing Activity Using SAR Imagery_: [arxiv.org/abs/2206.00897](https://arxiv.org/abs/2206.00897)
- **xView3 code + 5 winning models + baseline:** [github.com/DIUx-xView](https://github.com/DIUx-xView)
- **SARFish paper** (WACV 2024): [openaccess.thecvf.com](https://openaccess.thecvf.com/content/WACV2024W/CDL/papers/Luckett_The_SARFish_Dataset_and_Challenge_WACVW_2024_paper.pdf)
- **Krestenitis 2019** — _Oil Spill Identification from Satellite Images Using DNNs_, Remote Sensing 11(15):1762: [mdpi.com/2072-4292/11/15/1762](https://www.mdpi.com/2072-4292/11/15/1762)
- **Paolo et al., Nature 2024** — _Satellite mapping reveals extensive industrial activity at sea_ (the real GFW methods paper).
- **Raynor et al., Science 2025** — MPA fishing study (ecology/policy, not methods).
- **OpenDrift / OpenOil:** [opendrift.github.io](https://opendrift.github.io/)
- **Oil-spill SAR segmentation SOTA:** FA-MobileUNet (PMC11207802), improved DeepLabv3+ (PMC11397947), MOSSM attention model (Springer s10661-025-14676-1).
- **Zenodo oil dataset (instant):** [zenodo.org/records/8346860](https://zenodo.org/records/8346860)

---

## 13. Open items / next actions (as of 2026-06-23)

**Immediate (next Colab session):**
1. Fix EEZ download (`OGR_GEOJSON_MAX_OBJ_SIZE=0` or SHAPE-ZIP fallback)
2. Download MPA shapefiles
3. Run HRSID OBB label conversion (polygon → rotated box for YOLOv11m-OBB)
4. Write `src/data/sar_preprocess.py` (GEE live pipeline)
5. Download Zenodo Part I images (40.7 GB) → start oil model training

**Confirmed done (2026-06-23):**
- ✅ GFW token registered + API verified working
- ✅ 5,000 fishing events, 5,000 GAP events, 2,000 encounter events saved to CSV
- ✅ `fishing.potentialRisk` confirmed as M2 label, `gap.intentionalDisabling` as M1 label
- ✅ Krestenitis dropped — Zenodo binary is the only oil dataset
- ✅ Colab 100 GB confirmed sufficient for all training (no Lightning.ai needed)

**Confirm with supervisor:** must all 6 modules be delivered, or is a polished core acceptable?
(Phasing already protects either answer.)

_Last updated: 2026-06-23. Update this file whenever a decision changes — it is the contract
the build follows._
