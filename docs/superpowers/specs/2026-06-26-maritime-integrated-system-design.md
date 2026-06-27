# Maritime Security Intelligence — Integrated System & Frontend (Design Spec)

**Date:** 2026-06-26
**Status:** Approved design → ready for implementation plan
**Scope decision:** Approach **A — Integrated core** (see §2)

---

## 1. Goal

Wire the *already-trained* detection models (M1 ship detection, M3 oil segmentation) into **one
integrated Sentinel-1 pipeline** and expose it through a **professional Next.js frontend** backed
by a **FastAPI** service. The headline contribution is the **fusion layer**: linking a detected oil
slick to nearby *dark* (non-broadcasting) vessels as a candidate responsible party, with
explainable risk scoring.

This is "one system," not six notebooks or a results viewer.

## 2. Scope

### In scope (Approach A)
- One Sentinel-1 pipeline: GEE pull → land-mask → ship detection + oil segmentation → AIS matching
  → dark-vessel flagging → rule-based illegal-fishing zone check → fusion → risk scoring → Grad-CAM.
- FastAPI backend serving **precomputed** case-study outputs as GeoJSON + metrics JSON + overlay PNGs.
- Next.js + MapLibre frontend (restrained, professional) rendering all layers + detail/explain panels.
- 2–3 pre-baked case studies built from data we already hold.

### Out of scope (deferred — "B-later")
- **M2 XGBoost** — *dropped.* Illegal fishing is a rule-based zone-violation check, not a trained model.
- **M4 oil drift (OpenOil)** — deferred; needs Copernicus + ERA5 credentials.
- **Live GEE pull tab** — deferred; needs a GPU-backed live-inference path.
- **SHAP** — not needed without a tabular model; explainability = Grad-CAM + rule-contribution bars.
- Other SAR satellites (ICEYE/Capella/TerraSAR-X) — commercial/paid; Sentinel-1 only.

## 3. Locked design decisions (with rationale)

| Decision | Choice | Why |
|---|---|---|
| Sensor | **Sentinel-1 only** | Free/open; oil + ships from the *same* scene → auto co-registered + time-aligned. Other SAR is paid. S1A retires 2026-06-29 but S1C+S1D continue the same C-band format. |
| Dark-vessel timing | **Match AIS at the scene's known acquisition timestamp** | A SAR scene is timestamped to the second. Recency is cosmetic; correctness needs only image-time = AIS-time. No "fresh image"/multi-satellite needed. |
| AIS source | **GFW historical** (free, global, since 2012) | We already have a token + 5k GAP events with off/on positions + timestamps. NOAA Marine Cadastre is a US-only richer fallback. |
| Illegal fishing | **Rule-based zone violation** (`flag_zone_violations`, code exists) | A detected+AIS-matched fishing-type vessel inside a no-take MPA/restricted EEZ = illegal. No novelty-free ML. |
| Explainability | **Grad-CAM** (CV) + rule-contribution bars | SHAP over dense pixels is impractical; Grad-CAM is the correct CV explanation. |
| Sea-only | **GEE land mask** + offshore ROIs + skip land-heavy chips | Ports/buildings cause ship false positives; land masking removes most coastal FPs. |
| Demo data | **Precompute offline; serve static assets** | Fast, reliable demo with no GPU at presentation time. |
| Frontend | **Next.js + TS + Tailwind + MapLibre GL** (no token) | Nothing breaks on a missing API key; map is the product. |
| Deployment | **Local** (both servers on the machine) | Simplest, most reliable for a thesis defense. |

## 4. Pipeline

### 4.1 Build-time (run once per case study, on Colab/GPU)
1. **Pick scene** — region bbox + date with known ground truth, offshore/open water.
2. **GEE pull** — `COPERNICUS/S1_GRD`, VV+VH, IW mode, that date. Keep acquisition timestamp + CRS/affine.
3. **Preprocess** (`src/data/sar_preprocess.py`) — dB→linear → Lee filter → percentile-normalize →
   **land mask** (zero land via GEE water mask, e.g. `MODIS/006/MOD44W` or bathymetry < 0) →
   chip 512×512 (64px overlap, skip land-heavy/empty tiles).
4. **Ships (YOLO)** — best checkpoint (HF) on chips → boxes → NMS across chips → pixel→lat/lon via CRS.
5. **Oil (SegFormer)** — best checkpoint on chips → per-pixel mask → stitch → polygonize → lat/lon polygons.
6. **AIS pull (GFW)** — vessels broadcasting in bbox at scene time (±window): MMSI, position, type.
7. **Zones** — EEZ/MPA polygons (`world_eez.gpkg` + GFW `regions.mpa`).
8. **Match ships↔AIS** (`src/fusion/ais_matching.py`) — nearest AIS within ~500m gate, time-interpolated
   to scene time. Matched = normal; **unmatched = dark vessel**. Matched fishing-type inside a
   restricted zone = **illegal fishing**.
9. **Fusion** (`src/fusion/risk_scoring.py`) — oil polygon + dark vessel within range → candidate party.
10. **Risk scores** — Security = f(dark count, zone violations, intentionalDisabling); Environmental =
    f(oil area, proximity to coast/MPA).
11. **Explain** — Grad-CAM overlays (oil + ship) + rule-contribution bars.
12. **Serialize** — GeoJSON layers + `metrics.json` + overlay PNGs → static case-study assets.

### 4.2 Demo-time (instant, no GPU)
Next.js → user picks a case study → FastAPI serves precomputed assets → MapLibre renders layers + panels.

## 5. Architecture

```
maritime/
├─ src/                        # EXISTING — models, fusion, explain, data, viz (reused as-is)
├─ backend/                    # NEW — FastAPI
│  ├─ app.py                   #   GET /case-studies, GET /case-study/{id}
│  ├─ pipeline/run_scene.py    #   orchestrates §4.1 steps 2–12
│  └─ data/case_studies/{id}/  #   layers.geojson, metrics.json, overlays/*.png
├─ frontend/                   # NEW — Next.js (App Router, TS, Tailwind, MapLibre)
│  ├─ app/
│  ├─ components/              #   Map, LayerToggles, DetailPanel, ExplainPanel, MetricsStrip
│  └─ lib/                     #   api client + GeoJSON types
└─ checkpoints/                # YOLO + SegFormer best weights pulled from HF for pipeline runs
```

Python does ML + geospatial and emits GeoJSON; TypeScript only renders. The boundary is a small,
versioned GeoJSON + JSON contract (§6), so each side builds and tests independently.

## 6. Data contract (backend → frontend)

- `GET /case-studies` → `[{ id, title, sensor, acquired_utc, bbox, summary }]`
- `GET /case-study/{id}` → `{ meta, layers, metrics, explain }` where:
  - `layers`: a `FeatureCollection` per layer — `ships` (Point; props: matched bool, mmsi?, type?,
    confidence), `dark_vessels` (Point), `oil` (Polygon; props: area_km2), `ais_tracks` (LineString),
    `zones` (Polygon; props: kind=MPA/EEZ), `fusion_links` (LineString oil↔dark).
  - `metrics`: `{ oil_iou, yolo_map50, dark_count, slick_count, security_risk, environmental_risk }`.
  - `explain`: `{ gradcam_oil_png, gradcam_ship_png, risk_factors: [{label, weight}] }`.

## 7. Frontend design principles (restrained, professional)

Reference bar: Linear / Vercel / Felt / a serious GIS tool — **not** a dashboard template, **not**
"AI slop." See memory `ui-professional-not-slop`.

- **Map is the product**; chrome is quiet. Thin left rail (case-study + layer toggles); a detail panel
  that appears only on selection; a slim bottom metrics strip.
- **Neutral base** (one slate/zinc scale). Semantic color only where it means something —
  dark vessel = red, oil = amber, normal vessel = muted — sparingly, on a desaturated dark basemap.
- **Type hierarchy** via size/weight, not color. One clean sans (Inter/Geist). Monospaced,
  right-aligned numbers in the metrics strip.
- **No** gradients, glows, pulsing markers, gauges, emoji icons, drop-shadow soup, 3D/rainbow charts.
- Explainability "why flagged" = plain horizontal bar list.
- Build with the **frontend-design** skill for genuine polish.

## 8. Case studies (candidates — verify exact scene/date at build time)

Built from data we already hold:
1. **Oil** — a Zenodo Part III test scene (real Sentinel-1, known oil ground-truth mask) → demonstrates
   segmentation + environmental risk.
2. **Dark vessel** — a GFW **GAP event** (`gap.intentionalDisabling = true`, off/on positions +
   timestamp from our 5k events) → pull the Sentinel-1 scene at that location/time → demonstrates
   detection + AIS matching + dark flagging.
3. **Fusion (stretch)** — a scene containing both an oil slick and a nearby vessel → demonstrates the
   headline oil↔dark-vessel link + combined risk. If no single real scene has both, demonstrate the
   fusion logic on the oil scene with an AIS-derived nearby-vessel overlay (clearly labeled).

Exact scene IDs/dates + Sentinel-1/GFW coverage are confirmed during the build-time run.

## 9. Testing

- **Backend contract:** schema-validate `/case-study/{id}` output (GeoJSON well-formed, required props
  present, CRS = WGS84). Golden-file test per case study.
- **Pipeline units:** land-mask zeros land; pixel→lat/lon round-trips; AIS match gate (matched vs dark);
  polygonize produces valid geometries; risk scores in [0,1].
- **Frontend:** layers render + toggle; selecting an entity populates the detail panel; case-study switch
  reloads cleanly; loads against a fixture `/case-study` payload (no live backend needed for UI tests).

## 10. Risks

| Risk | Mitigation |
|---|---|
| GFW AIS coverage sparse at a chosen scene/time | Pick scenes from our existing GAP events (known AIS present); NOAA fallback for US waters. |
| Oil model domain shift on a non-Zenodo GEE scene | Use a Zenodo Part III scene for the oil case study (in-distribution); note cross-sensor as future work. |
| No single real scene has both oil + dark vessel | Fusion case study falls back to documented logic on the oil scene with labeled AIS overlay. |
| Pixel→geo wrong (Zenodo tiles not georeferenced) | Live geo-coords come from GEE scenes (carry CRS); Zenodo-only scenes shown in scene coords, clearly labeled. |
| Frontend drifts toward generic look | frontend-design skill + the restraint principles in §7; map-first layout. |

## 11. Future (Approach B, clearly-scoped upside)
- M4 OpenOil drift forecast (Copernicus + ERA5).
- Live "pull a fresh scene now" tab (GPU-backed inference endpoint).
- Cross-sensor generalization (fine-tune oil model on other SAR bands).
