# Plan 1: Backend Contract + Fixture Case Study + API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A runnable FastAPI service that serves a hand-authored fixture case study conforming to the design-spec data contract (§6), so the frontend (Plan 2) and the real pipeline (Plan 3) build against a locked, validated interface.

**Architecture:** A thin FastAPI app reads precomputed per-case-study assets from `backend/data/case_studies/<id>/` (meta.json, layers.geojson, metrics.json, explain.json, overlays/*.png) and returns them through two JSON endpoints plus a static mount for overlay images. Pydantic models define and validate the contract. No ML, no GEE — Plan 1 is the integration boundary only.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, Pydantic v2, pytest, httpx (TestClient).

## Global Constraints

- Python **3.12** (repo uses 3.12.5).
- Backend lives under `backend/`; do not modify existing `src/`.
- GeoJSON is **WGS84 (EPSG:4326)**, coordinate order **[lon, lat]**.
- Contract field names are **fixed** by the design spec §6 — copy verbatim; later plans depend on them.
- All file reads use `encoding="utf-8"`.
- CORS allows `http://localhost:3000` (the Next.js dev origin).
- Commit after every task with a `feat:`/`test:` message.

---

### Task 1: Backend scaffold + Pydantic contract + `/case-studies` listing

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/requirements.txt`
- Create: `backend/schemas.py`
- Create: `backend/app.py`
- Create: `backend/tests/__init__.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `backend/app.py:app` — the FastAPI instance.
  - `backend.app.DATA_DIR: Path` — `backend/data/case_studies`.
  - Pydantic models in `backend/schemas.py`: `CaseStudySummary`, `CaseStudyMeta`, `Metrics`, `RiskFactor`, `Explain`, `CaseStudyDetail` (exact fields below).
  - `GET /case-studies` → `list[CaseStudySummary]` (empty list when the data dir is absent/empty).

- [ ] **Step 1: Create the requirements file**

`backend/requirements.txt`:
```text
fastapi==0.115.5
uvicorn[standard]==0.32.1
pydantic==2.9.2
pytest==8.3.3
httpx==0.27.2
```

- [ ] **Step 2: Install the backend deps**

Run: `python -m pip install -r backend/requirements.txt`
Expected: installs without error (FastAPI, Uvicorn, Pydantic, pytest, httpx present).

- [ ] **Step 3: Create empty package markers**

`backend/__init__.py`: (empty file)
`backend/tests/__init__.py`: (empty file)

- [ ] **Step 4: Define the contract schemas**

`backend/schemas.py`:
```python
"""Pydantic models for the backend → frontend data contract (design spec §6)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CaseStudySummary(BaseModel):
    id: str
    title: str
    sensor: str
    acquired_utc: str
    bbox: list[float] = Field(min_length=4, max_length=4)  # [min_lon, min_lat, max_lon, max_lat]
    summary: str


class CaseStudyMeta(CaseStudySummary):
    """Same shape as the summary; lives in each case study's meta.json."""


class Metrics(BaseModel):
    oil_iou: float | None = None
    yolo_map50: float | None = None
    dark_count: int
    slick_count: int
    security_risk: float
    environmental_risk: float


class RiskFactor(BaseModel):
    label: str
    weight: float


class Explain(BaseModel):
    gradcam_oil_png: str | None = None
    gradcam_ship_png: str | None = None
    risk_factors: list[RiskFactor] = []


class CaseStudyDetail(BaseModel):
    meta: CaseStudyMeta
    layers: dict[str, dict]  # layer name -> GeoJSON FeatureCollection
    metrics: Metrics
    explain: Explain
```

- [ ] **Step 5: Write the failing test for the listing endpoint**

`backend/tests/test_api.py`:
```python
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_list_case_studies_returns_seeded_fixture():
    resp = client.get("/case-studies")
    assert resp.status_code == 200
    ids = [c["id"] for c in resp.json()]
    assert "fixture-gulf-001" in ids
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `python -m pytest backend/tests/test_api.py::test_list_case_studies_returns_seeded_fixture -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.app'` (app not created yet).

- [ ] **Step 7: Implement the app + listing endpoint**

`backend/app.py`:
```python
"""Maritime Intelligence API — serves precomputed case-study assets."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import CaseStudyDetail, CaseStudySummary

DATA_DIR = Path(__file__).parent / "data" / "case_studies"

app = FastAPI(title="Maritime Intelligence API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/case-studies", response_model=list[CaseStudySummary])
def list_case_studies() -> list[dict]:
    out: list[dict] = []
    if not DATA_DIR.exists():
        return out
    for d in sorted(DATA_DIR.iterdir()):
        meta = d / "meta.json"
        if d.is_dir() and meta.exists():
            out.append(_load_json(meta))
    return out


@app.get("/case-study/{cid}", response_model=CaseStudyDetail)
def get_case_study(cid: str) -> dict:
    d = DATA_DIR / cid
    if not (d.is_dir() and (d / "meta.json").exists()):
        raise HTTPException(status_code=404, detail=f"case study '{cid}' not found")
    return {
        "meta": _load_json(d / "meta.json"),
        "layers": _load_json(d / "layers.geojson"),
        "metrics": _load_json(d / "metrics.json"),
        "explain": _load_json(d / "explain.json"),
    }
```

(The detail endpoint is implemented here too; its test arrives in Task 3 once the fixture exists.)

- [ ] **Step 8: Run the test — still fails, but now on data, not import**

Run: `python -m pytest backend/tests/test_api.py::test_list_case_studies_returns_seeded_fixture -v`
Expected: FAIL — assertion error (`fixture-gulf-001` not in `[]`); the import now succeeds. The fixture is authored in Task 3, which flips this to PASS.

- [ ] **Step 9: Commit**

```bash
git add backend/__init__.py backend/requirements.txt backend/schemas.py backend/app.py backend/tests/__init__.py backend/tests/test_api.py
git commit -m "feat: backend scaffold, contract schemas, /case-studies endpoint"
```

---

### Task 2: `/case-study/{id}` 404 behavior

**Files:**
- Modify: `backend/tests/test_api.py` (append test)

**Interfaces:**
- Consumes: `GET /case-study/{cid}` from Task 1.
- Produces: verified 404 contract for unknown ids.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_api.py`:
```python
def test_get_unknown_case_study_returns_404():
    resp = client.get("/case-study/does-not-exist")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]
```

- [ ] **Step 2: Run it**

Run: `python -m pytest backend/tests/test_api.py::test_get_unknown_case_study_returns_404 -v`
Expected: PASS (the endpoint from Task 1 already raises 404 for missing dirs).

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_api.py
git commit -m "test: 404 for unknown case study id"
```

---

### Task 3: Author the fixture case study + golden contract test

**Files:**
- Create: `backend/data/case_studies/fixture-gulf-001/meta.json`
- Create: `backend/data/case_studies/fixture-gulf-001/layers.geojson`
- Create: `backend/data/case_studies/fixture-gulf-001/metrics.json`
- Create: `backend/data/case_studies/fixture-gulf-001/explain.json`
- Create: `backend/data/case_studies/fixture-gulf-001/overlays/.gitkeep`
- Modify: `backend/tests/test_api.py` (append golden test)

**Interfaces:**
- Consumes: `CaseStudyDetail` schema (Task 1), both endpoints.
- Produces: a contract-valid fixture with the **exact** layer names later plans/frontend rely on:
  `ships`, `dark_vessels`, `oil`, `ais_tracks`, `zones`, `fusion_links` — each a GeoJSON `FeatureCollection`.

- [ ] **Step 1: Author the case-study metadata**

`backend/data/case_studies/fixture-gulf-001/meta.json`:
```json
{
  "id": "fixture-gulf-001",
  "title": "Gulf of Mexico — fixture",
  "sensor": "Sentinel-1 (fixture)",
  "acquired_utc": "2024-03-12T06:14:00Z",
  "bbox": [-90.6, 28.0, -90.0, 28.5],
  "summary": "Synthetic fixture: 1 oil slick, 2 vessels (1 dark), 1 zone, 1 fusion link."
}
```

- [ ] **Step 2: Author the layers (GeoJSON FeatureCollections keyed by layer name)**

`backend/data/case_studies/fixture-gulf-001/layers.geojson`:
```json
{
  "ships": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-90.40, 28.22]},
        "properties": {"matched": true, "mmsi": "367123450", "type": "cargo", "confidence": 0.93}
      }
    ]
  },
  "dark_vessels": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-90.28, 28.31]},
        "properties": {"matched": false, "confidence": 0.88}
      }
    ]
  },
  "oil": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[-90.34, 28.27], [-90.30, 28.27], [-90.30, 28.30], [-90.34, 28.30], [-90.34, 28.27]]]},
        "properties": {"area_km2": 12.4}
      }
    ]
  },
  "ais_tracks": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[-90.45, 28.18], [-90.40, 28.22], [-90.35, 28.25]]},
        "properties": {"mmsi": "367123450"}
      }
    ]
  },
  "zones": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[-90.5, 28.05], [-90.1, 28.05], [-90.1, 28.45], [-90.5, 28.45], [-90.5, 28.05]]]},
        "properties": {"kind": "EEZ", "name": "fixture zone"}
      }
    ]
  },
  "fusion_links": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[-90.32, 28.285], [-90.28, 28.31]]},
        "properties": {"reason": "dark vessel within range of slick"}
      }
    ]
  }
}
```

- [ ] **Step 3: Author the metrics**

`backend/data/case_studies/fixture-gulf-001/metrics.json`:
```json
{
  "oil_iou": 0.49,
  "yolo_map50": 0.94,
  "dark_count": 1,
  "slick_count": 1,
  "security_risk": 0.72,
  "environmental_risk": 0.61
}
```

- [ ] **Step 4: Author the explainability block**

`backend/data/case_studies/fixture-gulf-001/explain.json`:
```json
{
  "gradcam_oil_png": "fixture-gulf-001/overlays/gradcam_oil.png",
  "gradcam_ship_png": "fixture-gulf-001/overlays/gradcam_ship.png",
  "risk_factors": [
    {"label": "AIS signal absence", "weight": 0.35},
    {"label": "Restricted-area presence", "weight": 0.25},
    {"label": "Proximity to slick", "weight": 0.20},
    {"label": "Vessel movement pattern", "weight": 0.12},
    {"label": "Slick size", "weight": 0.08}
  ]
}
```

- [ ] **Step 5: Keep the overlays dir tracked**

`backend/data/case_studies/fixture-gulf-001/overlays/.gitkeep`: (empty file)

- [ ] **Step 6: Write the golden contract test**

Append to `backend/tests/test_api.py`:
```python
from backend.schemas import CaseStudyDetail

EXPECTED_LAYERS = {"ships", "dark_vessels", "oil", "ais_tracks", "zones", "fusion_links"}


def test_fixture_detail_matches_contract():
    resp = client.get("/case-study/fixture-gulf-001")
    assert resp.status_code == 200
    detail = CaseStudyDetail.model_validate(resp.json())  # raises if contract violated
    assert detail.meta.id == "fixture-gulf-001"
    assert set(detail.layers.keys()) == EXPECTED_LAYERS
    for name, fc in detail.layers.items():
        assert fc["type"] == "FeatureCollection", f"{name} is not a FeatureCollection"
        assert isinstance(fc["features"], list)
    assert detail.metrics.dark_count == 1
    assert len(detail.explain.risk_factors) == 5
```

- [ ] **Step 7: Run the full test file**

Run: `python -m pytest backend/tests/test_api.py -v`
Expected: all PASS — including `test_list_case_studies_returns_seeded_fixture` from Task 1 (the fixture now exists).

- [ ] **Step 8: Commit**

```bash
git add backend/data/case_studies/fixture-gulf-001 backend/tests/test_api.py
git commit -m "feat: fixture case study + golden contract test"
```

---

### Task 4: Serve overlay images + run-script + manual smoke

**Files:**
- Modify: `backend/app.py` (mount static assets)
- Modify: `backend/tests/test_api.py` (append static-serving test)
- Create: `backend/data/case_studies/fixture-gulf-001/overlays/gradcam_oil.png`
- Create: `backend/data/case_studies/fixture-gulf-001/overlays/gradcam_ship.png`
- Create: `backend/README.md`

**Interfaces:**
- Consumes: `app`, `DATA_DIR` (Task 1); `explain.gradcam_*_png` relative paths (Task 3).
- Produces: `GET /assets/<id>/overlays/<file>.png` serving the overlay files referenced by `explain`.

- [ ] **Step 1: Generate two tiny placeholder PNGs**

Run:
```bash
python -c "import struct, zlib, pathlib; \
d=pathlib.Path('backend/data/case_studies/fixture-gulf-001/overlays'); \
raw=b''.join(b'\x00'+b'\x80\x80\x80'*4 for _ in range(4)); \
def chunk(t,b): import struct,zlib; return struct.pack('>I',len(b))+t+b+struct.pack('>I',zlib.crc32(t+b)&0xffffffff); \
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',4,4,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''); \
[ (d/n).write_bytes(png) for n in ('gradcam_oil.png','gradcam_ship.png') ]"
```
Expected: two 4×4 grey PNGs created. (If the one-liner is awkward in your shell, create any small valid PNGs at those two paths.)

- [ ] **Step 2: Write the failing static-serving test**

Append to `backend/tests/test_api.py`:
```python
def test_overlay_png_is_served():
    resp = client.get("/assets/fixture-gulf-001/overlays/gradcam_oil.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest backend/tests/test_api.py::test_overlay_png_is_served -v`
Expected: FAIL — 404 (no static mount yet).

- [ ] **Step 4: Mount the static assets directory**

In `backend/app.py`, add the import and mount (after the CORS middleware block):
```python
from fastapi.staticfiles import StaticFiles

# ... after app.add_middleware(...) and after DATA_DIR is defined:
if DATA_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DATA_DIR), name="assets")
```
Note: `DATA_DIR` is defined above the `app = FastAPI(...)` line in Task 1, so it is in scope here.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest backend/tests/test_api.py::test_overlay_png_is_served -v`
Expected: PASS.

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest backend/tests/ -v`
Expected: all PASS (4 tests).

- [ ] **Step 7: Write the backend README (run instructions)**

`backend/README.md`:
```markdown
# Maritime Intelligence API (backend)

Serves precomputed case-study assets to the frontend.

## Run
```bash
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app:app --reload --port 8000
```

- `GET /case-studies` — list of case-study summaries
- `GET /case-study/{id}` — full detail (meta, layers, metrics, explain)
- `GET /assets/{id}/overlays/{file}.png` — Grad-CAM overlay images

## Test
```bash
python -m pytest backend/tests/ -v
```

Case studies live in `backend/data/case_studies/<id>/` as
`meta.json`, `layers.geojson`, `metrics.json`, `explain.json`, `overlays/*.png`.
```

- [ ] **Step 8: Manual smoke (optional but recommended)**

Run: `python -m uvicorn backend.app:app --port 8000` then in another shell `curl http://localhost:8000/case-studies`
Expected: JSON array containing `fixture-gulf-001`.

- [ ] **Step 9: Commit**

```bash
git add backend/app.py backend/tests/test_api.py backend/data/case_studies/fixture-gulf-001/overlays backend/README.md
git commit -m "feat: serve overlay images + backend run/test docs"
```

---

## Self-Review

**1. Spec coverage (§6 data contract):** `/case-studies` (Task 1), `/case-study/{id}` with meta+layers+metrics+explain (Tasks 1, 3), 404 (Task 2), overlay images (Task 4). All six layer names (`ships`, `dark_vessels`, `oil`, `ais_tracks`, `zones`, `fusion_links`) authored and asserted (Task 3). Metrics + explain fields match §6. ✅ Plan 1 intentionally excludes the pipeline (Plan 3) and frontend (Plan 2).

**2. Placeholder scan:** No TBD/TODO; every code/test step shows complete content. ✅

**3. Type consistency:** `DATA_DIR`, `app`, `_load_json` consistent across Tasks 1 & 4. Layer-name set in the Task 3 test matches the fixture keys and §6. `explain.gradcam_oil_png` path in Task 3 matches the file created/served in Task 4. Schema field names match the JSON fixtures. ✅

---

## Next plans (written after Plan 1 is approved/executed)
- **Plan 2 — Frontend:** Next.js + TS + Tailwind + MapLibre rendering the contract; layer toggles, detail panel on select, metrics strip, Grad-CAM overlay toggle, restrained design (memory `ui-professional-not-slop`).
- **Plan 3 — Pipeline:** `backend/pipeline/run_scene.py` orchestrating `sar_preprocess` → YOLO + SegFormer (HF checkpoints) → polygonize → GFW AIS match → fusion + risk → Grad-CAM → serialize into the Task-3 asset layout, producing the real case studies (§8).
