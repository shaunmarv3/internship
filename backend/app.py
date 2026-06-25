"""Maritime Intelligence API -- serves precomputed case-study assets."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.schemas import CaseStudyDetail, CaseStudySummary

DATA_DIR = Path(__file__).parent / "data" / "case_studies"

app = FastAPI(title="Maritime Intelligence API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if DATA_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DATA_DIR), name="assets")


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
