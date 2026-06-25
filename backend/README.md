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
