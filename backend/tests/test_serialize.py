import json

from backend.pipeline.serialize import build_layers, write_case_study, LAYER_NAMES
from backend.schemas import CaseStudyDetail


def _sample_inputs():
    vessels = [
        {"lon": -90.40, "lat": 28.22, "dark_vessel": False, "matched_mmsi": "367123450",
         "vessel_type": "cargo", "conf": 0.93},
        {"lon": -90.28, "lat": 28.31, "dark_vessel": True, "matched_mmsi": None,
         "vessel_type": None, "conf": 0.88},
    ]
    oil = [{"coordinates": [[[-90.34, 28.27], [-90.30, 28.27], [-90.30, 28.30],
                             [-90.34, 28.30], [-90.34, 28.27]]], "area_km2": 12.4}]
    ais = [{"coordinates": [[-90.45, 28.18], [-90.40, 28.22]], "mmsi": "367123450"}]
    zones = [{"coordinates": [[[-90.5, 28.05], [-90.1, 28.05], [-90.1, 28.45],
                              [-90.5, 28.45], [-90.5, 28.05]]], "kind": "EEZ", "name": "z"}]
    fusion = [{"coordinates": [[-90.32, 28.285], [-90.28, 28.31]], "reason": "near slick"}]
    return vessels, oil, ais, zones, fusion


def test_build_layers_splits_dark_and_normal_vessels():
    vessels, oil, ais, zones, fusion = _sample_inputs()
    layers = build_layers(vessels, oil, ais, zones, fusion)
    assert set(layers.keys()) == set(LAYER_NAMES)
    assert len(layers["ships"]["features"]) == 1
    assert len(layers["dark_vessels"]["features"]) == 1
    assert layers["ships"]["features"][0]["properties"]["matched"] is True
    assert layers["dark_vessels"]["features"][0]["properties"]["matched"] is False
    # None props are dropped
    assert "mmsi" not in layers["dark_vessels"]["features"][0]["properties"]
    assert layers["oil"]["features"][0]["geometry"]["type"] == "Polygon"
    assert layers["ais_tracks"]["features"][0]["geometry"]["type"] == "LineString"


def test_write_case_study_round_trips_through_contract(tmp_path):
    vessels, oil, ais, zones, fusion = _sample_inputs()
    layers = build_layers(vessels, oil, ais, zones, fusion)
    meta = {
        "id": "test-001", "title": "T", "sensor": "Sentinel-1",
        "acquired_utc": "2024-03-12T06:14:00Z", "bbox": [-90.6, 28.0, -90.0, 28.5],
        "summary": "s",
    }
    metrics = {"oil_iou": None, "yolo_map50": None, "dark_count": 1, "slick_count": 1,
               "security_risk": 0.65, "environmental_risk": 0.4}
    explain = {"gradcam_oil_png": None, "gradcam_ship_png": None, "risk_factors": []}

    d = write_case_study(tmp_path / "test-001", meta, layers, metrics, explain)

    detail = CaseStudyDetail.model_validate({
        "meta": json.loads((d / "meta.json").read_text(encoding="utf-8")),
        "layers": json.loads((d / "layers.geojson").read_text(encoding="utf-8")),
        "metrics": json.loads((d / "metrics.json").read_text(encoding="utf-8")),
        "explain": json.loads((d / "explain.json").read_text(encoding="utf-8")),
    })
    assert detail.metrics.dark_count == 1
    assert set(detail.layers.keys()) == set(LAYER_NAMES)
    assert (d / "overlays").is_dir()
