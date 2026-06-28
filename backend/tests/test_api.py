from fastapi.testclient import TestClient

from backend.app import app
from backend.schemas import CaseStudyDetail

client = TestClient(app)

EXPECTED_LAYERS = {"ships", "dark_vessels", "oil", "ais_tracks", "zones", "fusion_links"}


def test_list_case_studies_returns_seeded_fixture():
    resp = client.get("/case-studies")
    assert resp.status_code == 200
    ids = [c["id"] for c in resp.json()]
    assert "fixture-gulf-001" in ids


def test_get_unknown_case_study_returns_404():
    resp = client.get("/case-study/does-not-exist")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


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


def test_process_requires_a_file():
    resp = client.post("/process")
    assert resp.status_code == 422  # FastAPI validation: file is required


def test_overlay_png_is_served():
    resp = client.get("/assets/fixture-gulf-001/overlays/gradcam_oil.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
