"""Pydantic models for the backend -> frontend data contract (design spec section 6)."""
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
