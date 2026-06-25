"""Raster mask -> GeoJSON polygons (oil slick vectorisation).

Used by the scene pipeline to turn a SegFormer scene-level oil mask into lon/lat
polygons for the 'oil' layer. Depends on rasterio + shapely (geo deps), so it is
kept out of serialize.py (which stays dependency-light and unit-tested).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from affine import Affine
from rasterio import features
from shapely.geometry import mapping, shape


def _px_ring_to_lonlat(ring: list[tuple[float, float]], transform: Affine) -> list[list[float]]:
    """Map a ring of (col, row) pixel coords to [lon, lat] via the scene affine."""
    out = []
    for x, y in ring:
        lon = transform.c + x * transform.a + y * transform.b
        lat = transform.f + x * transform.d + y * transform.e
        out.append([lon, lat])
    return out


def mask_to_polygons(
    mask: np.ndarray,
    transform: Affine,
    scale_m: float = 10.0,
    oil_class: int = 1,
    min_area_px: int = 50,
) -> list[dict[str, Any]]:
    """
    Vectorise a 2-D class mask into GeoJSON polygon records.

    mask: H×W int array (class indices); oil pixels == oil_class.
    transform: rasterio affine of the full scene (pixel -> CRS, expected EPSG:4326).
    scale_m: ground pixel size in metres (for area estimate).
    Returns: [{"coordinates": <GeoJSON Polygon rings in lon/lat>, "area_km2": float}, ...]
    """
    binary = (mask == oil_class).astype("uint8")
    polys: list[dict[str, Any]] = []
    # shapes() on the pixel grid (identity transform) so polygon .area is in pixels.
    for geom, val in features.shapes(binary, mask=binary.astype(bool), transform=Affine.identity()):
        if val != 1:
            continue
        poly = shape(geom)
        area_px = poly.area
        if area_px < min_area_px:
            continue
        rings = mapping(poly)["coordinates"]  # rings in (col,row) pixel space
        geo_rings = [_px_ring_to_lonlat(list(r), transform) for r in rings]
        polys.append({
            "coordinates": geo_rings,
            "area_km2": round(area_px * (scale_m ** 2) / 1e6, 3),
        })
    return polys
