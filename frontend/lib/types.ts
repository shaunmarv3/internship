// Mirror of the backend data contract (design spec section 6).

export interface SceneOverlay {
  png: string; // asset-relative path, e.g. "mauritius/overlays/scene.png"
  bounds: [number, number, number, number]; // [min_lon, min_lat, max_lon, max_lat]
  v?: number; // unix timestamp added at processing time — used as cache-buster
}

export interface CaseStudySummary {
  id: string;
  title: string;
  sensor: string;
  acquired_utc: string;
  bbox: [number, number, number, number]; // [min_lon, min_lat, max_lon, max_lat]
  summary: string;
  scene_overlay?: SceneOverlay | null;
}

export interface Metrics {
  oil_iou: number | null;
  yolo_map50: number | null;
  dark_count: number;
  slick_count: number;
  security_risk: number;
  environmental_risk: number;
}

export interface RiskFactor {
  label: string;
  weight: number;
}

export interface Explain {
  gradcam_oil_png: string | null;
  gradcam_ship_png: string | null;
  risk_factors: RiskFactor[];
}

export type LayerName =
  | "ships"
  | "dark_vessels"
  | "oil"
  | "ais_tracks"
  | "zones";

export interface CaseStudyDetail {
  meta: CaseStudySummary;
  layers: Record<LayerName, GeoJSON.FeatureCollection>;
  metrics: Metrics;
  explain: Explain;
}

export interface SelectedFeature {
  layer: LayerName;
  properties: Record<string, unknown>;
}
