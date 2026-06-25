"use client";

import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { CaseStudyDetail, LayerName, SelectedFeature } from "@/lib/types";

// Inline style: a dark background that ALWAYS renders (so the data layers show even
// fully offline) plus CARTO raster tiles for coastline context when online. This avoids
// the external vector-style/glyph/sprite fetch that can leave the map blank if it fails.
const STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    carto: {
      type: "raster",
      tiles: [
        "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
      ],
      tileSize: 256,
      attribution: "© OpenStreetMap, © CARTO",
    },
  },
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#0b1622" } },
    { id: "carto", type: "raster", source: "carto" },
  ],
};

// logical layer name -> maplibre layer ids
const GROUP: Record<LayerName, string[]> = {
  zones: ["zones-line"],
  oil: ["oil-fill", "oil-line"],
  ais_tracks: ["ais-line"],
  fusion_links: ["fusion-line"],
  ships: ["ships-circle"],
  dark_vessels: ["dark-circle"],
};

// clickable maplibre layer id -> logical layer name
const INTERACTIVE: Record<string, LayerName> = {
  "ships-circle": "ships",
  "dark-circle": "dark_vessels",
  "oil-fill": "oil",
};

function ensureSource(map: maplibregl.Map, name: string, data: GeoJSON.FeatureCollection) {
  const src = map.getSource(name) as maplibregl.GeoJSONSource | undefined;
  if (src) src.setData(data);
  else map.addSource(name, { type: "geojson", data });
}

function addLayer(map: maplibregl.Map, layer: maplibregl.LayerSpecification) {
  if (map.getLayer(layer.id)) map.removeLayer(layer.id);
  map.addLayer(layer);
}

function addData(map: maplibregl.Map, detail: CaseStudyDetail) {
  const L = detail.layers;
  (Object.keys(GROUP) as LayerName[]).forEach((n) => ensureSource(map, n, L[n]));

  addLayer(map, {
    id: "zones-line", type: "line", source: "zones",
    paint: { "line-color": "#475569", "line-width": 1, "line-opacity": 0.7, "line-dasharray": [2, 2] },
  });
  addLayer(map, {
    id: "oil-fill", type: "fill", source: "oil",
    paint: { "fill-color": "#f59e0b", "fill-opacity": 0.22 },
  });
  addLayer(map, {
    id: "oil-line", type: "line", source: "oil",
    paint: { "line-color": "#f59e0b", "line-width": 1.5, "line-opacity": 0.9 },
  });
  addLayer(map, {
    id: "ais-line", type: "line", source: "ais_tracks",
    paint: { "line-color": "#64748b", "line-width": 1.2, "line-opacity": 0.85 },
  });
  addLayer(map, {
    id: "fusion-line", type: "line", source: "fusion_links",
    paint: { "line-color": "#ef4444", "line-width": 1.4, "line-opacity": 0.9, "line-dasharray": [1.5, 1.5] },
  });
  addLayer(map, {
    id: "ships-circle", type: "circle", source: "ships",
    paint: { "circle-radius": 5, "circle-color": "#5b8fb0", "circle-stroke-color": "#0b1220", "circle-stroke-width": 1.5 },
  });
  addLayer(map, {
    id: "dark-circle", type: "circle", source: "dark_vessels",
    paint: { "circle-radius": 6, "circle-color": "#ef4444", "circle-stroke-color": "#1a0606", "circle-stroke-width": 1.5 },
  });

  const b = detail.meta.bbox;
  map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 64, duration: 600 });
}

export default function MapView({
  detail,
  visible,
  onSelect,
}: {
  detail: CaseStudyDetail | null;
  visible: Record<LayerName, boolean>;
  onSelect: (f: SelectedFeature | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const loadedRef = useRef(false);
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  // initialise the map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: STYLE,
      center: [-90.3, 28.25],
      zoom: 8.5,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.on("load", () => {
      loadedRef.current = true;
      map.resize();
    });

    const interactiveIds = Object.keys(INTERACTIVE);
    map.on("click", (e) => {
      const ids = interactiveIds.filter((id) => map.getLayer(id));
      const feats = ids.length ? map.queryRenderedFeatures(e.point, { layers: ids }) : [];
      if (!feats.length) {
        onSelectRef.current(null);
        return;
      }
      const f = feats[0];
      onSelectRef.current({
        layer: INTERACTIVE[f.layer.id],
        properties: (f.properties ?? {}) as Record<string, unknown>,
      });
    });
    map.on("mousemove", (e) => {
      const ids = interactiveIds.filter((id) => map.getLayer(id));
      const feats = ids.length ? map.queryRenderedFeatures(e.point, { layers: ids }) : [];
      map.getCanvas().style.cursor = feats.length ? "pointer" : "";
    });

    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
      loadedRef.current = false;
    };
  }, []);

  // render layers when the case study changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !detail) return;
    const apply = () => addData(map, detail);
    if (loadedRef.current) apply();
    else map.once("load", apply);
  }, [detail]);

  // apply layer visibility
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => {
      (Object.keys(GROUP) as LayerName[]).forEach((name) => {
        GROUP[name].forEach((id) => {
          if (map.getLayer(id)) {
            map.setLayoutProperty(id, "visibility", visible[name] ? "visible" : "none");
          }
        });
      });
    };
    if (loadedRef.current) apply();
    else map.once("load", apply);
  }, [visible, detail]);

  return <div ref={containerRef} className="absolute inset-0" />;
}
