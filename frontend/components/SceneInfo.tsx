"use client";

import type { CaseStudyDetail } from "@/lib/types";

// Degrees → km. Latitude is ~constant; longitude shrinks with cos(lat).
const NS_KM_PER_DEG = 110.57;
const EW_KM_PER_DEG = 111.32;

function fmtCoord(v: number, pos: string, neg: string): string {
  return `${Math.abs(v).toFixed(3)}° ${v >= 0 ? pos : neg}`;
}

function fmtKm(v: number): string {
  if (!isFinite(v)) return "—";
  return v < 1 ? `${(v * 1000).toFixed(0)} m` : `${v.toFixed(1)} km`;
}

function fmtArea(km2: number): string {
  if (km2 <= 0) return "—";
  return km2 < 1 ? `${(km2 * 1e6).toLocaleString()} m²` : `${km2.toFixed(2)} km²`;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5">
      <span className="text-[11px] text-zinc-500">{label}</span>
      <span className="font-mono text-[11px] text-zinc-200">{value}</span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border-b border-zinc-800 px-3 py-3">
      <h2 className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        {title}
      </h2>
      {children}
    </div>
  );
}

export default function SceneInfo({ detail }: { detail: CaseStudyDetail | null }) {
  if (!detail) {
    return (
      <div className="px-3 py-4 text-[11px] leading-snug text-zinc-600">
        Drop a scene above to run detection — metadata and oil-spill extent appear here.
      </div>
    );
  }

  const m = detail.meta;
  const [w, s, e, n] = m.bbox;
  const cLon = (w + e) / 2;
  const cLat = (s + n) / 2;
  const sceneEW = (e - w) * EW_KM_PER_DEG * Math.cos((cLat * Math.PI) / 180);
  const sceneNS = (n - s) * NS_KM_PER_DEG;

  const shipCount = detail.layers.ships.features.length;
  const darkCount = detail.metrics.dark_count;
  const slickCount = detail.metrics.slick_count;

  // Oil-spill extent: union bbox + total area across all slick polygons.
  const oilFeats = detail.layers.oil.features;
  let oMinLon = Infinity, oMinLat = Infinity, oMaxLon = -Infinity, oMaxLat = -Infinity;
  let totalArea = 0;
  for (const f of oilFeats) {
    totalArea += Number(f.properties?.area_km2) || 0;
    if (f.geometry.type !== "Polygon") continue;
    for (const ring of f.geometry.coordinates) {
      for (const [lon, lat] of ring as [number, number][]) {
        oMinLon = Math.min(oMinLon, lon); oMaxLon = Math.max(oMaxLon, lon);
        oMinLat = Math.min(oMinLat, lat); oMaxLat = Math.max(oMaxLat, lat);
      }
    }
  }
  const hasOil = oilFeats.length > 0 && isFinite(oMinLon);
  const oilMidLat = (oMinLat + oMaxLat) / 2;
  const oilEW = (oMaxLon - oMinLon) * EW_KM_PER_DEG * Math.cos((oilMidLat * Math.PI) / 180);
  const oilNS = (oMaxLat - oMinLat) * NS_KM_PER_DEG;
  const oilLength = Math.max(oilEW, oilNS);
  const oilWidth = Math.min(oilEW, oilNS);

  return (
    <div>
      <Section title="Scene">
        <Row label="ID" value={m.id} />
        <Row label="Sensor" value={m.sensor} />
        <Row label="Acquired" value={m.acquired_utc} />
      </Section>

      <Section title="Coverage">
        <Row label="Centre" value={`${fmtCoord(cLat, "N", "S")}, ${fmtCoord(cLon, "E", "W")}`} />
        <Row label="Extent" value={`${fmtKm(sceneEW)} × ${fmtKm(sceneNS)}`} />
      </Section>

      <Section title="Detections">
        <Row label="Vessels (AIS-matched)" value={String(shipCount)} />
        <Row label="Dark vessels" value={String(darkCount)} />
        <Row label="Oil slicks" value={String(slickCount)} />
      </Section>

      <Section title="Oil spill">
        {hasOil ? (
          <>
            <Row label="Affected area" value={fmtArea(totalArea)} />
            <Row label="Spread (L × W)" value={`${fmtKm(oilLength)} × ${fmtKm(oilWidth)}`} />
            <Row label="Slick patches" value={String(oilFeats.length)} />
          </>
        ) : (
          <p className="text-[11px] text-zinc-600">No oil detected in this scene.</p>
        )}
      </Section>
    </div>
  );
}
