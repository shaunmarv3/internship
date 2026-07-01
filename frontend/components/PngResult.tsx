"use client";

import { pngAssetUrl } from "@/lib/api";
import type { PngDetectResult } from "@/lib/types";

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

// Sidebar metadata for the PNG demo. Coverage is intentionally blank — a PNG has
// no georeferencing, so there is no centre/extent to report.
export function PngInfo({ result }: { result: PngDetectResult }) {
  const confs = result.confidences;
  const meanConf = confs.length
    ? confs.reduce((a, b) => a + b, 0) / confs.length
    : 0;

  return (
    <div>
      <Section title="Scene">
        <Row label="Source" value={result.filename} />
        <Row label="Sensor" value="HRSID (SAR)" />
        <Row label="Size" value={`${result.width} × ${result.height} px`} />
      </Section>

      <Section title="Coverage">
        <Row label="Centre" value="—" />
        <Row label="Extent" value="—" />
      </Section>

      <Section title="Detections">
        <Row label="Vessels (YOLO)" value={String(result.ship_count)} />
        <Row
          label="Mean confidence"
          value={confs.length ? meanConf.toFixed(2) : "—"}
        />
      </Section>
    </div>
  );
}

// Main-area image view — the annotated HRSID image, shown in place of the map.
export default function PngResult({ result }: { result: PngDetectResult }) {
  const url = pngAssetUrl(result.image) + (result.v ? `?v=${result.v}` : "");
  return (
    <div className="flex h-full w-full items-center justify-center bg-[#0b1622] p-4">
      <div className="relative max-h-full max-w-full">
        <img
          src={url}
          alt={`YOLO detections on ${result.filename}`}
          className="max-h-full max-w-full rounded border border-zinc-800 object-contain"
        />
        <div className="absolute left-2 top-2 rounded border border-zinc-700 bg-zinc-950/80 px-2 py-1 font-mono text-[11px] text-zinc-200">
          {result.ship_count} vessel{result.ship_count === 1 ? "" : "s"} · YOLO11m-OBB
        </div>
      </div>
    </div>
  );
}
