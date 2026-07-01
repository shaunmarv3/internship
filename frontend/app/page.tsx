"use client";

import { useCallback, useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { fetchCaseStudies, fetchCaseStudy } from "@/lib/api";
import type { CaseStudyDetail, LayerName, PngDetectResult } from "@/lib/types";
import SceneInfo from "@/components/SceneInfo";
import UploadScene from "@/components/UploadScene";
import UploadPng from "@/components/UploadPng";
import PngResult, { PngInfo } from "@/components/PngResult";

const MapView = dynamic(() => import("@/components/MapView"), { ssr: false });

const ALL_LAYERS: LayerName[] = [
  "ships",
  "dark_vessels",
  "oil",
  "ais_tracks",
  "zones",
];

// Layers are always on now (the toggle panel was removed) — pass a constant.
const ALL_VISIBLE = Object.fromEntries(
  ALL_LAYERS.map((l) => [l, true]),
) as Record<LayerName, boolean>;

function TopBar({ detail }: { detail: CaseStudyDetail | null }) {
  return (
    <header className="flex h-12 items-center justify-between border-b border-zinc-800 bg-zinc-950 px-4">
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-sm bg-zinc-400" />
        <h1 className="text-sm font-semibold tracking-tight text-zinc-100">
          Maritime Intelligence
        </h1>
      </div>
      {detail && (
        <div className="font-mono text-xs text-zinc-500">
          {detail.meta.sensor} · {detail.meta.acquired_utc}
        </div>
      )}
    </header>
  );
}

export default function Page() {
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<CaseStudyDetail | null>(null);
  const [pngResult, setPngResult] = useState<PngDetectResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load the most recently processed scene on first paint.
  useEffect(() => {
    fetchCaseStudies()
      .then((s) => {
        if (s.length) setActiveId(s[s.length - 1].id);
      })
      .catch((e) =>
        setError(`Cannot reach API — start the backend on :8000. (${e})`),
      );
  }, []);

  useEffect(() => {
    if (!activeId) return;
    fetchCaseStudy(activeId)
      .then(setDetail)
      .catch((e) => setError(String(e)));
  }, [activeId]);

  // Uploading a georeferenced scene clears any PNG-demo result (they share the main area).
  const handleUploaded = useCallback((id: string) => {
    setPngResult(null);
    setActiveId(id);
  }, []);

  const handlePngDetected = useCallback((r: PngDetectResult) => setPngResult(r), []);

  return (
    <div className="flex h-screen flex-col">
      <TopBar detail={pngResult ? null : detail} />
      <div className="flex min-h-0 flex-1">
        <aside className="flex w-72 flex-col overflow-y-auto border-r border-zinc-800 bg-zinc-950">
          <UploadScene onUploaded={handleUploaded} />
          <UploadPng onDetected={handlePngDetected} />
          {pngResult ? <PngInfo result={pngResult} /> : <SceneInfo detail={detail} />}
        </aside>

        <main className="relative min-w-0 flex-1">
          {error && (
            <div className="absolute left-3 top-3 z-10 max-w-sm rounded border border-red-900/60 bg-red-950/80 px-3 py-2 text-xs text-red-200">
              {error}
            </div>
          )}
          {pngResult ? (
            <PngResult result={pngResult} />
          ) : (
            <MapView detail={detail} visible={ALL_VISIBLE} onSelect={() => {}} />
          )}
        </main>
      </div>
    </div>
  );
}
