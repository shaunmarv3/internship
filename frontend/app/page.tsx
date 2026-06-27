"use client";

import { useCallback, useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { fetchCaseStudies, fetchCaseStudy } from "@/lib/api";
import type {
  CaseStudyDetail,
  CaseStudySummary,
  LayerName,
  SelectedFeature,
} from "@/lib/types";
import CaseStudyPicker from "@/components/CaseStudyPicker";
import LayerToggles from "@/components/LayerToggles";
import DetailPanel from "@/components/DetailPanel";
import MetricsStrip from "@/components/MetricsStrip";

const MapView = dynamic(() => import("@/components/MapView"), { ssr: false });

const ALL_LAYERS: LayerName[] = [
  "ships",
  "dark_vessels",
  "oil",
  "ais_tracks",
  "zones",
  "fusion_links",
];

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
  const [studies, setStudies] = useState<CaseStudySummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<CaseStudyDetail | null>(null);
  const [visible, setVisible] = useState<Record<LayerName, boolean>>(
    () =>
      Object.fromEntries(ALL_LAYERS.map((l) => [l, true])) as Record<
        LayerName,
        boolean
      >,
  );
  const [selected, setSelected] = useState<SelectedFeature | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCaseStudies()
      .then((s) => {
        setStudies(s);
        if (s.length) setActiveId(s[0].id);
      })
      .catch((e) =>
        setError(`Cannot reach API — start the backend on :8000. (${e})`),
      );
  }, []);

  useEffect(() => {
    if (!activeId) return;
    setSelected(null);
    fetchCaseStudy(activeId)
      .then(setDetail)
      .catch((e) => setError(String(e)));
  }, [activeId]);

  const toggle = useCallback(
    (l: LayerName) => setVisible((v) => ({ ...v, [l]: !v[l] })),
    [],
  );

  return (
    <div className="flex h-screen flex-col">
      <TopBar detail={detail} />
      <div className="flex min-h-0 flex-1">
        <aside className="flex w-72 flex-col overflow-y-auto border-r border-zinc-800 bg-zinc-950">
          <CaseStudyPicker
            studies={studies}
            activeId={activeId}
            onSelect={setActiveId}
          />
          <LayerToggles visible={visible} onToggle={toggle} />
        </aside>

        <main className="relative min-w-0 flex-1">
          {error && (
            <div className="absolute left-3 top-3 z-10 max-w-sm rounded border border-red-900/60 bg-red-950/80 px-3 py-2 text-xs text-red-200">
              {error}
            </div>
          )}
          <MapView detail={detail} visible={visible} onSelect={setSelected} />
        </main>

        <DetailPanel detail={detail} selected={selected} />
      </div>
      <MetricsStrip detail={detail} />
    </div>
  );
}
