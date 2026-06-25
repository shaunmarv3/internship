"use client";

import type { CaseStudyDetail } from "@/lib/types";

export default function MetricsStrip({ detail }: { detail: CaseStudyDetail | null }) {
  const m = detail?.metrics;
  const items: [string, string][] = m
    ? [
        ["Oil IoU", m.oil_iou != null ? m.oil_iou.toFixed(2) : "—"],
        ["YOLO mAP50", m.yolo_map50 != null ? m.yolo_map50.toFixed(2) : "—"],
        ["Dark vessels", String(m.dark_count)],
        ["Slicks", String(m.slick_count)],
      ]
    : [];

  return (
    <footer className="flex h-10 items-center gap-6 border-t border-zinc-800 bg-zinc-950 px-4 text-xs">
      {items.length === 0 ? (
        <span className="text-zinc-600">No scene loaded</span>
      ) : (
        items.map(([k, v]) => (
          <div key={k} className="flex items-center gap-2">
            <span className="text-zinc-500">{k}</span>
            <span className="font-mono text-zinc-200">{v}</span>
          </div>
        ))
      )}
    </footer>
  );
}
