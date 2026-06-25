"use client";

import type { CaseStudySummary } from "@/lib/types";

export default function CaseStudyPicker({
  studies,
  activeId,
  onSelect,
}: {
  studies: CaseStudySummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="border-b border-zinc-800 p-3">
      <h2 className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        Case studies
      </h2>
      <ul className="space-y-1">
        {studies.map((s) => (
          <li key={s.id}>
            <button
              onClick={() => onSelect(s.id)}
              className={`w-full rounded px-2 py-1.5 text-left text-sm transition-colors ${
                activeId === s.id
                  ? "bg-zinc-800 text-zinc-100"
                  : "text-zinc-400 hover:bg-zinc-900"
              }`}
            >
              <div className="font-medium leading-tight">{s.title}</div>
              <div className="text-[11px] text-zinc-500">{s.sensor}</div>
            </button>
          </li>
        ))}
        {studies.length === 0 && (
          <li className="px-2 text-xs text-zinc-600">No case studies — is the API running?</li>
        )}
      </ul>
    </div>
  );
}
