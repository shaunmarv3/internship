"use client";

import type { LayerName } from "@/lib/types";

const LABELS: Record<LayerName, string> = {
  ships: "Vessels",
  dark_vessels: "Dark vessels",
  oil: "Oil slick",
  ais_tracks: "AIS tracks",
  zones: "Zones",
  fusion_links: "Fusion links",
};

const DOT: Record<LayerName, string> = {
  ships: "#5b8fb0",
  dark_vessels: "#ef4444",
  oil: "#f59e0b",
  ais_tracks: "#64748b",
  zones: "#475569",
  fusion_links: "#ef4444",
};

const ORDER: LayerName[] = ["ships", "dark_vessels", "oil", "ais_tracks", "zones", "fusion_links"];

export default function LayerToggles({
  visible,
  onToggle,
}: {
  visible: Record<LayerName, boolean>;
  onToggle: (l: LayerName) => void;
}) {
  return (
    <div className="p-3">
      <h2 className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        Layers
      </h2>
      <ul className="space-y-0.5">
        {ORDER.map((l) => (
          <li key={l}>
            <label className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-sm text-zinc-300 hover:bg-zinc-900">
              <input
                type="checkbox"
                checked={visible[l]}
                onChange={() => onToggle(l)}
                className="accent-zinc-500"
              />
              <span className="h-2 w-2 rounded-full" style={{ background: DOT[l] }} />
              {LABELS[l]}
            </label>
          </li>
        ))}
      </ul>
    </div>
  );
}
