"use client";

import { assetUrl } from "@/lib/api";
import type { CaseStudyDetail, SelectedFeature } from "@/lib/types";

function Bar({ value, color }: { value: number; color: string }) {
  return (
    <div className="h-1.5 w-full rounded bg-zinc-800">
      <div
        className="h-full rounded"
        style={{ width: `${Math.round(value * 100)}%`, background: color }}
      />
    </div>
  );
}

function EntityCard({ f }: { f: SelectedFeature }) {
  const p = f.properties;
  const rows: [string, string][] = [];
  if (f.layer === "ships" || f.layer === "dark_vessels") {
    rows.push(["Status", f.layer === "dark_vessels" ? "Dark — no AIS" : "Broadcasting"]);
    if (p.mmsi) rows.push(["MMSI", String(p.mmsi)]);
    if (p.type) rows.push(["Type", String(p.type)]);
    if (p.confidence != null) rows.push(["Confidence", Number(p.confidence).toFixed(2)]);
  } else if (f.layer === "oil") {
    rows.push(["Feature", "Oil slick"]);
    if (p.area_km2 != null) rows.push(["Area", `${Number(p.area_km2).toFixed(1)} km²`]);
  }
  return (
    <div className="rounded border border-zinc-800 bg-zinc-900/50 p-3 text-sm">
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between py-0.5">
          <span className="text-zinc-500">{k}</span>
          <span className="font-mono text-zinc-200">{v}</span>
        </div>
      ))}
    </div>
  );
}

export default function DetailPanel({
  detail,
  selected,
}: {
  detail: CaseStudyDetail | null;
  selected: SelectedFeature | null;
}) {
  if (!detail) return null;
  const { metrics, explain } = detail;

  return (
    <aside className="flex w-80 flex-col gap-5 overflow-y-auto border-l border-zinc-800 bg-zinc-950 p-4">
      <section>
        <h2 className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
          Selected
        </h2>
        {selected ? (
          <EntityCard f={selected} />
        ) : (
          <p className="text-xs text-zinc-600">Click a vessel or slick on the map.</p>
        )}
      </section>

      <section>
        <h2 className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
          Risk assessment
        </h2>
        <div className="space-y-3 text-sm">
          <div>
            <div className="mb-1 flex justify-between text-zinc-400">
              <span>Security</span>
              <span className="font-mono text-zinc-200">{metrics.security_risk.toFixed(2)}</span>
            </div>
            <Bar value={metrics.security_risk} color="#ef4444" />
          </div>
          <div>
            <div className="mb-1 flex justify-between text-zinc-400">
              <span>Environmental</span>
              <span className="font-mono text-zinc-200">
                {metrics.environmental_risk.toFixed(2)}
              </span>
            </div>
            <Bar value={metrics.environmental_risk} color="#f59e0b" />
          </div>
        </div>
      </section>

      <section>
        <h2 className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
          Why flagged
        </h2>
        <ul className="space-y-2">
          {explain.risk_factors.map((r) => (
            <li key={r.label} className="text-xs">
              <div className="mb-1 flex justify-between text-zinc-400">
                <span>{r.label}</span>
                <span className="font-mono text-zinc-300">{Math.round(r.weight * 100)}%</span>
              </div>
              <Bar value={r.weight} color="#52525b" />
            </li>
          ))}
        </ul>
      </section>

      {(explain.gradcam_oil_png || explain.gradcam_ship_png) && (
        <section>
          <h2 className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
            Grad-CAM
          </h2>
          <div className="grid grid-cols-2 gap-2">
            {explain.gradcam_oil_png && (
              <figure>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={assetUrl(explain.gradcam_oil_png)}
                  alt="Grad-CAM oil saliency"
                  className="w-full rounded border border-zinc-800"
                />
                <figcaption className="mt-1 text-[10px] text-zinc-500">Oil</figcaption>
              </figure>
            )}
            {explain.gradcam_ship_png && (
              <figure>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={assetUrl(explain.gradcam_ship_png)}
                  alt="Grad-CAM vessel saliency"
                  className="w-full rounded border border-zinc-800"
                />
                <figcaption className="mt-1 text-[10px] text-zinc-500">Vessel</figcaption>
              </figure>
            )}
          </div>
        </section>
      )}
    </aside>
  );
}
