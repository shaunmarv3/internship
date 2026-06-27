import type { CaseStudyDetail, CaseStudySummary } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function fetchCaseStudies(): Promise<CaseStudySummary[]> {
  const r = await fetch(`${BASE}/case-studies`);
  if (!r.ok) throw new Error(`GET /case-studies -> ${r.status}`);
  return r.json();
}

export async function fetchCaseStudy(id: string): Promise<CaseStudyDetail> {
  const r = await fetch(`${BASE}/case-study/${id}`);
  if (!r.ok) throw new Error(`GET /case-study/${id} -> ${r.status}`);
  return r.json();
}

export function assetUrl(rel: string): string {
  return `${BASE}/assets/${rel}`;
}
