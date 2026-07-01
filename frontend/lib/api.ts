import type { CaseStudyDetail, CaseStudySummary, PngDetectResult } from "./types";

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

export function pngAssetUrl(rel: string): string {
  return `${BASE}/png-assets/${rel}`;
}

export async function detectPng(file: File): Promise<PngDetectResult> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/detect-png`, { method: "POST", body: fd });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return r.json();
}

export async function processScene(file: File): Promise<{ id: string }> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/process`, { method: "POST", body: fd });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return r.json();
}
