"use client";

import { useRef, useState } from "react";
import { detectPng } from "@/lib/api";
import type { PngDetectResult } from "@/lib/types";

export default function UploadPng({
  onDetected,
}: {
  onDetected: (r: PngDetectResult) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handle(file: File) {
    setBusy(true);
    setMsg("Running YOLO on HRSID image…");
    try {
      const res = await detectPng(file);
      setMsg(null);
      onDetected(res);
    } catch (e) {
      setMsg(`Failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="border-b border-zinc-800 p-3">
      <h2 className="mb-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        YOLO demo (HRSID .jpg)
      </h2>
      <p className="mb-2 text-[11px] leading-snug text-zinc-600">
        Detector-only exhibit on native-resolution SAR — no CFAR, no oil model, no map.
      </p>
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const f = e.dataTransfer.files?.[0];
          if (f && !busy) handle(f);
        }}
        onClick={() => !busy && inputRef.current?.click()}
        className={`cursor-pointer rounded border border-dashed px-3 py-5 text-center text-xs transition-colors ${
          busy
            ? "border-zinc-700 text-zinc-600"
            : "border-zinc-700 text-zinc-500 hover:border-zinc-500 hover:text-zinc-400"
        }`}
      >
        {busy ? "Detecting…" : "Drop an HRSID .jpg, or click to choose"}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handle(f);
        }}
      />
      {msg && <p className="mt-2 text-[11px] leading-snug text-zinc-500">{msg}</p>}
    </div>
  );
}
