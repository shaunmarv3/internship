"use client";

import { useRef, useState } from "react";
import { processScene } from "@/lib/api";

export default function UploadScene({ onUploaded }: { onUploaded: (id: string) => void }) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handle(file: File) {
    setBusy(true);
    setMsg("Processing scene — detection + segmentation…");
    try {
      const { id } = await processScene(file);
      setMsg(null);
      onUploaded(id);
    } catch (e) {
      setMsg(`Failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="border-b border-zinc-800 p-3">
      <h2 className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        Process a scene
      </h2>
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
        {busy ? "Processing…" : "Drop a Sentinel-1 .tif, or click to choose"}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".tif,.tiff"
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
