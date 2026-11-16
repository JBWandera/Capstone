"use client";

import { useState } from "react";
import { useDetector } from "@/lib/useDetector";
import { DEFAULT_THRESHOLD, liteMode, setLiteMode } from "@/lib/detector";
import { DetectPhoto } from "@/components/DetectPhoto";
import { DetectVideo } from "@/components/DetectVideo";
import { LiveCamera } from "@/components/LiveCamera";

const modes = [
  { id: "photo", label: "Photo" },
  { id: "video", label: "Video" },
  { id: "live", label: "Live camera" },
] as const;
type Mode = (typeof modes)[number]["id"];

export default function Home() {
  const det = useDetector();
  const [mode, setMode] = useState<Mode>("photo");
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [dashcam, setDashcam] = useState(true);

  return (
    <div className="pt-8 space-y-6">
      <section className="space-y-2">
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">
          Find potholes before they find you.
        </h1>
        <p className="text-muted max-w-2xl">
          AfroViewz spots potholes in road photos, recorded video and the live camera. The model runs inside your
          browser, so it works offline and nothing is uploaded.
        </p>
      </section>

      <div className="flex flex-wrap items-center gap-3">
        <div className="inline-flex bg-panel border border-line rounded-lg p-1">
          {modes.map((m) => (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className={`px-4 py-1.5 rounded-md text-sm transition-colors ${
                mode === m.id ? "bg-amber text-amber-ink font-medium" : "text-muted hover:text-ink"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        {mode !== "photo" && (
          <label className="flex items-center gap-2 text-sm text-muted" title="Look only at the band of road where potholes appear in dashcam footage">
            <input type="checkbox" checked={dashcam} onChange={(e) => setDashcam(e.target.checked)} className="accent-amber" />
            Dashcam mode
          </label>
        )}
        <label className="flex items-center gap-3 text-sm text-muted ml-auto">
          Sensitivity
          <input
            type="range"
            min={0.15}
            max={0.8}
            step={0.05}
            value={Number((0.95 - threshold).toFixed(2))}
            onChange={(e) => setThreshold(Number((0.95 - Number(e.target.value)).toFixed(2)))}
            className="accent-amber w-32"
            aria-label="Sensitivity"
          />
          <span className="tabular w-16 text-ink">min {Math.round(threshold * 100)}%</span>
        </label>
      </div>

      {det.status === "loading" && (
        <div className="bg-panel border border-line rounded-xl p-6 text-sm">
          <div className="mb-3">Loading the detector (once, then it works offline)…</div>
          <div className="h-1.5 bg-line rounded-full overflow-hidden">
            <div className="h-full bg-amber transition-all" style={{ width: `${Math.round(det.progress * 100)}%` }} />
          </div>
        </div>
      )}
      {det.status === "error" && (
        <div className="bg-panel border border-danger rounded-xl p-6 text-sm">
          The detector could not load: {det.message}
        </div>
      )}
      {det.status === "ready" && (
        <>
          {mode === "photo" && <DetectPhoto detector={det.detector} threshold={threshold} />}
          {mode === "video" && <DetectVideo detector={det.detector} threshold={threshold} dashcam={dashcam} />}
          {mode === "live" && <LiveCamera detector={det.detector} threshold={threshold} dashcam={dashcam} />}
          {det.recovered && (
            <p className="text-sm bg-panel border border-amber/50 rounded-lg p-3">
              The last live session stopped unexpectedly, so AfroViewz switched to lite mode (processor only). It is a
              little slower but uses much less memory.
            </p>
          )}
          <p className="text-xs text-muted">
            Running on {det.detector.backend === "webgpu" ? "the graphics chip (WebGPU)" : "the processor (WebAssembly)"}
            {liteMode() ? ", lite mode" : ""}.{" "}
            <button
              className="underline hover:text-ink"
              onClick={() => {
                setLiteMode(!liteMode());
                location.reload();
              }}
            >
              {liteMode() ? "Use the graphics chip" : "Switch to lite mode"}
            </button>
          </p>
        </>
      )}
    </div>
  );
}
