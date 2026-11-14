"use client";

import { useEffect, useRef, useState } from "react";
import type { PotholeDetector } from "@/lib/detector";
import { toSourceScale } from "@/lib/detector";
import { drawBoxes } from "@/lib/draw";
import { IouTracker } from "@/lib/tracker";

type Hit = { id: number; t: number; score: number; thumb: string };

export function DetectVideo({ detector, threshold }: { detector: PotholeDetector; threshold: number }) {
  const video = useRef<HTMLVideoElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const tracker = useRef(new IouTracker());
  const running = useRef(false);
  const thr = useRef(threshold);
  thr.current = threshold;
  const [url, setUrl] = useState<string | null>(null);
  const [count, setCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [hits, setHits] = useState<Hit[]>([]);

  useEffect(() => () => void (url && URL.revokeObjectURL(url)), [url]);

  function onFile(f: File | undefined) {
    if (!f) return;
    tracker.current.reset();
    setCount(0);
    setHits([]);
    setUrl(URL.createObjectURL(f));
  }

  // Process the frame that is on screen, as fast as the device allows.
  async function loop() {
    const v = video.current;
    const c = overlay.current;
    if (!v || !c || running.current) return;
    running.current = true;
    let frames = 0;
    let t0 = performance.now();
    while (!v.paused && !v.ended) {
      const d = await detector.detect(v, thr.current);
      const boxes = toSourceScale(d, v);
      const { active, fresh } = tracker.current.update(boxes, v.currentTime * 1000);
      drawBoxes(
        c,
        active.filter((t) => t.confirmed).map((t) => ({ ...t.box, label: `#${t.id}` })),
        v.videoWidth,
        v.videoHeight,
      );
      if (fresh.length) {
        setCount(tracker.current.confirmedCount);
        const cv = document.createElement("canvas");
        cv.width = 160;
        cv.height = 90;
        const thumbs = fresh.map((t) => {
          const b = t.box;
          const w = b.x2 - b.x1;
          const h = b.y2 - b.y1;
          const ctx = cv.getContext("2d")!;
          ctx.drawImage(v, b.x1 - w * 0.6, b.y1 - h * 0.6, w * 2.2, h * 2.2, 0, 0, 160, 90);
          return { id: t.id, t: v.currentTime, score: t.bestScore, thumb: cv.toDataURL("image/jpeg", 0.7) };
        });
        setHits((h) => [...thumbs, ...h].slice(0, 60));
      }
      frames++;
      const now = performance.now();
      if (now - t0 > 1000) {
        setFps((frames * 1000) / (now - t0));
        frames = 0;
        t0 = now;
      }
      await new Promise((r) => requestAnimationFrame(r));
    }
    running.current = false;
  }

  return (
    <div className="space-y-4">
      <label className="block border border-dashed border-line rounded-xl p-6 text-center cursor-pointer hover:border-amber transition-colors">
        <input type="file" accept="video/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
        <div className="font-medium">Choose a dashcam or phone video</div>
        <div className="text-sm text-muted mt-1">
          Each pothole is tracked across frames and counted once.
        </div>
      </label>

      {url && (
        <div className="relative bg-black rounded-xl overflow-hidden">
          <video
            ref={video}
            src={url}
            controls
            playsInline
            muted
            className="w-full max-h-[70vh] object-contain"
            onPlay={loop}
            onSeeked={() => tracker.current.reset()}
          />
          <canvas ref={overlay} className="absolute inset-0 w-full h-full pointer-events-none" />
        </div>
      )}

      {url && (
        <div className="flex items-center gap-4 text-sm tabular">
          <span>
            <b className="text-amber text-lg">{count}</b> unique pothole{count === 1 ? "" : "s"}
          </span>
          <span className="text-muted">{fps ? `${fps.toFixed(1)} frames per second` : "Press play"}</span>
        </div>
      )}

      {hits.length > 0 && (
        <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
          {hits.map((h) => (
            <button
              key={h.id}
              onClick={() => {
                if (video.current) video.current.currentTime = h.t;
              }}
              className="text-left bg-panel rounded-lg overflow-hidden border border-line hover:border-amber"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={h.thumb} alt={`Pothole ${h.id}`} className="w-full aspect-video object-cover" />
              <div className="px-2 py-1 text-xs text-muted tabular">
                #{h.id} at {h.t.toFixed(1)} s
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
