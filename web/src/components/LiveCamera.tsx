"use client";

import { useEffect, useRef, useState } from "react";
import type { PotholeDetector } from "@/lib/detector";
import { toSourceScale } from "@/lib/detector";
import { drawBoxes } from "@/lib/draw";
import { IouTracker } from "@/lib/tracker";
import { addReport, cropThumb, currentFix, newId, startLocation, stopLocation } from "@/lib/reports";

export function LiveCamera({ detector, threshold }: { detector: PotholeDetector; threshold: number }) {
  const video = useRef<HTMLVideoElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const live = useRef(false);
  const tracker = useRef(new IouTracker(0.2, 3, 6));
  const thr = useRef(threshold);
  thr.current = threshold;
  const [on, setOn] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [count, setCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [gps, setGps] = useState<string>("waiting for location");
  const [autoSave, setAutoSave] = useState(true);
  const auto = useRef(autoSave);
  auto.current = autoSave;

  async function start() {
    setErr(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      stream.current = s;
      const v = video.current!;
      v.srcObject = s;
      await v.play();
      startLocation((p) => setGps(`±${Math.round(p.coords.accuracy)} m`));
      tracker.current.reset();
      setCount(0);
      live.current = true;
      setOn(true);
      loop();
    } catch (e) {
      setErr(
        e instanceof DOMException && e.name === "NotAllowedError"
          ? "Camera permission was refused. Allow it in the browser settings and try again."
          : `Could not open the camera: ${String((e as Error)?.message ?? e)}`,
      );
    }
  }

  function stop() {
    live.current = false;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    stopLocation();
    setOn(false);
    const c = overlay.current;
    c?.getContext("2d")?.clearRect(0, 0, c.width, c.height);
  }

  useEffect(() => () => stop(), []); // eslint-disable-line react-hooks/exhaustive-deps

  async function loop() {
    const v = video.current!;
    const c = overlay.current!;
    let frames = 0;
    let t0 = performance.now();
    while (live.current) {
      const d = await detector.detect(v, thr.current);
      const boxes = toSourceScale(d, v);
      const { active, fresh } = tracker.current.update(boxes, performance.now());
      drawBoxes(
        c,
        active.filter((t) => t.confirmed).map((t) => t.box),
        v.videoWidth,
        v.videoHeight,
      );
      if (fresh.length) {
        setCount(tracker.current.confirmedCount);
        if (navigator.vibrate) navigator.vibrate(60);
        if (auto.current) {
          const fix = currentFix();
          for (const t of fresh) {
            addReport({
              id: newId(),
              createdAt: new Date().toISOString(),
              ...fix,
              score: t.bestScore,
              source: "live",
              thumb: cropThumb(v, t.box),
            });
          }
        }
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
  }

  return (
    <div className="space-y-4">
      <div className="relative bg-black rounded-xl overflow-hidden aspect-video">
        <video ref={video} playsInline muted className="w-full h-full object-contain" />
        <canvas ref={overlay} className="absolute inset-0 w-full h-full pointer-events-none" />
        {!on && (
          <div className="absolute inset-0 grid place-items-center p-6 text-center">
            <div className="space-y-3">
              <p className="text-sm text-muted max-w-sm">
                Mount the phone on the dashboard facing the road. Potholes are counted once each and, with location
                on, pinned to the map.
              </p>
              <button onClick={start} className="px-5 py-2.5 rounded-lg bg-amber text-amber-ink font-semibold">
                Start camera
              </button>
            </div>
          </div>
        )}
      </div>
      {err && <p className="text-danger text-sm">{err}</p>}
      {on && (
        <div className="flex flex-wrap items-center gap-4 text-sm tabular">
          <span>
            <b className="text-amber text-lg">{count}</b> pothole{count === 1 ? "" : "s"}
          </span>
          <span className="text-muted">{fps.toFixed(1)} fps</span>
          <span className="text-muted">GPS {gps}</span>
          <label className="flex items-center gap-2 text-muted">
            <input type="checkbox" checked={autoSave} onChange={(e) => setAutoSave(e.target.checked)} />
            Save to map
          </label>
          <button onClick={stop} className="ml-auto px-3 py-1.5 rounded-md border border-line hover:border-danger">
            Stop
          </button>
        </div>
      )}
    </div>
  );
}
