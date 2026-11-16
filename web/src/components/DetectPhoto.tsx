"use client";

import { useEffect, useRef, useState } from "react";
import type { Box, PotholeDetector } from "@/lib/detector";
import { toSourceScale } from "@/lib/detector";
import { drawBoxes, renderAnnotated } from "@/lib/draw";
import { addReport, cropThumb, newId } from "@/lib/reports";

type Result = { boxes: Box[]; ms: number; w: number; h: number };

/** Read GPS from EXIF if the photo has it (JPEG APP1, little or big endian). */
async function exifGps(file: File): Promise<{ lat: number; lon: number } | null> {
  try {
    const buf = new DataView(await file.slice(0, 256 * 1024).arrayBuffer());
    if (buf.getUint16(0) !== 0xffd8) return null;
    let off = 2;
    while (off < buf.byteLength - 4) {
      const marker = buf.getUint16(off);
      const len = buf.getUint16(off + 2);
      if (marker === 0xffe1 && buf.getUint32(off + 4) === 0x45786966) {
        const t = off + 10;
        const le = buf.getUint16(t) === 0x4949;
        const u16 = (p: number) => buf.getUint16(t + p, le);
        const u32 = (p: number) => buf.getUint32(t + p, le);
        const ifd0 = u32(4);
        let gpsOff = 0;
        for (let i = 0; i < u16(ifd0); i++) {
          const e = ifd0 + 2 + i * 12;
          if (u16(e) === 0x8825) gpsOff = u32(e + 8);
        }
        if (!gpsOff) return null;
        const tags: Record<number, number[] | string> = {};
        for (let i = 0; i < u16(gpsOff); i++) {
          const e = gpsOff + 2 + i * 12;
          const tag = u16(e);
          const type = u16(e + 2);
          if (type === 2) tags[tag] = String.fromCharCode(buf.getUint8(t + e + 8));
          if (type === 5) {
            const p = u32(e + 8);
            tags[tag] = [0, 1, 2].map((k) => u32(p + k * 8) / u32(p + k * 8 + 4));
          }
        }
        const dms = (v: number[] | string | undefined) =>
          Array.isArray(v) ? v[0] + v[1] / 60 + v[2] / 3600 : NaN;
        let lat = dms(tags[2]);
        let lon = dms(tags[4]);
        if (tags[1] === "S") lat = -lat;
        if (tags[3] === "W") lon = -lon;
        return Number.isFinite(lat) && Number.isFinite(lon) ? { lat, lon } : null;
      }
      off += 2 + len;
    }
  } catch {}
  return null;
}

export function DetectPhoto({ detector, threshold }: { detector: PotholeDetector; threshold: number }) {
  const imgRef = useRef<HTMLImageElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [res, setRes] = useState<Result | null>(null);
  const [all, setAll] = useState<Box[]>([]);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const [needsLocation, setNeedsLocation] = useState(false);

  async function run() {
    const img = imgRef.current;
    if (!img) return;
    setBusy(true);
    // Run once at a low threshold, then filter in the UI as the slider moves.
    const d = await detector.detect(img, 0.1);
    const boxes = toSourceScale(d, img);
    setAll(boxes);
    setRes({ boxes, ms: d.ms, w: img.naturalWidth, h: img.naturalHeight });
    setBusy(false);
  }

  const shown = all.filter((b) => b.score >= threshold);

  useEffect(() => {
    const c = overlay.current;
    if (c && res) drawBoxes(c, shown, res.w, res.h);
  });

  useEffect(() => () => void (url && URL.revokeObjectURL(url)), [url]);

  function onFile(f: File | undefined) {
    if (!f) return;
    setFile(f);
    setRes(null);
    setAll([]);
    setSaved(null);
    setNeedsLocation(false);
    setUrl(URL.createObjectURL(f));
  }

  function hereNow(): Promise<{ lat: number; lon: number; acc: number } | null> {
    return new Promise((resolve) => {
      if (!("geolocation" in navigator)) return resolve(null);
      navigator.geolocation.getCurrentPosition(
        (p) => resolve({ lat: p.coords.latitude, lon: p.coords.longitude, acc: Math.round(p.coords.accuracy) }),
        () => resolve(null),
        { enableHighAccuracy: true, timeout: 15000, maximumAge: 30000 },
      );
    });
  }

  async function save(useHere = false) {
    const img = imgRef.current;
    if (!img || !file) return;
    const exif = await exifGps(file);
    const here = !exif && useHere ? await hereNow() : null;
    if (!exif && !useHere) {
      setNeedsLocation(true);
      return;
    }
    setNeedsLocation(false);
    for (const b of shown) {
      await addReport({
        id: newId(),
        createdAt: new Date(exif ? file.lastModified || Date.now() : Date.now()).toISOString(),
        lat: exif?.lat ?? here?.lat ?? null,
        lon: exif?.lon ?? here?.lon ?? null,
        accuracyM: here?.acc ?? null,
        score: b.score,
        source: "photo",
        thumb: cropThumb(img, b),
      });
    }
    setSaved(
      exif || here
        ? `Saved ${shown.length} to the map.`
        : `Saved ${shown.length}, but without a location, so they are in the exports but not on the map.`,
    );
  }

  async function saveNoLocation() {
    const img = imgRef.current;
    if (!img || !file) return;
    for (const b of shown) {
      await addReport({
        id: newId(),
        createdAt: new Date().toISOString(),
        lat: null,
        lon: null,
        accuracyM: null,
        score: b.score,
        source: "photo",
        thumb: cropThumb(img, b),
      });
    }
    setSaved(`Saved ${shown.length} without a location: they are in the CSV export but not on the map.`);
  }

  function downloadAnnotated() {
    const img = imgRef.current;
    if (!img || !res) return;
    const c = renderAnnotated(img, res.w, res.h, shown);
    c.toBlob((b) => {
      if (!b) return;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(b);
      a.download = `afroviewz-${(file?.name ?? "photo").replace(/\.\w+$/, "")}.jpg`;
      a.click();
    }, "image/jpeg", 0.9);
  }

  return (
    <div className="space-y-4">
      <label
        className="block border border-dashed border-line rounded-xl p-6 text-center cursor-pointer hover:border-amber transition-colors"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          onFile(e.dataTransfer.files?.[0]);
        }}
      >
        <input type="file" accept="image/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
        <div className="font-medium">Choose or drop a road photo</div>
        <div className="text-sm text-muted mt-1">JPEG or PNG. Phone photos with location are placed on the map.</div>
      </label>

      {url && (
        <div className="relative bg-black rounded-xl overflow-hidden">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img ref={imgRef} src={url} alt="Uploaded road" className="w-full max-h-[70vh] object-contain" onLoad={run} />
          <canvas ref={overlay} className="absolute inset-0 w-full h-full pointer-events-none" />
          {busy && (
            <div className="absolute inset-0 grid place-items-center bg-black/40 text-sm">Looking for potholes…</div>
          )}
        </div>
      )}

      {res && (
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <span className="tabular">
            <b className="text-amber text-lg">{shown.length}</b> pothole{shown.length === 1 ? "" : "s"} found in{" "}
            {Math.round(res.ms)} ms
          </span>
          <div className="ml-auto flex gap-2">
            <button onClick={downloadAnnotated} className="px-3 py-1.5 rounded-md border border-line hover:border-amber">
              Download image
            </button>
            <button
              onClick={() => save(false)}
              disabled={!shown.length}
              className="px-3 py-1.5 rounded-md bg-amber text-amber-ink font-medium disabled:opacity-40"
            >
              Save reports
            </button>
          </div>
          {needsLocation && (
            <div className="w-full bg-panel border border-line rounded-lg p-3 space-y-2">
              <p className="text-muted">
                This photo carries no location (iPhone removes it when a photo is picked in the browser). Are you at
                the spot where it was taken?
              </p>
              <div className="flex gap-2">
                <button onClick={() => save(true)} className="px-3 py-1.5 rounded-md bg-amber text-amber-ink font-medium">
                  Use my current location
                </button>
                <button
                  onClick={async () => {
                    setNeedsLocation(false);
                    await saveNoLocation();
                  }}
                  className="px-3 py-1.5 rounded-md border border-line hover:border-amber"
                >
                  Save without location
                </button>
              </div>
            </div>
          )}
          {saved && <div className="w-full text-muted">{saved}</div>}
        </div>
      )}
    </div>
  );
}
