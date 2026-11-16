// AfroViewz detector, main thread side. The model itself runs in
// detector.worker.ts so the interface never freezes while it works.

export type Box = { x1: number; y1: number; x2: number; y2: number; score: number };
export type Detection = { boxes: Box[]; ms: number; width: number; height: number; offsetY: number };
export type Backend = "webgpu" | "wasm";

/** Road band for dashcam style frames, as fractions of image height (same as
 *  src/afroviewz/data/roi.py). The model was trained on frames cut this way. */
export const BAND: [number, number] = [0.35, 0.75];

export const MODEL_URL = process.env.NEXT_PUBLIC_MODEL_URL ?? "/models/afroviewz-nano-roi-fp16.onnx";
/** INT8 version for devices without WebGPU: about twice as fast on the
 *  processor, 0.623 test mAP50 against 0.643 for the full model. */
export const MODEL_URL_CPU = process.env.NEXT_PUBLIC_MODEL_URL_CPU ?? "/models/afroviewz-nano-roi-int8.onnx";
export const DEFAULT_THRESHOLD = 0.35;
/** Longest side read back from photos (full detail, the result is shown once). */
export const PHOTO_MAX_SIDE = 1920;
/** Longest side read back from video and camera frames: small enough to be
 *  quick and light on memory, large enough not to change the results. */
export const STREAM_MAX_SIDE = 960;

const LITE_KEY = "afroviewz-lite";

/** Lite mode runs on the CPU only. It is switched on automatically if a live
 *  session ended in a crash (some phones run out of graphics memory). */
export function liteMode(): boolean {
  try {
    return localStorage.getItem(LITE_KEY) === "1";
  } catch {
    return false;
  }
}
export function setLiteMode(on: boolean): void {
  try {
    if (on) localStorage.setItem(LITE_KEY, "1");
    else localStorage.removeItem(LITE_KEY);
  } catch {}
}

type Pending = { resolve: (r: { boxes: Box[]; ms: number }) => void; reject: (e: Error) => void };

export class PotholeDetector {
  private pending = new Map<number, Pending>();
  private nextId = 1;
  backend: Backend = "wasm";
  resolution = 384;

  private constructor(private worker: Worker) {}

  static create(opts: { url?: string; onProgress?: (f: number) => void } = {}): Promise<PotholeDetector> {
    const worker = new Worker(new URL("./detector.worker.ts", import.meta.url), { type: "module" });
    const det = new PotholeDetector(worker);
    return new Promise((resolve, reject) => {
      worker.onmessage = (e: MessageEvent) => {
        const m = e.data;
        if (m.type === "progress") opts.onProgress?.(m.value);
        else if (m.type === "ready") {
          det.backend = m.backend;
          det.resolution = m.resolution;
          worker.onmessage = (ev) => det.onMessage(ev);
          resolve(det);
        } else if (m.type === "error") reject(new Error(m.message));
      };
      worker.onerror = (e) => reject(new Error(e.message || "detector worker failed to start"));
      const modelUrl = new URL(opts.url ?? MODEL_URL, location.href).href;
      const cpuModelUrl = new URL(MODEL_URL_CPU, location.href).href;
      worker.postMessage({ type: "init", modelUrl, cpuModelUrl, inApp: "Capacitor" in window, forceWasm: liteMode() });
    });
  }

  private onMessage(e: MessageEvent) {
    const m = e.data;
    const p = this.pending.get(m.id);
    if (!p) return;
    this.pending.delete(m.id);
    if (m.type === "result") p.resolve({ boxes: m.boxes, ms: m.ms });
    else p.reject(new Error(m.message));
  }

  async detectPixels(px: ImageData, threshold = DEFAULT_THRESHOLD): Promise<{ boxes: Box[]; ms: number }> {
    const id = this.nextId++;
    const buf = px.data.buffer as ArrayBuffer;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage(
        { type: "detect", id, width: px.width, height: px.height, data: buf, threshold },
        [buf],
      );
    });
  }

  /** Detect on an <img>, <video> frame, canvas or ImageBitmap. With `band`,
   *  only that horizontal strip goes to the model (dashcam mode) and boxes are
   *  shifted back into full frame coordinates. */
  async detect(
    src: CanvasImageSource,
    threshold = DEFAULT_THRESHOLD,
    band?: [number, number],
    maxSide = PHOTO_MAX_SIDE,
  ): Promise<Detection> {
    const px = readPixels(src, band, maxSide);
    const r = await this.detectPixels(px.data, threshold);
    return {
      boxes: px.offsetY ? r.boxes.map((b) => ({ ...b, y1: b.y1 + px.offsetY, y2: b.y2 + px.offsetY })) : r.boxes,
      ms: r.ms,
      width: px.data.width,
      height: px.fullHeight,
      offsetY: px.offsetY,
    };
  }
}

let scratch: HTMLCanvasElement | null = null;

export function sourceSize(src: CanvasImageSource): [number, number] {
  if (src instanceof HTMLVideoElement) return [src.videoWidth, src.videoHeight];
  if (src instanceof HTMLImageElement) return [src.naturalWidth, src.naturalHeight];
  const s = src as { width: number; height: number };
  return [Number(s.width), Number(s.height)];
}

/** Read the pixels the model needs: the whole frame, or only the road band,
 *  scaled so the longest side of the frame is at most maxSide. */
export function readPixels(
  src: CanvasImageSource,
  band?: [number, number],
  maxSide = PHOTO_MAX_SIDE,
): { data: ImageData; offsetY: number; fullHeight: number } {
  const [w0, h0] = sourceSize(src);
  const f = Math.min(1, maxSide / Math.max(w0, h0));
  const sy = band ? Math.round(band[0] * h0) : 0;
  const sh = band ? Math.round(band[1] * h0) - sy : h0;
  const w = Math.max(1, Math.round(w0 * f));
  const h = Math.max(1, Math.round(sh * f));
  scratch ??= document.createElement("canvas");
  if (scratch.width !== w || scratch.height !== h) {
    scratch.width = w;
    scratch.height = h;
  }
  const ctx = scratch.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(src, 0, sy, w0, sh, 0, 0, w, h);
  return { data: ctx.getImageData(0, 0, w, h), offsetY: sy * f, fullHeight: Math.round(h0 * f) };
}

/** Boxes from readPixels space back to the source's native pixels. */
export function toSourceScale(d: Detection, src: CanvasImageSource): Box[] {
  const [w0] = sourceSize(src);
  const f = w0 / d.width;
  return d.boxes.map((b) => ({ ...b, x1: b.x1 * f, y1: b.y1 * f, x2: b.x2 * f, y2: b.y2 * f }));
}
