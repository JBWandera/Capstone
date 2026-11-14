// AfroViewz in-browser pothole detector.
//
// Same contract as src/afroviewz/infer/onnx_rfdetr.py, checked there against
// rfdetr's own predict (identical counts and scores on 40 test images):
//   input  "input"  float32 [1,3,R,R]  RGB, plain bilinear resize to R x R
//                   (no antialiasing, align_corners=false), /255, ImageNet norm
//   output "dets"   [1,Q,4]  cx, cy, w, h as fractions of the image
//   output "labels" [1,Q,K]  class logits, sigmoid gives scores
// Single class model: the score of a query is its best class score.
// DETR needs no NMS.

import type * as OrtNs from "onnxruntime-web";

export type Box = { x1: number; y1: number; x2: number; y2: number; score: number };
export type Detection = { boxes: Box[]; ms: number; width: number; height: number };
export type Backend = "webgpu" | "wasm";

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export const MODEL_URL =
  process.env.NEXT_PUBLIC_MODEL_URL ?? "/models/afroviewz-nano-fp16.onnx";
export const DEFAULT_THRESHOLD = 0.35;
const MAX_SIDE = 1920; // cap on the frame size we read back from the canvas

let ort: typeof OrtNs | null = null;

async function loadOrt(): Promise<typeof OrtNs> {
  if (ort) return ort;
  const mod = await import("onnxruntime-web/webgpu");
  mod.env.wasm.wasmPaths = "/ort/";
  mod.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort = mod as unknown as typeof OrtNs;
  return ort;
}

/** Fetch the model once and keep it in the Cache API so the app works offline. */
async function fetchModel(url: string, onProgress?: (f: number) => void): Promise<Uint8Array> {
  const cacheName = "afroviewz-model-v1";
  let res: Response | undefined;
  if ("caches" in self) {
    const cache = await caches.open(cacheName);
    res = await cache.match(url);
    if (!res) {
      const net = await fetch(url);
      if (!net.ok) throw new Error(`model download failed: ${net.status}`);
      await cache.put(url, net.clone());
      res = net;
    }
  } else {
    res = await fetch(url);
  }
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total || !onProgress) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got);
    got += value.length;
    onProgress(got / total);
  }
  return out;
}

export class PotholeDetector {
  private constructor(
    private session: OrtNs.InferenceSession,
    private ortMod: typeof OrtNs,
    readonly resolution: number,
    readonly backend: Backend,
  ) {}

  static async create(opts: { url?: string; onProgress?: (f: number) => void } = {}) {
    const o = await loadOrt();
    const bytes = await fetchModel(opts.url ?? MODEL_URL, opts.onProgress);
    const tryBackends: Backend[] = "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"];
    let lastErr: unknown;
    for (const backend of tryBackends) {
      try {
        const session = await o.InferenceSession.create(bytes, {
          executionProviders: [backend],
          graphOptimizationLevel: "all",
        });
        const meta = session.inputMetadata?.[0] as { shape?: (number | string)[] } | undefined;
        const r = Number(meta?.shape?.[3]) || 384;
        return new PotholeDetector(session, o, r, backend);
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr;
  }

  /** Plain bilinear resize (align_corners=false, no antialias) + normalise to NCHW. */
  private preprocess(px: ImageData): Float32Array {
    const R = this.resolution;
    const { width: W, height: H, data } = px;
    const out = new Float32Array(3 * R * R);
    const sx = W / R;
    const sy = H / R;
    const plane = R * R;
    for (let y = 0; y < R; y++) {
      let fy = (y + 0.5) * sy - 0.5;
      if (fy < 0) fy = 0;
      let y0 = Math.floor(fy);
      if (y0 > H - 1) y0 = H - 1;
      const y1 = Math.min(y0 + 1, H - 1);
      const wy = fy - y0;
      for (let x = 0; x < R; x++) {
        let fx = (x + 0.5) * sx - 0.5;
        if (fx < 0) fx = 0;
        let x0 = Math.floor(fx);
        if (x0 > W - 1) x0 = W - 1;
        const x1 = Math.min(x0 + 1, W - 1);
        const wx = fx - x0;
        const i00 = (y0 * W + x0) * 4;
        const i01 = (y0 * W + x1) * 4;
        const i10 = (y1 * W + x0) * 4;
        const i11 = (y1 * W + x1) * 4;
        const o = y * R + x;
        for (let c = 0; c < 3; c++) {
          const top = data[i00 + c] * (1 - wx) + data[i01 + c] * wx;
          const bot = data[i10 + c] * (1 - wx) + data[i11 + c] * wx;
          const v = top * (1 - wy) + bot * wy;
          out[c * plane + o] = (v / 255 - MEAN[c]) / STD[c];
        }
      }
    }
    return out;
  }

  async detectPixels(px: ImageData, threshold = DEFAULT_THRESHOLD, topK = 100): Promise<Detection> {
    const t0 = performance.now();
    const R = this.resolution;
    const input = new this.ortMod.Tensor("float32", this.preprocess(px), [1, 3, R, R]);
    const res = await this.session.run({ input });
    const dets = res.dets.data as Float32Array;
    const logits = res.labels.data as Float32Array;
    const [, Q, K] = res.labels.dims as number[];
    const cand: { q: number; s: number }[] = [];
    for (let q = 0; q < Q; q++) {
      let best = -Infinity;
      for (let k = 0; k < K; k++) best = Math.max(best, logits[q * K + k]);
      const s = 1 / (1 + Math.exp(-best));
      if (s >= threshold) cand.push({ q, s });
    }
    cand.sort((a, b) => b.s - a.s);
    const W = px.width;
    const H = px.height;
    const boxes = cand.slice(0, topK).map(({ q, s }) => {
      const cx = dets[q * 4];
      const cy = dets[q * 4 + 1];
      const w = dets[q * 4 + 2];
      const h = dets[q * 4 + 3];
      const clampX = (v: number) => Math.min(Math.max(v, 0), W);
      const clampY = (v: number) => Math.min(Math.max(v, 0), H);
      return {
        x1: clampX((cx - w / 2) * W),
        y1: clampY((cy - h / 2) * H),
        x2: clampX((cx + w / 2) * W),
        y2: clampY((cy + h / 2) * H),
        score: s,
      };
    });
    return { boxes, ms: performance.now() - t0, width: W, height: H };
  }

  /** Detect on an <img>, <video> frame, canvas or ImageBitmap. */
  async detect(src: CanvasImageSource, threshold = DEFAULT_THRESHOLD): Promise<Detection> {
    const px = readPixels(src);
    return this.detectPixels(px, threshold);
  }
}

let scratch: HTMLCanvasElement | null = null;

export function sourceSize(src: CanvasImageSource): [number, number] {
  if (src instanceof HTMLVideoElement) return [src.videoWidth, src.videoHeight];
  if (src instanceof HTMLImageElement) return [src.naturalWidth, src.naturalHeight];
  const s = src as { width: number; height: number };
  return [Number(s.width), Number(s.height)];
}

/** Read pixels at native size (capped), so the model sees the same bilinear
 *  sampling as in training rather than the browser's own downscale filter. */
export function readPixels(src: CanvasImageSource): ImageData {
  const [w0, h0] = sourceSize(src);
  const f = Math.min(1, MAX_SIDE / Math.max(w0, h0));
  const w = Math.round(w0 * f);
  const h = Math.round(h0 * f);
  scratch ??= document.createElement("canvas");
  scratch.width = w;
  scratch.height = h;
  const ctx = scratch.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(src, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

/** Boxes from readPixels space back to the source's native pixels. */
export function toSourceScale(d: Detection, src: CanvasImageSource): Box[] {
  const [w0] = sourceSize(src);
  const f = w0 / d.width;
  return d.boxes.map((b) => ({ ...b, x1: b.x1 * f, y1: b.y1 * f, x2: b.x2 * f, y2: b.y2 * f }));
}
