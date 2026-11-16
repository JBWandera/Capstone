/// <reference lib="webworker" />
// Runs the AfroViewz detector off the main thread, so the camera preview,
// video playback and buttons stay smooth while the model works.
//
// Same contract as src/afroviewz/infer/onnx_rfdetr.py (checked there against
// rfdetr's own predict): plain bilinear resize to R x R (align_corners=false,
// no antialias), /255, ImageNet norm, NCHW; score = sigmoid of the best logit
// per query; boxes are cx, cy, w, h fractions. No NMS.

import type * as OrtNs from "onnxruntime-web";

type Box = { x1: number; y1: number; x2: number; y2: number; score: number };
type InitMsg = { type: "init"; modelUrl: string; cpuModelUrl: string; inApp: boolean; forceWasm: boolean };
type DetectMsg = { type: "detect"; id: number; width: number; height: number; data: ArrayBuffer; threshold: number };

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let ort: typeof OrtNs;
let session: OrtNs.InferenceSession;
let R = 384;
let input: Float32Array;

const post = (m: unknown, t?: Transferable[]) => (self as unknown as Worker).postMessage(m, t ?? []);

async function fetchModel(url: string, useCache: boolean): Promise<Uint8Array> {
  let res: Response | undefined;
  if (useCache && "caches" in self) {
    const cache = await caches.open("afroviewz-model-v3");
    res = await cache.match(url);
    if (!res) {
      const net = await fetch(url);
      if (!net.ok) throw new Error(`model download failed: ${net.status}`);
      await cache.put(url, net.clone());
      res = net;
    }
  } else {
    res = await fetch(url);
    if (!res.ok) throw new Error(`model download failed: ${res.status}`);
  }
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got);
    got += value.length;
    post({ type: "progress", value: got / total });
  }
  return out;
}

async function hasWebGPU(): Promise<boolean> {
  const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    return !!(await gpu.requestAdapter());
  } catch {
    return false;
  }
}

async function init(m: InitMsg) {
  ort = (await import("onnxruntime-web/webgpu")) as unknown as typeof OrtNs;
  ort.env.wasm.wasmPaths = new URL("/ort/", self.location.origin).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const order: ("webgpu" | "wasm")[] = !m.forceWasm && (await hasWebGPU()) ? ["webgpu", "wasm"] : ["wasm"];
  let lastErr: unknown;
  for (const backend of order) {
    try {
      // The graphics chip gets the full model; the processor gets the INT8 one.
      const bytes = await fetchModel(backend === "webgpu" ? m.modelUrl : m.cpuModelUrl, !m.inApp);
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: [backend],
        graphOptimizationLevel: "all",
      });
      const meta = session.inputMetadata?.[0] as { shape?: (number | string)[] } | undefined;
      R = Number(meta?.shape?.[3]) || 384;
      input = new Float32Array(3 * R * R);
      // One warm-up run so the first real frame is not slow.
      await session.run({ input: new ort.Tensor("float32", input, [1, 3, R, R]) });
      post({ type: "ready", backend, resolution: R });
      return;
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr;
}

function preprocess(W: number, H: number, data: Uint8ClampedArray) {
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
        input[c * plane + o] = ((top * (1 - wy) + bot * wy) / 255 - MEAN[c]) / STD[c];
      }
    }
  }
}

async function detect(m: DetectMsg) {
  const t0 = performance.now();
  const W = m.width;
  const H = m.height;
  preprocess(W, H, new Uint8ClampedArray(m.data));
  const res = await session.run({ input: new ort.Tensor("float32", input, [1, 3, R, R]) });
  const dets = res.dets.data as Float32Array;
  const logits = res.labels.data as Float32Array;
  const [, Q, K] = res.labels.dims as number[];
  const cand: { q: number; s: number }[] = [];
  for (let q = 0; q < Q; q++) {
    let best = -Infinity;
    for (let k = 0; k < K; k++) best = Math.max(best, logits[q * K + k]);
    const s = 1 / (1 + Math.exp(-best));
    if (s >= m.threshold) cand.push({ q, s });
  }
  cand.sort((a, b) => b.s - a.s);
  const clamp = (v: number, hi: number) => Math.min(Math.max(v, 0), hi);
  const boxes: Box[] = cand.slice(0, 100).map(({ q, s }) => {
    const cx = dets[q * 4];
    const cy = dets[q * 4 + 1];
    const w = dets[q * 4 + 2];
    const h = dets[q * 4 + 3];
    return {
      x1: clamp((cx - w / 2) * W, W),
      y1: clamp((cy - h / 2) * H, H),
      x2: clamp((cx + w / 2) * W, W),
      y2: clamp((cy + h / 2) * H, H),
      score: s,
    };
  });
  for (const t of Object.values(res)) (t as { dispose?: () => void }).dispose?.();
  // Hand the pixel buffer back so the main thread can reuse it.
  post({ type: "result", id: m.id, boxes, ms: performance.now() - t0, data: m.data }, [m.data]);
}

self.onmessage = async (e: MessageEvent<InitMsg | DetectMsg>) => {
  const m = e.data;
  try {
    if (m.type === "init") await init(m);
    else if (m.type === "detect") await detect(m);
  } catch (err) {
    post({ type: "error", id: (m as DetectMsg).id, message: String((err as Error)?.message ?? err) });
  }
};
