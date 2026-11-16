// Runs before dev and build.
// 1. Copies the onnxruntime-web WebAssembly runtime into public/ort.
// 2. Downloads the detector model into public/models if it is missing, from
//    the public Hugging Face repo JBWan94/afroviewz-pothole-detector.
import { copyFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

const ortSrc = join(root, "node_modules", "onnxruntime-web", "dist");
const ortDst = join(root, "public", "ort");
mkdirSync(ortDst, { recursive: true });
for (const f of readdirSync(ortSrc)) {
  if (/^ort-wasm-simd-threaded(\.\w+)?\.(mjs|wasm)$/.test(f)) copyFileSync(join(ortSrc, f), join(ortDst, f));
}

// MapLibre loads its worker from a separate file the bundler does not place.
const mlDst = join(root, "public", "maplibre");
mkdirSync(mlDst, { recursive: true });
copyFileSync(join(root, "node_modules", "maplibre-gl", "dist", "maplibre-gl-worker.mjs"), join(mlDst, "maplibre-gl-worker.mjs"));

// Full model for WebGPU, INT8 model for devices that run on the processor.
for (const MODEL of ["afroviewz-nano-roi-fp16.onnx", "afroviewz-nano-roi-int8.onnx"]) {
  const url = `https://huggingface.co/JBWan94/afroviewz-pothole-detector/resolve/main/${MODEL}`;
  const dst = join(root, "public", "models", MODEL);
  if (existsSync(dst)) continue;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`[prepare-assets] ${MODEL} download failed: ${res.status}`);
  mkdirSync(dirname(dst), { recursive: true });
  writeFileSync(dst, Buffer.from(await res.arrayBuffer()));
  console.log(`[prepare-assets] downloaded ${MODEL}`);
}
