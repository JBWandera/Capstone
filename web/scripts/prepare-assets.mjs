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

const MODEL = "afroviewz-nano-roi-fp16.onnx";
const MODEL_URL = `https://huggingface.co/JBWan94/afroviewz-pothole-detector/resolve/main/${MODEL}`;
const modelDst = join(root, "public", "models", MODEL);
if (!existsSync(modelDst)) {
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`[prepare-assets] model download failed: ${res.status}`);
  mkdirSync(dirname(modelDst), { recursive: true });
  writeFileSync(modelDst, Buffer.from(await res.arrayBuffer()));
  console.log(`[prepare-assets] downloaded ${MODEL}`);
}
