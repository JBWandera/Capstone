// Runs before dev and build.
// 1. Copies the onnxruntime-web WebAssembly runtime into public/ort.
// 2. Downloads the detector model into public/models if it is missing.
//    The model lives in the private Hugging Face repo JBWan94/afroviewz-runs,
//    so HF_TOKEN must be set (locally or as a Vercel environment variable).
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

const MODEL = "afroviewz-nano-fp16.onnx";
const modelDst = join(root, "public", "models", MODEL);
if (!existsSync(modelDst)) {
  const token = process.env.HF_TOKEN;
  if (!token) {
    console.warn(`[prepare-assets] ${MODEL} missing and HF_TOKEN not set; the app will not find its model.`);
  } else {
    const url = `https://huggingface.co/JBWan94/afroviewz-runs/resolve/main/app/${MODEL}`;
    const res = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
    if (!res.ok) throw new Error(`[prepare-assets] model download failed: ${res.status}`);
    mkdirSync(dirname(modelDst), { recursive: true });
    writeFileSync(modelDst, Buffer.from(await res.arrayBuffer()));
    console.log(`[prepare-assets] downloaded ${MODEL}`);
  }
}
