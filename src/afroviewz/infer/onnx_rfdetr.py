"""Run an exported RF-DETR ONNX model with onnxruntime.

The contract (checked against rfdetr's own predict on the same image):
  input  "input"  float32 [1, 3, R, R]  RGB resized to R x R (no letterbox),
                  scaled to 0..1 then normalised with ImageNet mean and std
  output "dets"   float32 [1, Q, 4]     cx, cy, w, h as fractions of the image
  output "labels" float32 [1, Q, K]     class logits; sigmoid gives scores
Detection: take the top Q (query, class) pairs by score, keep the pothole
class, map boxes back to the original image size. No NMS is needed (DETR).
The web app (web/src/lib/detector.ts) implements the same steps in TypeScript.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Detections:
    xyxy: np.ndarray       # (n, 4) pixels in the original image
    confidence: np.ndarray  # (n,)

    def __len__(self) -> int:
        return len(self.confidence)


class OnnxDetector:
    def __init__(self, path: str, class_ids: tuple[int, ...] = (1,), providers: list[str] | None = None):
        import onnxruntime as ort
        self.session = ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.resolution = int(inp.shape[-1])
        self.class_ids = class_ids

    def preprocess(self, image: Image.Image) -> np.ndarray:
        x = np.asarray(image.convert("RGB").resize((self.resolution, self.resolution), Image.BILINEAR), np.float32)
        x = (x / 255.0 - MEAN) / STD
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def __call__(self, image: Image.Image, threshold: float = 0.5, top_k: int = 100) -> Detections:
        w, h = image.size
        dets, logits = self.session.run(None, {self.input_name: self.preprocess(image)})
        prob = 1.0 / (1.0 + np.exp(-logits[0][:, list(self.class_ids)]))  # (Q, len(class_ids))
        score = prob.max(axis=1)
        order = np.argsort(-score)[:top_k]
        order = order[score[order] >= threshold]
        cx, cy, bw, bh = dets[0, order].T
        xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1)
        xyxy = np.clip(xyxy, 0, [w, h, w, h])
        return Detections(xyxy=xyxy.astype(np.float32), confidence=score[order].astype(np.float32))
