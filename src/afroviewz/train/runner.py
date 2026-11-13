"""Run one experiment from configs/experiments.yaml, resumably, then predict on
valid and test so notebook 07 can evaluate every run the same way.

Colab drops free sessions without warning. Every run writes into
runs/<id>/ on Drive; calling run() again resumes from runs/<id>/last.ckpt.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm.auto import tqdm

from .. import paths
from ..data import coco as C
from ..eval import detection as E

RFDETR_KEYS = {"epochs", "batch_size", "grad_accum_steps", "lr", "lr_encoder", "lr_scheduler", "warmup_epochs",
               "early_stopping", "early_stopping_patience", "early_stopping_use_ema", "checkpoint_interval",
               "seed", "resolution"}


def load_experiments(path: str | Path | None = None) -> tuple[dict, dict]:
    path = Path(path) if path else paths.CONFIG_DIR / "experiments.yaml"
    cfg = yaml.safe_load(Path(path).read_text())
    return cfg["defaults"], cfg["experiments"]


def experiment_config(exp_id: str) -> dict:
    defaults, exps = load_experiments()
    cfg = dict(defaults)
    cfg.update(exps[exp_id])
    return cfg


def runs_root() -> Path:
    return paths.data_root().parent / "runs"


def _rfdetr_class(name: str):
    import rfdetr
    if not hasattr(rfdetr, name):
        options = sorted(n for n in dir(rfdetr) if n.startswith("RFDETR"))
        raise AttributeError(f"rfdetr has no {name}; available: {options}")
    return getattr(rfdetr, name)


# ------------------------------------------------------------------ training
def train_rfdetr(exp_id: str, dataset_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    cfg = experiment_config(exp_id)
    out_dir = Path(out_dir or runs_root() / exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    if (out_dir / "DONE").exists():
        print(f"{exp_id}: already trained")
        return out_dir
    kwargs = {k: v for k, v in cfg.items() if k in RFDETR_KEYS}
    resolution = kwargs.pop("resolution", None)
    model = _rfdetr_class(cfg["model"])(**({"resolution": resolution} if resolution else {}))
    last = out_dir / "last.ckpt"
    if last.exists():
        kwargs["resume"] = str(last)
        print(f"{exp_id}: resuming from {last}")
    t0 = time.time()
    model.train(dataset_dir=str(dataset_dir), output_dir=str(out_dir), **kwargs)
    (out_dir / "DONE").write_text(json.dumps({"train_seconds": round(time.time() - t0)}))
    return out_dir


def best_checkpoint(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
        if (out_dir / name).exists():
            return out_dir / name
    raise FileNotFoundError(f"no best checkpoint in {out_dir}")


def load_rfdetr(exp_id: str, out_dir: str | Path | None = None):
    cfg = experiment_config(exp_id)
    out_dir = Path(out_dir or runs_root() / exp_id)
    kw = {"pretrain_weights": str(best_checkpoint(out_dir))}
    if cfg.get("resolution"):
        kw["resolution"] = cfg["resolution"]
    model = _rfdetr_class(cfg["model"])(**kw)
    try:
        model.optimize_for_inference()
    except Exception as exc:  # noqa: BLE001  optional speed up; plain inference still works
        print("optimize_for_inference skipped:", exc)
    return model


# ------------------------------------------------------------------ YOLO baseline
def coco_split_to_yolo(split_dir: Path, out_dir: Path) -> None:
    """Write YOLO labels for one exported split (images are symlinked)."""
    coco = C.load(split_dir / "_annotations.coco.json")
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a["bbox"])
    for im in coco["images"]:
        src = split_dir / im["file_name"]
        dst = out_dir / "images" / im["file_name"]
        if not dst.exists():
            dst.symlink_to(src.resolve())
        w, h = im["width"], im["height"]
        lines = [f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                 for x, y, bw, bh in by_img.get(im["id"], [])]
        (out_dir / "labels" / Path(im["file_name"]).with_suffix(".txt").name).write_text("\n".join(lines))


def train_yolo(exp_id: str, dataset_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    from ultralytics import YOLO
    cfg = experiment_config(exp_id)
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir or runs_root() / exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        print(f"{exp_id}: already trained")
        return out_dir
    yolo_root = dataset_dir.parent / f"{dataset_dir.name}_yolo"
    for split in ("train", "valid", "test"):
        if not (yolo_root / split / "labels").exists():
            coco_split_to_yolo(dataset_dir / split, yolo_root / split)
    data_yaml = yolo_root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(yolo_root), "train": "train/images", "val": "valid/images",
                                         "test": "test/images", "names": {0: "pothole"}}))
    last = out_dir / "train" / "weights" / "last.pt"
    t0 = time.time()
    if last.exists():
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(cfg["model"]).train(data=str(data_yaml), imgsz=cfg["imgsz"], epochs=cfg["epochs"], batch=cfg["batch"],
                                 project=str(out_dir), name="train", exist_ok=True, seed=42, patience=8)
    (out_dir / "DONE").write_text(json.dumps({"train_seconds": round(time.time() - t0)}))
    return out_dir


# ------------------------------------------------------------------ predictions
def predict_split(predict_fn, split_dir: str | Path, batch: int = 8) -> tuple[list[dict], float]:
    """predict_fn(list of PIL images) -> list of (xyxy array, scores array).
    Returns COCO prediction records and mean milliseconds per image."""
    split_dir = Path(split_dir)
    coco = C.load(split_dir / "_annotations.coco.json")
    preds, elapsed, n = [], 0.0, 0
    ims = coco["images"]
    for i in tqdm(range(0, len(ims), batch), desc=split_dir.name):
        chunk = ims[i:i + batch]
        pil = [Image.open(split_dir / im["file_name"]).convert("RGB") for im in chunk]
        t0 = time.perf_counter()
        outs = predict_fn(pil)
        elapsed += time.perf_counter() - t0
        n += len(chunk)
        for im, (xyxy, scores) in zip(chunk, outs):
            preds.extend(E.to_prediction_records(im["id"], xyxy, scores))
    return preds, 1000 * elapsed / max(n, 1)


def rfdetr_predict_fn(model, threshold: float = 0.001):
    def fn(images):
        dets = model.predict(images, threshold=threshold)
        dets = dets if isinstance(dets, list) else [dets]
        return [(d.xyxy, d.confidence) for d in dets]
    return fn


def yolo_predict_fn(model, threshold: float = 0.001, imgsz: int = 640):
    def fn(images):
        res = model.predict(images, conf=threshold, imgsz=imgsz, verbose=False, max_det=100)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()) for r in res]
    return fn


def save_predictions(out_dir: str | Path, split: str, preds: list[dict], ms_per_image: float) -> None:
    out_dir = Path(out_dir)
    (out_dir / f"preds_{split}.json").write_text(json.dumps(preds))
    timing_path = out_dir / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    timing[split] = round(ms_per_image, 2)
    timing_path.write_text(json.dumps(timing, indent=2))
