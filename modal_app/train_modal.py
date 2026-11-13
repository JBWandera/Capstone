"""Train AfroViewz models on Modal GPUs, driven from the command line.

    modal run modal_app/train_modal.py --exp E2_rfdetr_nano
    modal run modal_app/train_modal.py --exp E2_rfdetr_nano --detach   (keeps running after you disconnect)

What happens on the GPU machine:
1. the dataset zip is pulled from the private Hugging Face dataset repo
   (secret "huggingface" holds HF_TOKEN) to the machine's local disk;
2. the experiment from configs/experiments.yaml is trained; checkpoints go to
   the Modal volume "afroviewz" under runs/<exp>/, so a restarted job resumes;
3. predictions for the valid and test splits (and the Uganda holdout when it
   exists) are saved next to the checkpoints for notebook 07.

Same code path as the Colab notebooks (afroviewz.train.runner); only the
place it runs changes.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_DATASET = "JBWan94/afroviewz-data"
DATASET = "afroviewz-v1"
VOL_ROOT = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "rfdetr[train,loggers,onnx]>=1.8",
        "supervision>=0.26",
        "pycocotools>=2.0.7",
        "ultralytics>=8.4",
        "huggingface_hub>=0.30",
        "pandas", "pyyaml", "imagehash", "tqdm",
    )
    .env({"AFROVIEWZ_DATA": f"{VOL_ROOT}/data", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(REPO_ROOT / "configs", "/root/configs")
    .add_local_python_source("afroviewz", copy=False)
)

app = modal.App("afroviewz-train", image=image)
volume = modal.Volume.from_name("afroviewz", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def _fetch_dataset() -> Path:
    """Download and unpack the dataset zip to local disk (fast I/O for training)."""
    import zipfile
    from huggingface_hub import hf_hub_download
    local = Path("/tmp/datasets")
    target = local / DATASET / "rfdetr"
    if not (target / "train" / "_annotations.coco.json").exists():
        z = hf_hub_download(HF_DATASET, f"processed/{DATASET}.zip", repo_type="dataset",
                            token=os.environ["HF_TOKEN"], local_dir="/tmp/hf")
        with zipfile.ZipFile(z) as f:
            f.extractall(local)
    return target


def _point_configs_at_image() -> None:
    from afroviewz import paths
    paths.CONFIG_DIR = Path("/root/configs")


@app.function(gpu="L4", timeout=6 * 3600, volumes={VOL_ROOT: volume}, secrets=[hf_secret],
              retries=modal.Retries(max_retries=2, initial_delay=30.0))
def train(exp_id: str) -> dict:
    import json
    _point_configs_at_image()
    from afroviewz.train import runner

    dataset_dir = _fetch_dataset()
    cfg = runner.experiment_config(exp_id)
    if cfg["family"] == "yolo":
        run_dir = runner.train_yolo(exp_id, dataset_dir)
        from ultralytics import YOLO
        model = YOLO(str(run_dir / "train" / "weights" / "best.pt"))
        predict = runner.yolo_predict_fn(model, imgsz=cfg["imgsz"])
    else:
        run_dir = runner.train_rfdetr(exp_id, dataset_dir)
        volume.commit()
        model = runner.load_rfdetr(exp_id, run_dir)
        predict = runner.rfdetr_predict_fn(model)
    for split in ("valid", "test"):
        preds, ms = runner.predict_split(predict, dataset_dir / split)
        runner.save_predictions(run_dir, split, preds, ms)
    volume.commit()
    return {"run": exp_id, "files": sorted(p.name for p in run_dir.iterdir()),
            "timing": json.loads((run_dir / "timing.json").read_text())}


@app.function(gpu="L4", timeout=3600, volumes={VOL_ROOT: volume}, secrets=[hf_secret])
def evaluate_checkpoint(name: str, checkpoint_path: str, model_class: str = "RFDETRBase") -> dict:
    """Predict valid/test with an existing checkpoint stored on the volume (E0: the v1 capstone model)."""
    import json
    import rfdetr
    _point_configs_at_image()
    from afroviewz.train import runner
    dataset_dir = _fetch_dataset()
    model = getattr(rfdetr, model_class)(pretrain_weights=checkpoint_path)
    model.optimize_for_inference()
    run_dir = Path(VOL_ROOT) / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    for split in ("valid", "test"):
        preds, ms = runner.predict_split(runner.rfdetr_predict_fn(model), dataset_dir / split)
        runner.save_predictions(run_dir, split, preds, ms)
    volume.commit()
    return json.loads((run_dir / "timing.json").read_text())


@app.function(gpu="L4", timeout=1800, volumes={VOL_ROOT: volume}, secrets=[hf_secret])
def export_onnx(exp_id: str) -> list[str]:
    """Export the best checkpoint of a run to ONNX (for the API and the browser)."""
    _point_configs_at_image()
    from afroviewz.train import runner
    run_dir = runner.runs_root() / exp_id
    model = runner.load_rfdetr(exp_id, run_dir)
    out = run_dir / "export"
    model.export(output_dir=str(out))
    volume.commit()
    return sorted(str(p) for p in out.rglob("*"))


@app.local_entrypoint()
def main(exp: str = "E2_rfdetr_nano", export: bool = False):
    if export:
        print(export_onnx.remote(exp))
    else:
        print(train.remote(exp))
