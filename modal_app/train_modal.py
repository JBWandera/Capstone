"""Train AfroViewz models on Modal GPUs, driven from the command line.

    modal run --detach modal_app/train_modal.py --exp E2_rfdetr_nano

--detach keeps the job running after the terminal closes; follow it with
`modal app logs <app id>` (the id is printed at launch, or see `modal app list`).

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
HF_RUNS = "JBWan94/afroviewz-runs"  # private model repo: one folder per run
DATASET = "afroviewz-v1"
VOL_ROOT = "/vol"
# Preferred GPU types in order: Modal takes the first with free capacity.
# L4 and A10 cost about USD 0.80 and 1.10 per hour, T4 about 0.59 but slower.
GPU = os.environ.get("AFROVIEWZ_GPU", "L4,A10,T4").split(",")

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
    # the whole package folder, subpackages included (add_local_python_source
    # left out afroviewz.data on the first try)
    .add_local_dir(REPO_ROOT / "src" / "afroviewz", "/root/afroviewz",
                   ignore=["**/__pycache__/**", "**/*.pyc"])
)

app = modal.App("afroviewz-train", image=image)
volume = modal.Volume.from_name("afroviewz", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def _fetch_dataset(name: str = DATASET) -> Path:
    """Download and unpack a dataset zip to local disk (fast I/O for training)."""
    import zipfile
    from huggingface_hub import hf_hub_download
    local = Path("/tmp/datasets")
    target = local / name / "rfdetr"
    if not (target / "train" / "_annotations.coco.json").exists():
        z = hf_hub_download(HF_DATASET, f"processed/{name}.zip", repo_type="dataset",
                            token=os.environ["HF_TOKEN"], local_dir="/tmp/hf")
        with zipfile.ZipFile(z) as f:
            f.extractall(local)
    return target


PUBLISH_FILES = ("config.json", "metrics.csv", "timing.json", "DONE", "curves.png", "results.csv",
                 "preds_valid.json.gz", "preds_test.json.gz", "preds_holdout.json.gz", "status.json")


def _publish(run_dir: Path, exp_id: str, with_weights: bool = False) -> None:
    """Copy a run's small files (and optionally its best weights) to the private
    Hugging Face repo, so results can be read without Modal access."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(HF_RUNS, repo_type="model", private=True, exist_ok=True)
    files = [p for p in run_dir.rglob("*") if p.is_file() and p.name in PUBLISH_FILES]
    if with_weights:
        files += [p for p in run_dir.glob("checkpoint_best_*.pth")] + list((run_dir / "train" / "weights").glob("best.pt"))
        files += [p for p in (run_dir / "export").rglob("*") if p.is_file()]
    if not files:
        return
    from huggingface_hub import CommitOperationAdd
    ops = [CommitOperationAdd(path_in_repo=f"runs/{exp_id}/{p.relative_to(run_dir)}", path_or_fileobj=str(p))
           for p in files]
    api.create_commit(HF_RUNS, operations=ops, commit_message=f"{exp_id}: {len(ops)} files")


def _progress_publisher(run_dir: Path, exp_id: str, every_s: int = 600):
    """Background thread: publish metrics every 10 minutes while training runs."""
    import json
    import threading
    import time
    stop = threading.Event()

    def loop():
        while not stop.wait(every_s):
            try:
                (run_dir / "status.json").write_text(json.dumps({"exp": exp_id, "state": "training",
                                                                 "updated": time.strftime("%Y-%m-%d %H:%M:%S")}))
                _publish(run_dir, exp_id)
            except Exception as exc:  # noqa: BLE001  progress reports must never stop training
                print("progress publish failed:", exc, flush=True)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return stop


def _point_configs_at_image() -> None:
    from afroviewz import paths
    paths.CONFIG_DIR = Path("/root/configs")


@app.function(gpu=GPU, timeout=10 * 3600, volumes={VOL_ROOT: volume}, secrets=[hf_secret],
              retries=modal.Retries(max_retries=2, initial_delay=30.0))
def train(exp_id: str) -> dict:
    import json
    _point_configs_at_image()
    from afroviewz.train import runner

    cfg = runner.experiment_config(exp_id)
    eval_dir = _fetch_dataset()                              # always scored on the standard split
    dataset_dir = _fetch_dataset(cfg.get("dataset", DATASET))  # what the model trains on
    run_dir = runner.runs_root() / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stop = _progress_publisher(run_dir, exp_id)
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
        if cfg.get("roi_inference"):
            from afroviewz.data.roi import roi_predict_fn
            predict = roi_predict_fn(predict)
    stop.set()
    for split in ("valid", "test"):
        preds, ms = runner.predict_split(predict, eval_dir / split)
        runner.save_predictions(run_dir, split, preds, ms)
    (run_dir / "status.json").write_text(json.dumps({"exp": exp_id, "state": "done"}))
    volume.commit()
    _publish(run_dir, exp_id, with_weights=True)
    out = {"run": exp_id, "files": sorted(p.name for p in run_dir.iterdir()),
           "timing": json.loads((run_dir / "timing.json").read_text())}
    print("RESULT", out, flush=True)
    return out


@app.function(gpu=GPU, timeout=3600, volumes={VOL_ROOT: volume}, secrets=[hf_secret])
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
    (run_dir / "status.json").write_text(json.dumps({"exp": name, "state": "done"}))
    volume.commit()
    _publish(run_dir, name)
    return json.loads((run_dir / "timing.json").read_text())


@app.function(gpu=GPU, timeout=1800, volumes={VOL_ROOT: volume}, secrets=[hf_secret])
def export_onnx(exp_id: str) -> list[str]:
    """Export the best checkpoint of a run to ONNX (for the API and the browser)."""
    _point_configs_at_image()
    from afroviewz.train import runner
    run_dir = runner.runs_root() / exp_id
    model = runner.load_rfdetr(exp_id, run_dir)
    out = run_dir / "export"
    model.export(output_dir=str(out))
    volume.commit()
    _publish(run_dir, exp_id, with_weights=True)
    return sorted(str(p) for p in out.rglob("*"))


@app.function(gpu=GPU, timeout=1800, secrets=[hf_secret])
def smoke() -> dict:
    """Five minute end to end check on a 64 image subset: data, training, prediction, ONNX export."""
    import shutil
    import subprocess
    import time
    _point_configs_at_image()
    from rfdetr import RFDETRNano
    from afroviewz.data import coco as C
    from afroviewz.eval import detection as E
    from afroviewz.train import runner
    t0 = time.time()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    full = _fetch_dataset()
    t_data = time.time() - t0
    small = Path("/tmp/smoke")
    for split, n in (("train", 64), ("valid", 24), ("test", 24)):
        c = C.load(full / split / "_annotations.coco.json")
        sub = C.subset(c, [im["id"] for im in c["images"][:n]])
        (small / split).mkdir(parents=True, exist_ok=True)
        for im in sub["images"]:
            shutil.copy(full / split / im["file_name"], small / split / im["file_name"])
        C.save(sub, small / split / "_annotations.coco.json")
    model = RFDETRNano()
    model.train(dataset_dir=str(small), output_dir="/tmp/smoke_out", epochs=1, batch_size=8,
                grad_accum_steps=1, tensorboard=False)
    t_train = time.time() - t0 - t_data
    model = RFDETRNano(pretrain_weights=str(runner.best_checkpoint("/tmp/smoke_out")))
    preds, ms = runner.predict_split(runner.rfdetr_predict_fn(model), small / "test")
    m = E.coco_metrics(C.load(small / "test" / "_annotations.coco.json"), preds)
    model.export(output_dir="/tmp/smoke_export")
    out = {"gpu": gpu, "data_s": round(t_data), "train_1_epoch_64_imgs_s": round(t_train),
           "test_preds": len(preds), "ms_per_img": round(ms, 1), "mAP50_after_1_epoch": round(m["AP50"], 3),
           "onnx": [p.name for p in Path("/tmp/smoke_export").rglob("*.onnx")]}
    print("RESULT", out, flush=True)
    pub = Path("/tmp/smoke_pub")
    pub.mkdir(exist_ok=True)
    import json
    (pub / "status.json").write_text(json.dumps({"exp": "smoke", "state": "done", **out}))
    _publish(pub, "smoke")
    return out


@app.local_entrypoint()
def main(exp: str = "E2_rfdetr_nano", export: bool = False):
    if export:
        print(export_onnx.remote(exp))
    else:
        print(train.remote(exp))
