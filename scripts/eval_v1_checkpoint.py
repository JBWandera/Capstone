import sys, time, json
sys.path.insert(0, "/home/claude/capstone/src")
from pathlib import Path
from afroviewz.train import runner
from rfdetr import RFDETRBase
out = Path("/home/claude/afz/runs_local/E0_v1_capstone")
model = RFDETRBase(pretrain_weights="/home/claude/afz/v1/weights/checkpoint_best_total.pth", resolution=560)
fn = runner.rfdetr_predict_fn(model)
D = Path("/home/claude/afz/data/processed/afroviewz-v1/rfdetr")
for split in sys.argv[1:]:
    t = time.time()
    preds, ms = runner.predict_split(fn, D / split, batch=4)
    runner.save_predictions(out, split, preds, ms)
    print(split, len(preds), "preds", round(time.time() - t), "s", flush=True)
(out / "config.json").write_text(json.dumps({"exp": "E0_v1_capstone", "model": "RFDETRBase", "resolution": 560,
    "note": "original capstone model (v1), trained on the capstone Roboflow set only"}))
