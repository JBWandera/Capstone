"""Start the Modal jobs listed in ops/queue.yaml (run by .github/workflows/train.yml).

queue.yaml:
    launch:
      - {kind: train, exp: E2_rfdetr_nano}
      - {kind: smoke}
      - {kind: export, exp: E2_rfdetr_small}
Each entry is started with .spawn() on the deployed app "afroviewz-train" and
runs on its own; results are published to the Hugging Face repo
JBWan94/afroviewz-runs by the job itself.
"""

import sys

import modal
import yaml

APP = "afroviewz-train"


def main(path: str = "ops/queue.yaml") -> None:
    queue = yaml.safe_load(open(path)) or {}
    for item in queue.get("launch", []):
        kind = item.get("kind", "train")
        if kind == "train":
            call = modal.Function.from_name(APP, "train").spawn(item["exp"])
        elif kind == "smoke":
            call = modal.Function.from_name(APP, "smoke").spawn()
        elif kind == "export":
            call = modal.Function.from_name(APP, "export_onnx").spawn(item["exp"])
        elif kind == "evaluate_checkpoint":
            call = modal.Function.from_name(APP, "evaluate_checkpoint").spawn(
                item["name"], item["checkpoint_path"], item.get("model_class", "RFDETRBase"))
        else:
            print("skipping unknown kind", item)
            continue
        print(f"started {item} as {call.object_id}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
