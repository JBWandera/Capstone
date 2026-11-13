"""Uganda holdout labelling through Roboflow Annotate.

Flow:
1. choose candidates: Mapillary frames, optionally ranked by a trained model so
   frames that likely show potholes are labelled first (pothole free frames
   are kept too, as negatives);
2. upload them to a Roboflow project, with the model's boxes as pre-labels;
3. a person checks and corrects every box in Roboflow Annotate (the labels are
   ground truth only after this human review);
4. export COCO and add it as the uganda_holdout test set (never trained on).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import coco as C


def rank_candidates(preds: dict[int, list[float]], coco: dict, n_positive: int = 350, n_negative: int = 150,
                    seed: int = 0) -> list[int]:
    """preds: image_id -> scores of predicted boxes. Takes the n_positive images
    with the highest max score, plus n_negative random images from the rest so
    the holdout also measures false alarms on ordinary Ugandan streets."""
    import random
    ranked = sorted(coco["images"], key=lambda im: -max(preds.get(im["id"], [0.0]) or [0.0]))
    pos = [im["id"] for im in ranked[:n_positive]]
    rest = [im["id"] for im in ranked[n_positive:]]
    rng = random.Random(seed)
    return pos + rng.sample(rest, min(n_negative, len(rest)))


def upload(api_key: str, project_slug: str, coco: dict, images_root: str | Path, image_ids: list[int],
           prelabels: dict[int, list[list[float]]] | None = None, batch_name: str = "uganda-holdout",
           create_if_missing: bool = True) -> str:
    """Upload images (and optional pre-label boxes in xywh) to a Roboflow project."""
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace()
    try:
        project = ws.project(project_slug)
    except Exception:  # noqa: BLE001  project does not exist yet
        if not create_if_missing:
            raise
        project = ws.create_project(project_name=project_slug, project_type="object-detection",
                                    project_license="CC BY-SA 4.0", annotation="pothole")
    images_root = Path(images_root)
    by_id = {im["id"]: im for im in coco["images"]}
    with tempfile.TemporaryDirectory() as tmp:
        for img_id in image_ids:
            im = by_id[img_id]
            path = images_root / im["file_name"]
            ann_path = None
            if prelabels and prelabels.get(img_id):
                one = C.new_coco()
                one["images"] = [{"id": 1, "file_name": path.name, "width": im["width"], "height": im["height"]}]
                one["annotations"] = [C.make_annotation(i, 1, b) for i, b in enumerate(prelabels[img_id], 1)]
                ann_path = Path(tmp) / f"{path.stem}.json"
                ann_path.write_text(json.dumps(one))
            project.single_upload(image_path=str(path), annotation_path=str(ann_path) if ann_path else None,
                                  batch_name=batch_name, split="test", num_retry_uploads=3)
    return project.id
