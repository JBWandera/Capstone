"""Detection evaluation.

Two rules the v1 capstone broke, and that this module enforces:
1. mAP is computed on ALL predictions down to a very low score (0.001). AP is
   the area under the precision / recall curve; dropping predictions below
   0.5 first cuts the curve short and understates AP.
2. The operating threshold for the app (and for precision, recall, F1) is
   chosen on the validation split, then applied unchanged to test.

Small test sets give noisy numbers, so bootstrap confidence intervals
(resampling images) are reported next to every headline metric.
"""

from __future__ import annotations

import contextlib
import copy
import io

import numpy as np
import pandas as pd

from ..data import coco as C

METRIC_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def to_prediction_records(image_id: int, xyxy: np.ndarray, scores: np.ndarray) -> list[dict]:
    out = []
    for (x0, y0, x1, y1), s in zip(np.asarray(xyxy, float), np.asarray(scores, float)):
        out.append({"image_id": int(image_id), "category_id": C.POTHOLE_ID,
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], "score": float(s)})
    return out


def _gt_for_pycocotools(gt: dict) -> dict:
    g = copy.deepcopy(gt)
    g["categories"] = [dict(C.POTHOLE_CATEGORY)]
    for a in g["annotations"]:
        a["category_id"] = C.POTHOLE_ID
        a.setdefault("iscrowd", 0)
        a.setdefault("area", a["bbox"][2] * a["bbox"][3])
        a["segmentation"] = a.get("segmentation") or []
    g.setdefault("info", {})
    g.setdefault("licenses", [])
    return g


def coco_metrics(gt: dict, preds: list[dict], image_ids: list[int] | None = None,
                 max_dets: int = 100) -> dict[str, float]:
    """Standard COCO box metrics. image_ids restricts evaluation (used by the bootstrap)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    g = _gt_for_pycocotools(gt)
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = g
        cg.createIndex()
        if not preds:
            return {k: 0.0 for k in METRIC_NAMES}
        cd = cg.loadRes(copy.deepcopy(preds))
        ev = COCOeval(cg, cd, "bbox")
        if image_ids is not None:
            ev.params.imgIds = list(image_ids)
        ev.params.maxDets = [1, 10, max_dets]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {k: float(v) for k, v in zip(METRIC_NAMES, ev.stats)}


def bootstrap(gt: dict, preds: list[dict], n: int = 100, seed: int = 0,
              metrics: tuple[str, ...] = ("AP", "AP50")) -> pd.DataFrame:
    """Percentile 95% intervals from resampling test images with replacement.

    pycocotools needs unique image ids, so each resample is rebuilt with fresh
    ids for repeated images.
    """
    rng = np.random.default_rng(seed)
    ids = [im["id"] for im in gt["images"]]
    anns_by, preds_by = {}, {}
    for a in gt["annotations"]:
        anns_by.setdefault(a["image_id"], []).append(a)
    for p in preds:
        preds_by.setdefault(p["image_id"], []).append(p)
    rows = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        g = {"images": [], "annotations": [], "categories": gt["categories"]}
        pr = []
        aid = 1
        for new_id, old in enumerate(pick, 1):
            g["images"].append({"id": new_id, "width": 1, "height": 1})
            for a in anns_by.get(old, []):
                g["annotations"].append(dict(a, id=aid, image_id=new_id))
                aid += 1
            pr.extend(dict(p, image_id=new_id) for p in preds_by.get(old, []))
        m = coco_metrics(g, pr)
        rows.append({k: m[k] for k in metrics})
    df = pd.DataFrame(rows)
    return pd.DataFrame({"low": df.quantile(0.025), "high": df.quantile(0.975), "std": df.std()})


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xywh boxes a (n,4) and b (m,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax0, ay0, ax1, ay1 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(ax0[:, None], bx0[None]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(ay0[:, None], by0[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / union, 0)


def match_predictions(gt: dict, preds: list[dict], iou_thr: float = 0.5) -> tuple[pd.DataFrame, int]:
    """Greedy matching by score, as in COCO. Returns (one row per prediction with
    score and tp flag, number of ground truth boxes)."""
    gts: dict[int, np.ndarray] = {}
    for a in gt["annotations"]:
        gts.setdefault(a["image_id"], []).append(a["bbox"])
    gts = {k: np.array(v, float) for k, v in gts.items()}
    rows = []
    by_img: dict[int, list[dict]] = {}
    for p in preds:
        by_img.setdefault(p["image_id"], []).append(p)
    for img_id, ps in by_img.items():
        ps = sorted(ps, key=lambda p: -p["score"])
        g = gts.get(img_id, np.zeros((0, 4)))
        ious = _iou_matrix(np.array([p["bbox"] for p in ps], float), g)
        taken = np.zeros(len(g), bool)
        for i, p in enumerate(ps):
            tp = False
            if len(g):
                cand = np.where(~taken, ious[i], -1)
                j = int(cand.argmax())
                if cand[j] >= iou_thr:
                    taken[j] = True
                    tp = True
            rows.append({"image_id": img_id, "score": p["score"], "tp": tp})
    return pd.DataFrame(rows, columns=["image_id", "score", "tp"]), len(gt["annotations"])


def threshold_table(gt: dict, preds: list[dict], iou_thr: float = 0.5,
                    thresholds: np.ndarray | None = None) -> pd.DataFrame:
    m, n_gt = match_predictions(gt, preds, iou_thr)
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        sel = m[m.score >= t]
        tp = int(sel.tp.sum())
        fp = int(len(sel) - tp)
        fn = int(n_gt - tp)
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / n_gt if n_gt else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    return pd.DataFrame(rows)


def pick_threshold(table: pd.DataFrame, min_recall: float | None = None) -> float:
    """Best F1 by default. With min_recall, the highest threshold that still
    reaches that recall (missing a pothole costs more than a false alarm)."""
    if min_recall is not None:
        ok = table[table.recall >= min_recall]
        if len(ok):
            return float(ok.threshold.max())
    return float(table.loc[table.f1.idxmax(), "threshold"])


def size_bins_at(gt: dict, resolution: int) -> dict[int, str]:
    """small / medium / large for each GT box, measured after the model's square resize."""
    sizes = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    out = {}
    for a in gt["annotations"]:
        w, h = sizes[a["image_id"]]
        side = np.sqrt(a["bbox"][2] * resolution / w * a["bbox"][3] * resolution / h)
        out[a["id"]] = "small" if side < 32 else "medium" if side < 96 else "large"
    return out
