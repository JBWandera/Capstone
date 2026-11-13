"""Evaluation sanity checks on hand made predictions."""

import numpy as np

from afroviewz.data import coco as C
from afroviewz.eval import detection as E


def _gt():
    g = C.new_coco()
    g["images"] = [{"id": i, "width": 640, "height": 480, "file_name": f"{i}.jpg"} for i in range(1, 5)]
    g["annotations"] = [C.make_annotation(i, i, [100, 100, 80, 60]) for i in range(1, 5)]
    return g


def test_perfect_predictions_score_one():
    g = _gt()
    preds = [p for i in range(1, 5) for p in E.to_prediction_records(i, np.array([[100, 100, 180, 160]]), np.array([0.9]))]
    m = E.coco_metrics(g, preds)
    assert m["AP50"] > 0.99 and m["AP"] > 0.99


def test_low_score_predictions_count_for_map_but_not_at_threshold():
    g = _gt()
    preds = []
    for i in range(1, 5):
        score = 0.9 if i <= 2 else 0.2  # two correct boxes have low confidence
        preds += E.to_prediction_records(i, np.array([[100, 100, 180, 160]]), np.array([score]))
    full = E.coco_metrics(g, preds)["AP50"]
    cut = E.coco_metrics(g, [p for p in preds if p["score"] >= 0.5])["AP50"]
    assert full > 0.99 and cut < 0.6  # the v1 mistake: cutting at 0.5 halves AP here
    table = E.threshold_table(g, preds)
    assert table.loc[table.threshold == 0.5, "recall"].item() == 0.5
    assert E.pick_threshold(table) <= 0.2


def test_bootstrap_interval_contains_point_estimate():
    g = _gt()
    preds = []
    for i in range(1, 5):
        box = [100, 100, 180, 160] if i != 4 else [400, 300, 450, 350]
        preds += E.to_prediction_records(i, np.array([box]), np.array([0.8]))
    point = E.coco_metrics(g, preds)["AP50"]
    ci = E.bootstrap(g, preds, n=30, metrics=("AP50",))
    assert ci.loc["AP50", "low"] <= point <= ci.loc["AP50", "high"]
