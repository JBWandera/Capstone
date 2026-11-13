"""End to end checks of the data pipeline on small synthetic sources."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from afroviewz.data import build, coco as C, converters, dedup, mapillary, split


def _img(path: Path, w: int, h: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.random((h // 8, w // 8, 3)) * 255).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).resize((w, h), Image.NEAREST).save(path)


@pytest.fixture()
def roboflow_export(tmp_path: Path) -> Path:
    root = tmp_path / "rf"
    for split_name, seeds in {"train": [1, 1, 2, 3, 4], "valid": [5], "test": [6]}.items():
        d = root / split_name
        images, anns = [], []
        for i, s in enumerate(seeds):
            # seed 1 twice: two augmented copies of the same upload
            name = f"photo{s}_jpg.rf.{split_name}{i}.jpg"
            _img(d / name, 640, 480, s)
            images.append({"id": i, "file_name": name, "width": 640, "height": 480})
            anns.append({"id": i, "image_id": i, "category_id": 1, "bbox": [100, 100, 50, 40], "area": 2000, "iscrowd": 0})
        anns.append({"id": 99, "image_id": 0, "category_id": 2, "bbox": [10, 10, 5, 5], "area": 25, "iscrowd": 0})
        C.save({"images": images, "annotations": anns, "categories": [
            {"id": 0, "name": "potholes", "supercategory": "none"},
            {"id": 1, "name": "pothole", "supercategory": "potholes"},
            {"id": 2, "name": "object", "supercategory": "potholes"}]}, d / "_annotations.coco.json")
    return root


@pytest.fixture()
def supervisely_project(tmp_path: Path) -> Path:
    root = tmp_path / "sly" / "ds1_simplex-train"
    for k in range(12):
        name = f"positive_G00{89830 + k}.JPG" if k < 8 else f"negative_G00{70000 + k}.JPG"
        _img(root / "img" / name, 1600, 1200, 100 + k)
        objects = []
        if k < 8:
            objects.append({"classTitle": "pothole", "geometryType": "rectangle",
                            "points": {"exterior": [[800, 900], [700, 820]], "interior": []}})
        ann = {"size": {"width": 1600, "height": 1200}, "objects": objects}
        (root / "ann").mkdir(parents=True, exist_ok=True)
        (root / "ann" / f"{name}.json").write_text(json.dumps(ann))
    return tmp_path / "sly"


def test_roboflow_conversion_groups_augmented_copies(roboflow_export: Path):
    coco, root = converters.from_roboflow_export(roboflow_export, "rf")
    assert len(coco["images"]) == 7
    assert len(coco["annotations"]) == 10  # "object" boxes are dropped later, in prepare_source
    groups = [im["group"] for im in coco["images"]]
    assert groups.count("rf:photo1_jpg") == 2
    assert all((root / im["file_name"]).exists() for im in coco["images"])


def test_supervisely_rectangle_is_normalised(supervisely_project: Path):
    coco, root = converters.from_supervisely(supervisely_project, "sly",
                                             group_fn=converters.stellenbosch_group_key, max_negatives=2)
    assert len([im for im in coco["images"]]) == 10  # 8 positive + 2 capped negatives
    ann = coco["annotations"][0]
    assert ann["bbox"] == [700.0, 820.0, 100.0, 80.0]


def test_clean_clips_and_drops():
    coco = C.new_coco()
    coco["images"] = [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}]
    coco["annotations"] = [
        C.make_annotation(1, 1, [90, 90, 30, 30]),   # clipped to 10x10
        C.make_annotation(2, 1, [5, 5, 2, 2]),       # tiny
        C.make_annotation(3, 1, [90, 90, 10, 10]),   # duplicate of 1 after clipping
    ]
    out, rep = C.clean_annotations(coco, min_side_px=4)
    assert rep.clipped == 1 and rep.dropped_tiny == 1 and rep.dropped_duplicate == 1
    assert out["annotations"][0]["bbox"] == [90, 90, 10, 10]


def test_dedup_pairs_and_clusters():
    hashes = {1: 0b0, 2: 0b111, 3: (1 << 63) | 0xFF00FF, 4: 0b1}
    pairs = dedup.near_duplicate_pairs(hashes, max_distance=3)
    found = {(a, b) for a, b, _ in pairs}
    assert {(1, 2), (1, 4), (2, 4)} <= found and all(3 not in p for p in found)
    cl = dedup.cluster(hashes, 3)
    assert cl[1] == cl[2] == cl[4] and cl[3] == 3


def test_grouped_split_has_no_leakage():
    coco = C.new_coco()
    coco["images"] = [{"id": i, "file_name": f"{i}.jpg", "source": "s", "group": f"g{i // 3}"} for i in range(300)]
    coco["images"] += [{"id": 1000 + i, "file_name": f"h{i}.jpg", "source": "uganda", "group": "u"} for i in range(5)]
    splits = split.grouped_split(coco, holdout_sources=("uganda",))
    assert not split.check_no_leakage(coco, splits)
    assert len(splits["holdout_uganda"]) == 5
    assert 220 <= len(splits["train"]) <= 260 and len(splits["valid"]) >= 20 and len(splits["test"]) >= 20


def test_mapillary_thin_spacing():
    imgs = [{"id": str(i), "sequence": "s1", "captured_at": i, "thumb_2048_url": "u", "is_pano": False,
             "computed_geometry": {"coordinates": [32.58 + i * 0.00005, 0.31]}} for i in range(100)]
    kept = mapillary.thin(imgs, min_spacing_m=15, max_per_sequence=10)
    assert len(kept) == 10
    lons = [d["computed_geometry"]["coordinates"][0] for d in kept]
    assert all(b > a for a, b in zip(lons, lons[1:]))


def test_build_end_to_end(tmp_path: Path, roboflow_export: Path, supervisely_project: Path):
    rf_coco, rf_root = converters.from_roboflow_export(roboflow_export, "rf")
    sly_coco, sly_root = converters.from_supervisely(supervisely_project, "sly",
                                                     group_fn=converters.stellenbosch_group_key, max_negatives=3)
    interim = tmp_path / "interim"
    rep_rf = build.prepare_source("rf", rf_coco, rf_root, interim / "rf", max_side=800)
    assert rep_rf["categories"]["dropped_by_category"] == {"object": 3}
    build.prepare_source("sly", sly_coco, sly_root, interim / "sly", max_side=800)
    out = tmp_path / "processed"
    report = build.build_from_interim({"rf": interim / "rf", "sly": interim / "sly"}, out)

    allc = C.load(out / "all.coco.json")
    # downscaled from 1600 to 800: the 100x80 Stellenbosch box becomes 50x40
    sly_ids = {im["id"] for im in allc["images"] if im["source"] == "sly"}
    box = next(a["bbox"] for a in allc["annotations"] if a["image_id"] in sly_ids)
    assert box == pytest.approx([350, 410, 50, 40])
    assert all(im["width"] <= 800 for im in allc["images"])
    # the two identical augmented copies (seed 1) collapse into one image
    assert report["dedup"]["exact_repeats_dropped"] >= 1
    assert not split.check_no_leakage(allc, {sp: [im["id"] for im in allc["images"] if im["split"] == sp]
                                             for sp in {im["split"] for im in allc["images"]}})
    for name in report["splits"]:
        exported = C.load(out / "rfdetr" / name / "_annotations.coco.json")
        assert [c["id"] for c in exported["categories"]] == [0, 1]
        for im in exported["images"]:
            assert (out / "rfdetr" / name / im["file_name"]).exists()
