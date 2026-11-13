"""COCO helpers: the one format every AfroViewz source is converted to.

Conventions
-----------
* One category: pothole, id 1.
* Every image record carries extra fields:
    source  name of the source in configs/sources.yaml
    group   split group; images in the same group always land in the same split
            (frames of one drive, augmented copies of one photo, near duplicates)
* file_name is relative to the images root of the dataset it belongs to.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

POTHOLE_ID = 1
POTHOLE_CATEGORY = {"id": POTHOLE_ID, "name": "pothole", "supercategory": "road_damage"}
POTHOLE_NAME = re.compile(r"pot\s*-?\s*hole", re.IGNORECASE)


def new_coco(description: str = "AfroViewz") -> dict:
    return {
        "info": {"description": description},
        "licenses": [],
        "categories": [dict(POTHOLE_CATEGORY)],
        "images": [],
        "annotations": [],
    }


def load(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(coco: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f)


def polygon_area(flat: list[float]) -> float:
    """Shoelace area of a flat [x1, y1, x2, y2, ...] polygon."""
    xs, ys = flat[0::2], flat[1::2]
    n = len(xs)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) / 2.0


def bbox_from_polygons(polys: list[list[float]]) -> list[float]:
    xs = [v for p in polys for v in p[0::2]]
    ys = [v for p in polys for v in p[1::2]]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def make_annotation(ann_id: int, image_id: int, bbox: list[float],
                    segmentation: list[list[float]] | None = None) -> dict:
    ann = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": POTHOLE_ID,
        "bbox": [float(v) for v in bbox],
        "iscrowd": 0,
    }
    if segmentation:
        ann["segmentation"] = [[float(v) for v in p] for p in segmentation]
        ann["area"] = float(sum(polygon_area(p) for p in segmentation))
    else:
        ann["segmentation"] = []
        ann["area"] = float(bbox[2] * bbox[3])
    return ann


def mask_kind(segmentation) -> str:
    """'polygon' for a real outline, 'box' for a polygon that is only the
    bounding rectangle (common when box labels are merged into a segmentation
    project), 'none' when there is no polygon. Only 'polygon' masks are used to
    train the severity (segmentation) model."""
    if not segmentation or not isinstance(segmentation, list):
        return "none"
    polys = [p for p in segmentation if isinstance(p, list) and len(p) >= 6]
    if not polys:
        return "none"
    if len(polys) == 1:
        pts = list(zip(polys[0][0::2], polys[0][1::2]))
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]  # closed ring: last point repeats the first
        xs = sorted({round(x) for x, _ in pts})
        ys = sorted({round(y) for _, y in pts})
        near = lambda vals: len(vals) <= 2 or (len(vals) <= 4 and vals[-1] - vals[0] > 2 and  # noqa: E731
                                               all(min(abs(v - vals[0]), abs(v - vals[-1])) <= 1 for v in vals))
        if len(pts) <= 5 and near(xs) and near(ys):
            return "box"
    return "polygon"


def normalize_categories(coco: dict, keep_names: tuple[str, ...] = ()) -> tuple[dict, dict]:
    """Map every pothole category to id 1 and drop everything else.

    Roboflow exports add a parent category (id 0, often named after the
    project) with no annotations, and some community datasets carry extra
    classes ("object", "crack"). Annotations are kept when their category name
    matches "pothole" or is listed in keep_names (some uploads name the class
    "0"; see keep_categories in configs/sources.yaml). Returns (new_coco, report).
    """
    keep_ids = {c["id"] for c in coco["categories"]
                if POTHOLE_NAME.search(c["name"]) or c["name"] in keep_names}
    out = copy.deepcopy(coco)
    out["categories"] = [dict(POTHOLE_CATEGORY)]
    kept, dropped = [], Counter()
    names = {c["id"]: c["name"] for c in coco["categories"]}
    for ann in out["annotations"]:
        if ann["category_id"] in keep_ids:
            ann["category_id"] = POTHOLE_ID
            kept.append(ann)
        else:
            dropped[names.get(ann["category_id"], str(ann["category_id"]))] += 1
    out["annotations"] = kept
    for ann in kept:
        ann["mask"] = mask_kind(ann.get("segmentation"))
    report = {
        "masks": dict(Counter(a["mask"] for a in kept)),
        "categories_in": [c["name"] for c in coco["categories"]],
        "kept_annotations": len(kept),
        "dropped_by_category": dict(dropped),
    }
    return out, report


@dataclass
class CleanReport:
    clipped: int = 0
    dropped_degenerate: int = 0
    dropped_tiny: int = 0
    dropped_duplicate: int = 0
    images_without_size: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def clean_annotations(coco: dict, min_side_px: float = 4.0, dup_iou: float = 0.95) -> tuple[dict, CleanReport]:
    """Clip boxes to the image, drop degenerate, tiny and duplicate boxes."""
    rep = CleanReport()
    out = copy.deepcopy(coco)
    sizes = {im["id"]: (im.get("width"), im.get("height")) for im in out["images"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in out["annotations"]:
        w_img, h_img = sizes.get(ann["image_id"], (None, None))
        x, y, w, h = ann["bbox"]
        if w_img and h_img:
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(w_img), x + w), min(float(h_img), y + h)
            if max(abs(x0 - x), abs(y0 - y), abs((x1 - x0) - w), abs((y1 - y0) - h)) > 0.5:
                rep.clipped += 1  # changes under half a pixel are rounding, not clipping
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
        else:
            rep.images_without_size += 1
        if w <= 0 or h <= 0:
            rep.dropped_degenerate += 1
            continue
        if min(w, h) < min_side_px:
            rep.dropped_tiny += 1
            continue
        ann["bbox"] = [x, y, w, h]
        if not ann.get("segmentation"):
            ann["area"] = w * h
        dup = any(iou(ann["bbox"], other["bbox"]) >= dup_iou for other in by_image[ann["image_id"]])
        if dup:
            rep.dropped_duplicate += 1
            continue
        by_image[ann["image_id"]].append(ann)
    out["annotations"] = [a for anns in by_image.values() for a in anns]
    return out, rep


def iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def merge(parts: Iterable[tuple[str, dict]]) -> dict:
    """Merge several COCO dicts into one, re-numbering ids.

    Each part is (prefix, coco). file_name becomes "<prefix>/<file_name>", so
    the merged images root holds one folder per part.
    """
    out = new_coco("AfroViewz merged")
    next_img, next_ann = 1, 1
    for prefix, coco in parts:
        id_map = {}
        for im in coco["images"]:
            new = dict(im)
            new["id"] = next_img
            new["file_name"] = f"{prefix}/{im['file_name']}" if prefix else im["file_name"]
            id_map[im["id"]] = next_img
            out["images"].append(new)
            next_img += 1
        for ann in coco["annotations"]:
            if ann["image_id"] not in id_map:
                continue
            new = dict(ann)
            new["id"] = next_ann
            new["image_id"] = id_map[ann["image_id"]]
            out["annotations"].append(new)
            next_ann += 1
    return out


def subset(coco: dict, image_ids: Iterable[int]) -> dict:
    keep = set(image_ids)
    out = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
    out["images"] = [im for im in coco["images"] if im["id"] in keep]
    out["annotations"] = [a for a in coco["annotations"] if a["image_id"] in keep]
    return out


def summary(coco: dict) -> dict:
    n_ann = Counter(a["image_id"] for a in coco["annotations"])
    n_img = len(coco["images"])
    return {
        "images": n_img,
        "annotations": len(coco["annotations"]),
        "images_with_potholes": sum(1 for im in coco["images"] if n_ann[im["id"]] > 0),
        "negative_images": sum(1 for im in coco["images"] if n_ann[im["id"]] == 0),
        "by_source": dict(Counter(im.get("source", "?") for im in coco["images"])),
    }
