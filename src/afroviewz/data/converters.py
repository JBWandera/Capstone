"""Convert each source's native format into AfroViewz COCO (see coco.py).

All converters return (coco, images_root): file_name in the COCO dict is
relative to images_root.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from PIL import Image

from . import coco as C

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------- Roboflow COCO
def roboflow_group_key(file_name: str) -> str:
    """Roboflow names copies of one upload 'name_jpg.rf.<hash>.jpg'.

    Everything before '.rf.' identifies the original photo, so augmented
    copies share a group and can never leak across splits.
    """
    base = Path(file_name).name
    return base.split(".rf.")[0] if ".rf." in base else Path(base).stem


def from_roboflow_export(export_dir: str | Path, source: str) -> tuple[dict, Path]:
    """Read a Roboflow COCO export (train/ valid/ test/ folders) as one dataset.

    The original split is ignored on purpose: AfroViewz re-splits all sources
    together with grouping, so the old split is only kept as metadata.
    """
    export_dir = Path(export_dir)
    parts = []
    for split_dir in sorted(p for p in export_dir.iterdir() if p.is_dir()):
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        coco = C.load(ann_path)
        for im in coco["images"]:
            im["source"] = source
            im["group"] = f"{source}:{roboflow_group_key(im['file_name'])}"
            im["orig_split"] = split_dir.name
        parts.append((split_dir.name, coco))
    if not parts:
        raise FileNotFoundError(f"No _annotations.coco.json found under {export_dir}")
    merged = C.merge(parts)
    # keep the original categories: prepare_source() maps them to pothole and
    # reports what was dropped. Roboflow uses one category list for all splits.
    cats = {json.dumps(p[1]["categories"], sort_keys=True) for p in parts}
    if len(cats) != 1:
        raise ValueError(f"{export_dir}: splits disagree on categories")
    merged["categories"] = parts[0][1]["categories"]
    return merged, export_dir


# ---------------------------------------------------------------- Supervisely
def _sly_objects_to_annotations(objects: list[dict], image_id: int, next_id: int) -> list[dict]:
    anns = []
    for obj in objects:
        if not C.POTHOLE_NAME.search(obj.get("classTitle", "")):
            continue
        geom = obj.get("geometryType")
        ext = obj.get("points", {}).get("exterior", [])
        if geom == "rectangle" and len(ext) == 2:
            (x0, y0), (x1, y1) = ext
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            anns.append(C.make_annotation(next_id + len(anns), image_id, [x0, y0, x1 - x0, y1 - y0]))
        elif geom == "polygon" and len(ext) >= 3:
            flat = [float(v) for pt in ext for v in pt]
            anns.append(C.make_annotation(next_id + len(anns), image_id, C.bbox_from_polygons([flat]), [flat]))
    return anns


def stellenbosch_group_key(file_name: str, frames_per_group: int = 50) -> str:
    """Stellenbosch frames come from GoPro drives named like G0089830.JPG.

    Consecutive numbers are consecutive frames, so frames are grouped in
    blocks of `frames_per_group` to keep near identical neighbours together.
    """
    m = re.search(r"G(\d+)", Path(file_name).stem)
    if not m:
        return Path(file_name).stem
    return f"G{int(m.group(1)) // frames_per_group}"


def from_supervisely(project_dir: str | Path, source: str, group_fn=None,
                     max_negatives: int | None = None, seed: int = 42) -> tuple[dict, Path]:
    """Read a Supervisely project (dataset-tools download).

    Layout: <project>/<dataset>/img/<image> and <project>/<dataset>/ann/<image>.json
    """
    project_dir = Path(project_dir)
    group_fn = group_fn or (lambda name: Path(name).stem)
    out = C.new_coco(source)
    negatives = []
    next_img, next_ann = 1, 1
    for ann_dir in sorted(project_dir.rglob("ann")):
        img_dir = ann_dir.parent / "img"
        if not img_dir.is_dir():
            continue
        for ann_file in sorted(ann_dir.glob("*.json")):
            img_name = ann_file.name[: -len(".json")]
            img_path = img_dir / img_name
            if not img_path.exists():
                continue
            data = json.loads(ann_file.read_text(encoding="utf-8"))
            size = data.get("size", {})
            image = {
                "id": next_img,
                "file_name": str(img_path.relative_to(project_dir)).replace("\\", "/"),
                "width": size.get("width"),
                "height": size.get("height"),
                "source": source,
                "group": f"{source}:{ann_dir.parent.name}:{group_fn(img_name)}",
                "orig_split": ann_dir.parent.name,
            }
            anns = _sly_objects_to_annotations(data.get("objects", []), next_img, next_ann)
            if anns:
                out["images"].append(image)
                out["annotations"].extend(anns)
                next_ann += len(anns)
            else:
                negatives.append(image)
            next_img += 1
    if max_negatives is not None and len(negatives) > max_negatives:
        rng = random.Random(seed)
        negatives = rng.sample(negatives, max_negatives)
    out["images"].extend(negatives)
    out["images"].sort(key=lambda im: im["id"])
    return out, project_dir


# ---------------------------------------------------------------- YOLO
def from_yolo(images_dir: str | Path, labels_dir: str | Path, class_names: list[str],
              source: str, group_fn=None) -> tuple[dict, Path]:
    """Read YOLO detection (cls cx cy w h) or segmentation (cls x1 y1 x2 y2 ...) labels."""
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    group_fn = group_fn or (lambda name: Path(name).stem)
    pothole_idx = {i for i, n in enumerate(class_names) if C.POTHOLE_NAME.search(n)}
    out = C.new_coco(source)
    next_img, next_ann = 1, 1
    for img_path in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        with Image.open(img_path) as im:
            w, h = im.size
        rel = str(img_path.relative_to(images_dir)).replace("\\", "/")
        out["images"].append({"id": next_img, "file_name": rel, "width": w, "height": h,
                              "source": source, "group": f"{source}:{group_fn(img_path.name)}"})
        label_path = labels_dir / Path(rel).with_suffix(".txt")
        if label_path.exists():
            for line in label_path.read_text().split("\n"):
                vals = line.split()
                if len(vals) < 5 or int(float(vals[0])) not in pothole_idx:
                    continue
                nums = [float(v) for v in vals[1:]]
                if len(nums) == 4:
                    cx, cy, bw, bh = nums
                    bbox = [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]
                    out["annotations"].append(C.make_annotation(next_ann, next_img, bbox))
                else:
                    flat = [v * (w if i % 2 == 0 else h) for i, v in enumerate(nums)]
                    out["annotations"].append(
                        C.make_annotation(next_ann, next_img, C.bbox_from_polygons([flat]), [flat]))
                next_ann += 1
        next_img += 1
    return out, images_dir


# ---------------------------------------------------------------- plain images (holdout before labelling)
def from_image_folder(images_dir: str | Path, source: str, group_fn=None) -> tuple[dict, Path]:
    """Images with no labels yet, such as Mapillary frames waiting for labelling."""
    images_dir = Path(images_dir)
    group_fn = group_fn or (lambda name: Path(name).stem)
    out = C.new_coco(source)
    for i, img_path in enumerate(sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS), 1):
        with Image.open(img_path) as im:
            w, h = im.size
        out["images"].append({"id": i, "file_name": str(img_path.relative_to(images_dir)).replace("\\", "/"),
                              "width": w, "height": h, "source": source,
                              "group": f"{source}:{group_fn(img_path.name)}"})
    return out, images_dir
