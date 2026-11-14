"""Grouped train / valid / test split and export to the RF-DETR folder layout.

Groups (one drive, one original photo and its augmented copies, one near
duplicate cluster) are assigned whole to a split, so nothing leaks. Each
source is split on its own, so every split has the same source mix.
"""

from __future__ import annotations

import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

from . import coco as C


def _stable_order(keys: list[str], seed: int) -> list[str]:
    return sorted(keys, key=lambda k: hashlib.sha1(f"{seed}:{k}".encode()).hexdigest())


def grouped_split(coco: dict, ratios=(0.8, 0.1, 0.1), seed: int = 42,
                  holdout_sources: tuple[str, ...] = ()) -> dict[str, list[int]]:
    """Return {"train": [ids], "valid": [...], "test": [...], "holdout_<src>": [...]}.

    Within each source, groups are visited largest first (ties in a seeded hash
    order) and each goes to the split furthest below its share of both images
    and pothole boxes, so dense groups do not pile into one split.
    """
    names = ("train", "valid", "test")
    n_boxes = defaultdict(int)
    for a in coco["annotations"]:
        n_boxes[a["image_id"]] += 1
    out: dict[str, list[int]] = defaultdict(list)
    groups: dict[str, list[int]] = defaultdict(list)
    group_sources: dict[str, list[str]] = defaultdict(list)
    for im in coco["images"]:
        src = im.get("source", "unknown")
        if src in holdout_sources:
            out[f"holdout_{src}"].append(im["id"])
            continue
        g = im.get("group", str(im["id"]))
        groups[g].append(im["id"])
        group_sources[g].append(src)
    # a group can span sources after near duplicate merging; it is placed with
    # the source most of its images come from, and goes to one split as a whole
    by_source: dict[str, list[str]] = defaultdict(list)
    for g, srcs in group_sources.items():
        # most common source; ties broken by name so every run gives the same answer
        by_source[max(sorted(set(srcs)), key=srcs.count)].append(g)

    for src, gnames in sorted(by_source.items()):
        size = {g: (len(groups[g]), sum(n_boxes[i] for i in groups[g])) for g in gnames}
        tot_img = sum(v[0] for v in size.values())
        tot_box = sum(v[1] for v in size.values())
        target = {n: (r * tot_img, r * tot_box) for n, r in zip(names, ratios)}
        got = {n: [0, 0] for n in names}

        def need(n):
            # how far below its share of images and of potholes this split is
            ti, tb = target[n]
            return (ti - got[n][0]) / max(ti, 1e-9) + (tb - got[n][1]) / max(tb, 1e-9)

        # big groups first so they do not overshoot a small split late in the pass
        order = sorted(_stable_order(gnames, seed), key=lambda g: -(size[g][0] + size[g][1]))
        for g in order:
            pick = max(names, key=need)
            out[pick].extend(groups[g])
            got[pick][0] += size[g][0]
            got[pick][1] += size[g][1]
    return dict(out)


def frozen_split(coco: dict, manifest_csv, holdout_sources: tuple[str, ...] = (), **kwargs) -> dict[str, list[int]]:
    """Use the split recorded in a manifest (file_name, split) for every image it
    lists; images it does not list are split by grouped_split. This keeps the
    published dataset fixed once models have been trained on it."""
    import csv
    with open(manifest_csv, encoding="utf-8") as f:
        fixed = {r["file_name"]: r["split"] for r in csv.DictReader(f)}
    known = [im for im in coco["images"] if im["file_name"] in fixed]
    rest = {k: v for k, v in coco.items() if k != "images"}
    rest["images"] = [im for im in coco["images"] if im["file_name"] not in fixed]
    out: dict[str, list[int]] = defaultdict(list)
    for im in known:
        out[fixed[im["file_name"]]].append(im["id"])
    if rest["images"]:
        for sp, ids in grouped_split(rest, holdout_sources=holdout_sources, **kwargs).items():
            out[sp].extend(ids)
    return dict(out)


def check_no_leakage(coco: dict, splits: dict[str, list[int]]) -> list[str]:
    group_of = {im["id"]: im.get("group") for im in coco["images"]}
    seen: dict[str, str] = {}
    problems = []
    for split, ids in splits.items():
        for i in ids:
            g = group_of[i]
            if g in seen and seen[g] != split:
                problems.append(f"group {g} in {seen[g]} and {split}")
            seen.setdefault(g, split)
    return problems


def export_rfdetr(coco: dict, images_root: str | Path, splits: dict[str, list[int]],
                  out_dir: str | Path, link: bool = False) -> dict[str, dict]:
    """Write <out>/<split>/_annotations.coco.json with images beside it.

    RF-DETR reads the Roboflow COCO layout. Roboflow exports carry a parent
    category with id 0 and the real class with id 1; RF-DETR expects the same,
    so that pattern is reproduced here. Files are flattened to
    '<source>__<path_with_underscores>.jpg' to keep names unique.
    """
    images_root, out_dir = Path(images_root), Path(out_dir)
    stats = {}
    for split, ids in splits.items():
        sub = C.subset(coco, ids)
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for im in sub["images"]:
            flat = im["file_name"].replace("/", "__")
            src, dst = images_root / im["file_name"], split_dir / flat
            if not dst.exists():
                if link:
                    dst.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst)
            im["file_name"] = flat
        sub["categories"] = [
            {"id": 0, "name": "potholes", "supercategory": "none"},
            {"id": 1, "name": "pothole", "supercategory": "potholes"},
        ]
        C.save(sub, split_dir / "_annotations.coco.json")
        stats[split] = C.summary(sub)
    return stats
