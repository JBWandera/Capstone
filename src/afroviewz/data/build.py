"""The data cleaning pipeline.

Notebook 01, per source:   prepare_source()
    source COCO -> fix categories -> clean boxes -> copy + downscale to
    interim/<source>/ (images + coco.json + report.json)

Notebook 02, all sources:  build_from_interim()
    merge -> near duplicate clusters -> drop exact repeats -> grouped split ->
    RF-DETR export to processed/<name>/

Every step writes a JSON report so the docs quote real numbers.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import coco as C
from . import dedup, images, split


def clean_source(coco: dict, min_side_px: float = 4.0, keep_categories: tuple[str, ...] = ()) -> tuple[dict, dict]:
    coco, cat_rep = C.normalize_categories(coco, keep_categories)
    coco, rep = C.clean_annotations(coco, min_side_px=min_side_px)
    return coco, {"categories": cat_rep, "boxes": rep.as_dict()}


def one_per_group(coco: dict, splits: tuple[str, ...], copies: int = 3, seed: int = 42) -> tuple[dict, int]:
    """Keep one image per original photo among images whose orig_split is in `splits`.

    Roboflow augmentation writes `copies` versions of each training photo
    (flipped, rotated 90 degrees, cropped). AfroViewz augments at training
    time, so baked-in copies would only repeat photos and add sideways roads.
    Two different uploads can share a file name, so a name group of 6 holds two
    photos: round(n / copies) images are kept per group.
    """
    import hashlib
    from collections import defaultdict
    members: dict[str, list] = defaultdict(list)
    for im in coco["images"]:
        if im.get("orig_split") in splits:
            key = hashlib.sha1(f"{seed}:{im['file_name']}".encode()).hexdigest()
            members[im["group"]].append((key, im["id"]))
    keep_ids = set()
    for items in members.values():
        n_keep = max(1, round(len(items) / copies))
        keep_ids.update(i for _, i in sorted(items)[:n_keep])
    ids = [im["id"] for im in coco["images"] if im.get("orig_split") not in splits or im["id"] in keep_ids]
    return C.subset(coco, ids), len(coco["images"]) - len(ids)


def prepare_source(name: str, coco: dict, images_root: str | Path, interim_dir: str | Path,
                   max_side: int = 1280, min_side_px: float = 4.0,
                   keep_categories: tuple[str, ...] = (), one_per_group_splits: tuple[str, ...] = ()) -> dict:
    """Clean one source and store a downscaled copy in interim_dir."""
    interim_dir = Path(interim_dir)
    before = C.summary(coco)
    dropped_aug = 0
    if one_per_group_splits:
        coco, dropped_aug = one_per_group(coco, tuple(one_per_group_splits))
    coco, rep = clean_source(coco, min_side_px, tuple(keep_categories))
    rep["augmented_copies_dropped"] = dropped_aug
    coco, problems = images.materialize(coco, images_root, interim_dir / "images", max_side=max_side)
    rep.update({"before": before, "after": C.summary(coco), "image_problem_count": len(problems),
                "image_problems": problems[:50], "max_side": max_side})
    C.save(coco, interim_dir / "coco.json")
    (interim_dir / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep


def load_interim(interim_dir: str | Path) -> tuple[dict, Path]:
    interim_dir = Path(interim_dir)
    return C.load(interim_dir / "coco.json"), interim_dir / "images"


def build_from_interim(interim: dict[str, Path], out_root: str | Path, holdout: tuple[str, ...] = (),
                       dup_distance: int = 6, max_mad: float = 6.0, seed: int = 42,
                       link_images: bool = False, export_root: str | Path | None = None) -> dict:
    """interim: {source_name: interim_dir}. Writes all.coco.json and the report to
    out_root, and the RF-DETR copy to export_root/rfdetr (default out_root). On
    Colab export_root is the local disk, so the image copies never fill Drive."""
    out_root = Path(out_root)
    export_root = Path(export_root) if export_root else out_root
    out_root.mkdir(parents=True, exist_ok=True)
    parts, roots = [], {}
    for name, d in interim.items():
        coco, root = load_interim(d)
        for im in coco["images"]:
            im["file_name"] = f"{name}/{im['file_name']}"
        roots[name] = root
        parts.append(("", coco))
    merged = C.merge(parts)

    # one folder per source under images/, linked to the interim copies (no extra disk)
    images_root = export_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    for name, root in roots.items():
        link = images_root / name
        if not link.exists():
            link.symlink_to(Path(root).resolve(), target_is_directory=True)

    hashes, thumbs = dedup.phash_images(merged, images_root, with_thumbs=True)
    clusters = dedup.cluster(hashes, dup_distance, thumbs=thumbs, max_mad=max_mad)
    drop = dedup.exact_duplicates_to_drop(merged, clusters, hashes, thumbs=thumbs)
    merged = C.subset(merged, [im["id"] for im in merged["images"] if im["id"] not in drop])
    groups_merged = dedup.apply_clusters_to_groups(merged, clusters)
    sizes = Counter(clusters[im["id"]] for im in merged["images"])
    for im in merged["images"]:
        im["phash"] = f"{hashes[im['id']]:016x}"
        im["dup_cluster"] = clusters[im["id"]]
    report: dict = {"dedup": {
        "phash_max_distance": dup_distance,
        "pixel_max_mad": max_mad,
        "largest_cluster": max(sizes.values()),
        "exact_repeats_dropped": len(drop),
        "near_duplicate_clusters": sum(1 for n in sizes.values() if n > 1),
        "images_in_clusters": sum(n for n in sizes.values() if n > 1),
        "groups_merged_by_near_duplicates": groups_merged,
    }}

    splits = split.grouped_split(merged, seed=seed, holdout_sources=holdout)
    leaks = split.check_no_leakage(merged, splits)
    if leaks:
        raise RuntimeError(f"split leakage: {leaks[:5]}")
    split_of = {i: sp for sp, ids in splits.items() for i in ids}
    for im in merged["images"]:
        im["split"] = split_of[im["id"]]
    C.save(merged, out_root / "all.coco.json")
    report["splits"] = split.export_rfdetr(merged, images_root, splits, export_root / "rfdetr", link=link_images)
    report["total"] = C.summary(merged)
    report["seed"] = seed
    (out_root / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def zip_export(export_root: str | Path, zip_path: str | Path, name: str) -> Path:
    """Zip <export_root>/rfdetr as <name>/rfdetr/... (what the training notebooks unpack)."""
    import zipfile
    export_root, zip_path = Path(export_root), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:
        for f in sorted((export_root / "rfdetr").rglob("*")):
            if f.is_file():
                z.write(f, Path(name) / f.relative_to(export_root))
    return zip_path


def split_manifest(all_coco: dict):
    """One row per image: where it came from and which split it is in."""
    import pandas as pd
    return pd.DataFrame([{"file_name": im["file_name"], "source": im["source"], "split": im["split"],
                          "group": im["group"], "phash": im.get("phash")} for im in all_coco["images"]])
