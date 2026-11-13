"""Download each source and convert it to AfroViewz COCO.

Usage (Colab, see notebooks/01_data_acquisition.ipynb):
    cfg = load_config()
    download("capstone_v1", cfg, roboflow_key=...)
    coco, root = to_coco("capstone_v1", cfg)
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .. import paths
from . import converters, mapillary


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else paths.CONFIG_DIR / "sources.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["sources"]


def dataset_ninja_url(name: str) -> str:
    """The archive URL dataset-tools would use, read from its bundled index."""
    import json
    from dataset_tools.repo import download as dt_download
    with open(dt_download.PATH_DOWNLOAD_URLS, encoding="utf-8") as f:
        return json.load(f)[name]["download_sly_url"]


def _find_supervisely_project(root: Path) -> Path:
    """dataset-tools unpacks into a folder whose name varies; find the one holding ann/ dirs."""
    hits = sorted({p.parent.parent for p in root.rglob("ann") if p.is_dir()})
    if not hits:
        raise FileNotFoundError(f"No Supervisely ann/ folders under {root}")
    import os
    return Path(os.path.commonpath([str(h) for h in hits]))


def download(name: str, cfg: dict, roboflow_key: str | None = None,
             mapillary_token: str | None = None) -> Path:
    src = cfg[name]
    dst = paths.raw_dir(name)
    kind = src["download"]
    if kind != "roboflow":  # the Roboflow SDK skips the download if the folder already exists
        dst.mkdir(parents=True, exist_ok=True)

    if kind == "roboflow":
        from roboflow import Roboflow  # imported here so the core package stays light
        if not roboflow_key:
            raise ValueError("ROBOFLOW_API_KEY is required")
        project = Roboflow(api_key=roboflow_key).workspace(src["workspace"]).project(src["project"])
        project.version(src["version"]).download(src["export_format"], location=str(dst), overwrite=True)

    elif kind == "dataset-tools":
        # Stream the archive and downscale on the fly (see stream.py): needs
        # about 2 GB of disk instead of about 46 GB.
        from .stream import stream_supervisely_tar
        stream_supervisely_tar(dataset_ninja_url(src["dataset_tools_name"]), dst,
                               max_side=src.get("max_side", 1600))

    elif kind == "mapillary":
        if not mapillary_token:
            raise ValueError("MAPILLARY_TOKEN is required")
        import json
        manifest = paths.CONFIG_DIR / "manifests" / f"{name}.csv"
        if manifest.exists():
            print(f"Downloading the {name} images listed in {manifest.name}")
            mapillary.download_manifest(mapillary_token, manifest, dst)
            return dst
        caps = src.get("max_images_per_area", {})
        for area, bbox in src["areas"].items():
            cache = dst / f"{area}_search.json"
            if cache.exists():
                found = json.loads(cache.read_text())
            else:
                found = mapillary.search(mapillary_token, bbox)
                cache.write_text(json.dumps(found))
            kept = mapillary.thin(found, src.get("min_spacing_m", 15), src.get("max_images_per_sequence", 40),
                                  max_total=caps.get(area), min_year=src.get("min_year"))
            print(f"{area}: {len(found)} images found, {len(kept)} kept after thinning")
            mapillary.download(kept, dst, area)

    elif kind == "manual":
        print(f"Put the files for {name} in {dst}")
    else:
        raise ValueError(f"Unknown download kind {kind!r}")
    return dst


def to_coco(name: str, cfg: dict) -> tuple[dict, Path]:
    src = cfg[name]
    raw = paths.raw_dir(name)
    fmt = src["format"]
    if fmt == "coco":
        return converters.from_roboflow_export(raw, name)
    if fmt == "supervisely":
        project = _find_supervisely_project(raw)
        return converters.from_supervisely(project, name, group_fn=converters.stellenbosch_group_key,
                                           max_negatives=src.get("max_negatives"))
    if fmt == "images_only":
        coco, root = converters.from_image_folder(raw, name, group_fn=lambda n: n)
        _attach_mapillary_meta(coco, raw / "metadata.csv")
        return coco, root
    if fmt == "video":
        frames = extract_frames(raw, paths.interim_dir(name) / "frames", src.get("frame_every_s", 1.0))
        return converters.from_image_folder(frames, name, group_fn=lambda n: n.split("__")[0])
    raise ValueError(f"Unknown format {fmt!r}")


def _attach_mapillary_meta(coco: dict, meta_csv: Path) -> None:
    """Copy GPS, sequence and creator onto each image; group by sequence."""
    import csv
    if not meta_csv.exists():
        return
    with open(meta_csv, encoding="utf-8") as f:
        rows = {r["file_name"]: r for r in csv.DictReader(f)}
    for im in coco["images"]:
        r = rows.get(im["file_name"])
        if r:
            im.update({"lat": float(r["lat"]), "lon": float(r["lon"]), "area": r["area"],
                       "sequence": r["sequence"], "creator": r["creator"], "license": r["license"],
                       "url": r["url"], "group": f"{im['source']}:{r['sequence']}"})


def extract_frames(video_dir: Path, out_dir: Path, every_s: float = 1.0) -> Path:
    """One frame every `every_s` seconds from each .mp4/.mov; names '<video>__<t>.jpg'."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    for vid in sorted(p for p in video_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}):
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps * every_s))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                cv2.imwrite(str(out_dir / f"{vid.stem}__{i // step:05d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            i += 1
        cap.release()
    return out_dir


def prepare_kwargs(name: str, cfg: dict) -> dict:
    """Per source settings for build.prepare_source, read from sources.yaml."""
    src = cfg[name]
    return {
        "max_side": src.get("max_side", 1280),
        "keep_categories": tuple(str(c) for c in src.get("keep_categories", ())),
        "one_per_group_splits": tuple(src.get("one_per_group_splits", ())),
    }
