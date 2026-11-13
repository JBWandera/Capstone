"""Image level work: check every file opens, record true sizes, downscale.

Downscaling matters here: the Stellenbosch frames are 3680x2760, which wastes
Drive space and Colab I/O. No model in this project reads more than about
880 px, so images are stored at max_side=1280 and boxes are scaled to match.
"""

from __future__ import annotations

import copy
from pathlib import Path

from PIL import Image, ImageOps
from tqdm.auto import tqdm


def _scale_ann(ann: dict, s: float) -> dict:
    ann = dict(ann)
    ann["bbox"] = [v * s for v in ann["bbox"]]
    if ann.get("segmentation"):
        ann["segmentation"] = [[v * s for v in poly] for poly in ann["segmentation"]]
    ann["area"] = ann["area"] * s * s
    return ann


def materialize(coco: dict, src_root: str | Path, dst_root: str | Path, prefix: str = "",
                max_side: int = 1280, quality: int = 90) -> tuple[dict, list[dict]]:
    """Copy images into dst_root/<prefix>/..., downscaling to max_side.

    Applies EXIF rotation, converts to RGB JPEG, rewrites width, height,
    file_name and scales annotations. Unreadable files are dropped and
    reported. Returns (new_coco, problems).
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    out = copy.deepcopy(coco)
    anns_by_img: dict[int, list[dict]] = {}
    for a in out["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    kept_images, kept_anns, problems = [], [], []
    for im in tqdm(out["images"], desc=f"images {prefix or ''}".strip()):
        src = src_root / im["file_name"]
        rel = Path(prefix) / Path(im["file_name"]).with_suffix(".jpg") if prefix else Path(im["file_name"]).with_suffix(".jpg")
        dst = dst_root / rel
        try:
            with Image.open(src) as pil:
                w0, h0 = pil.size
                stored = (im.get("width"), im.get("height"))
                if stored == (h0, w0) and stored != (w0, h0):
                    # Labels were drawn on the EXIF rotated view: rotate pixels to match.
                    pil = ImageOps.exif_transpose(pil)
                    w0, h0 = pil.size
                    problems.append({"file": str(src), "issue": "exif_rotated_to_match_labels"})
                elif stored[0] and stored != (w0, h0):
                    problems.append({"file": str(src), "issue": f"size_mismatch stored={stored} actual={(w0, h0)}"})
                pil = pil.convert("RGB")  # saving a fresh JPEG also drops EXIF, so viewers cannot rotate it again
                s = min(1.0, max_side / max(w0, h0))
                if s < 1.0:
                    pil = pil.resize((round(w0 * s), round(h0 * s)), Image.LANCZOS)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    pil.save(dst, "JPEG", quality=quality)
                w1, h1 = pil.size
        except Exception as exc:  # noqa: BLE001  any unreadable file is reported and skipped
            problems.append({"file": str(src), "issue": f"unreadable: {exc.__class__.__name__}"})
            continue
        # labels live in the stored size frame when one is given
        base_w = im.get("width") or w0
        sx = w1 / base_w
        im.update({"file_name": str(rel).replace("\\", "/"), "width": w1, "height": h1,
                   "orig_width": w0, "orig_height": h0})
        kept_images.append(im)
        kept_anns.extend(_scale_ann(a, sx) for a in anns_by_img.get(im["id"], []))
    out["images"], out["annotations"] = kept_images, kept_anns
    return out, problems
