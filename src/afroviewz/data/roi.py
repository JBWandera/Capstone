"""Road band crop for dashcam frames (experiment E4).

EDA (notebook 03) showed that dashcam potholes sit in a narrow horizontal band
(box centres at 45 to 60 percent of image height) and are tiny once the whole
frame is squashed to the model's square input. Cutting that band out before
the resize keeps the full width but makes every pothole about 2.5 times taller
to the model.

Training: dashcam images in the RF-DETR export are replaced by their band.
Inference: the same band is cut, the model runs on it, and boxes are shifted
back to full frame coordinates, so the result is scored against the usual
full frame labels (a pothole outside the band counts as missed).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from . import coco as C

BAND = (0.35, 0.75)          # top and bottom of the band, as fractions of image height
DASHCAM_SOURCES = ("stellenbosch",)


def is_dashcam(file_name: str, sources: tuple[str, ...] = DASHCAM_SOURCES) -> bool:
    return file_name.split("__", 1)[0] in sources


def band_box(height: int, band: tuple[float, float] = BAND) -> tuple[int, int]:
    return int(round(band[0] * height)), int(round(band[1] * height))


def build_roi_export(src_export: str | Path, dst_export: str | Path, band: tuple[float, float] = BAND,
                     min_visible: float = 0.5) -> dict:
    """Copy an RF-DETR export, replacing dashcam images by their road band.

    A box keeps its part inside the band; it is dropped when less than
    min_visible of its height is inside.
    """
    src_export, dst_export = Path(src_export), Path(dst_export)
    stats = {}
    for split_dir in sorted(p for p in src_export.iterdir() if p.is_dir()):
        coco = C.load(split_dir / "_annotations.coco.json")
        out_dir = dst_export / split_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        by_img: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        new_anns, cropped, dropped = [], 0, 0
        for im in coco["images"]:
            src, dst = split_dir / im["file_name"], out_dir / im["file_name"]
            anns = by_img.get(im["id"], [])
            if not is_dashcam(im["file_name"]):
                if not dst.exists():
                    shutil.copy2(src, dst)
                new_anns.extend(anns)
                continue
            top, bottom = band_box(im["height"], band)
            if not dst.exists():
                with Image.open(src) as pil:
                    pil.crop((0, top, im["width"], bottom)).save(dst, "JPEG", quality=92)
            im["height"] = bottom - top
            im["roi_band"] = [top, bottom]
            cropped += 1
            for a in anns:
                x, y, w, h = a["bbox"]
                y0, y1 = max(y, top), min(y + h, bottom)
                if h <= 0 or (y1 - y0) / h < min_visible:
                    dropped += 1
                    continue
                b = dict(a)
                b["bbox"] = [x, y0 - top, w, y1 - y0]
                b["area"] = w * (y1 - y0)
                b["segmentation"] = []
                new_anns.append(b)
        coco["annotations"] = new_anns
        C.save(coco, out_dir / "_annotations.coco.json")
        stats[split_dir.name] = {"images": len(coco["images"]), "dashcam_cropped": cropped,
                                 "boxes": len(new_anns), "boxes_dropped_outside_band": dropped}
    return stats


def roi_predict_fn(base_fn, band: tuple[float, float] = BAND):
    """Wrap a predict_fn so dashcam images are predicted on their band only.

    base_fn(list of PIL) -> list of (xyxy, scores). The returned function takes
    (images, file_names); runner.predict_split passes the names because the
    function has wants_names = True.
    """
    import numpy as np

    def fn(images, names):
        inputs, offsets = [], []
        for pil, name in zip(images, names):
            if is_dashcam(name):
                top, bottom = band_box(pil.height, band)
                inputs.append(pil.crop((0, top, pil.width, bottom)))
                offsets.append(top)
            else:
                inputs.append(pil)
                offsets.append(0)
        outs = base_fn(inputs)
        res = []
        for (xyxy, scores), off in zip(outs, offsets):
            xyxy = np.asarray(xyxy, dtype=float).copy()
            if len(xyxy):
                xyxy[:, [1, 3]] += off
            res.append((xyxy, scores))
        return res
    fn.wants_names = True
    return fn
