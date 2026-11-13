"""Download a Supervisely .tar and downscale it on the fly.

The Stellenbosch archive is about 23 GB of 3680x2760 JPEGs. Downloading it,
unpacking it and only then shrinking it needs about 46 GB of free disk, more
than a free Colab runtime spares. Instead the tar is read as a stream: each
image is shrunk in memory and each annotation is scaled by the same factor, so
the disk only ever holds the small copy (about 2 GB at max_side 1600).
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import requests
from PIL import Image
from tqdm.auto import tqdm


def _scale(size: dict, max_side: int) -> float:
    return min(1.0, max_side / max(size["width"], size["height"]))


def _scale_ann(data: dict, max_side: int) -> dict:
    s = _scale(data["size"], max_side)
    if s < 1.0:
        data["size"] = {"width": round(data["size"]["width"] * s), "height": round(data["size"]["height"] * s)}
        for obj in data.get("objects", []):
            pts = obj.get("points", {})
            for key in ("exterior", "interior"):
                if key == "exterior":
                    pts[key] = [[x * s, y * s] for x, y in pts.get(key, [])]
                else:
                    pts[key] = [[[x * s, y * s] for x, y in ring] for ring in pts.get(key, [])]
    data["scale_from_original"] = s
    return data


def stream_supervisely_tar(url: str, dst: str | Path, max_side: int = 1600, quality: int = 90,
                           chunk: int = 1 << 20) -> dict:
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    done_flag = dst / ".complete"
    if done_flag.exists():
        return json.loads(done_flag.read_text())
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0)) or None
    raw = r.raw
    raw.decode_content = True
    counts = {"images": 0, "annotations": 0, "other": 0, "skipped_existing": 0}
    pbar = tqdm(total=total, unit="B", unit_scale=True, desc="stream")

    class _Counting(io.RawIOBase):
        def readable(self):
            return True

        def readinto(self, b):
            data = raw.read(len(b))
            n = len(data)
            b[:n] = data
            pbar.update(n)
            return n

    stream = io.BufferedReader(_Counting(), buffer_size=chunk)
    with tarfile.open(fileobj=stream, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name)
            out = dst / name
            f = tar.extractfile(member)
            if f is None:
                continue
            payload = f.read()
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                counts["skipped_existing"] += 1
                continue
            if name.parent.name == "img":
                with Image.open(io.BytesIO(payload)) as im:
                    # let the JPEG decoder skip detail we would throw away anyway
                    # (DCT scaling): about 3x faster on 10 MP frames
                    s0 = min(1.0, max_side / max(im.size))
                    im.draft("RGB", (int(im.width * s0), int(im.height * s0)))
                    im = im.convert("RGB")
                    s = min(1.0, max_side / max(im.size))
                    if s < 1.0:
                        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
                    im.save(out, "JPEG", quality=quality)
                counts["images"] += 1
            elif name.parent.name == "ann" and name.suffix == ".json":
                data = _scale_ann(json.loads(payload), max_side)
                out.write_text(json.dumps(data))
                counts["annotations"] += 1
            else:
                out.write_bytes(payload)
                counts["other"] += 1
            if (counts["images"] + counts["annotations"]) % 500 == 0:
                pbar.set_postfix(images=counts["images"], anns=counts["annotations"])
    pbar.close()
    counts["max_side"] = max_side
    done_flag.write_text(json.dumps(counts))
    return counts
