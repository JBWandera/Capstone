"""Ugandan street level images from Mapillary (CC BY-SA 4.0).

Used for the Uganda holdout test set: never trained on, labelled in Roboflow,
then used to measure how the model does on our own roads. Each image keeps its
GPS position, capture time, sequence and creator, so the app map can show
real Kampala detections and every image can be attributed.

Needs a free client access token: https://www.mapillary.com/dashboard/developers
Store it as the Colab secret MAPILLARY_TOKEN.
"""

from __future__ import annotations

import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import requests
from tqdm.auto import tqdm

API = "https://graph.mapillary.com/images"
FIELDS = "id,thumb_2048_url,computed_geometry,geometry,captured_at,sequence,creator,is_pano,camera_type,compass_angle"


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def tiles(bbox: list[float], step: float = 0.02) -> list[list[float]]:
    x0, y0, x1, y1 = bbox
    out = []
    nx, ny = math.ceil((x1 - x0) / step), math.ceil((y1 - y0) / step)
    for i in range(nx):
        for j in range(ny):
            out.append([x0 + i * step, y0 + j * step, min(x1, x0 + (i + 1) * step), min(y1, y0 + (j + 1) * step)])
    return out


def _query(token: str, bbox: list[float], limit: int, session: requests.Session) -> list[dict] | None:
    params = {"access_token": token, "fields": FIELDS, "bbox": ",".join(f"{v:.6f}" for v in bbox), "limit": limit}
    for attempt in range(4):
        r = session.get(API, params=params, timeout=60)
        if r.status_code == 200:
            return r.json().get("data", [])
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        return None  # too large or another error: the caller splits the tile
    return None


def search(token: str, bbox: list[float], limit: int = 2000, step: float = 0.02,
           min_step: float = 0.0025) -> list[dict]:
    """All images in bbox. A tile that errors or comes back full is split in four."""
    session = requests.Session()
    found: dict[str, dict] = {}
    queue = tiles(bbox, step)
    pbar = tqdm(total=len(queue), desc="mapillary tiles")
    while queue:
        t = queue.pop()
        data = _query(token, t, limit, session)
        size = t[2] - t[0]
        if (data is None or len(data) >= limit) and size / 2 >= min_step:
            half = size / 2
            queue.extend(tiles(t, half))
            pbar.total += 4
        for d in data or []:
            found[d["id"]] = d
        pbar.update(1)
    pbar.close()
    return list(found.values())


def _lonlat(d: dict) -> tuple[float, float] | None:
    g = d.get("computed_geometry") or d.get("geometry")
    if not g:
        return None
    lon, lat = g["coordinates"][:2]
    return lon, lat


def thin(images: list[dict], min_spacing_m: float = 15.0, max_per_sequence: int = 40,
         max_total: int | None = None, min_year: int | None = None) -> list[dict]:
    """Keep frames at least min_spacing_m apart within each sequence, so the
    test set is not 30 copies of the same stretch of road. With max_total,
    sequences are visited in turn (one frame from each, then the next round),
    so the cap spreads over as many drives and streets as possible."""
    import datetime as dt
    by_seq: dict[str, list[dict]] = defaultdict(list)
    for d in images:
        if d.get("is_pano") or not d.get("thumb_2048_url") or _lonlat(d) is None:
            continue
        if min_year and d.get("captured_at"):
            if dt.datetime.fromtimestamp(d["captured_at"] / 1000, dt.timezone.utc).year < min_year:
                continue
        by_seq[d.get("sequence", d["id"])].append(d)
    kept = []
    for seq, items in by_seq.items():
        items.sort(key=lambda d: d.get("captured_at", 0))
        last, chosen = None, []
        for d in items:
            ll = _lonlat(d)
            if last is None or haversine_m(*last, *ll) >= min_spacing_m:
                chosen.append(d)
                last = ll
        if len(chosen) > max_per_sequence:  # spread evenly along the drive
            stride = len(chosen) / max_per_sequence
            chosen = [chosen[int(i * stride)] for i in range(max_per_sequence)]
        kept.append(chosen)
    if max_total is None:
        return [d for seq in kept for d in seq]
    out, i = [], 0
    while len(out) < max_total and any(i < len(seq) for seq in kept):
        for seq in kept:
            if i < len(seq) and len(out) < max_total:
                out.append(seq[i])
        i += 1
    return out


def download(images: list[dict], out_dir: str | Path, area: str) -> Path:
    """Save JPEGs to out_dir/<area>/ and append rows to out_dir/metadata.csv."""
    out_dir = Path(out_dir)
    img_dir = out_dir / area
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.csv"
    new_file = not meta_path.exists()
    cols = ["image_id", "file_name", "area", "lon", "lat", "captured_at", "sequence",
            "creator", "compass_angle", "camera_type", "license", "url"]
    session = requests.Session()
    with open(meta_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        for d in tqdm(images, desc=f"download {area}"):
            fn = img_dir / f"{d['id']}.jpg"
            if not fn.exists():
                r = session.get(d["thumb_2048_url"], timeout=60)
                if r.status_code != 200:
                    continue
                fn.write_bytes(r.content)
            lon, lat = _lonlat(d)
            w.writerow({
                "image_id": d["id"], "file_name": f"{area}/{fn.name}", "area": area,
                "lon": lon, "lat": lat, "captured_at": d.get("captured_at"),
                "sequence": d.get("sequence"), "creator": (d.get("creator") or {}).get("username"),
                "compass_angle": d.get("compass_angle"), "camera_type": d.get("camera_type"),
                "license": "CC BY-SA 4.0", "url": f"https://www.mapillary.com/app/?pKey={d['id']}",
            })
    return meta_path


def download_manifest(token: str, manifest_csv: str | Path, out_dir: str | Path) -> Path:
    """Re-download exactly the images listed in a committed manifest.

    Thumbnail URLs from a search expire, so each image id is looked up again.
    This makes the Uganda holdout reproducible: everyone gets the same photos.
    """
    import pandas as pd
    out_dir = Path(out_dir)
    man = pd.read_csv(manifest_csv)
    session = requests.Session()
    rows = []
    for _, r in tqdm(man.iterrows(), total=len(man), desc="mapillary manifest"):
        dst = out_dir / r["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            meta = session.get(f"https://graph.mapillary.com/{r['image_id']}",
                               params={"access_token": token, "fields": "thumb_2048_url"}, timeout=60)
            if meta.status_code != 200 or "thumb_2048_url" not in meta.json():
                continue  # removed from Mapillary since the manifest was made
            img = session.get(meta.json()["thumb_2048_url"], timeout=60)
            if img.status_code != 200:
                continue
            dst.write_bytes(img.content)
        rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "metadata.csv", index=False)
    return out_dir / "metadata.csv"
