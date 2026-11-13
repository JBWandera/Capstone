"""Near duplicate detection with perceptual hashes.

Why: public pothole sets are full of repeats (the same photo uploaded to
several Roboflow projects, augmented copies, consecutive video frames). If a
near copy of a test image sits in train, test scores are inflated. Images in
one duplicate cluster are given one shared group, so the splitter keeps them
together, and exact repeats across sources can be dropped.

Method: 64 bit pHash per image. Two hashes within Hamming distance d <= 7
must agree on at least one of eight 8 bit bands (pigeonhole), so candidate
pairs are found by bucketing on bands, then confirmed by the full distance.

pHash alone over-merges dashcam footage: the car bonnet and sky fill most of
every frame, so frames of different roads can hash alike and chain into
clusters of hundreds. Every candidate pair is therefore also checked on
pixels: the mean absolute difference (MAD, 0 to 255) of 64x64 grey
thumbnails must be small (default 6) for a near duplicate, and almost zero
(default 2) for an exact repeat.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image
from tqdm.auto import tqdm


def phash_images(coco: dict, images_root: str | Path, with_thumbs: bool = False):
    """pHash of every image; with_thumbs also returns 64x64 grey thumbnails."""
    import numpy as np
    root = Path(images_root)
    out, thumbs = {}, {}
    for im in tqdm(coco["images"], desc="phash"):
        with Image.open(root / im["file_name"]) as pil:
            out[im["id"]] = int(str(imagehash.phash(pil)), 16)
            if with_thumbs:
                thumbs[im["id"]] = np.asarray(pil.convert("L").resize((64, 64)), np.float32)
    return (out, thumbs) if with_thumbs else out


def pixel_mad(thumbs: dict, a: int, b: int) -> float:
    import numpy as np
    return float(np.abs(thumbs[a] - thumbs[b]).mean())


class _UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_pairs(hashes: dict[int, int], max_distance: int = 6) -> list[tuple[int, int, int]]:
    if max_distance > 7:
        raise ValueError("band bucketing guarantees recall only for max_distance <= 7")
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for img_id, h in hashes.items():
        for band in range(8):
            buckets[(band, (h >> (8 * band)) & 0xFF)].append(img_id)
    seen, pairs = set(), []
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = sorted((ids[i], ids[j]))
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                d = bin(hashes[a] ^ hashes[b]).count("1")
                if d <= max_distance:
                    pairs.append((a, b, d))
    return pairs


def cluster(hashes: dict[int, int], max_distance: int = 6, thumbs: dict | None = None,
            max_mad: float = 6.0) -> dict[int, int]:
    """Map image id -> cluster id (the smallest image id in its cluster)."""
    uf = _UnionFind()
    for img_id in hashes:
        uf.find(img_id)
    for a, b, _ in near_duplicate_pairs(hashes, max_distance):
        if thumbs is None or pixel_mad(thumbs, a, b) <= max_mad:
            uf.union(a, b)
    return {img_id: uf.find(img_id) for img_id in hashes}


def apply_clusters_to_groups(coco: dict, clusters: dict[int, int]) -> int:
    """Merge split groups that are linked by near duplicates. Returns groups merged."""
    uf = _UnionFind()
    group_ids: dict[str, int] = {}
    for im in coco["images"]:
        group_ids.setdefault(im["group"], len(group_ids))
    rep_group: dict[int, int] = {}
    for im in coco["images"]:
        g = group_ids[im["group"]]
        c = clusters.get(im["id"], im["id"])
        if c in rep_group:
            uf.union(rep_group[c], g)
        else:
            rep_group[c] = g
            uf.find(g)
    names = {v: k for k, v in group_ids.items()}
    merged = 0
    for im in coco["images"]:
        g = group_ids[im["group"]]
        root = uf.find(g)
        if root != g:
            im["group"] = names[root]
    merged = len(group_ids) - len({uf.find(g) for g in group_ids.values()})
    return merged


def exact_duplicates_to_drop(coco: dict, clusters: dict[int, int], hashes: dict[int, int],
                             thumbs: dict | None = None, max_mad: float = 2.0) -> set[int]:
    """Within a cluster, images with an identical hash (and, given thumbs, near
    identical pixels) are true repeats: keep the one with the most boxes (ties:
    lowest id) and drop the rest."""
    n_boxes = defaultdict(int)
    for a in coco["annotations"]:
        n_boxes[a["image_id"]] += 1
    by_key: dict[tuple[int, int], list[int]] = defaultdict(list)
    for img_id, c in clusters.items():
        by_key[(c, hashes[img_id])].append(img_id)
    drop = set()
    for ids in by_key.values():
        if len(ids) > 1:
            keep = sorted(ids, key=lambda i: (-n_boxes[i], i))[0]
            drop.update(i for i in ids if i != keep and (thumbs is None or pixel_mad(thumbs, i, keep) <= max_mad))
    return drop
