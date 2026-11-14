// Small IoU tracker so a pothole seen across many video frames is counted once.
// A track is confirmed after `minHits` matched frames and dropped after
// `maxMissed` frames without a match.

import type { Box } from "./detector";

export type Track = {
  id: number;
  box: Box;
  hits: number;
  missed: number;
  confirmed: boolean;
  bestScore: number;
  firstSeenMs: number;
};

function iou(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

export class IouTracker {
  private tracks: Track[] = [];
  private nextId = 1;
  confirmedCount = 0;

  constructor(
    private iouThreshold = 0.2,
    private minHits = 3,
    private maxMissed = 8,
  ) {}

  reset() {
    this.tracks = [];
    this.nextId = 1;
    this.confirmedCount = 0;
  }

  /** Returns tracks that are newly confirmed in this frame. */
  update(boxes: Box[], tMs: number): { active: Track[]; fresh: Track[] } {
    const fresh: Track[] = [];
    const used = new Set<number>();
    // Greedy matching, highest IoU first.
    const pairs: { t: number; d: number; v: number }[] = [];
    this.tracks.forEach((tr, ti) =>
      boxes.forEach((b, di) => {
        const v = iou(tr.box, b);
        if (v >= this.iouThreshold) pairs.push({ t: ti, d: di, v });
      }),
    );
    pairs.sort((a, b) => b.v - a.v);
    const matchedTracks = new Set<number>();
    for (const p of pairs) {
      if (matchedTracks.has(p.t) || used.has(p.d)) continue;
      matchedTracks.add(p.t);
      used.add(p.d);
      const tr = this.tracks[p.t];
      tr.box = boxes[p.d];
      tr.hits += 1;
      tr.missed = 0;
      tr.bestScore = Math.max(tr.bestScore, boxes[p.d].score);
      if (!tr.confirmed && tr.hits >= this.minHits) {
        tr.confirmed = true;
        this.confirmedCount += 1;
        fresh.push(tr);
      }
    }
    this.tracks.forEach((tr, ti) => {
      if (!matchedTracks.has(ti)) tr.missed += 1;
    });
    this.tracks = this.tracks.filter((t) => t.missed <= this.maxMissed);
    boxes.forEach((b, di) => {
      if (used.has(di)) return;
      this.tracks.push({
        id: this.nextId++,
        box: b,
        hits: 1,
        missed: 0,
        confirmed: this.minHits <= 1,
        bestScore: b.score,
        firstSeenMs: tMs,
      });
    });
    return { active: this.tracks.filter((t) => t.missed === 0), fresh };
  }
}
