import type { Box } from "./detector";

/** Draw boxes on an overlay canvas sized to the displayed media. */
export function drawBoxes(
  canvas: HTMLCanvasElement,
  boxes: (Box & { label?: string })[],
  srcW: number,
  srcH: number,
): void {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  // object-fit: contain mapping
  const s = Math.min(cw / srcW, ch / srcH);
  const ox = (cw - srcW * s) / 2;
  const oy = (ch - srcH * s) / 2;
  ctx.lineWidth = 2.5;
  ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
  for (const b of boxes) {
    const x = ox + b.x1 * s;
    const y = oy + b.y1 * s;
    const w = (b.x2 - b.x1) * s;
    const h = (b.y2 - b.y1) * s;
    ctx.strokeStyle = "#f5a524";
    ctx.fillStyle = "rgba(245,165,36,0.14)";
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 4);
    ctx.fill();
    ctx.stroke();
    const text = b.label ?? `${Math.round(b.score * 100)}%`;
    const tw = ctx.measureText(text).width + 10;
    const ty = y > 20 ? y - 20 : y + h + 2;
    ctx.fillStyle = "#f5a524";
    ctx.beginPath();
    ctx.roundRect(x, ty, tw, 18, 4);
    ctx.fill();
    ctx.fillStyle = "#1a1204";
    ctx.fillText(text, x + 5, ty + 13);
  }
}

/** Burn boxes into a copy of the image for download or sharing. */
export function renderAnnotated(src: CanvasImageSource, w: number, h: number, boxes: Box[]): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(src, 0, 0, w, h);
  const lw = Math.max(2, Math.round(Math.max(w, h) / 400));
  ctx.lineWidth = lw;
  ctx.font = `600 ${lw * 6}px ui-sans-serif, system-ui, sans-serif`;
  for (const b of boxes) {
    ctx.strokeStyle = "#f5a524";
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    const t = `pothole ${Math.round(b.score * 100)}%`;
    const tw = ctx.measureText(t).width + lw * 4;
    ctx.fillStyle = "#f5a524";
    ctx.fillRect(b.x1, Math.max(0, b.y1 - lw * 9), tw, lw * 9);
    ctx.fillStyle = "#1a1204";
    ctx.fillText(t, b.x1 + lw * 2, Math.max(lw * 7, b.y1 - lw * 2));
  }
  return c;
}
