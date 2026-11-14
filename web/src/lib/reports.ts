// Pothole reports kept on the device (IndexedDB), exportable as GeoJSON or CSV.
// Nothing leaves the phone unless the user exports or shares it.

export type Report = {
  id: string;
  createdAt: string; // ISO time
  lat: number | null;
  lon: number | null;
  accuracyM: number | null;
  score: number;
  source: "photo" | "video" | "live";
  thumb: string; // small JPEG data URL of the crop
  note?: string;
};

const DB = "afroviewz";
const STORE = "reports";

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE, { keyPath: "id" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const r = fn(db.transaction(STORE, mode).objectStore(STORE));
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

export const addReport = (r: Report) => tx("readwrite", (s) => s.put(r));
export const deleteReport = (id: string) => tx("readwrite", (s) => s.delete(id));
export const clearReports = () => tx("readwrite", (s) => s.clear());
export async function listReports(): Promise<Report[]> {
  const all = await tx<Report[]>("readonly", (s) => s.getAll() as IDBRequest<Report[]>);
  return all.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export function newId(): string {
  return crypto.randomUUID();
}

/** Last known position, refreshed by watchPosition while the camera runs. */
let lastFix: GeolocationPosition | null = null;
let watchId: number | null = null;

export function startLocation(onFix?: (p: GeolocationPosition) => void): void {
  if (!("geolocation" in navigator) || watchId !== null) return;
  watchId = navigator.geolocation.watchPosition(
    (p) => {
      lastFix = p;
      onFix?.(p);
    },
    () => {},
    { enableHighAccuracy: true, maximumAge: 5000, timeout: 20000 },
  );
}

export function stopLocation(): void {
  if (watchId !== null) navigator.geolocation.clearWatch(watchId);
  watchId = null;
}

export function currentFix(): { lat: number | null; lon: number | null; accuracyM: number | null } {
  if (!lastFix) return { lat: null, lon: null, accuracyM: null };
  const c = lastFix.coords;
  return { lat: c.latitude, lon: c.longitude, accuracyM: Math.round(c.accuracy) };
}

/** Crop a detection to a small JPEG thumbnail. */
export function cropThumb(
  src: CanvasImageSource,
  b: { x1: number; y1: number; x2: number; y2: number },
  size = 160,
): string {
  const pad = 0.4;
  const w = b.x2 - b.x1;
  const h = b.y2 - b.y1;
  const sx = Math.max(0, b.x1 - w * pad);
  const sy = Math.max(0, b.y1 - h * pad);
  const sw = w * (1 + 2 * pad);
  const sh = h * (1 + 2 * pad);
  const c = document.createElement("canvas");
  const f = size / Math.max(sw, sh);
  c.width = Math.max(1, Math.round(sw * f));
  c.height = Math.max(1, Math.round(sh * f));
  c.getContext("2d")!.drawImage(src, sx, sy, sw, sh, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.7);
}

export function toGeoJSON(rs: Report[]): string {
  return JSON.stringify(
    {
      type: "FeatureCollection",
      features: rs
        .filter((r) => r.lat !== null && r.lon !== null)
        .map((r) => ({
          type: "Feature",
          geometry: { type: "Point", coordinates: [r.lon, r.lat] },
          properties: {
            id: r.id,
            createdAt: r.createdAt,
            score: Number(r.score.toFixed(3)),
            source: r.source,
            accuracyM: r.accuracyM,
            note: r.note ?? "",
          },
        })),
    },
    null,
    2,
  );
}

export function toCSV(rs: Report[]): string {
  const head = "id,created_at,lat,lon,accuracy_m,score,source,note";
  const esc = (s: string) => `"${s.replace(/"/g, '""')}"`;
  const rows = rs.map((r) =>
    [r.id, r.createdAt, r.lat ?? "", r.lon ?? "", r.accuracyM ?? "", r.score.toFixed(3), r.source, esc(r.note ?? "")].join(","),
  );
  return [head, ...rows].join("\n");
}

export function download(name: string, text: string, type: string): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
