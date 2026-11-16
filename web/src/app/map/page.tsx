"use client";

import { useEffect, useRef, useState } from "react";
import "maplibre-gl/dist/maplibre-gl.css";
import type { GeoJSONSource, Map as MlMap } from "maplibre-gl";
import { Report, clearReports, deleteReport, download, listReports, toCSV, toGeoJSON } from "@/lib/reports";

const STYLE = "https://tiles.openfreemap.org/styles/dark";
const KAMPALA: [number, number] = [32.5825, 0.3476];

export default function MapPage() {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<MlMap | null>(null);
  const [reports, setReports] = useState<Report[]>([]);
  const [selected, setSelected] = useState<Report | null>(null);
  const [loaded, setLoaded] = useState(false);
  const reportsRef = useRef<Report[]>([]);
  reportsRef.current = reports;

  async function refresh() {
    setReports(await listReports());
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const ml = await import("maplibre-gl");
      ml.setWorkerUrl(new URL("/maplibre/maplibre-gl-worker.mjs", location.href).href);
      if (cancelled || !el.current || map.current) return;
      const m = new ml.Map({ container: el.current, style: STYLE, center: KAMPALA, zoom: 11.5, attributionControl: { compact: true } });
      m.addControl(new ml.NavigationControl({ showCompass: false }), "top-right");
      m.addControl(new ml.GeolocateControl({ positionOptions: { enableHighAccuracy: true } }), "top-right");
      m.on("load", () => {
        m.addSource("potholes", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
        m.addLayer({
          id: "potholes-halo",
          type: "circle",
          source: "potholes",
          paint: { "circle-radius": 14, "circle-color": "#f5a524", "circle-opacity": 0.18 },
        });
        m.addLayer({
          id: "potholes",
          type: "circle",
          source: "potholes",
          paint: {
            "circle-radius": 6,
            "circle-color": "#f5a524",
            "circle-stroke-color": "#1a1204",
            "circle-stroke-width": 1.5,
          },
        });
        m.on("click", "potholes", (e) => {
          const id = e.features?.[0]?.properties?.id as string | undefined;
          if (id) setSelected((prev) => reportsRef.current.find((r) => r.id === id) ?? prev);
        });
        m.on("mouseenter", "potholes", () => (m.getCanvas().style.cursor = "pointer"));
        m.on("mouseleave", "potholes", () => (m.getCanvas().style.cursor = ""));
        setLoaded(true);
      });
      map.current = m;
    })();
    return () => {
      cancelled = true;
      map.current?.remove();
      map.current = null;
    };
  }, []);

  useEffect(() => {
    const m = map.current;
    if (!m || !loaded) return;
    const src = m.getSource("potholes") as GeoJSONSource | undefined;
    src?.setData(JSON.parse(toGeoJSON(reports)));
    const pts = reports.filter((r) => r.lat !== null && r.lon !== null);
    if (pts.length) {
      const lons = pts.map((r) => r.lon!);
      const lats = pts.map((r) => r.lat!);
      m.fitBounds(
        [
          [Math.min(...lons), Math.min(...lats)],
          [Math.max(...lons), Math.max(...lats)],
        ],
        { padding: 60, maxZoom: 16, duration: 600 },
      );
    }
  }, [reports, loaded]);

  const located = reports.filter((r) => r.lat !== null).length;
  const stamp = new Date().toISOString().slice(0, 10);

  return (
    <div className="pt-6 space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Pothole map</h1>
          <p className="text-sm text-muted tabular">
            {reports.length} report{reports.length === 1 ? "" : "s"} on this device, {located} with location
          </p>
        </div>
        <div className="ml-auto flex gap-2 text-sm">
          <button
            disabled={!located}
            onClick={() => download(`afroviewz-potholes-${stamp}.geojson`, toGeoJSON(reports), "application/geo+json")}
            className="px-3 py-1.5 rounded-md border border-line hover:border-amber disabled:opacity-40"
          >
            GeoJSON
          </button>
          <button
            disabled={!reports.length}
            onClick={() => download(`afroviewz-potholes-${stamp}.csv`, toCSV(reports), "text/csv")}
            className="px-3 py-1.5 rounded-md border border-line hover:border-amber disabled:opacity-40"
          >
            CSV
          </button>
          <button
            disabled={!reports.length}
            onClick={async () => {
              if (confirm("Delete all reports on this device?")) {
                await clearReports();
                setSelected(null);
                refresh();
              }
            }}
            className="px-3 py-1.5 rounded-md border border-line hover:border-danger disabled:opacity-40"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="relative">
        <div ref={el} className="h-[65vh] rounded-xl overflow-hidden border border-line" />
        {selected && (
          <div className="absolute left-3 bottom-3 w-64 bg-panel border border-line rounded-xl overflow-hidden shadow-xl">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={selected.thumb} alt="Detected pothole" className="w-full aspect-video object-cover" />
            <div className="p-3 text-xs space-y-1 tabular">
              <div>
                <b className="text-amber">{Math.round(selected.score * 100)}%</b> confidence, from {selected.source}
              </div>
              <div className="text-muted">{new Date(selected.createdAt).toLocaleString()}</div>
              {selected.accuracyM !== null && <div className="text-muted">GPS accuracy ±{selected.accuracyM} m</div>}
              <div className="flex gap-2 pt-1">
                <button onClick={() => setSelected(null)} className="text-muted hover:text-ink">
                  Close
                </button>
                <button
                  onClick={async () => {
                    await deleteReport(selected.id);
                    setSelected(null);
                    refresh();
                  }}
                  className="ml-auto text-danger"
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {reports.length === 0 && (
        <p className="text-sm text-muted">
          No reports yet. Use the live camera with location on, or save detections from a phone photo that has GPS.
        </p>
      )}
    </div>
  );
}
