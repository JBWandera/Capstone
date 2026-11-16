"use client";

import { useEffect, useState } from "react";
import { PotholeDetector, liteMode, setLiteMode } from "./detector";

type State =
  | { status: "loading"; progress: number }
  | { status: "ready"; detector: PotholeDetector; recovered: boolean }
  | { status: "error"; message: string };

const LIVE_KEY = "afroviewz-live-running";

/** Set while the live camera is running on screen. If the page starts again
 *  and the flag is still there, the last live session crashed the tab. */
export function markLiveRunning(on: boolean): void {
  try {
    if (on) localStorage.setItem(LIVE_KEY, String(Date.now()));
    else localStorage.removeItem(LIVE_KEY);
  } catch {}
}

let recovered = false;
function checkCrash(): void {
  try {
    if (localStorage.getItem(LIVE_KEY) && !liteMode()) {
      setLiteMode(true);
      recovered = true;
    }
    localStorage.removeItem(LIVE_KEY);
  } catch {}
}

let shared: Promise<PotholeDetector> | null = null;
const listeners = new Set<(f: number) => void>();

function getDetector(): Promise<PotholeDetector> {
  if (!shared) {
    checkCrash();
    shared = PotholeDetector.create({ onProgress: (f) => listeners.forEach((l) => l(f)) }).catch((e) => {
      shared = null;
      throw e;
    });
  }
  return shared;
}

/** One model for the whole app, loaded on first use and cached offline. */
export function useDetector(): State {
  const [state, setState] = useState<State>({ status: "loading", progress: 0 });
  useEffect(() => {
    let alive = true;
    const onP = (f: number) => alive && setState({ status: "loading", progress: f });
    listeners.add(onP);
    getDetector()
      .then((d) => alive && setState({ status: "ready", detector: d, recovered }))
      .catch((e) => alive && setState({ status: "error", message: String(e?.message ?? e) }));
    return () => {
      alive = false;
      listeners.delete(onP);
    };
  }, []);
  return state;
}
