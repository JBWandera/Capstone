"use client";

import { useEffect, useState } from "react";
import { PotholeDetector } from "./detector";

type State =
  | { status: "loading"; progress: number }
  | { status: "ready"; detector: PotholeDetector }
  | { status: "error"; message: string };

let shared: Promise<PotholeDetector> | null = null;
const listeners = new Set<(f: number) => void>();

function getDetector(): Promise<PotholeDetector> {
  shared ??= PotholeDetector.create({ onProgress: (f) => listeners.forEach((l) => l(f)) }).catch((e) => {
    shared = null;
    throw e;
  });
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
      .then((d) => alive && setState({ status: "ready", detector: d }))
      .catch((e) => alive && setState({ status: "error", message: String(e?.message ?? e) }));
    return () => {
      alive = false;
      listeners.delete(onP);
    };
  }, []);
  return state;
}
