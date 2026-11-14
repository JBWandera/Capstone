import type { CapacitorConfig } from "@capacitor/cli";

// The Android app wraps the static export in out/. Everything, including the
// model, ships inside the APK, so it works with no network at all.
const config: CapacitorConfig = {
  appId: "app.afroviewz",
  appName: "AfroViewz",
  webDir: "out",
  backgroundColor: "#111214",
  android: { allowMixedContent: false },
};

export default config;
