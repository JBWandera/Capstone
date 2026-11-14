import type { MetadataRoute } from "next";

export const dynamic = "force-static";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "AfroViewz",
    short_name: "AfroViewz",
    description: "Pothole detection on your phone, offline.",
    start_url: "/",
    display: "standalone",
    orientation: "any",
    background_color: "#111214",
    theme_color: "#111214",
    icons: [
      { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
