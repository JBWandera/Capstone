import type { NextConfig } from "next";

// Static export: the whole app is client side (the model runs in the browser),
// so `out/` can be served by Vercel and wrapped by Capacitor for Android.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
