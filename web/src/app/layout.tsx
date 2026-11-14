import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";
import { RegisterSW } from "@/components/RegisterSW";

export const metadata: Metadata = {
  title: "AfroViewz: pothole detection",
  description:
    "Detect potholes in photos, video and live camera, on the device. Built and evaluated on African road imagery.",
  applicationName: "AfroViewz",
  appleWebApp: { capable: true, title: "AfroViewz", statusBarStyle: "black-translucent" },
};

export const viewport: Viewport = {
  themeColor: "#111214",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">
        <RegisterSW />
        <Nav />
        <main className="flex-1 w-full max-w-5xl mx-auto px-4 pb-16">{children}</main>
        <footer className="border-t border-line text-xs text-muted py-6 px-4 text-center">
          AfroViewz runs the model on your device. Images and video never leave it.
        </footer>
      </body>
    </html>
  );
}
