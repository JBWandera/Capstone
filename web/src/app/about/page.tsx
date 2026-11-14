import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = { title: "How AfroViewz works" };

const steps = [
  {
    n: "01",
    title: "Data acquisition",
    body: "Four sources: the Stellenbosch dashcam set (CC0, South Africa), two Roboflow pothole sets, and street imagery from Mapillary across Kampala, Entebbe, Masaka, Mbale and Lira kept apart as a Uganda test set.",
  },
  {
    n: "02",
    title: "Cleaning",
    body: "One class, one format (COCO). Boxes clipped to the image, slivers under 4 px dropped, duplicate boxes merged, augmented copies collapsed to one original, near duplicate images found with perceptual hashing and a pixel check.",
  },
  {
    n: "03",
    title: "Leak free splits",
    body: "Images from the same drive or the same original photo always land in the same split. 80 / 10 / 10 by group, balanced on both images and boxes, frozen in a manifest so every result is reproducible.",
  },
  {
    n: "04",
    title: "Exploratory analysis",
    body: "Most dashcam potholes are tiny: 98% are small at the model's input size, with a median of 9 px. That one finding drove the experiments on input resolution and road cropping.",
  },
  {
    n: "05",
    title: "Modelling",
    body: "RF-DETR (ICLR 2026) in Nano, Small and Medium sizes against a YOLO26 baseline, trained on free cloud GPUs with early stopping on the validation set.",
  },
  {
    n: "06",
    title: "Evaluation",
    body: "COCO mAP on the untouched test split with 95% bootstrap intervals, per source and per pothole size, with the operating threshold picked on validation only.",
  },
];

const results = [
  { model: "RF-DETR Small, road band crop", map50: "0.694", ci: "0.655 to 0.742", ms: "8.5" },
  { model: "RF-DETR Small, 704 px input", map50: "0.667", ci: "0.624 to 0.715", ms: "13.9" },
  { model: "RF-DETR Medium", map50: "0.628", ci: "0.584 to 0.679", ms: "9.7" },
  { model: "YOLO26 Small", map50: "0.600", ci: "0.556 to 0.648", ms: "6.9" },
  { model: "RF-DETR Small", map50: "0.578", ci: "0.542 to 0.627", ms: "8.0" },
  { model: "RF-DETR Nano (in this app)", map50: "0.522", ci: "0.476 to 0.575", ms: "5.5" },
];

export default function About() {
  return (
    <div className="pt-8 space-y-12">
      <section className="space-y-3 max-w-3xl">
        <h1 className="text-3xl font-semibold tracking-tight">How it works</h1>
        <p className="text-muted">
          AfroViewz is a capstone project rebuilt as a working product. Every number below comes from a test set the
          model never saw during training or tuning.
        </p>
      </section>

      <section className="grid sm:grid-cols-2 gap-4">
        {steps.map((s) => (
          <div key={s.n} className="bg-panel border border-line rounded-xl p-5">
            <div className="text-amber text-sm font-mono">{s.n}</div>
            <h2 className="font-semibold mt-1">{s.title}</h2>
            <p className="text-sm text-muted mt-2 leading-relaxed">{s.body}</p>
          </div>
        ))}
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Test results</h2>
        <p className="text-sm text-muted">592 test images, 846 potholes. Speed is on an NVIDIA L4 GPU.</p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm tabular">
            <thead className="text-muted text-left border-b border-line">
              <tr>
                <th className="py-2 pr-4 font-normal">Model</th>
                <th className="py-2 pr-4 font-normal">mAP at IoU 0.5</th>
                <th className="py-2 pr-4 font-normal">95% interval</th>
                <th className="py-2 font-normal">ms per image</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.model} className="border-b border-line/60">
                  <td className="py-2 pr-4">{r.model}</td>
                  <td className="py-2 pr-4 text-amber">{r.map50}</td>
                  <td className="py-2 pr-4 text-muted">{r.ci}</td>
                  <td className="py-2">{r.ms}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-muted">
          Cropping dashcam frames to the band of road where potholes appear was the biggest single gain: dashcam
          accuracy rose from 0.46 to 0.66 at no cost in speed. The app ships the Nano model because it runs smoothly
          on a phone; a Nano trained with the road band crop is next.
        </p>
      </section>

      <section className="space-y-2">
        <h2 className="text-xl font-semibold">Privacy</h2>
        <p className="text-sm text-muted max-w-3xl">
          The model is downloaded once and runs on your device. Photos, video and camera frames are never uploaded.
          Saved reports stay in this browser until you export them.
        </p>
      </section>

      <Link href="/" className="inline-block px-5 py-2.5 rounded-lg bg-amber text-amber-ink font-semibold">
        Try it
      </Link>
    </div>
  );
}
