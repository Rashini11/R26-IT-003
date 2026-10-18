import { Maximize2, Minimize2 } from "lucide-react";

export default function MediaFitToggle({ mode = "fit", onChange, compact = false }) {
  return (
    <div className={`media-fit-toggle ${compact ? "media-fit-toggle--compact" : ""}`} role="group" aria-label="Image framing mode">
      <button
        type="button"
        className={mode === "fit" ? "active" : ""}
        onClick={() => onChange?.("fit")}
        aria-pressed={mode === "fit"}
        title="Fit: show the entire image"
      >
        <Minimize2 size={12} />
        {!compact && <span>FIT</span>}
      </button>
      <button
        type="button"
        className={mode === "fill" ? "active" : ""}
        onClick={() => onChange?.("fill")}
        aria-pressed={mode === "fill"}
        title="Fill: use the whole frame and crop overflow"
      >
        <Maximize2 size={12} />
        {!compact && <span>FILL</span>}
      </button>
    </div>
  );
}
