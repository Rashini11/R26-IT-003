// Draw vessel detection boxes over the source image.
export default function BoatDetectionOverlay({
  result,
  color,
  mediaFit = "fit",
  detections: cleanDetections = [],
}) {
  const [imageWidth, imageHeight] = result?.image_size || [];
  const detections = cleanDetections.filter((detection) => (
    Array.isArray(detection.box) && detection.box.length === 4
  ));

  if (!imageWidth || !imageHeight || detections.length === 0) return null;

  return (
    <svg
      className="boat-detection-overlay"
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio={mediaFit === "fill" ? "xMidYMid slice" : "xMidYMid meet"}
      aria-label="Boat detection labels"
    >
      {detections.map((detection, index) => {
        const [x1, y1, x2, y2] = detection.box;
        const boxWidth = Math.max(1, x2 - x1);
        const boxHeight = Math.max(1, y2 - y1);
        const label = detection.label;
        const labelWidth = Math.min(imageWidth - x1, Math.max(120, label.length * 8));
        const labelY = Math.max(18, y1);
        const isLocal = detection.label?.startsWith("Local ");
        const boxColor = isLocal ? "#00ffb3" : color;

        return (
          <g key={`${detection.label}-${index}`}>
            <rect
              x={x1}
              y={y1}
              width={boxWidth}
              height={boxHeight}
              fill="none"
              stroke={boxColor}
              strokeWidth={Math.max(2, imageWidth / 160)}
            />
            <rect
              x={x1}
              y={labelY - 18}
              width={labelWidth}
              height="18"
              fill={boxColor}
              opacity="0.92"
            />
            <text
              x={x1 + 5}
              y={labelY - 5}
              fill="#041018"
              fontSize={Math.max(10, imageWidth / 32)}
              fontWeight="700"
              fontFamily="monospace"
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
