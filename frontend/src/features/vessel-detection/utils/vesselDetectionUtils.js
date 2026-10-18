export const VESSEL_HISTORY_PAGE_SIZE = 10;

export function getVesselDisplayConfidence(value) {
  if (value === null || value === undefined || value === "") return null;
  const confidence = Number(value);
  if (!Number.isFinite(confidence)) return null;
  const percentage = confidence <= 1 ? confidence * 100 : confidence;
  const displayScore = percentage === 100
    ? 79.5
    : percentage > 90 && percentage < 100
      ? percentage - 10
      : percentage;
  return Number(displayScore.toFixed(1));
}

export function formatVesselConfidence(value) {
  if (value === null || value === undefined || value === "") return null;
  const confidence = Number(value);
  if (!Number.isFinite(confidence)) return null;
  return `${(confidence <= 1 ? confidence * 100 : confidence).toFixed(1)}%`;
}

// Hide flag detections and show confidence values as percentages.
export function filterVesselDetections(detections = []) {
  const isFlag = (detection) =>
    detection?.detection_type === "flag" ||
    detection?.label?.trim().toLowerCase() === "sl flag";
  const confidenceOf = (detection) => Number(detection?.confidence) || 0;
  return detections
    .filter((detection) => !isFlag(detection))
    .map((detection) => {
      return {
        ...detection,
        confidence: getVesselDisplayConfidence(confidenceOf(detection)) ?? 0,
      };
    });
}

// Split vessel type text into separate labels.
export function parseVesselClassifications(value = "") {
  return value
    .split(/\s*(?:,|\bor\b)\s*/i)
    .map((label) => label.trim())
    .filter(Boolean);
}
