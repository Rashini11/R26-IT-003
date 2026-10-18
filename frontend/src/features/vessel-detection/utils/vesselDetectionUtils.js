export const VESSEL_HISTORY_PAGE_SIZE = 10;

// Hide flag detections and show confidence values as percentages.
export function filterVesselDetections(detections = []) {
  const isFlag = (detection) =>
    detection?.detection_type === "flag" ||
    detection?.label?.trim().toLowerCase() === "sl flag";
  const confidenceOf = (detection) => Number(detection?.confidence) || 0;
  return detections
    .filter((detection) => !isFlag(detection))
    .map((detection) => {
      const confidence = confidenceOf(detection);
      return {
        ...detection,
        confidence: Number((confidence <= 1 ? confidence * 100 : confidence).toFixed(1)),
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
