export const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
export function eventKey(event) {
  return `${event.recording_id || event.simulation_id}:${event.frame_number}`;
}
export function mergeEvents(previous, incoming, limit = 10000) {
  const map = new Map(previous.map((event) => [eventKey(event), event]));
  incoming.forEach((event) => map.set(eventKey(event), event));
  return [...map.values()].sort((a, b) => a.frame_number - b.frame_number).slice(-limit);
}
export function sameSegment(a, b) {
  if (!a || !b || (a.recording_id || a.simulation_id) !== (b.recording_id || b.simulation_id)) return false;
  if (isNumber(a.cycle_index) && isNumber(b.cycle_index)) return a.cycle_index === b.cycle_index;
  return Date.parse(b.simulated_timestamp) >= Date.parse(a.simulated_timestamp);
}
export function playbackDelay(a, b, speed = 1) {
  let seconds;
  if (sameSegment(a, b)) seconds = (Date.parse(b.simulated_timestamp) - Date.parse(a.simulated_timestamp)) / 1000;
  if (!isNumber(seconds) || seconds <= 0) seconds = b?.simulated_interval_seconds || 10;
  return Math.max(16, seconds * 1000 / speed);
}
export function validPosition(p) {
  return p && isNumber(p.latitude) && isNumber(p.longitude)
    && Math.abs(p.latitude) <= 90 && Math.abs(p.longitude) <= 180;
}
export function localPosition(p, origin) {
  if (!validPosition(p) || !validPosition(origin)) return null;
  // Same equirectangular local approximation as backend geo.py, with longitude wrapping.
  let lon = p.longitude - origin.longitude;
  lon = ((lon + 540) % 360) - 180;
  const rad = Math.PI / 180;
  return { x: 6371000 * lon * rad * Math.cos(origin.latitude * rad),
    y: 6371000 * (p.latitude - origin.latitude) * rad };
}
export function trailSegments(events, selected, role, origin) {
  const segments = [];
  let current = [];
  let previous;
  for (const event of events) {
    if (event.frame_number > selected.frame_number) break;
    const state = event[role]?.current_state;
    const point = localPosition(state, origin);
    const gap = previous && event.frame_number !== previous.frame_number + 1;
    if (!sameSegment(event, selected) || !point || gap || (previous && !sameSegment(previous, event))) {
      if (current.length) segments.push(current);
      current = [];
    }
    if (sameSegment(event, selected) && point) current.push({ ...point, state });
    previous = event;
  }
  if (current.length) segments.push(current);
  return segments;
}
export function elapsedLabel(seconds) {
  if (!isNumber(seconds)) return "—";
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}
