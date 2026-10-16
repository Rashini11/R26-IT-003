import { useMemo, useState } from "react";
import { elapsedLabel, isNumber, localPosition, sameSegment, trailSegments, validPosition } from "./replayMath";
import "./trajectory.css";

const ROLES = [
  { key: "own_vessel", label: "Own vessel", color: "#56d6ed" },
  { key: "target_vessel", label: "Target vessel", color: "#ffbd74" },
];
function number(value, digits = 2) { return isNumber(value) ? value.toFixed(digits) : "—"; }

export default function TrajectoryPanel({ replay }) {
  const [selectedRole, setSelectedRole] = useState("own_vessel");
  const { event, events, index } = replay;
  const geometry = useMemo(() => {
    if (!event) return null;
    const segmentEvents = events.filter((e) => e.frame_number <= event.frame_number && sameSegment(e, event));
    const first = segmentEvents.find((e) => validPosition(e.own_vessel?.current_state)) || event;
    const origin = first.own_vessel?.current_state;
    if (!validPosition(origin)) return null;
    const roles = ROLES.map((role) => {
      const vessel = event[role.key];
      const prediction = vessel?.motion_prediction;
      return { ...role, vessel,
        current: localPosition(vessel?.current_state, origin),
        forecast: localPosition(prediction && {
          latitude: prediction.predicted_latitude, longitude: prediction.predicted_longitude,
        }, origin),
        trails: trailSegments(events, event, role.key, origin),
      };
    });
    const points = roles.flatMap((r) => [...r.trails.flat(), r.current, r.forecast]).filter(Boolean);
    if (!points.length) return null;
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const spanX = Math.max(200, maxX - minX), spanY = Math.max(200, maxY - minY);
    const scale = Math.min(760 / (spanX * 1.2), 340 / (spanY * 1.2));
    const centreX = (minX + maxX) / 2, centreY = (minY + maxY) / 2;
    return { origin, roles, scale,
      screen: (p) => ({ x: 460 + (p.x - centreX) * scale, y: 230 - (p.y - centreY) * scale }),
      worldX: (screenX) => centreX + (screenX - 460) / scale,
      worldY: (screenY) => centreY - (screenY - 230) / scale,
    };
  }, [events, event]);
  const selected = event?.[selectedRole];
  const state = selected?.current_state;
  const forecast = selected?.motion_prediction;
  const firstElapsed = events[0]?.replay_elapsed_seconds;
  const lastElapsed = events.at(-1)?.replay_elapsed_seconds;
  const duration = isNumber(firstElapsed) && isNumber(lastElapsed) ? lastElapsed - firstElapsed : null;
  const elapsed = isNumber(event?.replay_elapsed_seconds) && isNumber(firstElapsed) ? event.replay_elapsed_seconds - firstElapsed : null;
  const targetTime = forecast?.forecast_target_timestamp || null;
  return <section className="sim-panel trajectory-panel" aria-label="Vessel trajectory and replay">
    <div className="sim-panel-head"><span>VESSEL TRAJECTORIES & ENCOUNTER REPLAY</span><span role="status">{replay.playbackState}</span></div>
    <div className="trajectory-run-row">
      <label>Recording <select aria-label="Recording" value={replay.selectedRun} onChange={(e) => replay.setSelectedRun(e.target.value)}>
        <option value="">Current / most recent simulation</option>
        {replay.runs.map((run) => <option key={run.recording_id} value={run.recording_id}>
          {run.scenario_mode} · {run.started_at} · {run.recording_id.slice(0, 8)}
        </option>)}
      </select></label>
      <span>{replay.loading ? "Retrieving recorded events…" : `${events.length} events available`}</span>
    </div>
    <div className="trajectory-layout">
      <div>
        {geometry ? <svg className="trajectory-chart" viewBox="0 0 900 480" role="img" aria-label="Vessel movements in local east and north metres">
          <title>Local encounter plot with equal distance scale on both axes</title>
          <desc>Solid tracks show recorded movement up to the selected event. Hollow diamonds are GRU forecast endpoints. Dashed connectors are illustrative.</desc>
          <rect x="65" y="35" width="790" height="390" rx="8" fill="var(--trajectory-bg)" />
          {[0, 1, 2, 3, 4].map((n) => {
            const x = 85 + n * 185, y = 55 + n * 85;
            return <g key={n} className="trajectory-grid">
              <line x1={x} y1="45" x2={x} y2="405" /><line x1="80" y1={y} x2="840" y2={y} />
              <text x={x} y="445" textAnchor="middle">{Math.round(geometry.worldX(x))}</text>
              <text x="62" y={y + 4} textAnchor="end">{Math.round(geometry.worldY(y))}</text>
            </g>;
          })}
          <text x="460" y="472" textAnchor="middle" className="trajectory-axis">East (m)</text>
          <text x="18" y="230" textAnchor="middle" transform="rotate(-90 18 230)" className="trajectory-axis">North (m)</text>
          {geometry.roles.map((role) => <g key={role.key}>
            {role.trails.map((trail, i) => <g key={i}>
              <polyline points={trail.map((p) => { const q = geometry.screen(p); return `${q.x},${q.y}`; }).join(" ")}
                fill="none" stroke={role.color} strokeWidth="2.5" opacity="0.75" />
              {trail.filter((p, n) => n % Math.max(1, Math.ceil(trail.length / 150)) === 0).map((p, n) => {
                const q = geometry.screen(p); const observed = p.state.observed === true;
                return <circle key={n} cx={q.x} cy={q.y} r="2.5" stroke={role.color} fill={observed ? role.color : "var(--trajectory-bg)"}>
                  <title>{event.scenario_mode === "constructed" ? "Constructed scenario position" : observed ? "Observed AIS record" : "Resampled / source unspecified"}</title>
                </circle>;
              })}
            </g>)}
            {role.current && role.forecast && (() => {
              const a = geometry.screen(role.current), b = geometry.screen(role.forecast);
              return <g><line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={role.color} strokeWidth="2" strokeDasharray="7 5" />
                <path d={`M ${b.x} ${b.y - 8} l 8 8 -8 8 -8 -8 Z`} fill="var(--trajectory-bg)" stroke={role.color} strokeWidth="2.5"><title>{role.label} +5-minute forecast endpoint</title></path>
                <text x={b.x + 12} y={b.y - 8} fill={role.color} fontSize="12">+5 min</text></g>;
            })()}
            {role.current && (() => {
              const q = geometry.screen(role.current), course = role.vessel?.current_state?.course_degrees;
              return <g transform={`translate(${q.x} ${q.y})`}>
                <circle r={selectedRole === role.key ? 19 : 15} fill="var(--trajectory-bg)" stroke={role.color} opacity="0.95" />
                {isNumber(course) ? <path d="M 0 -12 L 8 10 L 0 6 L -8 10 Z" transform={`rotate(${course})`} fill={role.color} /> : <circle r="6" fill={role.color} />}
                <text x="22" y="5" fill={role.color} fontSize="12">{role.label}</text>
              </g>;
            })()}
          </g>)}
        </svg> : <div className="trajectory-empty">{event ? "Valid vessel coordinates are unavailable for this event." : "Start a simulation or select a recording to display vessel movements."}</div>}
        <div className="trajectory-legend"><span className="own">● Own vessel</span><span className="target">● Target vessel</span><span>━ Recorded track</span><span>◇ GRU endpoint</span><span>┄ Illustrative connector</span></div>
        <p className="trajectory-note">{event?.scenario_mode === "constructed" ? "Constructed encounter" : "Historical AIS replay"}; local position view, equal axis scale. Trails stop at the selected time and restart at cycle boundaries. Forecasts are endpoints, not a predicted multi-step route.</p>
      </div>
      <aside className="trajectory-details">
        <div className="trajectory-vessel-tabs">{ROLES.map((role) => <button key={role.key} onClick={() => setSelectedRole(role.key)} aria-pressed={selectedRole === role.key}>{role.label}</button>)}</div>
        <dl>
          <dt>MMSI from scenario</dt><dd>{state?.mmsi ?? "—"}</dd>
          <dt>Selected simulation time</dt><dd>{event?.simulated_timestamp || "—"}</dd>
          <dt>Latitude / longitude</dt><dd>{number(state?.latitude, 6)}, {number(state?.longitude, 6)}</dd>
          <dt>Speed / course</dt><dd>{number(state?.speed_knots)} kn / {number(state?.course_degrees, 1)}°</dd>
          <dt>Forecast input end time</dt><dd>{forecast?.input_end_timestamp || "Unavailable"}</dd>
          <dt>Forecast target time</dt><dd>{targetTime || "Unavailable"}</dd>
          <dt>Forecast state</dt><dd>{forecast ? `${forecast.predicted_motion_class || "—"} · ${number(forecast.predicted_speed_knots)} kn` : selected?.motion_error || "Collecting ten one-minute observations"}</dd>
          <dt>Current-state risk</dt><dd>{event?.encounter?.risk_level || "Unavailable"}</dd>
          <dt>DCPA / TCPA</dt><dd>{number(event?.encounter?.dcpa_nautical_miles, 3)} NM / {number(event?.encounter?.tcpa_minutes, 1)} min</dd>
        </dl>
        <p className="trajectory-note">Radar imagery is paired by scenario; it is not a verified match to this MMSI.</p>
      </aside>
    </div>
    <div className="trajectory-controls">
      <button disabled={!events.length} onClick={replay.restart}>Restart replay</button>
      <button aria-label="Previous event" disabled={index <= 0 || !events.length} onClick={() => replay.seek(index - 1)}>← Step</button>
      <button disabled={!events.length} onClick={replay.togglePlay}>{replay.playing ? "Pause replay" : "Play replay"}</button>
      <button aria-label="Next event" disabled={index >= events.length - 1 || !events.length} onClick={() => replay.seek(index + 1)}>Step →</button>
      <label>Speed <select aria-label="Replay speed" value={replay.speed} onChange={(e) => replay.setSpeed(Number(e.target.value))}>{[0.5, 1, 2, 4].map((n) => <option key={n} value={n}>{n}×</option>)}</select></label>
      <button disabled={!events.length && !replay.selectedRun} onClick={replay.latest}>Jump to latest</button>
    </div>
    <label className="trajectory-timeline">Selected event {events.length ? index + 1 : 0} / {events.length} · {elapsedLabel(elapsed)} / {elapsedLabel(duration)} simulated time retained
      <input aria-label="Replay timeline" type="range" min="0" max={Math.max(0, events.length - 1)} value={Math.max(0, index)} disabled={events.length < 2} onChange={(e) => replay.seek(Number(e.target.value))} />
    </label>
    <p className="trajectory-note">Replay 1× = one simulation second per playback second. Pausing or restarting replay does not pause or restart backend generation. Use the existing STOP button to stop generation.</p>
    {replay.notice && <p className="trajectory-notice" role="status">{replay.notice}</p>}
    {replay.error && <p className="sim-error" role="alert">{replay.error} Recorded events remain available for inspection.</p>}
  </section>;
}
