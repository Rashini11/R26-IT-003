import { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";
import { mergeEvents, playbackDelay } from "./replayMath";

function message(error) {
  const detail = error?.response?.data?.detail;
  return typeof detail === "string" ? detail : error?.message || "Simulation service unavailable.";
}

export default function useSimulationReplay(api) {
  const [status, setStatus] = useState(null);
  const [events, setEvents] = useState([]);
  const [fallback, setFallback] = useState(null);
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState("");
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [following, setFollowing] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(false);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const buffer = useRef({ id: null, events: [] });
  const refresh = useCallback(async () => setRefreshVersion((n) => n + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    let disposed = false;
    let timer;
    const options = { signal: controller.signal, timeout: 8000 };
    async function poll() {
      try {
        const { data: nextStatus } = await axios.get(`${api}/simulation/status`, options);
        if (disposed) return;
        setStatus(nextStatus);
        const id = selectedRun || nextStatus.recording_id;
        if (buffer.current.id !== id) {
          buffer.current = { id, events: [] };
          setEvents([]); setFallback(null); setSelectedFrame(null); setPlaying(false); setNotice("");
          setFollowing(true);
        }
        const runResponse = await axios.get(`${api}/simulation/replay/runs`, options);
        if (disposed) return;
        setRuns(runResponse.data.records || []);
        if (!id) {
          // A pre-installation active run can still be viewed, but cannot be replayed safely.
          const response = await axios.get(`${api}/simulation/latest`, options).catch((e) => {
            if (e.response?.status === 404) return null;
            throw e;
          });
          if (disposed) return;
          setFallback(response?.data || null);
          setNotice(response?.data ? "Start a new simulation to record this feature's replay timeline." : "Start a simulation to record vessel movements.");
        } else {
          setLoading(true);
          let after = buffer.current.events.at(-1)?.frame_number || 0;
          let more = true;
          let pages = 0;
          let incomplete = false;
          let database = false;
          while (more && pages++ < 8 && !disposed) {
            const response = await axios.get(`${api}/simulation/replay/events`, {
              ...options, params: { recording_id: id, after_frame: after, limit: 500 },
            });
            if (disposed) return;
            const data = response.data;
            const incoming = data.results || [];
            if (incoming.length && incoming[0].frame_number > after + 1) incomplete = true;
            incomplete ||= Boolean(data.truncated);
            database ||= Boolean(data.database_available);
            buffer.current.events = mergeEvents(buffer.current.events, incoming);
            const next = data.next_after_frame;
            more = Boolean(data.has_more) && next > after;
            after = next;
          }
          if (disposed) return;
          const all = buffer.current.events;
          setEvents([...all]);
          const gaps = all.some((e, i) => i && e.frame_number !== all[i - 1].frame_number + 1);
          const trimmed = all.length && all[0].frame_number > 1;
          setNotice([
            incomplete || gaps || trimmed ? "Partial recording: unavailable or trimmed events are not reconstructed." : "",
            more ? "Loading earlier recording pages…" : "",
            database ? "Recorded results reused; no replay inference." : "Memory recording only (last 2,000 backend events / 3 runs); not durable across backend restart.",
          ].filter(Boolean).join(" "));
        }
        setError("");
      } catch (e) {
        if (!disposed && !axios.isCancel(e)) setError(message(e));
      } finally {
        if (!disposed) { setLoading(false); timer = setTimeout(poll, 2000); }
      }
    }
    poll();
    return () => { disposed = true; controller.abort(); clearTimeout(timer); };
  }, [api, selectedRun, refreshVersion]);

  const index = following ? events.length - 1 : Math.max(0,
    events.findIndex((event) => event.frame_number >= selectedFrame));
  const event = events[index] || fallback;
  const nextEvent = events[index + 1];
  useEffect(() => {
    if (!playing || following || !events.length) return undefined;
    if (!nextEvent) {
      // Stay in replay mode at the tail. If generation continues, a later poll extends it.
      return undefined;
    }
    const timer = setTimeout(() => setSelectedFrame(nextEvent.frame_number),
      playbackDelay(event, nextEvent, speed));
    return () => clearTimeout(timer);
  }, [playing, following, event?.frame_number, event?.recording_id, nextEvent?.frame_number, speed]);

  function seek(next) {
    const bounded = Math.max(0, Math.min(events.length - 1, next));
    setFollowing(false); setPlaying(false); setSelectedFrame(events[bounded]?.frame_number ?? null);
  }
  function togglePlay() {
    if (playing) { setPlaying(false); return; }
    if (!events.length) return;
    if (following) { setSelectedFrame(events[index]?.frame_number); setFollowing(false); }
    setPlaying(true);
  }
  const completed = !status?.running || (selectedRun && selectedRun !== status?.recording_id);
  const atEnd = events.length > 0 && index === events.length - 1;
  const playbackState = !events.length ? "No recording" : following ? selectedRun ? "Following recording tail" : "Following latest"
    : playing ? atEnd ? completed ? "Replay completed" : "Waiting for recorded events" : "Replay playing"
    : "Replay paused";
  return { status, events, event, runs, selectedRun,
    setSelectedRun: (id) => { setSelectedRun(id); setEvents([]); setFallback(null); setSelectedFrame(null); setPlaying(false); setFollowing(true); },
    index, following,
    playing, speed, setSpeed, error, notice, loading, refresh, seek, togglePlay, playbackState,
    restart: () => seek(0),
    latest: () => { setSelectedRun(""); setFollowing(true); setPlaying(false); },
  };
}
