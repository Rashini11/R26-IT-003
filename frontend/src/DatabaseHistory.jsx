import {
  useCallback,
  useEffect,
  useState,
} from "react";

import axios from "axios";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? "" : "http://localhost:8000");


function formatDate(value) {
  if (value === null || value === undefined || value === "") {
    return "Not recorded";
  }

  try {
    return new Date(value).toLocaleString();
  } catch {
    return String(value);
  }
}

function textOrMissing(value) {
  if (value === null || value === undefined || value === "") {
    return "Not recorded";
  }
  return value;
}

function percentOrMissing(value) {
  if (value === null || value === undefined || value === "") {
    return "Not recorded";
  }
  return `${value}%`;
}


export function RadarDatabaseHistory() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadHistory = useCallback(async () => {
    try {
      setLoading(true);
      setError("");

      const response = await axios.get(
        `${API_BASE_URL}/radar/history`,
        {
          params: {
            limit: 20,
          },
        }
      );

      setRecords(
        response.data?.records || []
      );
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
        err.message ||
        "Unable to load Radar history."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  return (
    <section className="sim-panel sim-events">
      <div className="sim-panel-head">
        <span>Radar history</span>

        <button
          type="button"
          className="sim-button sim-button--ghost"
          onClick={loadHistory}
          disabled={loading}
        >
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="sim-waiting">
          {error}
        </div>
      )}

      <div className="sim-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>File</th>
              <th>Prediction</th>
              <th>Confidence</th>
              <th>Bird</th>
              <th>Ship</th>
              <th>Unknown</th>
              <th>Model</th>
            </tr>
          </thead>

          <tbody>
            {records.map(
              (record, index) => (
                <tr
                  key={
                    `${record.timestamp}-${index}`
                  }
                >
                  <td data-label="Time">
                    {formatDate(
                      record.timestamp
                    )}
                  </td>

                  <td data-label="File">
                    {textOrMissing(record.filename)}
                  </td>

                  <td data-label="Prediction">
                    {textOrMissing(record.final_prediction)}
                  </td>

                  <td data-label="Confidence">
                    {percentOrMissing(record.confidence)}
                  </td>

                  <td data-label="Bird">
                    {percentOrMissing(record.bird_probability)}
                  </td>

                  <td data-label="Ship">
                    {percentOrMissing(record.ship_probability)}
                  </td>
                  <td data-label="Unknown">
                    {percentOrMissing(record.unknown_probability)}
                  </td>

                  <td data-label="Model">
                    {textOrMissing(record.model_name)}
                  </td>
                </tr>
              )
            )}

            {!records.length && !loading && (
              <tr>
                <td colSpan={8}>
                  No stored radar records.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}


export function SimulationDatabaseHistory() {
  const [runs, setRuns] = useState([]);
  const [events, setEvents] = useState([]);
  const [selectedRun, setSelectedRun] =
    useState("");
  const [loading, setLoading] =
    useState(false);
  const [error, setError] =
    useState("");

  const loadRuns = useCallback(
    async () => {
      try {
        setLoading(true);
        setError("");

        const response = await axios.get(
          `${API_BASE_URL}/simulation/database/runs`,
          {
            params: {
              limit: 20,
            },
          }
        );

        const nextRuns =
          response.data?.records || [];

        setRuns(nextRuns);

        if (
          nextRuns.length &&
          !selectedRun
        ) {
          setSelectedRun(
            nextRuns[0].simulation_id
          );
        }
      } catch (err) {
        setError(
          err?.response?.data?.detail ||
          err.message ||
          "Unable to load simulation runs."
        );
      } finally {
        setLoading(false);
      }
    },
    [selectedRun]
  );

  const loadEvents = useCallback(
    async (simulationId) => {
      if (!simulationId) {
        setEvents([]);
        return;
      }

      try {
        setLoading(true);
        setError("");

        const response = await axios.get(
          `${API_BASE_URL}/simulation/database/events`,
          {
            params: {
              simulation_id:
                simulationId,
              limit: 100,
            },
          }
        );

        setEvents(
          response.data?.records || []
        );
      } catch (err) {
        setError(
          err?.response?.data?.detail ||
          err.message ||
          "Unable to load simulation events."
        );
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  useEffect(() => {
    if (selectedRun) {
      loadEvents(selectedRun);
    }
  }, [
    selectedRun,
    loadEvents,
  ]);

  return (
    <>
      <section className="sim-panel sim-events">
        <div className="sim-panel-head">
          <span>Simulation runs</span>

          <button
            type="button"
            className="sim-button sim-button--ghost"
            onClick={loadRuns}
            disabled={loading}
          >
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>

        {error && (
          <div className="sim-waiting">
            {error}
          </div>
        )}

        <div className="sim-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Status</th>
                <th>Mode</th>
                <th>Started</th>
                <th>Ended</th>
                <th>Frames</th>
                <th>Source</th>
              </tr>
            </thead>

            <tbody>
              {runs.map((run) => (
                <tr
                  key={run.simulation_id}
                  onClick={() =>
                    setSelectedRun(
                      run.simulation_id
                    )
                  }
                  style={{
                    cursor: "pointer",
                  }}
                >
                  <td data-label="Status">
                    {textOrMissing(run.status)}
                  </td>

                  <td data-label="Mode">
                    {textOrMissing(run.scenario_mode)}
                  </td>

                  <td data-label="Started">
                    {formatDate(
                      run.started_at
                    )}
                  </td>

                  <td data-label="Ended">
                    {formatDate(
                      run.ended_at
                    )}
                  </td>

                  <td data-label="Frames">
                    {run.frame_count == null ? "Not recorded" : run.frame_count}
                  </td>

                  <td data-label="Source">
                    {textOrMissing(run.sar_source)}
                  </td>
                </tr>
              ))}

              {!runs.length && !loading && (
                <tr>
                  <td colSpan={6}>No stored simulation runs.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="sim-panel sim-events">
        <div className="sim-panel-head">
          <span>Stored events</span>

          <span>
            {events.length} shown
          </span>
        </div>

        <div className="sim-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Frame</th>
                <th>Time</th>
                <th>Radar</th>
                <th>Conf.</th>
                <th>GRU</th>
                <th>DCPA</th>
                <th>TCPA</th>
                <th>Risk</th>
              </tr>
            </thead>

            <tbody>
              {[...events]
                .reverse()
                .map((event) => {
                  const prediction =
                    event.own_vessel
                      ?.motion_prediction;

                  return (
                    <tr
                      key={
                        `${event.simulation_id}`
                        + `-${event.frame_number}`
                      }
                    >
                      <td data-label="Frame">
                        {event.frame_number == null ? "Not recorded" : event.frame_number}
                      </td>

                      <td data-label="Time">
                        {
                          event
                            .simulated_timestamp
                            ?.slice(11, 19)
                          || "Not recorded"
                        }
                      </td>

                      <td data-label="Radar">
                        {textOrMissing(event.sar?.classification)}
                      </td>

                      <td data-label="Conf.">
                        {percentOrMissing(event.sar?.confidence)}
                      </td>

                      <td data-label="GRU">
                        {
                          prediction
                            ?.predicted_motion_class
                          || "waiting"
                        }
                      </td>

                      <td data-label="DCPA">
                        {
                          event.encounter
                            ?.dcpa_nautical_miles == null
                            ? "Not recorded"
                            : `${event.encounter.dcpa_nautical_miles} NM`
                        }
                      </td>

                      <td data-label="TCPA">
                        {
                          event.encounter
                            ?.tcpa_minutes == null
                            ? "Not recorded"
                            : `${event.encounter.tcpa_minutes} min`
                        }
                      </td>

                      <td data-label="Risk">
                        {textOrMissing(event.encounter?.risk_level)}
                      </td>
                    </tr>
                  );
                })}

              {!events.length && !loading && (
                <tr>
                  <td colSpan={8}>No stored simulation events.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
