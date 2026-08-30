import {
  useCallback,
  useEffect,
  useState,
} from "react";

import axios from "axios";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? "" : "http://localhost:8000");


function formatDate(value) {
  if (!value) return "—";

  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
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
        <span>
          DATABASE · RADAR HISTORY
        </span>

        <button
          type="button"
          onClick={loadHistory}
          disabled={loading}
        >
          {loading ? "LOADING..." : "REFRESH"}
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
              <th>TIME</th>
              <th>FILE</th>
              <th>PREDICTION</th>
              <th>CONFIDENCE</th>
              <th>BIRD</th>
              <th>SHIP</th>
              <th>UNKNOWN</th>
              <th>MODEL</th>
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
                  <td data-label="TIME">
                    {formatDate(
                      record.timestamp
                    )}
                  </td>

                  <td data-label="FILE">
                    {record.filename || "—"}
                  </td>

                  <td data-label="PREDICTION">
                    {record.final_prediction || "—"}
                  </td>

                  <td data-label="CONFIDENCE">
                    {record.confidence ?? "—"}%
                  </td>

                  <td data-label="BIRD">
                    {
                      record.bird_probability
                      ?? "—"
                    }%
                  </td>

                  <td data-label="SHIP">
                    {
                      record.ship_probability
                      ?? "—"
                    }%
                  </td>
                  <td data-label="UNKNOWN">
                    {
                      record.unknown_probability
                      ?? "—"
                    }%
                  </td>

                  <td data-label="MODEL">
                    {record.model_name || "—"}
                  </td>
                </tr>
              )
            )}

            {!records.length && !loading && (
              <tr>
                <td colSpan="7">
                  No stored Radar records.
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
          <span>
            DATABASE · SIMULATION RUNS
          </span>

          <button
            type="button"
            onClick={loadRuns}
            disabled={loading}
          >
            {loading ? "LOADING..." : "REFRESH"}
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
                <th>STATUS</th>
                <th>MODE</th>
                <th>STARTED</th>
                <th>ENDED</th>
                <th>FRAMES</th>
                <th>SOURCE</th>
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
                  <td data-label="STATUS">
                    {run.status || "—"}
                  </td>

                  <td data-label="MODE">
                    {
                      run.scenario_mode
                      || "—"
                    }
                  </td>

                  <td data-label="STARTED">
                    {formatDate(
                      run.started_at
                    )}
                  </td>

                  <td data-label="ENDED">
                    {formatDate(
                      run.ended_at
                    )}
                  </td>

                  <td data-label="FRAMES">
                    {run.frame_count ?? "—"}
                  </td>

                  <td data-label="SOURCE">
                    {run.sar_source || "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="sim-panel sim-events">
        <div className="sim-panel-head">
          <span>
            DATABASE · STORED EVENTS
          </span>

          <span>
            {events.length} shown
          </span>
        </div>

        <div className="sim-table-wrap">
          <table>
            <thead>
              <tr>
                <th>FRAME</th>
                <th>TIME</th>
                <th>RADAR</th>
                <th>CONF.</th>
                <th>GRU</th>
                <th>DCPA</th>
                <th>TCPA</th>
                <th>RISK</th>
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
                      <td data-label="FRAME">
                        {event.frame_number}
                      </td>

                      <td data-label="TIME">
                        {
                          event
                            .simulated_timestamp
                            ?.slice(11, 19)
                          || "—"
                        }
                      </td>

                      <td data-label="RADAR">
                        {
                          event.sar
                            ?.classification
                          || "—"
                        }
                      </td>

                      <td data-label="CONF.">
                        {
                          event.sar
                            ?.confidence
                          ?? "—"
                        }%
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
                            ?.dcpa_nautical_miles
                          ?? "—"
                        }{" "}
                        NM
                      </td>

                      <td data-label="TCPA">
                        {
                          event.encounter
                            ?.tcpa_minutes
                          ?? "—"
                        }{" "}
                        min
                      </td>

                      <td data-label="RISK">
                        {
                          event.encounter
                            ?.risk_level
                          || "—"
                        }
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
