import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Gauge,
  Loader,
  MapPin,
  Navigation,
  Play,
  RadioTower,
  RefreshCw,
  Ship,
  Square,
} from "lucide-react";
import MediaFitToggle from "./components/MediaFitToggle";

/*
 * SINGLE BACKEND CONFIGURATION
 *
 * All OceanIQ modules now use the same FastAPI application.
 *
 * Default:
 *   http://localhost:8000
 *
 * Optional deployment override:
 *   VITE_API_BASE_URL=http://server:8000
 */
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function fmt(value, digits = 2, fallback = "—") {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(Number(value))
  ) {
    return fallback;
  }

  return Number(value).toFixed(digits);
}

function riskClass(level) {
  return `sim-risk sim-risk--${String(
    level || "unknown"
  ).toLowerCase()}`;
}

function EnvironmentPanel({ environment }) {
  const atmospheric =
    environment?.atmospheric;

  const marine =
    environment?.marine;

  if (!environment) {
    return (
      <section className="sim-panel">
        <div className="sim-panel-head">
          <span>
            ENVIRONMENTAL CONDITIONS
          </span>
        </div>

        <div className="sim-waiting">
          Waiting for environmental data…
        </div>
      </section>
    );
  }

  return (
    <section className="sim-panel">

      <div className="sim-panel-head">
        <span>
          ENVIRONMENTAL CONDITIONS
        </span>

        <span className="sim-mono">
          {environment.provider || "—"}
        </span>
      </div>


      <div className="sim-history-row">
        <span>
          Encounter position
        </span>

        <strong>
          {fmt(
            environment.latitude,
            4
          )}
          {", "}
          {fmt(
            environment.longitude,
            4
          )}
        </strong>
      </div>


      <div className="sim-panel-head">
        <span>
          ATMOSPHERIC
        </span>
      </div>


      {atmospheric?.available ? (

        <div className="sim-kpi-grid">

          <div>
            <small>TEMPERATURE</small>
            <strong>
              {fmt(
                atmospheric.temperature_c,
                1
              )} °C
            </strong>
          </div>

          <div>
            <small>HUMIDITY</small>
            <strong>
              {fmt(
                atmospheric
                  .relative_humidity_percent,
                0
              )}%
            </strong>
          </div>

          <div>
            <small>WIND</small>
            <strong>
              {fmt(
                atmospheric.wind_speed_knots,
                2
              )} kn
            </strong>
          </div>

          <div>
            <small>WIND DIRECTION</small>
            <strong>
              {fmt(
                atmospheric
                  .wind_direction_degrees,
                0
              )}°
            </strong>
          </div>

          <div>
            <small>WIND GUST</small>
            <strong>
              {fmt(
                atmospheric.wind_gust_knots,
                2
              )} kn
            </strong>
          </div>

          <div>
            <small>VISIBILITY</small>
            <strong>
              {fmt(
                atmospheric.visibility_km,
                1
              )} km
            </strong>
          </div>

          <div>
            <small>PRESSURE</small>
            <strong>
              {fmt(
                atmospheric.pressure_msl_hpa,
                1
              )} hPa
            </strong>
          </div>

          <div>
            <small>RAIN</small>
            <strong>
              {fmt(
                atmospheric.rain_mm,
                2
              )} mm
            </strong>
          </div>

          <div>
            <small>CLOUD COVER</small>
            <strong>
              {fmt(
                atmospheric.cloud_cover_percent,
                0
              )}%
            </strong>
          </div>

        </div>

      ) : (

        <div className="sim-waiting">
          Atmospheric data unavailable.
        </div>

      )}


      <div
        className="sim-panel-head"
        style={{
          marginTop: "18px",
        }}
      >
        <span>
          MARINE
        </span>
      </div>


      {marine?.available ? (

        <div className="sim-kpi-grid">

          <div>
            <small>WAVE HEIGHT</small>
            <strong>
              {fmt(
                marine.wave_height_m,
                2
              )} m
            </strong>
          </div>

          <div>
            <small>WAVE DIRECTION</small>
            <strong>
              {fmt(
                marine
                  .wave_direction_degrees,
                0
              )}°
            </strong>
          </div>

          <div>
            <small>WAVE PERIOD</small>
            <strong>
              {fmt(
                marine.wave_period_seconds,
                1
              )} s
            </strong>
          </div>

          <div>
            <small>SWELL HEIGHT</small>
            <strong>
              {fmt(
                marine.swell_height_m,
                2
              )} m
            </strong>
          </div>

          <div>
            <small>SWELL DIRECTION</small>
            <strong>
              {fmt(
                marine
                  .swell_direction_degrees,
                0
              )}°
            </strong>
          </div>

          <div>
            <small>SWELL PERIOD</small>
            <strong>
              {fmt(
                marine.swell_period_seconds,
                1
              )} s
            </strong>
          </div>

          <div>
            <small>SEA TEMPERATURE</small>
            <strong>
              {fmt(
                marine
                  .sea_surface_temperature_c,
                1
              )} °C
            </strong>
          </div>

          <div>
            <small>CURRENT SPEED</small>
            <strong>
              {fmt(
                marine
                  .ocean_current_speed_knots,
                2
              )} kn
            </strong>
          </div>

          <div>
            <small>CURRENT DIRECTION</small>
            <strong>
              {fmt(
                marine
                  .ocean_current_direction_degrees,
                0
              )}°
            </strong>
          </div>

        </div>

      ) : (

        <div className="sim-waiting">
          {marine?.reason ||
            "Marine data unavailable at this location."}
        </div>

      )}


      <p className="sim-notice">
        Environmental observations are
        contextual decision-support data.
        They are not inputs to RadarTargetCNN
        or the GRU motion model.
      </p>

    </section>
  );
}


function VesselPanel({ label, vessel }) {
  const state = vessel?.current_state;
  const prediction = vessel?.motion_prediction;

  const progress = Math.min(
    10,
    vessel?.history_available_minutes || 0
  );

  return (
    <section className="sim-panel">
      <div className="sim-panel-head">
        <span>
          <Ship size={14} /> {label}
        </span>

        <span className="sim-mono">
          MMSI {state?.mmsi || "—"}
        </span>
      </div>

      <div className="sim-kpi-grid">
        <div>
          <small>LATITUDE</small>
          <strong>
            {fmt(state?.latitude, 6)}
          </strong>
        </div>

        <div>
          <small>LONGITUDE</small>
          <strong>
            {fmt(state?.longitude, 6)}
          </strong>
        </div>

        <div>
          <small>SPEED</small>
          <strong>
            {fmt(state?.speed_knots, 2)} kn
          </strong>
        </div>

        <div>
          <small>COURSE</small>
          <strong>
            {fmt(state?.course_degrees, 1)}°
          </strong>
        </div>
      </div>

      <div className="sim-history-row">
        <span>GRU history</span>

        <div className="sim-history-track">
          <i
            style={{
              width: `${progress * 10}%`,
            }}
          />
        </div>

        <strong>
          {progress}/10 min
        </strong>
      </div>

      {prediction ? (
        <div className="sim-forecast">
          <div>
            <small>+5 MIN LAT/LON</small>
            <strong>
              {fmt(
                prediction.predicted_latitude,
                6
              )}
              {", "}
              {fmt(
                prediction.predicted_longitude,
                6
              )}
            </strong>
          </div>

          <div>
            <small>FUTURE SPEED</small>
            <strong>
              {fmt(
                prediction.predicted_speed_knots,
                2
              )}{" "}
              kn
            </strong>
          </div>

          <div>
            <small>MOTION CLASS</small>
            <strong>
              {
                prediction.predicted_motion_class
              }
            </strong>
          </div>

          <div>
            <small>INFERENCE</small>
            <strong>
              {fmt(
                prediction.inference_time_ms,
                2
              )}{" "}
              ms
            </strong>
          </div>
        </div>
      ) : (
        <div className="sim-waiting">
          {vessel?.motion_error
            ? `GRU unavailable: ${vessel.motion_error}`
            : "Collecting ten one-minute observations…"}
        </div>
      )}
    </section>
  );
}

export default function LiveSimulation({ mediaFit = "fit", onMediaFitChange }) {
  const [mode, setMode] =
    useState("constructed");

  const [realInterval, setRealInterval] =
    useState(1);

  const [sarSource, setSarSource] =
    useState("ship");

  const [latest, setLatest] =
    useState(null);

  const [status, setStatus] =
    useState(null);

  const [history, setHistory] =
    useState([]);

  const [busy, setBusy] =
    useState(false);

  const [error, setError] =
    useState("");

  const running = Boolean(
    status?.running || latest?.running
  );

  const refresh = useCallback(
    async () => {
      const [
        statusResponse,
        latestResponse,
        historyResponse,
      ] = await Promise.allSettled([
        axios.get(
          `${API_BASE_URL}/simulation/status`,
          { timeout: 4000 }
        ),

        axios.get(
          `${API_BASE_URL}/simulation/latest`,
          { timeout: 4000 }
        ),

        axios.get(
          `${API_BASE_URL}/simulation/history?limit=12`,
          { timeout: 4000 }
        ),
      ]);

      if (
        statusResponse.status
        === "rejected"
      ) {
        setError(
          statusResponse.reason?.message
          || "Integrated backend unavailable."
        );
        return;
      }

      setStatus(
        statusResponse.value.data
      );

      if (
        latestResponse.status
        === "fulfilled"
      ) {
        setLatest(
          latestResponse.value.data
        );
      }

      if (
        historyResponse.status
        === "fulfilled"
      ) {
        setHistory(
          historyResponse.value.data.results
          || []
        );
      }

      setError("");
    },
    []
  );

  useEffect(() => {
    refresh();

    const timer = setInterval(
      refresh,
      2000
    );

    return () => clearInterval(
      timer
    );
  }, [refresh]);

  const start = async () => {
    setBusy(true);
    setError("");

    try {
      await axios.post(
        `${API_BASE_URL}/simulation/start`,
        {
          sar_source: sarSource,
          mode,
          real_interval_seconds:
            Number(realInterval),
          simulated_interval_seconds: 10,
          loop: true,
          seed: 42,
          scenario_minutes: 20,
        },
        {
          timeout: 120000,
        }
      );

      await refresh();

    } catch (requestError) {
      setError(
        requestError.response?.data?.detail
        || requestError.message
        || "Unable to start simulation."
      );

    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    setError("");

    try {
      await axios.post(
        `${API_BASE_URL}/simulation/stop`,
        {},
        {
          timeout: 5000,
        }
      );

      await refresh();

    } catch (requestError) {
      setError(
        requestError.response?.data?.detail
        || requestError.message
        || "Unable to stop simulation."
      );

    } finally {
      setBusy(false);
    }
  };

  const imageUrl = useMemo(
    () => {
      if (!latest?.frame_number) {
        return null;
      }

      return (
        `${API_BASE_URL}`
        + "/simulation/current-image"
        + `?frame=${latest.frame_number}`
      );
    },
    [latest?.frame_number]
  );

  const encounter =
    latest?.encounter;

  const sar =
    latest?.sar;

  return (
    <div className="simulation-page">

      <section className="sim-control-bar">

        <label>
          Scenario

          <select
            value={mode}
            onChange={(event) =>
              setMode(
                event.target.value
              )
            }
            disabled={
              running || busy
            }
          >
            <option value="constructed">
              Constructed encounter
            </option>

            <option value="actual">
              Actual AIS encounter
            </option>
          </select>
        </label>

        <label>
          SAR source

          <select
            value={sarSource}
            onChange={(event) =>
              setSarSource(
                event.target.value
              )
            }
            disabled={
              running || busy
            }
          >
            <option value="ship">
              Ship images
            </option>

            <option value="all">
              All classes
            </option>

            <option value="bird">
              Bird images
            </option>

            <option value="unknown">
              Unknown images
            </option>
          </select>
        </label>

        <label>
          Real interval

          <select
            value={realInterval}
            onChange={(event) =>
              setRealInterval(
                Number(
                  event.target.value
                )
              )
            }
            disabled={
              running || busy
            }
          >
            <option value={1}>
              1 second — accelerated
            </option>

            <option value={10}>
              10 seconds — viva demo
            </option>
          </select>
        </label>

        <button
          className="sim-button sim-button--start"
          onClick={start}
          disabled={
            running || busy
          }
        >
          {busy && !running ? (
            <Loader
              size={14}
              className="spin"
            />
          ) : (
            <Play size={14} />
          )}

          START
        </button>

        <button
          className="sim-button sim-button--stop"
          onClick={stop}
          disabled={
            !running || busy
          }
        >
          <Square size={13} />
          STOP
        </button>

        <button
          className="sim-button sim-button--ghost"
          onClick={refresh}
          disabled={busy}
        >
          <RefreshCw size={13} />
          REFRESH
        </button>

      </section>

      {error && (
        <div className="sim-error">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      <section className="sim-status-strip">

        <span
          className={
            running
              ? "sim-online"
              : "sim-offline"
          }
        >
          {running
            ? "STREAMING"
            : "STOPPED"}
        </span>

        <span>
          <RadioTower size={13} />
          Frame {
            latest?.frame_number || 0
          }
        </span>

        <span>
          <Clock3 size={13} />
          Real:{" "}
          {latest?.real_timestamp
            ? new Date(
                latest.real_timestamp
              ).toLocaleTimeString()
            : "—"}
        </span>

        <span>
          <Clock3 size={13} />
          Simulated:{" "}
          {
            latest?.simulated_timestamp
            || "—"
          }
        </span>

        <span className="sim-mono">
          {
            latest?.scenario_mode
            || mode
          }
        </span>

      </section>

      <div className="sim-top-grid">

        <section className="sim-radar-card">

          <div className="sim-panel-head">
            <span>
              <RadioTower size={14} />
              LIVE SAR FRAME
            </span>

            <span className="sim-mono">
              {sar?.filename || "—"}
            </span>
          </div>

          <div className="sim-media-toolbar">
            <MediaFitToggle mode={mediaFit} onChange={onMediaFitChange} compact />
          </div>

          <div className="sim-image-stage">
            {imageUrl ? (
              <img
                src={imageUrl}
                alt="Current simulated SAR frame"
                className={`sim-image sim-image--${mediaFit}`}
              />
            ) : (
              <div>
                <RadioTower size={42} />
                <p>
                  Start the simulation
                  to receive SAR frames
                </p>
              </div>
            )}
          </div>

        </section>

        <section className="sim-panel sim-classification-card">

          <div className="sim-panel-head">
            <span>
              <Activity size={14} />
              CLASSIFICATION
            </span>
          </div>

          {sar?.error ? (
            <div className="sim-error">
              <AlertTriangle size={16} />
              {sar.error}
            </div>
          ) : (
            <>
              <div className="sim-class-main">
                {
                  sar?.classification
                  || "Awaiting frame"
                }
              </div>

              <div className="sim-kpi-grid">

                <div>
                  <small>
                    BIRD PROBABILITY
                  </small>

                  <strong>
                    BIRD
                  </strong>

                  <em>
                    {fmt(
                      sar?.bird_probability,
                      2
                    )}
                    %
                  </em>
                </div>

                <div>
                  <small>
                    SHIP PROBABILITY
                  </small>

                  <strong>
                    SHIP
                  </strong>

                  <em>
                    {fmt(
                      sar?.ship_probability,
                      2
                    )}
                    %
                  </em>
                </div>

              </div>

              <div className="sim-decision">

                {sar?.classification
                  && sar.classification
                  !== "uncertain"
                  ? (
                    <CheckCircle2
                      size={15}
                    />
                  )
                  : (
                    <AlertTriangle
                      size={15}
                    />
                  )}

                {
                  sar?.agreement_status
                  || "Waiting for classification"
                }

              </div>
            </>
          )}

        </section>

        <section className="sim-panel sim-risk-card">

          <div className="sim-panel-head">
            <span>
              <Gauge size={14} />
              COLLISION RISK
            </span>
          </div>

          <div
            className={riskClass(
              encounter?.risk_level
            )}
          >
            {
              encounter?.risk_level
              || "—"
            }
          </div>

          <div className="sim-kpi-grid">

            <div>
              <small>SEPARATION</small>
              <strong>
                {fmt(
                  encounter
                    ?.current_separation_nautical_miles,
                  3
                )}{" "}
                NM
              </strong>
            </div>

            <div>
              <small>DCPA</small>
              <strong>
                {fmt(
                  encounter
                    ?.dcpa_nautical_miles,
                  3
                )}{" "}
                NM
              </strong>
            </div>

            <div>
              <small>TCPA</small>
              <strong>
                {fmt(
                  encounter?.tcpa_minutes,
                  1
                )}{" "}
                min
              </strong>
            </div>

            <div>
              <small>RELATION</small>
              <strong>
                {
                  encounter
                    ?.movement_relationship
                  || "—"
                }
              </strong>
            </div>

          </div>

          <div className="sim-risk-reasons">

            {(
              encounter?.risk_reasons
              || [
                "Waiting for encounter data.",
              ]
            ).map((reason) => (
              <p key={reason}>
                • {reason}
              </p>
            ))}

          </div>

        </section>

      </div>

      <EnvironmentPanel
        environment={latest?.environment}
      />

      <div className="sim-vessel-grid">

        <VesselPanel
          label="OWN VESSEL"
          vessel={
            latest?.own_vessel
          }
        />

        <VesselPanel
          label="TARGET VESSEL"
          vessel={
            latest?.target_vessel
          }
        />

      </div>


      <section className="sim-panel sim-events">

        <div className="sim-panel-head">
          <span>
            <Navigation size={14} />
            RECENT EVENTS
          </span>

          <span>
            {history.length} shown
          </span>
        </div>

        <div className="sim-table-wrap">
          <table>

            <thead>
              <tr>
                <th>FRAME</th>
                <th>SIM TIME</th>
                <th>SAR CLASS</th>
                <th>SEPARATION</th>
                <th>DCPA</th>
                <th>TCPA</th>
                <th>RISK</th>
              </tr>
            </thead>

            <tbody>

              {[...history]
                .reverse()
                .map((event) => (
                  <tr
                    key={
                      `${event.simulation_id}`
                      + `-${event.frame_number}`
                    }
                  >
                    <td data-label="FRAME">
                      {
                        event.frame_number
                      }
                    </td>

                    <td data-label="SIM TIME">
                      {
                        event
                          .simulated_timestamp
                          ?.slice(
                            11,
                            19
                          )
                        || "—"
                      }
                    </td>

                    <td data-label="SAR CLASS">
                      {
                        event.sar
                          ?.classification
                        || "error"
                      }
                    </td>

                    <td data-label="SEPARATION">
                      {fmt(
                        event.encounter
                          ?.current_separation_nautical_miles,
                        3
                      )}{" "}
                      NM
                    </td>

                    <td data-label="DCPA">
                      {fmt(
                        event.encounter
                          ?.dcpa_nautical_miles,
                        3
                      )}{" "}
                      NM
                    </td>

                    <td data-label="TCPA">
                      {fmt(
                        event.encounter
                          ?.tcpa_minutes,
                        1
                      )}{" "}
                      min
                    </td>

                    <td data-label="RISK">
                      <span
                        className={riskClass(
                          event.encounter
                            ?.risk_level
                        )}
                      >
                        {
                          event.encounter
                            ?.risk_level
                          || "—"
                        }
                      </span>
                    </td>
                  </tr>
                ))}

            </tbody>

          </table>
        </div>

      </section>

      <p className="sim-notice">
        <MapPin size={13} />
        SAR images provide target appearance
        classification. AIS sequences provide
        speed and trajectory. The link is a
        controlled research simulation.
      </p>

    </div>
  );
}
