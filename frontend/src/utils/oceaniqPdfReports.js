import axios from "axios";
import { jsPDF } from "jspdf";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? "" : "http://localhost:8000");


// ============================================================
// DATA HELPERS
// ============================================================

function asArray(data, keys = []) {
  if (Array.isArray(data)) return data;

  for (const key of keys) {
    if (Array.isArray(data?.[key])) {
      return data[key];
    }
  }

  return [];
}


function val(value, fallback = "N/A") {
  if (
    value === null ||
    value === undefined ||
    value === ""
  ) {
    return fallback;
  }

  return String(value);
}


function fixed(value, digits = 2) {
  const n = Number(value);

  return Number.isFinite(n)
    ? n.toFixed(digits)
    : "N/A";
}


function pct(value) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return "N/A";
  }

  const percentage =
    Math.abs(n) <= 1
      ? n * 100
      : n;

  return `${percentage.toFixed(2)}%`;
}


function dateValue(value) {
  if (!value) return "N/A";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return date.toLocaleString();
}


function newest(records) {
  if (!records.length) return {};

  const keys = [
    "created_at",
    "timestamp",
    "predicted_at",
    "ended_at",
    "started_at",
  ];

  return [...records].sort((a, b) => {
    const getTime = (record) => {
      for (const key of keys) {
        if (record?.[key]) {
          const time =
            new Date(record[key]).getTime();

          if (!Number.isNaN(time)) {
            return time;
          }
        }
      }

      return 0;
    };

    return getTime(b) - getTime(a);
  })[0];
}


// ============================================================
// PDF BRANDING
// ============================================================

const COLORS = {
  navy: [4, 20, 34],
  navy2: [8, 35, 54],
  cyan: [0, 188, 230],
  cyanDark: [0, 126, 160],
  white: [255, 255, 255],
  text: [33, 47, 61],
  muted: [91, 108, 123],
  line: [204, 218, 228],
  light: [241, 247, 250],
  danger: [190, 55, 55],
};


function drawLogo(doc) {
  // OceanIQ vector logo mark.
  doc.setFillColor(...COLORS.cyan);
  doc.roundedRect(
    14,
    9,
    13,
    13,
    2.5,
    2.5,
    "F"
  );

  doc.setDrawColor(...COLORS.white);
  doc.setLineWidth(0.7);

  // Small maritime / anchor-inspired mark.
  doc.circle(20.5, 13, 1.8, "S");

  doc.line(
    20.5,
    14.8,
    20.5,
    19
  );

  doc.line(
    17.5,
    17,
    20.5,
    19
  );

  doc.line(
    23.5,
    17,
    20.5,
    19
  );

  doc.setFont(
    "helvetica",
    "bold"
  );

  doc.setTextColor(
    ...COLORS.white
  );

  doc.setFontSize(14);

  doc.text(
    "OceanIQ",
    31,
    15
  );

  doc.setFont(
    "helvetica",
    "normal"
  );

  doc.setFontSize(6.8);

  doc.setTextColor(
    170,
    210,
    226
  );

  doc.text(
    "MARINE INTELLIGENCE",
    31,
    20
  );
}


function drawHeader(ctx) {
  const {
    doc,
    title,
    subtitle,
    reportCode,
  } = ctx;

  doc.setFillColor(
    ...COLORS.navy
  );

  doc.rect(
    0,
    0,
    210,
    31,
    "F"
  );

  drawLogo(doc);

  doc.setFont(
    "helvetica",
    "bold"
  );

  doc.setFontSize(13);

  doc.setTextColor(
    ...COLORS.white
  );

  doc.text(
    title,
    196,
    13,
    {
      align: "right",
    }
  );

  doc.setFont(
    "helvetica",
    "normal"
  );

  doc.setFontSize(7.5);

  doc.setTextColor(
    173,
    207,
    220
  );

  doc.text(
    subtitle,
    196,
    19,
    {
      align: "right",
    }
  );

  doc.setFontSize(6.5);

  doc.text(
    reportCode,
    196,
    24,
    {
      align: "right",
    }
  );

  doc.setDrawColor(
    ...COLORS.cyan
  );

  doc.setLineWidth(0.8);

  doc.line(
    0,
    31,
    210,
    31
  );

  ctx.y = 40;
}


function newPage(ctx) {
  ctx.doc.addPage();

  drawHeader(ctx);
}


function ensureSpace(
  ctx,
  required = 15
) {
  if (
    ctx.y + required > 278
  ) {
    newPage(ctx);
  }
}


function sectionTitle(
  ctx,
  number,
  title
) {
  ensureSpace(
    ctx,
    16
  );

  const { doc } = ctx;

  doc.setFillColor(
    ...COLORS.navy2
  );

  doc.roundedRect(
    14,
    ctx.y,
    182,
    9,
    1.5,
    1.5,
    "F"
  );

  doc.setTextColor(
    ...COLORS.white
  );

  doc.setFont(
    "helvetica",
    "bold"
  );

  doc.setFontSize(9);

  doc.text(
    `${number}. ${title.toUpperCase()}`,
    18,
    ctx.y + 6
  );

  ctx.y += 13;
}


function noteBox(
  ctx,
  text
) {
  ensureSpace(ctx, 18);

  const { doc } = ctx;

  const lines =
    doc.splitTextToSize(
      text,
      172
    );

  const height =
    7 + lines.length * 4;

  doc.setFillColor(
    ...COLORS.light
  );

  doc.setDrawColor(
    ...COLORS.line
  );

  doc.roundedRect(
    14,
    ctx.y,
    182,
    height,
    1.5,
    1.5,
    "FD"
  );

  doc.setFont(
    "helvetica",
    "normal"
  );

  doc.setFontSize(7.3);

  doc.setTextColor(
    ...COLORS.muted
  );

  doc.text(
    lines,
    18,
    ctx.y + 6
  );

  ctx.y +=
    height + 5;
}


// ============================================================
// TABLE HELPERS
// ============================================================

function keyValueTable(
  ctx,
  rows
) {
  const { doc } = ctx;

  const labelWidth = 59;
  const valueWidth = 123;

  for (const [label, value] of rows) {
    const valueLines =
      doc.splitTextToSize(
        val(value),
        valueWidth - 8
      );

    const rowHeight =
      Math.max(
        8,
        4 + valueLines.length * 4
      );

    ensureSpace(
      ctx,
      rowHeight
    );

    doc.setFillColor(
      ...COLORS.light
    );

    doc.setDrawColor(
      ...COLORS.line
    );

    doc.rect(
      14,
      ctx.y,
      labelWidth,
      rowHeight,
      "FD"
    );

    doc.setFillColor(
      255,
      255,
      255
    );

    doc.rect(
      14 + labelWidth,
      ctx.y,
      valueWidth,
      rowHeight,
      "FD"
    );

    doc.setFont(
      "helvetica",
      "bold"
    );

    doc.setFontSize(7.2);

    doc.setTextColor(
      ...COLORS.muted
    );

    doc.text(
      label.toUpperCase(),
      18,
      ctx.y + 5.3
    );

    doc.setFont(
      "helvetica",
      "normal"
    );

    doc.setFontSize(7.6);

    doc.setTextColor(
      ...COLORS.text
    );

    doc.text(
      valueLines,
      77,
      ctx.y + 5.3
    );

    ctx.y += rowHeight;
  }

  ctx.y += 5;
}


function dataTable(
  ctx,
  headers,
  rows,
  widths
) {
  const { doc } = ctx;

  const startX = 14;
  const headerHeight = 8;

  function drawTableHeader() {
    let x = startX;

    doc.setFillColor(
      ...COLORS.navy2
    );

    doc.setDrawColor(
      ...COLORS.navy2
    );

    headers.forEach(
      (header, index) => {
        doc.rect(
          x,
          ctx.y,
          widths[index],
          headerHeight,
          "FD"
        );

        doc.setFont(
          "helvetica",
          "bold"
        );

        doc.setFontSize(6.7);

        doc.setTextColor(
          ...COLORS.white
        );

        doc.text(
          String(header),
          x + 2,
          ctx.y + 5.2
        );

        x += widths[index];
      }
    );

    ctx.y += headerHeight;
  }


  ensureSpace(ctx, 18);

  drawTableHeader();


  rows.forEach(
    (row, rowIndex) => {
      const wrapped =
        row.map(
          (cell, index) =>
            doc.splitTextToSize(
              val(cell),
              widths[index] - 4
            )
        );

      const maxLines =
        Math.max(
          ...wrapped.map(
            (lines) =>
              lines.length
          )
        );

      const rowHeight =
        Math.max(
          8,
          4 + maxLines * 3.7
        );


      if (
        ctx.y + rowHeight > 278
      ) {
        newPage(ctx);

        drawTableHeader();
      }


      let x = startX;

      row.forEach(
        (_, index) => {
          if (
            rowIndex % 2 === 0
          ) {
            doc.setFillColor(
              247,
              250,
              252
            );
          } else {
            doc.setFillColor(
              255,
              255,
              255
            );
          }

          doc.setDrawColor(
            ...COLORS.line
          );

          doc.rect(
            x,
            ctx.y,
            widths[index],
            rowHeight,
            "FD"
          );

          doc.setFont(
            "helvetica",
            "normal"
          );

          doc.setFontSize(6.7);

          doc.setTextColor(
            ...COLORS.text
          );

          doc.text(
            wrapped[index],
            x + 2,
            ctx.y + 5
          );

          x += widths[index];
        }
      );

      ctx.y += rowHeight;
    }
  );

  ctx.y += 6;
}


// ============================================================
// FOOTER
// ============================================================

function finishReport(ctx) {
  const {
    doc,
    reportCode,
  } = ctx;

  const pages =
    doc.getNumberOfPages();

  for (
    let page = 1;
    page <= pages;
    page += 1
  ) {
    doc.setPage(page);

    doc.setDrawColor(
      ...COLORS.line
    );

    doc.setLineWidth(0.3);

    doc.line(
      14,
      286,
      196,
      286
    );

    doc.setFont(
      "helvetica",
      "normal"
    );

    doc.setFontSize(6.3);

    doc.setTextColor(
      ...COLORS.muted
    );

    doc.text(
      "OceanIQ - AI-Driven Multi-Sensor Decision Support System",
      14,
      291
    );

    doc.text(
      "Research Project R26-IT-003",
      105,
      291,
      {
        align: "center",
      }
    );

    doc.text(
      `${reportCode} | Page ${page} of ${pages}`,
      196,
      291,
      {
        align: "right",
      }
    );
  }
}


function createContext(
  title,
  subtitle,
  reportCode
) {
  const doc =
    new jsPDF({
      orientation: "portrait",
      unit: "mm",
      format: "a4",
    });

  const ctx = {
    doc,
    title,
    subtitle,
    reportCode,
    y: 40,
  };

  drawHeader(ctx);

  return ctx;
}


// ============================================================
// RADAR HELPERS
// ============================================================

function radarPrediction(record) {
  return (
    record?.final_prediction ||
    record?.binary_prediction ||
    record?.prediction ||
    record?.classification ||
    "N/A"
  );
}


function radarBird(record) {
  return (
    record?.bird_probability ??
    record?.probabilities?.bird ??
    record?.bird_prob
  );
}


function radarShip(record) {
  return (
    record?.ship_probability ??
    record?.probabilities?.ship ??
    record?.ship_prob
  );
}


// ============================================================
// RADAR REPORT
// ============================================================

export async function generateRadarReport() {
  const response =
    await axios.get(
      `${API_BASE_URL}/radar/history`,
      {
        withCredentials: true,
      }
    );

  const records =
    asArray(
      response.data,
      [
        "records",
        "history",
        "radar_predictions",
      ]
    );

  if (!records.length) {
    throw new Error(
      "No Radar classification records are available."
    );
  }

  const latest =
    newest(records);

  const ctx =
    createContext(
      "Radar Classification Report",
      "Shipboard Radar Object Classification",
      "RADAR REPORT"
    );

  const { doc } = ctx;


  sectionTitle(
    ctx,
    1,
    "Report Information"
  );

  keyValueTable(
    ctx,
    [
      [
        "Generated",
        new Date()
          .toLocaleString(),
      ],
      [
        "Report Type",
        "Radar Object Classification",
      ],
      [
        "Data Source",
        "OceanIQ MongoDB Radar Prediction History",
      ],
      [
        "Records Available",
        records.length,
      ],
    ]
  );


  sectionTitle(
    ctx,
    2,
    "Latest Classification"
  );

  keyValueTable(
    ctx,
    [
      [
        "Timestamp",
        dateValue(
          latest.created_at ||
          latest.timestamp ||
          latest.predicted_at
        ),
      ],
      [
        "Source",
        latest.source,
      ],
      [
        "Image / Filename",
        latest.filename,
      ],
      [
        "Final Prediction",
        radarPrediction(
          latest
        ).toUpperCase(),
      ],
      [
        "Model Confidence",
        pct(
          latest.confidence
        ),
      ],
      [
        "Bird Probability",
        pct(
          radarBird(latest)
        ),
      ],
      [
        "Ship Probability",
        pct(
          radarShip(latest)
        ),
      ],
    ]
  );


  sectionTitle(
    ctx,
    3,
    "Model Information"
  );

  keyValueTable(
    ctx,
    [
      [
        "Model",
        latest.model_name ||
        "RadarTargetCNN",
      ],
      [
        "Model Version",
        latest.model_version,
      ],
      [
        "Validation Accuracy",
        pct(
          latest.validation_accuracy
        ),
      ],
      [
        "Held-out Test Accuracy",
        pct(
          latest.test_accuracy ||
          latest.model_accuracy
        ),
      ],
      [
        "Macro Precision",
        latest.macro_precision != null
          ? fixed(
              latest.macro_precision,
              4
            )
          : "N/A",
      ],
      [
        "Macro Recall",
        latest.macro_recall != null
          ? fixed(
              latest.macro_recall,
              4
            )
          : "N/A",
      ],
      [
        "Macro F1",
        latest.macro_f1 != null
          ? fixed(
              latest.macro_f1,
              4
            )
          : "N/A",
      ],
    ]
  );


  sectionTitle(
    ctx,
    4,
    "Classification Probability"
  );

  dataTable(
    ctx,
    [
      "Class",
      "Probability",
      "Result",
    ],
    [
      [
        "Bird",
        pct(
          radarBird(latest)
        ),
        radarPrediction(latest)
          .toLowerCase() ===
        "bird"
          ? "Selected"
          : "-",
      ],
      [
        "Ship",
        pct(
          radarShip(latest)
        ),
        radarPrediction(latest)
          .toLowerCase() ===
        "ship"
          ? "Selected"
          : "-",
      ],
    ],
    [
      55,
      55,
      72,
    ]
  );


  sectionTitle(
    ctx,
    5,
    "Recent Classification History"
  );

  const recent =
    [...records]
      .slice(0, 12);

  dataTable(
    ctx,
    [
      "Time",
      "File",
      "Prediction",
      "Conf.",
      "Bird",
      "Ship",
    ],
    recent.map(
      (record) => [
        dateValue(
          record.created_at ||
          record.timestamp ||
          record.predicted_at
        ),
        record.filename,
        radarPrediction(
          record
        ),
        pct(
          record.confidence
        ),
        pct(
          radarBird(record)
        ),
        pct(
          radarShip(record)
        ),
      ]
    ),
    [
      37,
      43,
      27,
      23,
      26,
      26,
    ]
  );


  noteBox(
    ctx,
    "This report summarizes stored OceanIQ RadarTargetCNN classification results. Model evaluation metrics describe the finalized deployed Radar model and should be interpreted together with the documented held-out test methodology."
  );


  finishReport(ctx);


  const stamp =
    new Date()
      .toISOString()
      .replace(/[:.]/g, "-");

  doc.save(
    `OceanIQ_Radar_Report_${stamp}.pdf`
  );
}


// ============================================================
// SIMULATION HELPERS
// ============================================================

function vesselData(event, key) {
  const raw =
    event?.[key] || {};

  const state =
    raw.state || raw;

  return {
    raw,
    state,
    prediction:
      raw.motion_prediction ||
      state.motion_prediction ||
      null,
  };
}


function eventRadarPrediction(event) {
  return (
    event?.radar_classification ||
    event?.sar_classification ||
    event?.classification
      ?.final_prediction ||
    event?.classification
      ?.prediction ||
    event?.classification
      ?.binary_prediction ||
    "N/A"
  );
}


function eventRadarConfidence(event) {
  return (
    event?.radar_confidence ??
    event?.classification
      ?.confidence
  );
}


function predictionValue(
  prediction,
  keys
) {
  if (!prediction) {
    return "N/A";
  }

  for (const key of keys) {
    if (
      prediction[key] !==
        null &&
      prediction[key] !==
        undefined
    ) {
      return prediction[key];
    }
  }

  return "N/A";
}


// ============================================================
// LIVE SIMULATION REPORT
// ============================================================

export async function generateSimulationReport() {
  const [
    runsResponse,
    eventsResponse,
  ] = await Promise.all([
    axios.get(
      `${API_BASE_URL}/simulation/database/runs`,
      {
        withCredentials: true,
      }
    ),

    axios.get(
      `${API_BASE_URL}/simulation/database/events`,
      {
        withCredentials: true,
      }
    ),
  ]);


  const runs =
    asArray(
      runsResponse.data,
      [
        "runs",
        "records",
        "simulation_runs",
      ]
    );

  const allEvents =
    asArray(
      eventsResponse.data,
      [
        "events",
        "records",
        "simulation_events",
      ]
    );


  if (
    !runs.length &&
    !allEvents.length
  ) {
    throw new Error(
      "No Live Simulation records are available."
    );
  }


  const run =
    newest(runs);


  const simulationId =
    run.simulation_id ||
    run.scenario_id ||
    newest(allEvents)
      .simulation_id;


  let events =
    simulationId
      ? allEvents.filter(
          (event) =>
            event.simulation_id ===
            simulationId
        )
      : allEvents;


  if (!events.length) {
    events =
      allEvents;
  }


  events =
    [...events].sort(
      (a, b) =>
        Number(
          a.frame_number || 0
        ) -
        Number(
          b.frame_number || 0
        )
    );


  const latest =
    events[
      events.length - 1
    ] || {};


  const own =
    vesselData(
      latest,
      "own_vessel"
    );

  const target =
    vesselData(
      latest,
      "target_vessel"
    );


  const encounter =
    latest.encounter || {};

  const environment =
    latest.environment || {};

  const atmosphere =
    environment.atmospheric || {};

  const marine =
    environment.marine || {};


  const ctx =
    createContext(
      "Live Simulation Report",
      "AIS, GRU Motion Forecasting & Collision Risk",
      "SIMULATION REPORT"
    );


  const { doc } = ctx;


  sectionTitle(
    ctx,
    1,
    "Simulation Summary"
  );

  keyValueTable(
    ctx,
    [
      [
        "Generated",
        new Date()
          .toLocaleString(),
      ],
      [
        "Simulation ID",
        simulationId,
      ],
      [
        "Mode",
        run.mode ||
        latest.scenario?.mode,
      ],
      [
        "Status",
        run.status ||
        (latest.running
          ? "Running"
          : "Stopped"),
      ],
      [
        "Frame Count",
        run.frame_count ||
        latest.frame_number,
      ],
      [
        "Started",
        dateValue(
          run.started_at
        ),
      ],
      [
        "Ended",
        dateValue(
          run.ended_at
        ),
      ],
      [
        "Scenario",
        run.scenario_description ||
        latest.scenario
          ?.description,
      ],
    ]
  );


  sectionTitle(
    ctx,
    2,
    "Latest Radar Observation"
  );

  keyValueTable(
    ctx,
    [
      [
        "Radar Image",
        latest.sar_filename ||
        latest.filename,
      ],
      [
        "Classification",
        eventRadarPrediction(
          latest
        ),
      ],
      [
        "Confidence",
        pct(
          eventRadarConfidence(
            latest
          )
        ),
      ],
      [
        "Radar Model",
        latest.classification
          ?.model_name ||
        "RadarTargetCNN",
      ],
    ]
  );


  sectionTitle(
    ctx,
    3,
    "Vessel States"
  );

  dataTable(
    ctx,
    [
      "Vessel",
      "Latitude",
      "Longitude",
      "Speed",
      "Course",
      "History",
    ],
    [
      [
        "Own Vessel",
        fixed(
          own.state.latitude,
          5
        ),
        fixed(
          own.state.longitude,
          5
        ),
        `${fixed(
          own.state
            .speed_over_ground_knots ??
          own.state.sog,
          2
        )} kn`,
        `${fixed(
          own.state
            .course_over_ground_degrees ??
          own.state.cog,
          1
        )} deg`,
        `${val(
          own.raw
            .history_available_minutes ??
          own.state
            .history_available_minutes
        )} min`,
      ],

      [
        "Target Vessel",
        fixed(
          target.state.latitude,
          5
        ),
        fixed(
          target.state.longitude,
          5
        ),
        `${fixed(
          target.state
            .speed_over_ground_knots ??
          target.state.sog,
          2
        )} kn`,
        `${fixed(
          target.state
            .course_over_ground_degrees ??
          target.state.cog,
          1
        )} deg`,
        `${val(
          target.raw
            .history_available_minutes ??
          target.state
            .history_available_minutes
        )} min`,
      ],
    ],
    [
      30,
      32,
      32,
      28,
      28,
      32,
    ]
  );


  sectionTitle(
    ctx,
    4,
    "GRU Motion Forecast"
  );

  dataTable(
    ctx,
    [
      "Vessel",
      "Status",
      "Motion",
      "Future Lat",
      "Future Lon",
      "Future Speed",
    ],
    [
      [
        "Own Vessel",

        own.prediction
          ? "Available"
          : "Waiting",

        predictionValue(
          own.prediction,
          [
            "motion_class",
            "predicted_class",
            "class_name",
          ]
        ),

        fixed(
          predictionValue(
            own.prediction,
            [
              "future_latitude",
              "predicted_latitude",
              "latitude",
            ]
          ),
          5
        ),

        fixed(
          predictionValue(
            own.prediction,
            [
              "future_longitude",
              "predicted_longitude",
              "longitude",
            ]
          ),
          5
        ),

        `${fixed(
          predictionValue(
            own.prediction,
            [
              "speed_knots",
              "predicted_speed_knots",
              "future_speed_knots",
            ]
          ),
          2
        )} kn`,
      ],

      [
        "Target Vessel",

        target.prediction
          ? "Available"
          : "Waiting",

        predictionValue(
          target.prediction,
          [
            "motion_class",
            "predicted_class",
            "class_name",
          ]
        ),

        fixed(
          predictionValue(
            target.prediction,
            [
              "future_latitude",
              "predicted_latitude",
              "latitude",
            ]
          ),
          5
        ),

        fixed(
          predictionValue(
            target.prediction,
            [
              "future_longitude",
              "predicted_longitude",
              "longitude",
            ]
          ),
          5
        ),

        `${fixed(
          predictionValue(
            target.prediction,
            [
              "speed_knots",
              "predicted_speed_knots",
              "future_speed_knots",
            ]
          ),
          2
        )} kn`,
      ],
    ],
    [
      27,
      25,
      31,
      33,
      33,
      33,
    ]
  );


  sectionTitle(
    ctx,
    5,
    "Collision Risk Assessment"
  );

  keyValueTable(
    ctx,
    [
      [
        "DCPA",
        encounter.dcpa_nm != null
          ? `${fixed(
              encounter.dcpa_nm,
              3
            )} NM`
          : "N/A",
      ],
      [
        "TCPA",
        encounter.tcpa_minutes != null
          ? `${fixed(
              encounter.tcpa_minutes,
              2
            )} min`
          : "N/A",
      ],
      [
        "Risk Level",
        encounter.risk_level ||
        encounter.risk,
      ],
      [
        "Separation",
        encounter.separation_nm != null
          ? `${fixed(
              encounter.separation_nm,
              3
            )} NM`
          : "N/A",
      ],
    ]
  );


  sectionTitle(
    ctx,
    6,
    "Atmospheric Conditions"
  );

  dataTable(
    ctx,
    [
      "Parameter",
      "Value",
      "Parameter",
      "Value",
    ],
    [
      [
        "Temperature",
        atmosphere.temperature_c != null
          ? `${fixed(
              atmosphere.temperature_c,
              1
            )} C`
          : "N/A",

        "Humidity",
        atmosphere
          .relative_humidity_percent != null
          ? `${fixed(
              atmosphere
                .relative_humidity_percent,
              0
            )}%`
          : "N/A",
      ],

      [
        "Wind Speed",
        atmosphere
          .wind_speed_knots != null
          ? `${fixed(
              atmosphere
                .wind_speed_knots,
              2
            )} kn`
          : "N/A",

        "Wind Direction",
        atmosphere
          .wind_direction_degrees != null
          ? `${fixed(
              atmosphere
                .wind_direction_degrees,
              0
            )} deg`
          : "N/A",
      ],

      [
        "Wind Gust",
        atmosphere
          .wind_gust_knots != null
          ? `${fixed(
              atmosphere
                .wind_gust_knots,
              2
            )} kn`
          : "N/A",

        "Visibility",
        atmosphere.visibility_km != null
          ? `${fixed(
              atmosphere.visibility_km,
              1
            )} km`
          : "N/A",
      ],

      [
        "Pressure",
        atmosphere
          .pressure_msl_hpa != null
          ? `${fixed(
              atmosphere
                .pressure_msl_hpa,
              1
            )} hPa`
          : "N/A",

        "Rain",
        atmosphere.rain_mm != null
          ? `${fixed(
              atmosphere.rain_mm,
              2
            )} mm`
          : "N/A",
      ],
    ],
    [
      45,
      46,
      45,
      46,
    ]
  );


  sectionTitle(
    ctx,
    7,
    "Marine Conditions"
  );

  if (marine.available === false) {
    noteBox(
      ctx,
      marine.reason ||
      "Marine environmental information is unavailable for this encounter location."
    );
  } else {
    dataTable(
      ctx,
      [
        "Parameter",
        "Value",
        "Parameter",
        "Value",
      ],
      [
        [
          "Wave Height",
          marine.wave_height_m != null
            ? `${fixed(
                marine.wave_height_m,
                2
              )} m`
            : "N/A",

          "Wave Direction",
          marine
            .wave_direction_degrees != null
            ? `${fixed(
                marine
                  .wave_direction_degrees,
                0
              )} deg`
            : "N/A",
        ],

        [
          "Wave Period",
          marine
            .wave_period_seconds != null
            ? `${fixed(
                marine
                  .wave_period_seconds,
                1
              )} s`
            : "N/A",

          "Swell Height",
          marine.swell_height_m != null
            ? `${fixed(
                marine.swell_height_m,
                2
              )} m`
            : "N/A",
        ],

        [
          "Swell Direction",
          marine
            .swell_direction_degrees != null
            ? `${fixed(
                marine
                  .swell_direction_degrees,
                0
              )} deg`
            : "N/A",

          "Swell Period",
          marine
            .swell_period_seconds != null
            ? `${fixed(
                marine
                  .swell_period_seconds,
                1
              )} s`
            : "N/A",
        ],

        [
          "Sea Temperature",
          marine
            .sea_surface_temperature_c != null
            ? `${fixed(
                marine
                  .sea_surface_temperature_c,
                1
              )} C`
            : "N/A",

          "Current Speed",
          marine
            .ocean_current_speed_knots != null
            ? `${fixed(
                marine
                  .ocean_current_speed_knots,
                2
              )} kn`
            : "N/A",
        ],
      ],
      [
        45,
        46,
        45,
        46,
      ]
    );
  }


  sectionTitle(
    ctx,
    8,
    "Recent Simulation Events"
  );

  const recent =
    events.slice(-12);

  dataTable(
    ctx,
    [
      "Frame",
      "Radar",
      "Conf.",
      "DCPA",
      "TCPA",
      "Risk",
    ],
    recent.map(
      (event) => [
        event.frame_number,

        eventRadarPrediction(
          event
        ),

        pct(
          eventRadarConfidence(
            event
          )
        ),

        event.encounter?.dcpa_nm != null
          ? fixed(
              event.encounter
                .dcpa_nm,
              2
            )
          : "N/A",

        event.encounter
          ?.tcpa_minutes != null
          ? fixed(
              event.encounter
                .tcpa_minutes,
              1
            )
          : "N/A",

        event.encounter
          ?.risk_level ||
        event.encounter?.risk ||
        "N/A",
      ]
    ),
    [
      22,
      43,
      28,
      29,
      29,
      31,
    ]
  );


  noteBox(
    ctx,
    "Environmental observations are contextual decision-support information and are not direct inputs to RadarTargetCNN or the GRU motion forecasting model. DCPA/TCPA and risk assessment are calculated from vessel motion states."
  );


  finishReport(ctx);


  const stamp =
    new Date()
      .toISOString()
      .replace(/[:.]/g, "-");

  doc.save(
    `OceanIQ_Live_Simulation_Report_${stamp}.pdf`
  );
}


// ============================================================
// DATABASE REPORT
// ============================================================

export async function generateDatabaseReport() {
  const [
    radarResponse,
    runsResponse,
    eventsResponse,
  ] = await Promise.all([
    axios.get(
      `${API_BASE_URL}/radar/history`,
      { withCredentials: true }
    ),

    axios.get(
      `${API_BASE_URL}/simulation/database/runs`,
      { withCredentials: true }
    ),

    axios.get(
      `${API_BASE_URL}/simulation/database/events`,
      { withCredentials: true }
    ),
  ]);

  const radarRecords = asArray(
    radarResponse.data,
    [
      "records",
      "history",
      "radar_predictions",
    ]
  );

  const runs = asArray(
    runsResponse.data,
    [
      "runs",
      "records",
      "simulation_runs",
    ]
  );

  const events = asArray(
    eventsResponse.data,
    [
      "events",
      "records",
      "simulation_events",
    ]
  );

  const ctx = createContext(
    "Database Records Report",
    "Stored OceanIQ Intelligence Records",
    "DATABASE REPORT"
  );

  const { doc } = ctx;


  // ----------------------------------------------------------
  // DATABASE SUMMARY
  // ----------------------------------------------------------

  sectionTitle(
    ctx,
    1,
    "Database Summary"
  );

  keyValueTable(
    ctx,
    [
      [
        "Generated",
        new Date().toLocaleString(),
      ],
      [
        "Database",
        "OceanIQ MongoDB",
      ],
      [
        "Radar Records",
        radarRecords.length,
      ],
      [
        "Simulation Runs",
        runs.length,
      ],
      [
        "Simulation Events",
        events.length,
      ],
      [
        "Collections",
        "radar_predictions, simulation_runs, simulation_events",
      ],
    ]
  );


  // ----------------------------------------------------------
  // RADAR RECORDS
  // ----------------------------------------------------------

  sectionTitle(
    ctx,
    2,
    "Radar Classification Records"
  );

  if (!radarRecords.length) {
    noteBox(
      ctx,
      "No Radar classification records are currently stored."
    );
  } else {
    dataTable(
      ctx,
      [
        "Time",
        "Filename",
        "Prediction",
        "Confidence",
        "Model",
      ],

      radarRecords
        .slice(0, 20)
        .map((record) => [
          dateValue(
            record.created_at ||
            record.timestamp ||
            record.predicted_at
          ),

          record.filename,

          radarPrediction(record),

          pct(record.confidence),

          record.model_name ||
          "RadarTargetCNN",
        ]),

      [
        38,
        52,
        31,
        28,
        33,
      ]
    );
  }


  // ----------------------------------------------------------
  // SIMULATION RUNS
  // ----------------------------------------------------------

  sectionTitle(
    ctx,
    3,
    "Simulation Runs"
  );

  if (!runs.length) {
    noteBox(
      ctx,
      "No simulation runs are currently stored."
    );
  } else {
    dataTable(
      ctx,
      [
        "Simulation ID",
        "Mode",
        "Status",
        "Frames",
        "Started",
      ],

      runs
        .slice(0, 15)
        .map((run) => [
          run.simulation_id ||
          run.scenario_id,

          run.mode,

          run.status,

          run.frame_count,

          dateValue(
            run.started_at
          ),
        ]),

      [
        61,
        29,
        28,
        22,
        42,
      ]
    );
  }


  // ----------------------------------------------------------
  // SIMULATION EVENTS
  // ----------------------------------------------------------

  sectionTitle(
    ctx,
    4,
    "Recent Simulation Events"
  );

  if (!events.length) {
    noteBox(
      ctx,
      "No simulation events are currently stored."
    );
  } else {
    const recentEvents =
      [...events]
        .slice(-25)
        .reverse();

    dataTable(
      ctx,
      [
        "Frame",
        "Radar",
        "Conf.",
        "DCPA",
        "TCPA",
        "Risk",
      ],

      recentEvents.map(
        (event) => [
          event.frame_number,

          eventRadarPrediction(
            event
          ),

          pct(
            eventRadarConfidence(
              event
            )
          ),

          event.encounter
            ?.dcpa_nm != null
            ? fixed(
                event.encounter.dcpa_nm,
                2
              )
            : "N/A",

          event.encounter
            ?.tcpa_minutes != null
            ? fixed(
                event.encounter.tcpa_minutes,
                1
              )
            : "N/A",

          event.encounter
            ?.risk_level ||
          event.encounter
            ?.risk ||
          "N/A",
        ]
      ),

      [
        22,
        43,
        27,
        29,
        29,
        32,
      ]
    );
  }


  // ----------------------------------------------------------
  // LATEST ENVIRONMENTAL RECORD
  // ----------------------------------------------------------

  const latestEvent =
    events.length
      ? events[events.length - 1]
      : {};

  const environment =
    latestEvent.environment || {};

  const atmospheric =
    environment.atmospheric || {};

  const marine =
    environment.marine || {};


  sectionTitle(
    ctx,
    5,
    "Latest Stored Environmental Record"
  );

  if (!environment.available) {
    noteBox(
      ctx,
      "No environmental record is available in the latest stored simulation event."
    );
  } else {
    dataTable(
      ctx,
      [
        "Parameter",
        "Value",
        "Parameter",
        "Value",
      ],

      [
        [
          "Temperature",
          atmospheric.temperature_c != null
            ? `${fixed(
                atmospheric.temperature_c,
                1
              )} C`
            : "N/A",

          "Wind Speed",
          atmospheric.wind_speed_knots != null
            ? `${fixed(
                atmospheric.wind_speed_knots,
                2
              )} kn`
            : "N/A",
        ],

        [
          "Visibility",
          atmospheric.visibility_km != null
            ? `${fixed(
                atmospheric.visibility_km,
                1
              )} km`
            : "N/A",

          "Pressure",
          atmospheric.pressure_msl_hpa != null
            ? `${fixed(
                atmospheric.pressure_msl_hpa,
                1
              )} hPa`
            : "N/A",
        ],

        [
          "Wave Height",
          marine.wave_height_m != null
            ? `${fixed(
                marine.wave_height_m,
                2
              )} m`
            : "N/A",

          "Swell Height",
          marine.swell_height_m != null
            ? `${fixed(
                marine.swell_height_m,
                2
              )} m`
            : "N/A",
        ],

        [
          "Sea Temperature",
          marine.sea_surface_temperature_c != null
            ? `${fixed(
                marine.sea_surface_temperature_c,
                1
              )} C`
            : "N/A",

          "Ocean Current",
          marine.ocean_current_speed_knots != null
            ? `${fixed(
                marine.ocean_current_speed_knots,
                2
              )} kn`
            : "N/A",
        ],
      ],

      [
        45,
        46,
        45,
        46,
      ]
    );
  }


  // ----------------------------------------------------------
  // STORAGE INFORMATION
  // ----------------------------------------------------------

  sectionTitle(
    ctx,
    6,
    "Storage Information"
  );

  noteBox(
    ctx,
    "OceanIQ stores Radar classification results, simulation runs and per-frame simulation events in MongoDB. Simulation events may contain Radar classification, AIS vessel state, GRU prediction, DCPA/TCPA collision assessment and environmental context."
  );


  finishReport(ctx);

  const stamp =
    new Date()
      .toISOString()
      .replace(/[:.]/g, "-");

  doc.save(
    `OceanIQ_Database_Report_${stamp}.pdf`
  );
}
