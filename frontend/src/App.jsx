/**
 * ╔══════════════════════════════════════════════════════════╗
 * ║  OceanIQ — Marine AI Intelligence Platform               ║
 * ║  Research Project R26-IT-003                             ║
 * ║                                                          ║
 * ║  App.jsx — Main Application                              ║
 * ║  Theme: Maritime Defense Command Console                 ║
 * ║  Layout: Persistent left sidebar + main analysis area    ║
 * ╚══════════════════════════════════════════════════════════╝
 *
 * Install required packages:
 *   npm install axios lucide-react
 */

import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import { jsPDF } from "jspdf";
import {
  Anchor,
  Waves,
  RadioTower,
  Ship,
  Upload,
  Activity,
  Cpu,
  Shield,
  Eye,
  EyeOff,
  AlertTriangle,
  CheckCircle2,
  Loader,
  ScanEye,
  Zap,
  BarChart2,
  Crosshair,
  ChevronRight,
  Network,
  Globe,
  Layers,
  Navigation,
  UserRoundCog,
  LockKeyhole,
} from "lucide-react";
import LiveSimulation from "./LiveSimulation";
import ProtectedRoute from "./components/ProtectedRoute";
import LogoutButton from "./components/LogoutButton";
import AdminUsers from "./components/AdminUsers";
import { useAuth } from "./context/AuthContext";
import "./App.css";

/* ══════════════════════════════════════════════════════════
   BACKEND CONFIGURATION — do not modify endpoint names
   ══════════════════════════════════════════════════════════ */
const API_BASE_URL = "http://localhost:8000";

/* ══════════════════════════════════════════════════════════
   MODULE DEFINITIONS
   Each module maps to a backend endpoint and has its own
   color identity used throughout the UI.
   ══════════════════════════════════════════════════════════ */
const MODULES = {
  hull: {
    id: "hull",
    label: "Hull Defect",
    title: "Hull Defect Detection",
    endpoint: `${API_BASE_URL}/predict-hull-defect`,
    description: "CNN-based detection of corrosion, cracks, and biofouling, and paint damage with Grad-CAM explainability.",
    icon: Anchor,
    color: "#00d4ff",
    colorDim: "#00d4ff22",
    colorMid: "#00d4ff55",
    tag: "CNN · Grad-CAM",
    statusLabel: "HULL SCANNER",
  },
  sea: {
    id: "sea",
    label: "Sea State",
    title: "Sea State Classification",
    endpoint: `${API_BASE_URL}/predict-sea-state`,
    description: "Multi-class deep learning model classifying sea surface conditions from imagery.",
    icon: Waves,
    color: "#00ffb3",
    colorDim: "#00ffb322",
    colorMid: "#00ffb355",
    tag: "Multi-class CNN",
    statusLabel: "SEA ANALYZER",
  },
  boat: {
    id: "boat",
    label: "Boat Detection",
    title: "Vessel Detection",
    endpoint: `${API_BASE_URL}/predict-boat-detection`,
    description: "YOLO-based real-time object detection for maritime vessel identification.",
    icon: Ship,
    color: "#a78bfa",
    colorDim: "#a78bfa22",
    colorMid: "#a78bfa55",
    tag: "YOLO · Object Detection",
    statusLabel: "VESSEL TRACKER",
  },
  radar: {
    id: "radar",
    label: "Radar Objects",
    title: "Radar Object Classification",
    endpoint: `${API_BASE_URL}/predict-radar-object`,
    description: "Fused YOLO + CNN pipeline classifying radar signatures as bird, ship, or unknown.",
    icon: RadioTower,
    color: "#f97316",
    colorDim: "#f9731622",
    colorMid: "#f9731655",
    tag: "YOLO + CNN Fusion",
    statusLabel: "RADAR ANALYSIS",
  },
  simulation: {
    id: "simulation",
    label: "Live Simulation",
    title: "Live Maritime Simulation",
    description: "Streams SAR images, replays two AIS trajectories, forecasts motion with the selected GRU and assesses DCPA/TCPA collision risk.",
    icon: Navigation,
    color: "#00d4ff",
    colorDim: "#00d4ff22",
    colorMid: "#00d4ff55",
    tag: "SAR · AIS · GRU · CPA",
    statusLabel: "SIMULATION CONTROL",
  },
};

/* ══════════════════════════════════════════════════════════
   RADAR GRID CANVAS BACKGROUND
   Draws an animated radar sweep on a dot-grid background.
   This creates the "command center" feel of the dashboard.
   Uses requestAnimationFrame for smooth 60fps rendering.
   ══════════════════════════════════════════════════════════ */
function RadarCanvas() {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let frame;
    let angle = 0;

    const draw = () => {
      const W = (canvas.width = canvas.offsetWidth);
      const H = (canvas.height = canvas.offsetHeight);
      const cx = W / 2;
      const cy = H / 2;
      const R = Math.min(W, H) * 0.42;

      ctx.clearRect(0, 0, W, H);

      // Dot grid — creates the "tactical display" texture
      ctx.fillStyle = "rgba(0, 212, 255, 0.07)";
      const spacing = 28;
      for (let x = 0; x < W; x += spacing)
        for (let y = 0; y < H; y += spacing) {
          ctx.beginPath();
          ctx.arc(x, y, 1, 0, Math.PI * 2);
          ctx.fill();
        }

      // Concentric rings — radar range indicators
      [0.3, 0.55, 0.78, 1].forEach((s) => {
        ctx.beginPath();
        ctx.arc(cx, cy, R * s, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0, 212, 255, 0.09)";
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Cross-hairs — bearing lines
      ctx.strokeStyle = "rgba(0, 212, 255, 0.07)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke();

      // Sweep trail — drawn as many thin arcs fading out
      const sweepLength = Math.PI * 0.6;
      for (let i = 0; i < 60; i++) {
        const a = angle - (i / 60) * sweepLength;
        const alpha = (1 - i / 60) * 0.18;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, R, a, a + 0.045);
        ctx.closePath();
        ctx.fillStyle = `rgba(0, 212, 255, ${alpha})`;
        ctx.fill();
      }

      // Bright sweep line
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(angle) * R, cy + Math.sin(angle) * R);
      ctx.strokeStyle = "rgba(0, 212, 255, 0.65)";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Center beacon dot
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(0, 212, 255, 0.9)";
      ctx.fill();

      angle = (angle + 0.015) % (Math.PI * 2);
      frame = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(frame);
  }, []);

  return <canvas ref={ref} className="radar-canvas" aria-hidden="true" />;
}

/* ══════════════════════════════════════════════════════════
   SYSTEM CLOCK — live HH:MM:SS display in sidebar footer
   ══════════════════════════════════════════════════════════ */
function SystemClock() {
  const [time, setTime] = useState("");
  useEffect(() => {
    const tick = () => setTime(new Date().toTimeString().slice(0, 8));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return <span className="sys-clock">{time}</span>;
}

/* ══════════════════════════════════════════════════════════
   SYSTEM STATUS ROWS — shows AI subsystem states
   Blinking dots reinforce the "live system" aesthetic
   ══════════════════════════════════════════════════════════ */
const STATUS_LINES = [
  { label: "AI ENGINE",   state: "ONLINE" },
  { label: "MODEL CACHE", state: "READY" },
  { label: "INFERENCE",   state: "ARMED" },
  { label: "BACKEND",     state: "ACTIVE" },
];

function StatusRows() {
  return (
    <div className="status-rows">
      {STATUS_LINES.map((s) => (
        <div className="status-row" key={s.label}>
          <span className="status-blink" />
          <span className="status-key">{s.label}</span>
          <span className="status-val">{s.state}</span>
        </div>
      ))}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════
   CONFIDENCE RING — SVG circular gauge for confidence %
   Used in Hull and Sea State result displays.
   ══════════════════════════════════════════════════════════ */
function ConfidenceRing({ value, color }) {
  const r = 44;
  const circ = 2 * Math.PI * r;
  const filled = (Math.min(100, Math.max(0, value)) / 100) * circ;

  return (
    <div className="conf-ring-wrap">
      <svg viewBox="0 0 110 110" width="110" height="110">
        {/* Background track ring */}
        <circle cx="55" cy="55" r={r} fill="none"
          stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
        {/* Filled confidence arc */}
        <circle cx="55" cy="55" r={r} fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circ - filled}`}
          strokeDashoffset={circ * 0.25}
          style={{
            filter: `drop-shadow(0 0 8px ${color})`,
            transition: "stroke-dasharray 1.2s cubic-bezier(0.22,1,0.36,1)",
          }}
        />
        <text x="55" y="51" textAnchor="middle"
          fill="white" fontSize="17" fontWeight="700" fontFamily="'Space Mono', monospace">
          {Math.round(value)}%
        </text>
        <text x="55" y="66" textAnchor="middle"
          fill="rgba(255,255,255,0.35)" fontSize="7.5" fontFamily="'Space Mono', monospace"
          letterSpacing="1.5">
          CONFIDENCE
        </text>
      </svg>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════
   DATA ROW — key/value pair used in the results panel
   ══════════════════════════════════════════════════════════ */
function DataRow({ label, value, color }) {
  if (!value && value !== 0) return null;
  return (
    <div className="data-row">
      <span className="data-label">{label}</span>
      <span className="data-value" style={color ? { color } : {}}>{value}</span>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════
   PROBABILITY BAR — animated fill bar for class probs
   ══════════════════════════════════════════════════════════ */
function ProbBar({ label, value, color }) {
  return (
    <div className="prob-bar">
      <div className="prob-bar-header">
        <span>{label}</span>
        <strong style={{ color }}>{value}%</strong>
      </div>
      <div className="prob-bar-track">
        <div className="prob-bar-fill"
          style={{ width: `${value}%`, background: `linear-gradient(90deg, ${color}66, ${color})` }} />
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════
   MAIN APP COMPONENT
   ══════════════════════════════════════════════════════════ */
function AppContent() {
  const { user, canWrite, isAdmin, accessLevel } = useAuth();

  /* ── Application state ── */
  const [activeModule, setActiveModule] = useState("hull");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showGradcam, setShowGradcam] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false); // mobile sidebar toggle
  const [applyEnhancement, setApplyEnhancement] = useState(false);
  const [seaHistory, setSeaHistory] = useState([]);
  const [seaHistoryLoading, setSeaHistoryLoading] = useState(false);
  const [showAccessAdmin, setShowAccessAdmin] = useState(false);

  const mod = MODULES[activeModule];
  const ModIcon = mod.icon;

  /* ── Sea-state history helpers ── */
  const fetchSeaHistory = useCallback(async () => {
    try {
      setSeaHistoryLoading(true);
      const { data } = await axios.get(`${API_BASE_URL}/sea-state-history`);
      setSeaHistory(data.history || []);
    } catch (error) {
      console.error("Failed to load sea-state history:", error);
    } finally {
      setSeaHistoryLoading(false);
    }
  }, []);

  const clearSeaHistory = async () => {
    if (!canWrite) return;
    try {
      await axios.delete(`${API_BASE_URL}/sea-state-history`);
      setSeaHistory([]);
    } catch (error) {
      console.error("Failed to clear sea-state history:", error);
    }
  };

  /* ── File selection handler ── */
  const processFile = useCallback((f) => {
    if (!canWrite || !f || !f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setShowGradcam(false);
    setPreview(URL.createObjectURL(f));
  }, [canWrite]);

  const handleFileChange = (e) => processFile(e.target.files[0]);

  /* ── Drag and drop handlers ── */
  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    processFile(e.dataTransfer.files[0]);
  };

  /* ── Switch between AI modules — clears state ── */
  const switchModule = (id) => {
    setActiveModule(id);
    setFile(null);
    setPreview(null);
    setResult(null);
    setShowGradcam(false);
    setApplyEnhancement(false);
    setSidebarOpen(false);
    if (id === "sea") fetchSeaHistory();
  };

  /* ══════════════════════════════════════════════
     BACKEND CALL
     axios POST to the selected module endpoint.
     FormData carries the image file.
     All original endpoints are preserved.
     ══════════════════════════════════════════════ */
  const runPrediction = async () => {
    if (!file || loading || !canWrite) return;
    const form = new FormData();
    form.append("file", file);
    if (activeModule === "sea") {
      form.append("apply_enhancement", applyEnhancement);
    }
    try {
      setLoading(true);
      setResult(null);
      const { data } = await axios.post(mod.endpoint, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      if (data?.error) {
        setResult({ __error: true, message: data.error, details: data.validation });
      } else {
        setResult(data);
      }
      if (activeModule === "sea") fetchSeaHistory();
    } catch (err) {
      console.error("Prediction error:", err);
      setResult({
        __error: true,
        message: `Connection failed — is the backend running at ${API_BASE_URL}?`,
      });
    } finally {
      setLoading(false);
    }
  };

  /* ── Sea-state PDF report from the Sea-State branch ── */
  const generateSeaStatePDF = () => {
    if (!result || activeModule !== "sea" || result.__error) return;

    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 20;
    const contentWidth = pageWidth - margin * 2;
    let y = 20;

    const addFooter = () => {
      const pageNumber = doc.internal.getNumberOfPages();
      doc.setFontSize(8);
      doc.setTextColor(100, 100, 100);
      doc.text("OceanIQ - Marine AI Intelligence Platform", margin, pageHeight - 10);
      doc.text(`Page ${pageNumber}`, pageWidth - margin, pageHeight - 10, { align: "right" });
    };

    const checkPageSpace = (requiredSpace = 10) => {
      if (y + requiredSpace > pageHeight - 25) {
        addFooter();
        doc.addPage();
        y = 20;
      }
    };

    const addTitle = (title) => {
      checkPageSpace(15);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(13);
      doc.setTextColor(0, 120, 180);
      doc.text(title, margin, y);
      y += 8;
    };

    const addText = (text, indent = 0) => {
      if (text === undefined || text === null) return;
      const lines = doc.splitTextToSize(String(text), contentWidth - indent);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.setTextColor(30, 30, 30);
      lines.forEach((line) => {
        checkPageSpace(6);
        doc.text(line, margin + indent, y);
        y += 5.5;
      });
      y += 1;
    };

    const addKeyValue = (label, value) => {
      checkPageSpace(8);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.setTextColor(30, 30, 30);
      doc.text(`${label}:`, margin, y);
      doc.setFont("helvetica", "normal");
      const labelWidth = doc.getTextWidth(`${label}: `);
      const lines = doc.splitTextToSize(String(value ?? "N/A"), contentWidth - labelWidth);
      doc.text(lines[0], margin + labelWidth, y);
      y += 5.5;
      for (let i = 1; i < lines.length; i += 1) {
        checkPageSpace(6);
        doc.text(lines[i], margin + labelWidth, y);
        y += 5.5;
      }
      y += 1;
    };

    const addBullet = (text) => {
      const lines = doc.splitTextToSize(String(text), contentWidth - 8);
      lines.forEach((line, index) => {
        checkPageSpace(6);
        doc.text(index === 0 ? `• ${line}` : `  ${line}`, margin + 3, y);
        y += 5.5;
      });
      y += 1;
    };

    doc.setFont("helvetica", "bold");
    doc.setFontSize(20);
    doc.setTextColor(20, 20, 20);
    doc.text("OceanIQ Marine AI Intelligence Platform", pageWidth / 2, y, { align: "center" });
    y += 9;
    doc.setFontSize(15);
    doc.setTextColor(0, 120, 180);
    doc.text("Sea State Classification Report", pageWidth / 2, y, { align: "center" });
    y += 12;
    doc.setDrawColor(0, 160, 200);
    doc.line(margin, y, pageWidth - margin, y);
    y += 10;

    addTitle("Prediction Information");
    addKeyValue("File", result.filename || "N/A");
    addKeyValue("Timestamp", result.timestamp || new Date().toLocaleString());
    addKeyValue("Processing Time", `${result.processing_time ?? "N/A"} sec`);
    addKeyValue("Enhancement Applied", result.enhancement_applied ? "Yes" : "No");
    y += 3;

    addTitle("Sea State Classification");
    addKeyValue("Predicted Sea State", String(result.predicted_sea_state || "N/A").toUpperCase());
    addKeyValue("AI Confidence", `${result.confidence ?? 0}%`);
    y += 3;

    addTitle("Overall Risk");
    addKeyValue("Risk Level", result.risk_indicator?.level || "N/A");
    addKeyValue("Risk Score", `${result.risk_indicator?.score ?? "N/A"}/100`);
    addKeyValue("Risk Factors", result.risk_indicator?.reasons?.length ? result.risk_indicator.reasons.join(", ") : "None");
    y += 3;

    addTitle("Class Probabilities");
    Object.entries(result.probabilities || {}).forEach(([label, probability]) => {
      addKeyValue(label, `${probability}%`);
    });
    y += 3;

    addTitle("Image Quality Analysis");
    if (result.image_quality) {
      addKeyValue("Brightness", `${result.image_quality.brightness_status} (${result.image_quality.brightness_value})`);
      addKeyValue("Contrast", `${result.image_quality.contrast_status} (${result.image_quality.contrast_value})`);
      addKeyValue("Sharpness", `${result.image_quality.sharpness_status} (${result.image_quality.sharpness_value})`);
      addKeyValue("Visibility", result.image_quality.visibility_status);
    }
    y += 3;

    addTitle("Decision Support Recommendation");
    if (result.recommendation) {
      addKeyValue("Risk Level", result.recommendation.risk_level);
      addText(result.recommendation.message);
    }

    addTitle("Weather Suitability");
    if (result.weather_suitability) {
      addKeyValue("Condition", result.weather_suitability.condition);
      addKeyValue("Suitability Score", `${result.weather_suitability.score}/100`);
      if (result.weather_suitability.operations?.length) {
        addText("Suitable Operations:");
        result.weather_suitability.operations.forEach(addBullet);
      }
      addKeyValue("Reason", result.weather_suitability.reason);
    }

    addTitle("System Warnings");
    if (result.warnings?.length) result.warnings.forEach(addBullet);
    else addText("No system warnings.");

    addFooter();
    const filename = result.filename
      ? `Sea_State_Report_${result.filename.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_")}.pdf`
      : "Sea_State_Prediction_Report.pdf";
    doc.save(filename);
  };

  /* ══════════════════════════════════════════════
     RESULTS RENDERER
     Renders the correct result block based on the
     active module. All original result fields are
     preserved from the original frontend.
     ══════════════════════════════════════════════ */
  const renderResults = () => {
    if (!result) return null;

    /* ── Error state ── */
    if (result.__error) {
      return (
        <div className="result-section fade-in">
          <div className="result-error-card">
            <AlertTriangle size={22} />
            <div>
              <p className="re-title">ANALYSIS FAILED</p>
              <p className="re-msg">{result.message}</p>
            </div>
          </div>
        </div>
      );
    }

        return (
      <div className="result-section fade-in">

        {/* Result header bar */}
        <div
          className="result-header"
          style={{ borderColor: mod.colorMid }}
        >
          <div
            className="result-badge"
            style={{
              background: mod.colorDim,
              color: mod.color,
              borderColor: mod.colorMid,
            }}
          >
            <CheckCircle2 size={12} />
            ANALYSIS COMPLETE
          </div>

          <span
            className="result-hdr-module"
            style={{ color: mod.color }}
          >
            {mod.statusLabel}
          </span>
        </div>

        

        {/* ══════════════════════════
            HULL DEFECT results
            ══════════════════════════ */}
        {activeModule === "hull" && (
          
          <div className="result-body">

          {/* PRIMARY PREDICTION */}
          <div className="result-primary-row">
            <div
              className="result-pred-block"
              style={{ borderColor: mod.colorMid }}
            >
              <p className="rpb-eye">DETECTED CONDITION</p>

              <p
                className="rpb-val"
                style={{ color: mod.color }}
              >
                {result?.prediction}
              </p>

              <p className="rpb-sub">
                Conf: {Number(result?.confidence ?? 0).toFixed(2)}%
              </p>
            </div>

            <ConfidenceRing
              value={Number(result?.confidence ?? 0)}
              color={mod.color}
            />
          </div>

            {/* CLASS PROBABILITY DISTRIBUTION */}
            {result.probabilities && (
              <div className="prob-section">
                <p className="prob-title">
                  <BarChart2 size={13} />
                  CLASS PROBABILITY DISTRIBUTION
                </p>

                {Object.entries(result.probabilities).map(([label, value]) => (
                  <ProbBar
                    key={label}
                    label={label}
                    value={Number(value).toFixed(2)}
                    color={mod.color}
                  />
                ))}
              </div>
            )}

            {/* SECONDARY DATA */}
            <div className="data-rows-block">
              <DataRow
                label="RECOMMENDATION"
                value={result.recommendation}
              />

              <DataRow
                label="WARNING"
                value={result.warning}
                color="#f97316"
              />
            </div>

            {/* Hull class probability distribution */}
            {result.probabilities && (
              <div className="prob-section">
                <p className="prob-title">
                  <BarChart2 size={13} /> CLASS PROBABILITY DISTRIBUTION
                </p>

                {Object.entries(result.probabilities).map(([label, probability]) => {
                  const value = Number(probability);

                  return (
                    <ProbBar
                      key={label}
                      label={label}
                      value={value}
                      color={mod.color}
                    />
                  );
                })}
              </div>
            )}

            {/* GRAD-CAM */}
            {result.gradcam && (
              <div className="gradcam-block">

                <button
                  className="ghost-btn"
                  style={{ "--gc": mod.color }}
                  onClick={() => setShowGradcam(v => !v)}
                >
                  {showGradcam ? (
                    <EyeOff size={14} />
                  ) : (
                    <Eye size={14} />
                  )}

                  {showGradcam
                    ? "HIDE GRAD-CAM"
                    : "VIEW GRAD-CAM"}
                </button>

                {showGradcam && (
                  <div className="gradcam-img-wrap fade-in">

                    <p className="gradcam-caption">
                      <ScanEye size={12} />
                      Gradient-weighted Class Activation Map
                    </p>

                    <img
                      src={`data:image/jpeg;base64,${result.gradcam}`}
                      alt="Grad-CAM"
                    />

                  </div>
                )}

              </div>
            )}

          </div>
        )}

        {/* ══════════════════════════
            SEA STATE results
            ══════════════════════════ */}
        {activeModule === "sea" && (
          <div className="result-body">
            <div className="result-primary-row">
              <div className="result-pred-block" style={{ borderColor: mod.colorMid }}>
                <p className="rpb-eye">CLASSIFIED SEA STATE</p>
                <p className="rpb-val" style={{ color: mod.color }}>{result.predicted_sea_state}</p>
                <p className="rpb-sub">Confidence: {result.confidence}%</p>
              </div>
              <ConfidenceRing value={parseFloat(result.confidence)} color={mod.color} />
            </div>

            <div className="sea-meta-grid">
              <DataRow label="PROCESSING TIME" value={`${result.processing_time ?? "N/A"} sec`} />
              <DataRow label="ENHANCEMENT" value={result.enhancement_applied ? "Applied" : "Not applied"} />
              <DataRow label="VALIDATION" value={result.validation?.message || "N/A"} />
            </div>

            {result.risk_indicator && (
              <div className={`sea-risk-card risk-${String(result.risk_indicator.level || "unknown").toLowerCase().replaceAll(" ", "-")}`}>
                <div className="sea-risk-topline">
                  <span>OPERATIONAL RISK</span>
                  <strong>{result.risk_indicator.level}</strong>
                </div>
                <div className="sea-risk-track">
                  <div className="sea-risk-fill" style={{ width: `${result.risk_indicator.score || 0}%` }} />
                </div>
                <div className="sea-risk-score">{result.risk_indicator.score}/100</div>
                {result.risk_indicator.reasons?.length > 0 && (
                  <p className="sea-support-text">Factors: {result.risk_indicator.reasons.join(", ")}</p>
                )}
              </div>
            )}

            {result.probabilities && (
              <div className="prob-section">
                <p className="prob-title"><BarChart2 size={13} /> CLASS PROBABILITY DISTRIBUTION</p>
                {Object.entries(result.probabilities).map(([k, v]) => (
                  <ProbBar key={k} label={k} value={v} color={mod.color} />
                ))}
              </div>
            )}

            {result.image_quality && (
              <div className="sea-feature-card">
                <p className="sea-feature-title">IMAGE QUALITY ANALYSIS</p>
                <div className="sea-kv-grid">
                  <DataRow label="BRIGHTNESS" value={`${result.image_quality.brightness_status} (${result.image_quality.brightness_value})`} />
                  <DataRow label="CONTRAST" value={`${result.image_quality.contrast_status} (${result.image_quality.contrast_value})`} />
                  <DataRow label="SHARPNESS" value={`${result.image_quality.sharpness_status} (${result.image_quality.sharpness_value})`} />
                  <DataRow label="VISIBILITY" value={result.image_quality.visibility_status} />
                </div>
              </div>
            )}

            {result.recommendation && (
              <div className="sea-feature-card">
                <p className="sea-feature-title">DECISION SUPPORT RECOMMENDATION</p>
                <div className="sea-recommendation-row">
                  <span className="sea-level-badge">{result.recommendation.risk_level}</span>
                  <p>{result.recommendation.message}</p>
                </div>
              </div>
            )}

            {result.weather_suitability && (
              <div className="sea-feature-card">
                <p className="sea-feature-title">MARINE OPERATION SUITABILITY</p>
                <div className="sea-suitability-head">
                  <strong>{result.weather_suitability.condition}</strong>
                  <span>{result.weather_suitability.score}/100</span>
                </div>
                <div className="sea-risk-track">
                  <div className="sea-suitability-fill" style={{ width: `${result.weather_suitability.score}%` }} />
                </div>
                <div className="sea-operation-badges">
                  {result.weather_suitability.operations?.map((operation) => (
                    <span key={operation}>{operation}</span>
                  ))}
                </div>
                <p className="sea-support-text">{result.weather_suitability.reason}</p>
              </div>
            )}

            {result.warnings?.length > 0 && (
              <div className="sea-feature-card sea-warning-card">
                <p className="sea-feature-title">SYSTEM WARNINGS</p>
                {result.warnings.map((warning, index) => (
                  <p className="sea-warning-line" key={`${warning}-${index}`}>
                    <AlertTriangle size={13} /> {warning}
                  </p>
                ))}
              </div>
            )}

            <button className="sea-pdf-btn" onClick={generateSeaStatePDF}>
              DOWNLOAD SEA-STATE PDF REPORT
            </button>

            <div className="sea-history-section">
              <div className="sea-history-header">
                <p className="sea-feature-title">SEA-STATE PREDICTION HISTORY</p>
                <div className="sea-history-actions">
                  <button onClick={fetchSeaHistory} disabled={seaHistoryLoading}>
                    {seaHistoryLoading ? "LOADING…" : "REFRESH"}
                  </button>
                  <button className="danger" onClick={clearSeaHistory} disabled={!canWrite} title={!canWrite ? "Read-only access cannot clear history" : "Clear history"}>CLEAR</button>
                </div>
              </div>
              {seaHistory.length === 0 ? (
                <p className="sea-history-empty">No sea-state prediction history available.</p>
              ) : (
                <div className="sea-history-grid">
                  {seaHistory.slice(0, 6).map((item, index) => (
                    <div className="sea-history-card" key={`${item.timestamp}-${item.filename}-${index}`}>
                      <span>{item.timestamp}</span>
                      <strong>{item.predicted_sea_state}</strong>
                      <p>{item.filename}</p>
                      <p>{item.confidence}% confidence · {item.recommendation?.risk_level || "Unknown"} risk</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

<<<<<<< HEAD
        {mode === "boat" && (
          <>
            <div className="boat-summary-card">
              <div>
                <p className="section-label">Detection Summary</p>
                <h3>{result.status || "Detected"}</h3>
              </div>
              <div className="confidence-badge">
                {result.confidence ? `${result.confidence}%` : "—"}
              </div>
            </div>

            <ResultItem label="Status" value={result.status || "Detected"} />
            <ResultItem label="Confidence" value={`${result.confidence}%`} />
            <ResultItem label="Timestamp" value={result.timestamp || "N/A"} />
            <ResultItem label="Source" value={result.source || "Drone"} />
            <ResultItem
              label="Estimated Size"
              value={result.estimated_size || "Medium Vessel"}
            />
            <ResultItem
              label="Vessel Origin"
              value={result.vessel_origin || "Local Boat"}
            />

=======
        {/* ══════════════════════════
            BOAT DETECTION results
            ══════════════════════════ */}
        {activeModule === "boat" && (
          <div className="result-body">
>>>>>>> 1a1db97d707f055c0380e77678b4a731c5c8918c
            {result.results && result.results.length > 0 ? (
              <>
                <div className="result-primary-row">
                  <div className="result-pred-block" style={{ borderColor: mod.colorMid }}>
                    <p className="rpb-eye">VESSELS DETECTED</p>
                    <p className="rpb-val" style={{ color: mod.color }}>{result.count}</p>
                    <p className="rpb-sub">Objects identified in frame</p>
                  </div>
                  <div className="vessel-count-badge"
                    style={{ color: mod.color, borderColor: mod.colorMid, background: mod.colorDim }}>
                    <Ship size={26} strokeWidth={1.5} />
                    <span>{result.count} FOUND</span>
                  </div>
                </div>

                <div className="vessel-list">
                  {result.results.map((det, i) => (
                    <div key={i} className="vessel-item" style={{ borderColor: mod.colorMid }}>
                      <div className="vessel-index"
                        style={{ background: mod.colorDim, color: mod.color }}>
                        <Crosshair size={12} /> {i + 1}
                      </div>
                      <span className="vessel-label">{det.label}</span>
                      <span className="vessel-conf" style={{ color: mod.color }}>{det.confidence}%</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-result">
                <Ship size={38} opacity={0.2} />
                <p>No vessels detected in this frame.</p>
              </div>
            )}
          </div>
        )}

        {/* ══════════════════════════
            RADAR results
            ══════════════════════════ */}
        {activeModule === "radar" && (
          <div className="result-body">

            {/* Final fused prediction — the most prominent element */}
            <div className="radar-final-card"
              style={{ borderColor: mod.colorMid, background: mod.colorDim }}>
              <p className="rfc-eye">FINAL FUSED DECISION</p>
              <p className="rfc-val" style={{ color: mod.color }}>{result.final_prediction}</p>
              <p className="rfc-status">{result.decision_status}</p>
            </div>

            {/* YOLO vs CNN model breakdown side by side */}
            <div className="radar-model-row">
              <div className="radar-model-card" style={{ borderColor: "#00d4ff44" }}>
                <p className="rmc-label" style={{ color: "#00d4ff" }}>
                  <Zap size={11} /> YOLO MODEL
                </p>
                <p className="rmc-pred">{result.yolo_prediction}</p>
                <p className="rmc-conf" style={{ color: "#00d4ff" }}>{result.yolo_confidence}%</p>
              </div>
              <div className="radar-model-card" style={{ borderColor: "#00ffb344" }}>
                <p className="rmc-label" style={{ color: "#00ffb3" }}>
                  <Network size={11} /> CNN VERIFY
                </p>
                <p className="rmc-pred">{result.cnn_prediction}</p>
                <p className="rmc-conf" style={{ color: "#00ffb3" }}>{result.cnn_confidence}%</p>
              </div>
            </div>
          </div>
        )}
        </div>
        );
        };

  

  /* ════════════════════════════════════════════════════════
     FULL APPLICATION RENDER
     Layout: sidebar (fixed left) + main content (scrollable)
     ════════════════════════════════════════════════════════ */
  return (
    <div className="app-shell">

      {showAccessAdmin && isAdmin && (
        <AdminUsers onClose={() => setShowAccessAdmin(false)} />
      )}

      {/* Full-viewport animated radar canvas background */}
      <RadarCanvas />

      {/* Mobile sidebar backdrop overlay */}
      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      {/* ══════════════════════════════════════════
          LEFT SIDEBAR
          Contains: branding, module nav, system status
          ══════════════════════════════════════════ */}
      <aside className={`sidebar ${sidebarOpen ? "sidebar--open" : ""}`}>

        {/* OceanIQ brand mark */}
        <div className="sidebar-brand">
          <div className="sb-logo">
            <Anchor size={19} strokeWidth={2.5} />
          </div>
          <div>
            <p className="sb-name">OceanIQ</p>
            <p className="sb-tagline">Marine Intelligence</p>
          </div>
        </div>

        {/* Research project identifier */}
        <div className="sidebar-proj-tag">
          <Globe size={10} />
          R26-IT-003
        </div>

        {/* Module navigation — clicking switches the active AI module */}
        <p className="sidebar-sec-label">AI MODULES</p>
        <nav className="module-nav">
          {Object.values(MODULES).map((m) => {
            const Icon = m.icon;
            const active = activeModule === m.id;
            return (
              <button key={m.id}
                className={`mnb ${active ? "mnb--active" : ""}`}
                style={active ? { "--mc": m.color, "--mc-dim": m.colorDim } : {}}
                onClick={() => switchModule(m.id)}
              >
                {active && <span className="mnb-accent" style={{ background: m.color }} />}

                <div className="mnb-icon"
                  style={{
                    color: active ? m.color : "rgba(255,255,255,0.3)",
                    background: active ? m.colorDim : "transparent",
                  }}>
                  <Icon size={16} strokeWidth={1.8} />
                </div>

                <div className="mnb-text">
                  <span className="mnb-name"
                    style={{ color: active ? "#fff" : "rgba(255,255,255,0.5)" }}>
                    {m.label}
                  </span>
                  <span className="mnb-tag"
                    style={{ color: active ? m.color : "rgba(255,255,255,0.2)" }}>
                    {m.tag}
                  </span>
                </div>

                {active && <ChevronRight size={13} style={{ color: m.color, marginLeft: "auto", flexShrink: 0 }} />}
              </button>
            );
          })}
        </nav>

        <div style={{ flex: 1 }} />

        {/* System health status indicators */}
        <div className="sidebar-status-block">
          <p className="sidebar-sec-label">SYSTEM STATUS</p>
          <StatusRows />
        </div>

        {/* Live clock + version info */}
        <div className="sidebar-footer">
          <SystemClock />
          <span className="sidebar-ver">v2.0 · R26-IT-003</span>
        </div>
      </aside>

      {/* ══════════════════════════════════════════
          MAIN CONTENT AREA
          ══════════════════════════════════════════ */}
      <main className="main-area">

        {/* Top command bar — shows active module + system badges */}
        <header className="command-bar">
          {/* Hamburger for mobile sidebar toggle */}
          <button className="hamburger" onClick={() => setSidebarOpen(v => !v)}>
            <span /><span /><span />
          </button>

          <div className="cb-module-row">
            <div className="cb-icon"
              style={{ background: mod.colorDim, color: mod.color, borderColor: mod.colorMid }}>
              <ModIcon size={15} strokeWidth={1.8} />
            </div>
            <div>
              <p className="cb-title">{mod.title}</p>
              <p className="cb-tag">{mod.tag}</p>
            </div>
          </div>

          <div className="cb-badges">
            <div className="badge-live"><span className="live-dot" />LIVE</div>
            <div className="badge-secure"><Shield size={12} />SECURE</div>
            <div className={`badge-access ${canWrite ? "badge-access--write" : "badge-access--read"}`}>
              <LockKeyhole size={12} />
              {isAdmin ? "ADMIN" : accessLevel === "read_write" ? "READ / WRITE" : "READ ONLY"}
            </div>
            {isAdmin && (
              <button className="admin-access-btn" onClick={() => setShowAccessAdmin(true)}>
                <UserRoundCog size={13} /> ACCESS ADMIN
              </button>
            )}
            <span className="cb-user">{user?.username}</span>
            <LogoutButton />
          </div>
        </header>

        {/* Module description strip — shows what the selected model does */}
        <div className="desc-strip" style={{ borderColor: mod.colorMid }}>
          <Cpu size={13} style={{ color: mod.color, flexShrink: 0, marginTop: 1 }} />
          <p>{mod.description}</p>
        </div>

        {!canWrite && (
          <div className="readonly-banner">
            <LockKeyhole size={14} />
            <div>
              <strong>READ-ONLY ACCESS</strong>
              <span>You can review available data and history, but analysis uploads, deletes and simulation controls are blocked by the backend.</span>
            </div>
          </div>
        )}

        {activeModule === "simulation" ? (
          <div className={!canWrite ? "simulation-readonly" : ""}>
            <LiveSimulation />
          </div>
        ) : (
          <>
        {/* ── 3-panel analysis grid ── */}
        <div className="analysis-grid">

          {/* ─── PANEL 1: Upload ─── */}
          <section className="a-card upload-card">
            <div className="a-card-header">
              <span className="a-card-label"><Upload size={11} /> INPUT IMAGE</span>
              <span className="a-card-tag" style={{ color: mod.color }}>{mod.statusLabel}</span>
            </div>

            {/* Drag-and-drop zone */}
            <label
              className={`drop-zone ${dragOver ? "dz--over" : ""} ${file ? "dz--filled" : ""} ${!canWrite ? "dz--readonly" : ""}`}
              style={dragOver ? { borderColor: mod.color } : {}}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <input type="file" accept="image/*" onChange={handleFileChange} disabled={!canWrite} />

              {file ? (
                <div className="dz-loaded">
                  <CheckCircle2 size={24} style={{ color: mod.color }} />
                  <p className="dz-fname">{file.name}</p>
                  <p className="dz-change">Click to change</p>
                </div>
              ) : (
                <div className="dz-empty">
                  <div className="dz-upload-icon">
                    <Upload size={24} style={{ color: mod.color }} />
                  </div>
                  <p className="dz-main">Drop image here</p>
                  <p className="dz-sub">or click to browse</p>
                  <p className="dz-fmt">PNG · JPG · JPEG · BMP</p>
                </div>
              )}
            </label>

            {activeModule === "sea" && (
              <label className="sea-enhancement-toggle">
                <input
                  type="checkbox"
                  checked={applyEnhancement}
                  onChange={(e) => setApplyEnhancement(e.target.checked)}
                  disabled={!canWrite}
                />
                <span>Apply image enhancement before prediction</span>
              </label>
            )}

            <button
              className="run-btn"
              style={{ "--mc": mod.color, "--mc-dim": mod.colorDim }}
              onClick={runPrediction}
              disabled={!file || loading || !canWrite}
            >
              {loading
                ? <><Loader size={15} className="spin" /> ANALYZING…</>
                : <><Zap size={15} /> RUN ANALYSIS</>}
            </button>
          </section>

          {/* ─── PANEL 2: Preview ─── */}
          <section className="a-card preview-card">
            <div className="a-card-header">
              <span className="a-card-label"><Eye size={11} /> PREVIEW</span>
              {loading && (
                <span className="analyzing-pill" style={{ color: mod.color }}>
                  <Loader size={10} className="spin" /> PROCESSING
                </span>
              )}
            </div>

            <div className="preview-area">
              {preview ? (
                <div className="preview-img-wrap">
                  <img src={preview} alt="Preview" className="preview-img" />
                  {/* Scan-line overlay shown while model is running */}
                  {loading && (
                    <div className="scan-overlay">
                      <div className="scan-beam" style={{ "--sc": mod.color }} />
                      {/* Corner bracket indicators — HUD style */}
                      <div className="scan-corners" style={{ "--sc": mod.color }}>
                        <span className="sc sc-tl" />
                        <span className="sc sc-tr" />
                        <span className="sc sc-bl" />
                        <span className="sc sc-br" />
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="preview-placeholder">
                  <ModIcon size={40} strokeWidth={0.9}
                    style={{ color: mod.color, opacity: 0.18 }} />
                  <p className="pp-main">Awaiting input</p>
                  <p className="pp-sub">Upload an image to preview</p>
                </div>
              )}
            </div>
          </section>

          {/* ─── PANEL 3: Results ─── */}
          <section className="a-card results-card">
            <div className="a-card-header">
              <span className="a-card-label"><Activity size={11} /> INTELLIGENCE REPORT</span>
              <span className="a-card-tag" style={{ color: mod.color }}>
                <Layers size={10} /> {mod.tag}
              </span>
            </div>

            {/* Shimmer skeleton while loading */}
            {loading && !result && (
              <div className="skeleton-block fade-in">
                <div className="skel skel--wide" />
                <div className="skel skel--med" />
                <div className="skel skel--short" />
                <div className="skel skel--wide" style={{ marginTop: 18 }} />
                <div className="skel skel--med" />
              </div>
            )}

            {/* Empty state — before any prediction */}
            {!loading && !result && (
              <div className="results-empty">
                <ScanEye size={34} strokeWidth={1}
                  style={{ color: mod.color, opacity: 0.4 }} />
                <p className="results-empty-main">No analysis yet</p>
                <p className="results-empty-sub">
                  Upload an image and run analysis to see the AI output
                </p>
              </div>
            )}

            {/* Actual prediction results */}
            {renderResults()}
          </section>
        </div>
          </>
        )}

        {/* Footer strip */}
        <footer className="main-footer">
          <span>OceanIQ · Marine AI Intelligence Platform</span>
          <span>Research Project R26-IT-003</span>
        </footer>
      </main>
    </div>
  );

}


function App() {
  return (
    <ProtectedRoute>
      <AppContent />
    </ProtectedRoute>
  );
}

export default App;
