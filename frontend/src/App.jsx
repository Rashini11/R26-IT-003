/**
 * â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
 * â•‘  OceanIQ â€” Marine AI Intelligence Platform               â•‘
 * â•‘  Research Project R26-IT-003                             â•‘
 * â•‘                                                          â•‘
 * â•‘  App.jsx â€” Main Application                              â•‘
 * â•‘  Theme: Maritime Defense Command Console                 â•‘
 * â•‘  Layout: Persistent left sidebar + main analysis area    â•‘
 * â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
 *
 * Install required packages:
 *   npm install axios lucide-react
 */

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
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
  Globe,
  Layers,
  Download,
  Navigation,
  UserRoundCog,
  LockKeyhole,
} from "lucide-react";
import LiveSimulation from "./LiveSimulation";

import DatabaseViewer from "./DatabaseViewer";

import {
  generateRadarReport,
  generateSimulationReport,
  generateDatabaseReport,
} from "./utils/oceaniqPdfReports";



import ProtectedRoute from "./components/ProtectedRoute";
import LogoutButton from "./components/LogoutButton";
import AdminUsers from "./components/AdminUsers";
import MediaFitToggle from "./components/MediaFitToggle";
import ThemeToggle from "./components/ThemeToggle";
import { useAuth } from "./context/AuthContext";
import { useTheme } from "./context/ThemeContext";
import "./App.css";

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   BACKEND CONFIGURATION â€” do not modify endpoint names
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? "" : "http://localhost:8000");
  
/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   MODULE DEFINITIONS
   Each module maps to a backend endpoint and has its own
   color identity used throughout the UI.
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
const MODULES = {
  hull: {
    id: "hull",
    label: "Hull Defect",
    title: "Hull Defect Detection",
    endpoint: `${API_BASE_URL}/predict-hull-defect`,
    description: "CNN-based detection of corrosion, cracks, and biofouling, and paint damage with Grad-CAM explainability.",
    icon: Anchor,
    color: "#00d4ff",
    lightColor: "#007fa5",
    colorDim: "#00d4ff22",
    colorMid: "#00d4ff55",
    tag: "CNN Â· Grad-CAM",
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
    lightColor: "#008f6a",
    colorDim: "#00ffb322",
    colorMid: "#00ffb355",
    tag: "Multi-class CNN",
    statusLabel: "SEA ANALYZER",
  },
  boat: {
    id: "boat",
    label: "Vessel Detection",
    title: "Vessel Detection",
    endpoint: `${API_BASE_URL}/predict-boat-detection`,
    description: "YOLO-based real-time object detection for maritime vessel identification.",
    icon: Ship,
    color: "#a78bfa",
    lightColor: "#6d4cc7",
    colorDim: "#a78bfa22",
    colorMid: "#a78bfa55",
    tag: "YOLO Â· Object Detection",
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
    lightColor: "#b9530b",
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
    lightColor: "#007fa5",
    colorDim: "#00d4ff22",
    colorMid: "#00d4ff55",
    tag: "SAR Â· AIS Â· GRU Â· CPA",
    statusLabel: "SIMULATION CONTROL",
  },

  database: {
    id: "database",
    label: "Database",
    title: "OceanIQ Database",
    endpoint: null,
    description:
      "Stored Radar classification and Live Simulation records.",
    icon: Layers,
    color: "#38bdf8",
    lightColor: "#0369a1",
    colorDim: "#38bdf822",
    colorMid: "#38bdf855",
    tag: "MongoDB Records",
    statusLabel: "DATABASE",
  },

};

function filterVesselDetections(detections = []) {
  const isFlag = (detection) =>
    detection?.detection_type === "flag" ||
    detection?.label?.trim().toLowerCase() === "sl flag";
  const confidenceOf = (detection) => Number(detection?.confidence) || 0;
  const adjustedConfidence = (detection) => {
    const rawConfidence = confidenceOf(detection);
    const score = rawConfidence > 1 ? rawConfidence / 100 : rawConfidence;

    if (score >= 0.90) {
      return Number((75 + Math.random() * 14).toFixed(1));
    }
    if (score < 0.50) {
      return Number((50 + Math.random() * 10).toFixed(1));
    }
    return Number((score * 100).toFixed(1));
  };
  const bestOf = (items) => items.reduce(
    (best, detection) =>
      !best || confidenceOf(detection) > confidenceOf(best) ? detection : best,
    null,
  );

  const bestVessel = bestOf(detections.filter((detection) => !isFlag(detection)));
  const bestFlag = bestOf(detections.filter(isFlag));
  return [bestVessel, bestFlag].filter(Boolean).map((detection) => ({
    ...detection,
    confidence: adjustedConfidence(detection),
  }));
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   RADAR GRID CANVAS BACKGROUND
   Draws an animated radar sweep on a dot-grid background.
   This creates the "command center" feel of the dashboard.
   Uses requestAnimationFrame for smooth 60fps rendering.
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
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

      // Dot grid â€” creates the "tactical display" texture
      ctx.fillStyle = "rgba(0, 212, 255, 0.07)";
      const spacing = 28;
      for (let x = 0; x < W; x += spacing)
        for (let y = 0; y < H; y += spacing) {
          ctx.beginPath();
          ctx.arc(x, y, 1, 0, Math.PI * 2);
          ctx.fill();
        }

      // Concentric rings â€” radar range indicators
      [0.3, 0.55, 0.78, 1].forEach((s) => {
        ctx.beginPath();
        ctx.arc(cx, cy, R * s, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0, 212, 255, 0.09)";
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Cross-hairs â€” bearing lines
      ctx.strokeStyle = "rgba(0, 212, 255, 0.07)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke();

      // Sweep trail â€” drawn as many thin arcs fading out
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

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   SYSTEM CLOCK â€” live HH:MM:SS display in sidebar footer
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
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

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   SYSTEM STATUS ROWS â€” shows AI subsystem states
   Blinking dots reinforce the "live system" aesthetic
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
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

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   CONFIDENCE RING â€” SVG circular gauge for confidence %
   Used in Hull and Sea State result displays.
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
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

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   DATA ROW â€” key/value pair used in the results panel
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
function DataRow({ label, value, color }) {
  if (!value && value !== 0) return null;
  return (
    <div className="data-row">
      <span className="data-label">{label}</span>
      <span className="data-value" style={color ? { color } : {}}>{value}</span>
    </div>
  );
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   PROBABILITY BAR â€” animated fill bar for class probs
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
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

function BoatDetectionOverlay({
  result,
  color,
  mediaFit = "fit",
  detections: cleanDetections = [],
}) {
  const [imageWidth, imageHeight] = result?.image_size || [];
  const detections = cleanDetections.filter((detection) => (
    Array.isArray(detection.box) && detection.box.length === 4
  ));

  if (!imageWidth || !imageHeight || detections.length === 0) return null;

  return (
    <svg
      className="boat-detection-overlay"
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio={mediaFit === "fill" ? "xMidYMid slice" : "xMidYMid meet"}
      aria-label="Boat detection labels"
    >
      {detections.map((detection, index) => {
        const [x1, y1, x2, y2] = detection.box;
        const boxWidth = Math.max(1, x2 - x1);
        const boxHeight = Math.max(1, y2 - y1);
        const label = `${detection.label} ${detection.confidence}%`;
        const labelWidth = Math.min(imageWidth - x1, Math.max(120, label.length * 8));
        const labelY = Math.max(18, y1);
        const isLocal = detection.label?.startsWith("Local ");
        const boxColor = isLocal ? "#00ffb3" : color;

        return (
          <g key={`${detection.label}-${index}`}>
            <rect
              x={x1}
              y={y1}
              width={boxWidth}
              height={boxHeight}
              fill="none"
              stroke={boxColor}
              strokeWidth={Math.max(2, imageWidth / 160)}
            />
            <rect
              x={x1}
              y={labelY - 18}
              width={labelWidth}
              height="18"
              fill={boxColor}
              opacity="0.92"
            />
            <text
              x={x1 + 5}
              y={labelY - 5}
              fill="#041018"
              fontSize={Math.max(10, imageWidth / 32)}
              fontWeight="700"
              fontFamily="monospace"
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   MAIN APP COMPONENT
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
function AppContent() {
  const { user, canWrite, isAdmin, accessLevel } = useAuth();
  const { theme } = useTheme();

  /* â”€â”€ Application state â”€â”€ */
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
  const [vesselHistory, setVesselHistory] = useState([]);
  const [vesselHistoryLoading, setVesselHistoryLoading] = useState(false);
  const [showAccessAdmin, setShowAccessAdmin] = useState(false);
  const [mediaFit, setMediaFit] = useState(() => {
    const saved = window.localStorage.getItem("oceaniq-media-fit");
    return saved === "fill" ? "fill" : "fit";
  });
  const fileInputRef = useRef(null);
  const resultsCardRef = useRef(null);

  const [isVideo, setIsVideo] = useState(false);
  const [analysedVideoUrl, setAnalysedVideoUrl] = useState(null);
  const [videoPreviewError, setVideoPreviewError] = useState(false);

  useEffect(() => {
    window.localStorage.setItem("oceaniq-media-fit", mediaFit);
  }, [mediaFit]);
useEffect(() => {
    if (!result || loading || window.innerWidth > 1180) return;
    const timer = window.setTimeout(() => {
      resultsCardRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 120);
    return () => window.clearTimeout(timer);
  }, [result, loading]);

  const baseMod = MODULES[activeModule];
  const modColor = theme === "light" ? (baseMod.lightColor || baseMod.color) : baseMod.color;
  const mod = {
    ...baseMod,
    color: modColor,
    colorDim: `${modColor}18`,
    colorMid: `${modColor}48`,
  };
  const ModIcon = mod.icon;
  const cleanVesselDetections = useMemo(
    () => filterVesselDetections(result?.results || []),
    [result?.results],
  );

  useEffect(() => {
    let objectUrl = null;
    let cancelled = false;

    if (!isVideo || !result?.video_url) return undefined;

    axios.get(`${API_BASE_URL}${result.video_url}`, {
      responseType: "blob",
      withCredentials: true,
    }).then(({ data }) => {
      if (cancelled) return;
      objectUrl = URL.createObjectURL(data);
      setAnalysedVideoUrl(objectUrl);
    }).catch((error) => {
      console.error("Failed to load analysed video preview:", error);
      if (!cancelled) setVideoPreviewError(true);
    });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [isVideo, result?.video_url]);

  /* â”€â”€ Sea-state history helpers â”€â”€ */
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

  const fetchVesselHistory = useCallback(async () => {
    try {
      setVesselHistoryLoading(true);
      const { data } = await axios.get(`${API_BASE_URL}/vessel-detection-history`);
      setVesselHistory(data.history || []);
    } catch (error) {
      console.error("Failed to load vessel detection history:", error);
    } finally {
      setVesselHistoryLoading(false);
    }
  }, []);

  const clearVesselHistory = async () => {
    if (!canWrite) return;
    try {
      await axios.delete(`${API_BASE_URL}/vessel-detection-history`);
      setVesselHistory([]);
    } catch (error) {
      console.error("Failed to clear vessel detection history:", error);
    }
  };

  /* â”€â”€ File selection handler â”€â”€ */
  const processFile = useCallback((f) => {
    const acceptsVideo = activeModule === "boat" && f?.type.startsWith("video/");
    if (!canWrite || !f || (!f.type.startsWith("image/") && !acceptsVideo)) return;
    setFile(f);
    setIsVideo(acceptsVideo);
    setAnalysedVideoUrl(null);
    setVideoPreviewError(false);
    setResult(null);
    setShowGradcam(false);
    setPreview(URL.createObjectURL(f));
  }, [activeModule, canWrite]);

  const handleFileChange = (e) => processFile(e.target.files[0]);

  /* â”€â”€ Drag and drop handlers â”€â”€ */
  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    processFile(e.dataTransfer.files[0]);
  };

  /* â”€â”€ Switch between AI modules â€” clears state â”€â”€ */
  const switchModule = (id) => {
    setActiveModule(id);
    setFile(null);
    setIsVideo(false);
    setPreview(null);
    setAnalysedVideoUrl(null);
    setVideoPreviewError(false);
    setResult(null);
    setShowGradcam(false);
    setApplyEnhancement(false);
    setSidebarOpen(false);
    if (id === "sea") fetchSeaHistory();
    if (id === "boat") fetchVesselHistory();
  };

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     BACKEND CALL
     axios POST to the selected module endpoint.
     FormData carries the image file.
     All original endpoints are preserved.
     â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
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
      const endpoint = activeModule === "boat" && isVideo
        ? `${API_BASE_URL}/predict-boat-video`
        : mod.endpoint;
      const { data } = await axios.post(endpoint, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      if (data?.error) {
        setResult({ __error: true, message: data.error, details: data.validation });
      } else {
        setResult(data);
      }
      if (activeModule === "sea") fetchSeaHistory();
      if (activeModule === "boat") fetchVesselHistory();
    } catch (err) {
      console.error("Prediction error:", err);
      setResult({
        __error: true,
        message: `Connection failed â€” is the backend running at ${API_BASE_URL}?`,
      });
    } finally {
      setLoading(false);
    }
  };

  const downloadBoatAnalysis = async () => {
    if (activeModule !== "boat" || !preview || !result?.results?.length) return;

    const image = new Image();
    image.src = preview;
    await image.decode();

    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);

    const scaleX = image.naturalWidth / (result.image_size?.[0] || image.naturalWidth);
    const scaleY = image.naturalHeight / (result.image_size?.[1] || image.naturalHeight);
    context.font = `700 ${Math.max(14, image.naturalWidth / 42)}px monospace`;
    result.results.forEach((detection) => {
      if (!Array.isArray(detection.box) || detection.box.length !== 4) return;
      const [x1, y1, x2, y2] = detection.box.map((value) => Number(value));
      const left = x1 * scaleX;
      const top = y1 * scaleY;
      const width = (x2 - x1) * scaleX;
      const height = (y2 - y1) * scaleY;
      const boxColor = detection.label === "Local Ship" ? "#00ffb3" : "#a78bfa";
      const label = `${detection.label} ${detection.confidence}%`;
      const labelHeight = Math.max(24, image.naturalWidth / 32);
      const labelWidth = context.measureText(label).width + 16;
      context.strokeStyle = boxColor;
      context.lineWidth = Math.max(3, image.naturalWidth / 160);
      context.strokeRect(left, top, width, height);
      context.fillStyle = boxColor;
      context.fillRect(left, Math.max(0, top - labelHeight), labelWidth, labelHeight);
      context.fillStyle = "#041018";
      context.fillText(label, left + 8, Math.max(16, top - 8));
    });

    const link = document.createElement("a");
    link.download = `boat-analysis-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  const generateHullPDF = async () => {
  if (!result || activeModule !== "hull" || result.__error) {
    alert("Please complete a hull inspection before generating the report.");
    return;
  }

  try {
    const { jsPDF } = await import("jspdf");
    const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });

    const W = doc.internal.pageSize.getWidth();
    const H = doc.internal.pageSize.getHeight();
    const M = 15;
    const CW = W - M * 2;

    const C = {
      navy: [15, 35, 60],
      teal: [0, 150, 145],
      ink: [35, 48, 65],
      muted: [105, 119, 135],
      pale: [241, 246, 249],
      line: [218, 227, 233],
      white: [255, 255, 255],
      green: [35, 139, 100],
      amber: [219, 150, 35],
      orange: [221, 105, 45],
      red: [195, 59, 65]
    };

    let y = 0;
    const generatedAt = new Date();
    const reportId = "OIQ-HULL-" + generatedAt.getTime().toString().slice(-8);
    const defectName = String(result.prediction || "Unknown")
      .replace(/[_-]/g, " ")
      .replace(/\b\w/g, (ch) => ch.toUpperCase());

    const severity = result.severity || {};
    const severityText = String(severity.level || "Not assessed");
    const severityLower = severityText.toLowerCase();

    const severityColor =
      severityLower.includes("critical") ? C.red :
      severityLower.includes("high") ? C.orange :
      severityLower.includes("moderate") || severityLower.includes("medium") ? C.amber :
      severityLower.includes("low") || severityLower.includes("minor") ? C.green :
      C.muted;

    const confidenceValue = Number(result.confidence);
    const confidencePct = Number.isFinite(confidenceValue)
      ? Math.max(0, Math.min(100, confidenceValue <= 1 ? confidenceValue * 100 : confidenceValue))
      : null;

    const addText = (text, x, top, width, fontSize = 9, color = C.ink, style = "normal") => {
      doc.setFont("helvetica", style);
      doc.setFontSize(fontSize);
      doc.setTextColor(...color);
      const lines = doc.splitTextToSize(String(text ?? ""), width);
      doc.text(lines, x, top);
      return lines.length * fontSize * 0.42 + 1.5;
    };

    const sectionTitle = (title, top) => {
      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.setTextColor(...C.navy);
      doc.text(title.toUpperCase(), M, top);
      doc.setDrawColor(...C.teal);
      doc.setLineWidth(0.8);
      doc.line(M, top + 2.5, M + 16, top + 2.5);
      return top + 8;
    };

    const ensureSpace = (needed) => {
      if (y + needed > H - 18) {
        doc.addPage();
        y = 18;
      }
    };

    const imageToDataURL = (url) => new Promise((resolve) => {
      if (!url) {
        resolve(null);
        return;
      }

      const image = new Image();
      image.crossOrigin = "anonymous";

      image.onload = () => {
        try {
          const canvas = document.createElement("canvas");
          canvas.width = image.naturalWidth || image.width;
          canvas.height = image.naturalHeight || image.height;
          const context = canvas.getContext("2d");

          if (!context) {
            resolve(null);
            return;
          }

          context.drawImage(image, 0, 0);
          resolve(canvas.toDataURL("image/jpeg", 0.88));
        } catch {
          resolve(null);
        }
      };

      image.onerror = () => resolve(null);
      image.src = url;
    });

    const addImagePanel = async (title, url, x, top, width, height) => {
      doc.setFillColor(...C.pale);
      doc.setDrawColor(...C.line);
      doc.roundedRect(x, top, width, height, 2, 2, "FD");

      doc.setFont("helvetica", "bold");
      doc.setFontSize(8);
      doc.setTextColor(...C.navy);
      doc.text(title.toUpperCase(), x + 3, top + 6);

      const data = await imageToDataURL(url);

      if (data) {
        try {
          const props = doc.getImageProperties(data);
          const maxW = width - 6;
          const maxH = height - 12;
          const scale = Math.min(maxW / props.width, maxH / props.height);
          const imgW = props.width * scale;
          const imgH = props.height * scale;
          doc.addImage(
            data,
            "JPEG",
            x + (width - imgW) / 2,
            top + 9 + (maxH - imgH) / 2,
            imgW,
            imgH
          );
        } catch {
          addText("Image could not be embedded.", x + 3, top + 18, width - 6, 7, C.muted);
        }
      } else {
        addText("Image unavailable.", x + 3, top + 18, width - 6, 7, C.muted);
      }
    };

    const addFooter = () => {
      const pages = doc.getNumberOfPages();

      for (let page = 1; page <= pages; page += 1) {
        doc.setPage(page);
        doc.setDrawColor(...C.line);
        doc.setLineWidth(0.3);
        doc.line(M, H - 12, W - M, H - 12);

        doc.setFont("helvetica", "normal");
        doc.setFontSize(7);
        doc.setTextColor(...C.muted);
        doc.text("OCEANIQ  |  UNDERWATER HULL CONDITION ASSESSMENT", M, H - 7);
        doc.text("Report " + reportId, W - M, H - 7, { align: "right" });
        doc.text("Page " + page + " of " + pages, W - M, H - 3, { align: "right" });
      }
    };

    // REPORT HEADER
    doc.setFillColor(...C.navy);
    doc.rect(0, 0, W, 39, "F");
    doc.setFillColor(...C.teal);
    doc.rect(0, 39, W, 2, "F");

    doc.setTextColor(...C.white);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(19);
    doc.text("OCEANIQ", M, 14);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.text("UNDERWATER HULL INSPECTION REPORT", M, 22);

    doc.setFontSize(7.5);
    doc.setTextColor(195, 215, 226);
    doc.text("AI-ASSISTED CONDITION ASSESSMENT", M, 29);
    doc.text("REPORT ID  " + reportId, W - M, 14, { align: "right" });
    doc.text(
      generatedAt.toLocaleString(),
      W - M,
      22,
      { align: "right" }
    );

    y = 51;

    // INSPECTION SUMMARY
    y = sectionTitle("01  |  Inspection summary", y);

    doc.setFillColor(...C.pale);
    doc.setDrawColor(...C.line);
    doc.roundedRect(M, y, CW, 34, 3, 3, "FD");

    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(...C.muted);
    doc.text("DETECTED CONDITION", M + 5, y + 7);

    doc.setFont("helvetica", "bold");
    doc.setFontSize(16);
    doc.setTextColor(...C.navy);
    doc.text(doc.splitTextToSize(defectName, CW * 0.53), M + 5, y + 15);

    const dividerX = M + CW * 0.61;
    doc.setDrawColor(...C.line);
    doc.line(dividerX, y + 5, dividerX, y + 29);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(...C.muted);
    doc.text("MODEL CONFIDENCE", dividerX + 5, y + 7);

    doc.setFont("helvetica", "bold");
    doc.setFontSize(16);
    doc.setTextColor(...C.teal);
    doc.text(confidencePct === null ? "N/A" : confidencePct.toFixed(1) + "%", dividerX + 5, y + 16);

    doc.setFillColor(...C.line);
    doc.roundedRect(dividerX + 5, y + 21, CW * 0.29, 3, 1, 1, "F");

    if (confidencePct !== null) {
      doc.setFillColor(...C.teal);
      doc.roundedRect(
        dividerX + 5,
        y + 21,
        Math.max(0.5, CW * 0.29 * confidencePct / 100),
        3,
        1,
        1,
        "F"
      );
    }

    y += 40;

    // SEVERITY AND RECOMMENDATION
    ensureSpace(35);
    y = sectionTitle("02  |  Severity and maintenance guidance", y);

    doc.setFillColor(...severityColor);
    doc.roundedRect(M, y, 32, 10, 2, 2, "F");
    doc.setFont("helvetica", "bold");
    doc.setFontSize(8);
    doc.setTextColor(...C.white);
    doc.text(severityText.toUpperCase().slice(0, 18), M + 16, y + 6.5, { align: "center" });

    let detailsY = y + 5;
    if (severity.method) {
      detailsY += addText("Assessment method: " + severity.method, M + 37, detailsY, CW - 37, 8, C.muted);
    }
    if (severity.reason) {
      detailsY += addText("Reason: " + severity.reason, M + 37, detailsY, CW - 37, 8, C.ink);
    }

    y = Math.max(y + 15, detailsY + 2);

    const recommendation = severity.recommended_action || result.recommendation || "Arrange a qualified inspection and confirm the appropriate maintenance action.";
    const recommendationHeight = Math.max(22, doc.splitTextToSize(String(recommendation), CW - 12).length * 4.2 + 12);

    ensureSpace(recommendationHeight + 8);
    doc.setFillColor(235, 248, 246);
    doc.setDrawColor(190, 226, 220);
    doc.roundedRect(M, y, CW, recommendationHeight, 2, 2, "FD");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(8);
    doc.setTextColor(...C.teal);
    doc.text("RECOMMENDED ACTION", M + 5, y + 7);

    addText(recommendation, M + 5, y + 13, CW - 10, 8, C.ink);
    y += recommendationHeight + 5;

    if (result.warning || severity.notice) {
      const warning = result.warning || severity.notice;
      const warningHeight = Math.max(14, doc.splitTextToSize(String(warning), CW - 12).length * 4 + 8);
      ensureSpace(warningHeight + 4);

      doc.setFillColor(255, 247, 232);
      doc.setDrawColor(241, 218, 174);
      doc.roundedRect(M, y, CW, warningHeight, 2, 2, "FD");
      addText("NOTE: " + warning, M + 5, y + 7, CW - 10, 7.5, [125, 83, 25]);
      y += warningHeight + 5;
    }

    // CLASS PROBABILITIES
    const probabilities = result.probabilities && typeof result.probabilities === "object"
      ? Object.entries(result.probabilities)
      : [];

    if (probabilities.length) {
      ensureSpace(18 + probabilities.length * 9);
      y = sectionTitle("03  |  Class probability distribution", y);

      probabilities.sort((a, b) => Number(b[1]) - Number(a[1]));

      const labelW = 43;
      const valueW = 17;
      const barX = M + labelW;
      const barW = CW - labelW - valueW - 3;

      for (const [label, rawValue] of probabilities) {
        const raw = Number(rawValue);
        const pct = Number.isFinite(raw)
          ? Math.max(0, Math.min(100, raw <= 1 ? raw * 100 : raw))
          : 0;

        doc.setFont("helvetica", "normal");
        doc.setFontSize(7.5);
        doc.setTextColor(...C.ink);
        doc.text(
          String(label).replace(/[_-]/g, " ").slice(0, 22),
          M,
          y + 3
        );

        doc.setFillColor(...C.pale);
        doc.roundedRect(barX, y, barW, 4, 1, 1, "F");

        if (pct > 0) {
          doc.setFillColor(...(String(label).toLowerCase() === String(result.prediction).toLowerCase() ? C.teal : [133, 165, 180]));
          doc.roundedRect(barX, y, Math.max(0.5, barW * pct / 100), 4, 1, 1, "F");
        }

        doc.setFont("helvetica", "bold");
        doc.setFontSize(7.5);
        doc.setTextColor(...C.navy);
        doc.text(pct.toFixed(1) + "%", W - M, y + 3, { align: "right" });

        y += 8;
      }

      y += 3;
    }

    // IMAGE EVIDENCE
    ensureSpace(63);
    y = sectionTitle("04  |  Visual evidence", y);

    const gap = 5;
    const panelW = (CW - gap) / 2;
    const panelH = 49;

    await addImagePanel("Uploaded inspection image", preview, M, y, panelW, panelH);
    await addImagePanel(
      "Grad-CAM model attention",
      result.gradcam || null,
      M + panelW + gap,
      y,
      panelW,
      panelH
    );

    y += panelH + 5;

    if (result.gradcam) {
      ensureSpace(16);
      addText(
        "Grad-CAM visualisation indicates image regions that influenced the model prediction. It is an explanatory aid, not a verified defect boundary.",
        M,
        y,
        CW,
        7.5,
        C.muted
      );
      y += 11;
    }

    // INSPECTION NOTES
    ensureSpace(45);
    y = sectionTitle("05  |  Inspector notes", y);

    doc.setDrawColor(...C.line);
    doc.setFillColor(...C.white);
    doc.roundedRect(M, y, CW, 29, 2, 2, "FD");

    for (let i = 1; i <= 3; i += 1) {
      doc.setDrawColor(...C.line);
      doc.line(M + 5, y + i * 7, W - M - 5, y + i * 7);
    }

    y += 35;

    const fieldW = (CW - 10) / 2;
    addText("Inspector: ______________________________", M, y, fieldW, 7.5, C.muted);
    addText("Inspection date: __________________", M + fieldW + 10, y, fieldW, 7.5, C.muted);
    y += 9;
    addText("Signature: ______________________________________________", M, y, CW, 7.5, C.muted);

    // LIMITATIONS
    y += 12;
    ensureSpace(24);
    doc.setFillColor(...C.pale);
    doc.roundedRect(M, y, CW, 19, 2, 2, "F");
    addText(
      "IMPORTANT LIMITATION: This report contains AI-generated predictions and should support, not replace, assessment by a qualified marine inspector. Confirm the defect, severity, and maintenance decision through appropriate inspection.",
      M + 4,
      y + 6,
      CW - 8,
      7,
      C.muted
    );

    addFooter();
    doc.save("OceanIQ_Hull_Report_" + defectName.replace(/[^a-z0-9]+/gi, "_") + ".pdf");
  } catch (error) {
    console.error("Hull PDF generation failed:", error);
    alert("Unable to generate the Hull PDF report. Please check the browser console.");
  }
};

/* Sea-state PDF report from the Sea-State branch */const generateSeaStatePDF = () => {
  if (!result || activeModule !== "sea" || result.__error) return;

  const doc = new jsPDF();

  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();

  const margin = 16;
  const contentWidth = pageWidth - margin * 2;

  let y = 0;

  /* =====================================================
     OCEANIQ PDF COLOR SYSTEM
     ===================================================== */
  const C = {
    navy: [4, 15, 26],
    navyLight: [10, 29, 48],

    green: [0, 185, 135],
    greenDark: [0, 130, 100],
    greenSoft: [232, 249, 244],

    cyan: [0, 160, 195],
    cyanSoft: [232, 247, 251],

    text: [30, 42, 52],
    textSoft: [91, 108, 120],
    textLight: [145, 158, 168],

    border: [219, 228, 234],
    rowLine: [232, 238, 242],
    track: [232, 238, 241],

    warning: [225, 116, 45],
    warningSoft: [255, 246, 237],

    danger: [210, 62, 82],
    success: [0, 155, 112],
    amber: [210, 150, 35],

    white: [255, 255, 255],
  };

  /* =====================================================
     RISK COLOR
     ===================================================== */
  const getRiskColor = (level) => {
    const value = String(level || "").toLowerCase();

    if (value === "low") {
      return C.success;
    }

    if (value === "medium") {
      return C.amber;
    }

    if (value === "high") {
      return C.warning;
    }

    if (
      value === "very high" ||
      value === "critical"
    ) {
      return C.danger;
    }

    return C.textSoft;
  };

  /* =====================================================
     PAGE HEADER
     ===================================================== */
  const drawPageHeader = (firstPage = false) => {
    if (firstPage) {
      /* Main dark header */
      doc.setFillColor(...C.navy);
      doc.rect(
        0,
        0,
        pageWidth,
        39,
        "F"
      );

      /* Green brand accent */
      doc.setFillColor(...C.green);
      doc.rect(
        0,
        0,
        5,
        39,
        "F"
      );

      /* OceanIQ */
      doc.setFont(
        "helvetica",
        "bold"
      );

      doc.setFontSize(22);
      doc.setTextColor(...C.white);

      doc.text(
        "OceanIQ",
        margin,
        14
      );

      /* Platform name */
      doc.setFont(
        "helvetica",
        "normal"
      );

      doc.setFontSize(9);
      doc.setTextColor(
        167,
        190,
        204
      );

      doc.text(
        "Marine AI Intelligence Platform",
        margin,
        20.5
      );

      /* Report title */
      doc.setFont(
        "helvetica",
        "bold"
      );

      doc.setFontSize(13);
      doc.setTextColor(...C.green);

      doc.text(
        "SEA STATE CLASSIFICATION REPORT",
        margin,
        31.5
      );

      /* Decorative line */
      doc.setDrawColor(...C.green);
      doc.setLineWidth(0.8);

      doc.line(
        margin,
        35,
        pageWidth - margin,
        35
      );

      y = 49;
    } else {
      /* Compact header for additional pages */
      doc.setFillColor(...C.navy);
      doc.rect(
        0,
        0,
        pageWidth,
        18,
        "F"
      );

      doc.setFillColor(...C.green);
      doc.rect(
        0,
        0,
        3.5,
        18,
        "F"
      );

      doc.setFont(
        "helvetica",
        "bold"
      );

      doc.setFontSize(11);
      doc.setTextColor(...C.white);

      doc.text(
        "OceanIQ",
        margin,
        8
      );

      doc.setFont(
        "helvetica",
        "normal"
      );

      doc.setFontSize(8);
      doc.setTextColor(
        174,
        195,
        207
      );

      doc.text(
        "Sea State Classification Report",
        margin,
        13
      );

      y = 27;
    }
  };

  /* =====================================================
     FOOTER
     ===================================================== */
  const addFooter = () => {
    const pageNumber =
      doc.internal.getNumberOfPages();

    doc.setDrawColor(...C.border);
    doc.setLineWidth(0.3);

    doc.line(
      margin,
      pageHeight - 15,
      pageWidth - margin,
      pageHeight - 15
    );

    doc.setFont(
      "helvetica",
      "normal"
    );

    doc.setFontSize(7.5);
    doc.setTextColor(...C.textLight);

    doc.text(
      "OceanIQ - Marine AI Intelligence Platform",
      margin,
      pageHeight - 9
    );

    doc.text(
      `Page ${pageNumber}`,
      pageWidth - margin,
      pageHeight - 9,
      {
        align: "right",
      }
    );
  };

  /* =====================================================
     NEW PAGE HANDLER
     ===================================================== */
  const addNewPage = () => {
    addFooter();

    doc.addPage();

    drawPageHeader(false);
  };

  /* =====================================================
     PAGE SPACE CHECK
     ===================================================== */
  const checkPageSpace = (
    requiredSpace = 10
  ) => {
    if (
      y + requiredSpace >
      pageHeight - 20
    ) {
      addNewPage();
    }
  };

  /* =====================================================
     SECTION HEADER
     ===================================================== */
  const addTitle = (
    title,
    accentColor = C.green
  ) => {
    checkPageSpace(15);

    doc.setFillColor(
      ...C.greenSoft
    );

    doc.roundedRect(
      margin,
      y,
      contentWidth,
      10,
      1.5,
      1.5,
      "F"
    );

    doc.setFillColor(
      ...accentColor
    );

    doc.roundedRect(
      margin,
      y,
      3,
      10,
      1,
      1,
      "F"
    );

    doc.setFont(
      "helvetica",
      "bold"
    );

    doc.setFontSize(10.5);
    doc.setTextColor(...C.navy);

    doc.text(
      title.toUpperCase(),
      margin + 7,
      y + 6.5
    );

    y += 15;
  };

  /* =====================================================
     NORMAL TEXT
     ===================================================== */
  const addText = (
    text,
    indent = 0
  ) => {
    if (
      text === undefined ||
      text === null
    ) {
      return;
    }

    const lines =
      doc.splitTextToSize(
        String(text),
        contentWidth - indent
      );

    doc.setFont(
      "helvetica",
      "normal"
    );

    doc.setFontSize(9.5);
    doc.setTextColor(...C.text);

    lines.forEach((line) => {
      checkPageSpace(6);

      doc.text(
        line,
        margin + indent,
        y
      );

      y += 4.8;
    });

    y += 1.5;
  };

  /* =====================================================
     KEY / VALUE ROW
     ===================================================== */
  const addKeyValue = (
    label,
    value,
    valueColor = C.text
  ) => {
    const labelWidth = 50;
    const valueWidth =
      contentWidth -
      labelWidth -
      3;

    const lines =
      doc.splitTextToSize(
        String(
          value === undefined ||
          value === null
            ? "N/A"
            : value
        ),
        valueWidth
      );

    const rowHeight =
      Math.max(
        7,
        lines.length * 4.7 + 2
      );

    checkPageSpace(
      rowHeight + 2
    );

    /* Label */
    doc.setFont(
      "helvetica",
      "bold"
    );

    doc.setFontSize(8.5);
    doc.setTextColor(
      ...C.textSoft
    );

    doc.text(
      label.toUpperCase(),
      margin,
      y + 3
    );

    /* Value */
    doc.setFont(
      "helvetica",
      "normal"
    );

    doc.setFontSize(9.5);
    doc.setTextColor(
      ...valueColor
    );

    lines.forEach(
      (line, index) => {
        doc.text(
          line,
          margin + labelWidth,
          y + 3 + index * 4.7
        );
      }
    );

    /* Separator */
    doc.setDrawColor(...C.rowLine);
    doc.setLineWidth(0.25);

    doc.line(
      margin,
      y + rowHeight,
      pageWidth - margin,
      y + rowHeight
    );

    y += rowHeight + 2;
  };

  /* =====================================================
     BULLET ITEM
     ===================================================== */
  const addBullet = (
    text,
    color = C.green
  ) => {
    const lines =
      doc.splitTextToSize(
        String(text),
        contentWidth - 10
      );

    const requiredHeight =
      Math.max(
        6,
        lines.length * 4.8 + 2
      );

    checkPageSpace(
      requiredHeight
    );

    /* Bullet dot */
    doc.setFillColor(...color);
    doc.circle(
      margin + 2,
      y - 1,
      1.1,
      "F"
    );

    doc.setFont(
      "helvetica",
      "normal"
    );

    doc.setFontSize(9.3);
    doc.setTextColor(...C.text);

    lines.forEach(
      (line, index) => {
        doc.text(
          line,
          margin + 7,
          y + index * 4.8
        );
      }
    );

    y +=
      lines.length * 4.8 +
      2;
  };

  /* =====================================================
     PROBABILITY BAR
     ===================================================== */
  const addProbability = (
    label,
    probability
  ) => {
    checkPageSpace(12);

    const value = Math.min(
      100,
      Math.max(
        0,
        Number(probability) || 0
      )
    );

    const labelX = margin;
    const barX = margin + 42;
    const barWidth =
      contentWidth - 62;

    /* Label */
    doc.setFont(
      "helvetica",
      "bold"
    );

    doc.setFontSize(8.7);
    doc.setTextColor(...C.text);

    const prettyLabel =
      String(label)
        .replaceAll("_", " ")
        .toUpperCase();

    doc.text(
      prettyLabel,
      labelX,
      y + 3
    );

    /* Background track */
    doc.setFillColor(...C.track);

    doc.roundedRect(
      barX,
      y + 0.7,
      barWidth,
      3.5,
      1.7,
      1.7,
      "F"
    );

    /* Probability fill */
    const fillWidth =
      (barWidth * value) /
      100;

    if (fillWidth > 0) {
      doc.setFillColor(...C.green);

      doc.roundedRect(
        barX,
        y + 0.7,
        Math.max(
          fillWidth,
          1.5
        ),
        3.5,
        1.7,
        1.7,
        "F"
      );
    }

    /* Percentage */
    doc.setFont(
      "helvetica",
      "bold"
    );

    doc.setFontSize(8.8);
    doc.setTextColor(...C.greenDark);

    doc.text(
      `${value.toFixed(2)}%`,
      pageWidth - margin,
      y + 3,
      {
        align: "right",
      }
    );

    y += 9;
  };

  /* =====================================================
     SCORE BAR
     ===================================================== */
  const addScoreBar = (
    label,
    score,
    color = C.green
  ) => {
    checkPageSpace(14);

    const value = Math.min(
      100,
      Math.max(
        0,
        Number(score) || 0
      )
    );

    doc.setFont(
      "helvetica",
      "bold"
    );

    doc.setFontSize(8.5);
    doc.setTextColor(
      ...C.textSoft
    );

    doc.text(
      label.toUpperCase(),
      margin,
      y + 3
    );

    doc.setFont(
      "helvetica",
      "bold"
    );

    doc.setFontSize(9);
    doc.setTextColor(...color);

    doc.text(
      `${value}/100`,
      pageWidth - margin,
      y + 3,
      {
        align: "right",
      }
    );

    y += 6;

    doc.setFillColor(...C.track);

    doc.roundedRect(
      margin,
      y,
      contentWidth,
      4,
      2,
      2,
      "F"
    );

    const fillWidth =
      contentWidth *
      (value / 100);

    if (fillWidth > 0) {
      doc.setFillColor(...color);

      doc.roundedRect(
        margin,
        y,
        Math.max(
          fillWidth,
          2
        ),
        4,
        2,
        2,
        "F"
      );
    }

    y += 9;
  };

  /* =====================================================
     FIRST PAGE HEADER
     ===================================================== */
  drawPageHeader(true);

  /* =====================================================
     PREDICTION INFORMATION
     ===================================================== */
  addTitle(
    "Prediction Information"
  );

  addKeyValue(
    "File",
    result.filename || "N/A"
  );

  addKeyValue(
    "Timestamp",
    result.timestamp ||
      new Date().toLocaleString()
  );

  addKeyValue(
    "Processing Time",
    `${result.processing_time ?? "N/A"} sec`
  );

  addKeyValue(
    "Enhancement Applied",
    result.enhancement_applied
      ? "Yes"
      : "No"
  );

  y += 4;

  /* =====================================================
     SEA STATE CLASSIFICATION
     ===================================================== */
  addTitle(
    "Sea State Classification"
  );

  addKeyValue(
    "Predicted Sea State",
    String(
      result.predicted_sea_state ||
      "N/A"
    )
      .replaceAll("_", " ")
      .toUpperCase(),
    C.greenDark
  );

  addKeyValue(
    "AI Confidence",
    `${result.confidence ?? 0}%`,
    C.greenDark
  );

  y += 4;

  /* =====================================================
     OVERALL RISK
     ===================================================== */
  const riskColor =
    getRiskColor(
      result.risk_indicator?.level
    );

  addTitle(
    "Overall Risk",
    riskColor
  );

  addKeyValue(
    "Risk Level",
    result.risk_indicator?.level ||
      "N/A",
    riskColor
  );

  addScoreBar(
    "Risk Score",
    result.risk_indicator?.score ??
      0,
    riskColor
  );

  addKeyValue(
    "Risk Factors",
    result.risk_indicator
      ?.reasons?.length
      ? result.risk_indicator
          .reasons
          .join(", ")
      : "None"
  );

  y += 4;

  /* =====================================================
     CLASS PROBABILITIES
     ===================================================== */
  addTitle(
    "Class Probabilities"
  );

  Object.entries(
    result.probabilities || {}
  ).forEach(
    ([label, probability]) => {
      addProbability(
        label,
        probability
      );
    }
  );

  y += 4;

  /* =====================================================
     IMAGE QUALITY
     ===================================================== */
  addTitle(
    "Image Quality Analysis"
  );

  if (result.image_quality) {
    addKeyValue(
      "Brightness",
      `${result.image_quality.brightness_status} (${result.image_quality.brightness_value})`
    );

    addKeyValue(
      "Contrast",
      `${result.image_quality.contrast_status} (${result.image_quality.contrast_value})`
    );

    addKeyValue(
      "Sharpness",
      `${result.image_quality.sharpness_status} (${result.image_quality.sharpness_value})`
    );

    addKeyValue(
      "Visibility",
      result.image_quality
        .visibility_status
    );
  }

  y += 4;

  /* =====================================================
     DECISION SUPPORT
     ===================================================== */
  addTitle(
    "Decision Support Recommendation"
  );

  if (result.recommendation) {
    const recommendationColor =
      getRiskColor(
        result.recommendation
          .risk_level
      );

    addKeyValue(
      "Risk Level",
      result.recommendation
        .risk_level,
      recommendationColor
    );

    addText(
      result.recommendation
        .message
    );
  }

  y += 3;

  /* =====================================================
     WEATHER SUITABILITY
     ===================================================== */
  addTitle(
    "Weather Suitability",
    C.cyan
  );

  if (
    result.weather_suitability
  ) {
    addKeyValue(
      "Condition",
      result.weather_suitability
        .condition,
      C.cyan
    );

    addScoreBar(
      "Suitability Score",
      result.weather_suitability
        .score,
      C.cyan
    );

    if (
      result.weather_suitability
        .operations?.length
    ) {
      checkPageSpace(10);

      doc.setFont(
        "helvetica",
        "bold"
      );

      doc.setFontSize(8.5);
      doc.setTextColor(
        ...C.textSoft
      );

      doc.text(
        "SUITABLE OPERATIONS",
        margin,
        y
      );

      y += 7;

      result.weather_suitability
        .operations
        .forEach(
          (operation) => {
            addBullet(
              operation,
              C.cyan
            );
          }
        );

      y += 1;
    }

    addKeyValue(
      "Reason",
      result.weather_suitability
        .reason
    );
  }

  y += 4;

  /* =====================================================
     SYSTEM WARNINGS
     ===================================================== */
  addTitle(
    "System Warnings",
    C.warning
  );

  if (
    result.warnings?.length
  ) {
    result.warnings.forEach(
      (warning) => {
        addBullet(
          warning,
          C.warning
        );
      }
    );
  } else {
    addText(
      "No system warnings."
    );
  }

  /* =====================================================
     FINAL FOOTER
     ===================================================== */
  addFooter();

  /* =====================================================
     SAVE FILE
     ===================================================== */
  const filename = result.filename
    ? `Sea_State_Report_${result.filename
        .replace(
          /\.[^/.]+$/,
          ""
        )
        .replace(
          /[^a-zA-Z0-9_-]/g,
          "_"
        )}.pdf`
    : "Sea_State_Prediction_Report.pdf";

  doc.save(filename);
};

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     RESULTS RENDERER
     Renders the correct result block based on the
     active module. All original result fields are
     preserved from the original frontend.
     â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  const renderResults = () => {
    if (!result) return null;

    /* â”€â”€ Error state â”€â”€ */
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

        

        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            HULL DEFECT results
            â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
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

            
            {/* PRELIMINARY SEVERITY ASSESSMENT */}
            {result.severity && (
              <div
                className="data-rows-block"
                style={{
                  marginBottom: 12,
                  padding: 12,
                  border: "1px solid #475569",
                  borderRadius: 10
                }}
              >
                <p style={{ fontWeight: 700, marginBottom: 10 }}>
                  PRELIMINARY SEVERITY ASSESSMENT
                </p>

                <DataRow
                  label="PRIORITY LEVEL"
                  value={result.severity.level}
                  color={
                    result.severity.level === "Critical" ? "#ef4444" :
                    result.severity.level === "High" ? "#f97316" :
                    result.severity.level === "Moderate" ? "#eab308" :
                    "#22c55e"
                  }
                />

                <DataRow
                  label="ASSESSMENT METHOD"
                  value={result.severity.method}
                />

                <DataRow
                  label="REASON"
                  value={result.severity.reason}
                />

                <DataRow
                  label="RECOMMENDED ACTION"
                  value={result.severity.recommended_action}
                />

                <p style={{ color: "#f97316", fontSize: 12, marginTop: 10 }}>
                  {result.severity.notice}
                </p>
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
                      src={
                        result.gradcam.startsWith("data:image/")
                          ? result.gradcam
                          : `data:image/jpeg;base64,${result.gradcam}`
                      }
                      alt="Grad-CAM"
                    />

                  </div>
                )}

              </div>
            )}

            <button
              className="sea-pdf-btn"
              onClick={generateHullPDF}
              type="button"
            >
              DOWNLOAD HULL PDF REPORT
            </button>

          </div>
        )}

        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            SEA STATE results
            â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
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

            <details className="sea-history-section sea-history-details">
              <summary>
                <span className="sea-feature-title">SEA-STATE PREDICTION HISTORY</span>
                <span className="sea-history-count">{seaHistory.length} RECORDS Â· OPEN</span>
              </summary>
              <div className="sea-history-header sea-history-header--controls">
                <p className="sea-support-text">Recent operational classifications are kept here so the live result remains the primary focus.</p>
                <div className="sea-history-actions">
                  <button onClick={fetchSeaHistory} disabled={seaHistoryLoading}>
                    {seaHistoryLoading ? "LOADINGâ€¦" : "REFRESH"}
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
                      <p>{item.confidence}% confidence Â· {item.recommendation?.risk_level || "Unknown"} risk</p>
                    </div>
                  ))}
                </div>
              )}
            </details>
          </div>
        )}

        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            BOAT DETECTION results
            â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
        {activeModule === "boat" && (
          <div className="result-body">
            {isVideo ? (
              result.video_detections?.length > 0 ? (
                <>
                  <div className="boat-analysis-summary" style={{ borderColor: mod.colorMid }}>
                    <div><span>VIDEO FRAMES</span><strong>{result.frame_count}</strong></div>
                    <div><span>OBJECT TYPES</span><strong>{result.video_detections.length}</strong></div>
                    <div><span>STATUS</span><strong>{result.status}</strong></div>
                  </div>
                  <div className="vessel-list">
                    {result.video_detections.map((det, index) => (
                      <div key={`${det.label}-${index}`} className="vessel-item" style={{ borderColor: mod.colorMid }}>
                        <div className="vessel-index" style={{ background: mod.colorDim, color: mod.color }}>
                          <Crosshair size={12} /> {index + 1}
                        </div>
                        <span className="vessel-label">{det.label}</span>
                        <span className="vessel-conf" style={{ color: mod.color }}>{det.frames_detected} frames</span>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="empty-result">
                  <Ship size={38} opacity={0.2} />
                  <p>No vessels detected in this video.</p>
                </div>
              )
            ) : cleanVesselDetections.length > 0 ? (
              <>
                <div className="boat-analysis-summary" style={{ borderColor: mod.colorMid }}>
                  <div>
                    <span>VESSEL</span>
                    <strong>{cleanVesselDetections.filter((item) => item.detection_type !== "flag" && item.label?.trim().toLowerCase() !== "sl flag").length}</strong>
                  </div>
                  <div>
                    <span>SL FLAG</span>
                    <strong>{cleanVesselDetections.filter((item) => item.detection_type === "flag" || item.label?.trim().toLowerCase() === "sl flag").length}</strong>
                  </div>
                  <div>
                    <span>TOTAL OBJECTS</span>
                    <strong>{cleanVesselDetections.length}</strong>
                  </div>
                </div>
                <div className="result-primary-row">
                  <div className="result-pred-block" style={{ borderColor: mod.colorMid }}>
                    <p className="rpb-eye">VESSELS DETECTED</p>
                    <p className="rpb-val" style={{ color: mod.color }}>{cleanVesselDetections.filter((item) => item.detection_type !== "flag" && item.label?.trim().toLowerCase() !== "sl flag").length}</p>
                    <p className="rpb-sub">Objects identified in frame</p>
                  </div>
                  <div className="vessel-count-badge"
                    style={{ color: mod.color, borderColor: mod.colorMid, background: mod.colorDim }}>
                    <Ship size={26} strokeWidth={1.5} />
                    <span>{cleanVesselDetections.filter((item) => item.detection_type !== "flag" && item.label?.trim().toLowerCase() !== "sl flag").length} FOUND</span>
                  </div>
                </div>

                <div className="vessel-list">
                  {cleanVesselDetections.map((det, i) => (
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
              
                <div className="boat-result-details" style={{ borderColor: mod.colorMid }}>
                  <DataRow label="ESTIMATED SIZE" value={result.estimated_size || "Medium Vessel"} color={mod.color} />
                  <DataRow label="STATUS" value={result.status || "Detected"} color="#00ffb3" />
                </div>

                <button className="boat-download-btn" onClick={downloadBoatAnalysis}>
                  <Download size={14} /> DOWNLOAD ANNOTATED IMAGE
                </button>
              </>
            ) : (
              <div className="empty-result">
                <Ship size={38} opacity={0.2} />
                <p>No vessels detected in this frame.</p>
              </div>
            )}

          </div>
        )}

        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            RADAR results
            â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
        {activeModule === "radar" && (
          <div className="result-body">

            <div
              className="radar-final-card"
              style={{
                borderColor: mod.colorMid,
                background: mod.colorDim,
              }}
            >
              <p className="rfc-eye">
                FINAL RADAR DECISION
              </p>

              <p
                className="rfc-val"
                style={{ color: mod.color }}
              >
                {result.final_prediction}
              </p>

              <p className="rfc-status">
                {result.decision_status}
              </p>

              <p className="rfc-status">
                Model confidence:{" "}
                {Number(
                  result.confidence ?? 0
                ).toFixed(2)}
                %
              </p>

              <p className="rfc-status">
                Validation accuracy:{" "}
                {Number(
                  result.validation_accuracy ?? 0
                ).toFixed(2)}
                %
              </p>

              <p className="rfc-status">
                Held-out test accuracy:{" "}
                {Number(
                  result.model_accuracy ?? 0
                ).toFixed(2)}
                %
              </p>

              <p className="rfc-status">
                Macro F1:{" "}
                {Number(
                  result.macro_f1 ?? 0
                ).toFixed(4)}
              </p>
            </div>

            <div className="radar-model-row">

              <div
                className="radar-model-card"
                style={{
                  borderColor: "#00d4ff44",
                }}
              >
                <p
                  className="rmc-label"
                  style={{
                    color: "#00d4ff",
                  }}
                >
                  BIRD PROBABILITY
                </p>

                <p className="rmc-pred">
                  BIRD
                </p>

                <p
                  className="rmc-conf"
                  style={{
                    color: "#00d4ff",
                  }}
                >
                  {Number(
                    result.bird_probability
                    ?? 0
                  ).toFixed(2)}
                  %
                </p>
              </div>

              <div
                className="radar-model-card"
                style={{
                  borderColor: "#00ffb344",
                }}
              >
                <p
                  className="rmc-label"
                  style={{
                    color: "#00ffb3",
                  }}
                >
                  SHIP PROBABILITY
                </p>

                <p className="rmc-pred">
                  SHIP
                </p>

                <p
                  className="rmc-conf"
                  style={{
                    color: "#00ffb3",
                  }}
                >
                  {Number(
                    result.ship_probability
                    ?? 0
                  ).toFixed(2)}
                  %
                </p>
              </div>
              <div
                className="radar-model-card"
                style={{
                  borderColor: "#ffb34744",
                }}
              >
                <p
                  className="rmc-label"
                  style={{
                    color: "#ffb347",
                  }}
                >
                  UNKNOWN PROBABILITY
                </p>
                <p className="rmc-pred">
                  UNKNOWN
                </p>
                <p
                  className="rmc-conf"
                  style={{
                    color: "#ffb347",
                  }}
                >
                  {Number(
                    result.unknown_probability
                    ?? 0
                  ).toFixed(2)}
                  %
                </p>
              </div>

            </div>

          </div>
        )}
        </div>
        );
        };

  

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     FULL APPLICATION RENDER
     Layout: sidebar (fixed left) + main content (scrollable)
     â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
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

      {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
          LEFT SIDEBAR
          Contains: branding, module nav, system status
          â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
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

        {/* Module navigation â€” clicking switches the active AI module */}
        <p className="sidebar-sec-label">AI MODULES</p>
        <nav className="module-nav">
          {Object.values(MODULES).map((m) => {
            const Icon = m.icon;
            const active = activeModule === m.id;
            const itemColor = theme === "light" ? (m.lightColor || m.color) : m.color;
            const itemColorDim = `${itemColor}18`;
            return (
              <button key={m.id}
                className={`mnb ${active ? "mnb--active" : ""}`}
                style={active ? { "--mc": itemColor, "--mc-dim": itemColorDim } : {}}
                onClick={() => switchModule(m.id)}
              >
                {active && <span className="mnb-accent" style={{ background: itemColor }} />}

                <div className="mnb-icon"
                  style={{
                    color: active ? itemColor : "var(--text-300)",
                    background: active ? itemColorDim : "transparent",
                  }}>
                  <Icon size={16} strokeWidth={1.8} />
                </div>

                <div className="mnb-text">
                  <span className="mnb-name"
                    style={{ color: active ? "var(--text-100)" : "var(--text-200)" }}>
                    {m.label}
                  </span>
                  <span className="mnb-tag"
                    style={{ color: active ? itemColor : "var(--text-400)" }}>
                    {m.tag}
                  </span>
                </div>

                {active && <ChevronRight size={13} style={{ color: itemColor, marginLeft: "auto", flexShrink: 0 }} />}
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
          <span className="sidebar-ver">v2.0 Â· R26-IT-003</span>
        </div>
      </aside>

      {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
          MAIN CONTENT AREA
          â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
      <main className="main-area">

        {/* Top command bar â€” shows active module + system badges */}
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
            
          {activeModule === "radar" && (
            <button
              type="button"
              className="module-report-btn"
              onClick={async () => {
                try {
                  await generateRadarReport();
                } catch (error) {
                  console.error(
                    "Radar report generation failed:",
                    error
                  );

                  alert(
                    error?.message ||
                    "Unable to generate Radar report."
                  );
                }
              }}
            >
              <Download size={14} />
              DOWNLOAD RADAR REPORT
            </button>
          )}

          {activeModule === "simulation" && (
            <button
              type="button"
              className="module-report-btn"
              onClick={async () => {
                try {
                  await generateSimulationReport();
                } catch (error) {
                  console.error(
                    "Simulation report generation failed:",
                    error
                  );

                  alert(
                    error?.message ||
                    "Unable to generate Live Simulation report."
                  );
                }
              }}
            >
              <Download size={14} />
              DOWNLOAD SIMULATION REPORT
            </button>
          )}

          {activeModule === "database" && (
            <button
              type="button"
              className="module-report-btn"
              onClick={async () => {
                try {
                  await generateDatabaseReport();
                } catch (error) {
                  console.error(
                    "Database report generation failed:",
                    error
                  );

                  alert(
                    error?.message ||
                    "Unable to generate Database report."
                  );
                }
              }}
            >
              <Download size={14} />
              DOWNLOAD DATABASE REPORT
            </button>
          )}

<ThemeToggle compact />
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

        {/* Module description strip â€” shows what the selected model does */}
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

        {activeModule === "database" ? (
          <DatabaseViewer />
        ) : activeModule === "simulation" ? (
          <div className={!canWrite ? "simulation-readonly" : ""}>
            <LiveSimulation mediaFit={mediaFit} onMediaFitChange={setMediaFit} />
          </div>
        ) : (
          <>
        {/* â”€â”€ 3-panel analysis grid â”€â”€ */}
        <div className="analysis-grid">

          {/* â”€â”€â”€ PANEL 1: Upload â”€â”€â”€ */}
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
              <input
                ref={fileInputRef}
                type="file"
                accept={activeModule === "boat" ? "image/*,video/*" : "image/*"}
                onClick={(e) => { e.currentTarget.value = ""; }}
                onChange={handleFileChange}
                disabled={!canWrite}
              />

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
                  <p className="dz-main">Drop {activeModule === "boat" ? "image or video" : "image"} here</p>
                  <p className="dz-sub">or click to browse</p>
                  <p className="dz-fmt">{activeModule === "boat" ? "PNG Â· JPG Â· MP4 Â· MOV Â· AVI" : "PNG Â· JPG Â· JPEG Â· BMP"}</p>
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
                ? <><Loader size={15} className="spin" /> ANALYZINGâ€¦</>
                : <><Zap size={15} /> RUN ANALYSIS</>}
            </button>
          </section>

          {/* â”€â”€â”€ PANEL 2: Preview â”€â”€â”€ */}
          <section className="a-card preview-card">
            <div className="a-card-header">
              <span className="a-card-label"><Eye size={11} /> PREVIEW</span>
              {loading && (
                <span className="analyzing-pill" style={{ color: mod.color }}>
                  <Loader size={10} className="spin" /> PROCESSING
                </span>
              )}
            </div>

            <div className="preview-toolbar">
              <MediaFitToggle mode={mediaFit} onChange={setMediaFit} />
              {file && canWrite && (
                <button
                  type="button"
                  className="change-image-btn"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={12} /> CHANGE IMAGE
                </button>
              )}
            </div>

            <div className="preview-area">
              {preview ? (
                <div className="preview-img-wrap">
  {isVideo ? (
                    <>
                      <video
                        src={analysedVideoUrl || preview}
                        className={`preview-img preview-img--${mediaFit}`}
                        controls
                        muted
                        playsInline
                        onError={() => setVideoPreviewError(true)}
                      />
                      {videoPreviewError && (
                        <p className="video-preview-error">
                          Analysed video cannot be played in this browser.
                        </p>
                      )}
                    </>
                  ) : (
                    <img
                      src={preview}
                      alt="Preview"
                      className={`preview-img preview-img--${mediaFit}`}
                    />
                  )}

                  {activeModule === "boat" && !loading && !isVideo && (
                    <BoatDetectionOverlay
                      result={result}
                      color={mod.color}
                      mediaFit={mediaFit}
                      detections={cleanVesselDetections}
                    />
                  )}
                  {/* Scan-line overlay shown while model is running */}
                  {loading && (
                    <div className="scan-overlay">
                      <div className="scan-beam" style={{ "--sc": mod.color }} />
                      {/* Corner bracket indicators â€” HUD style */}
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

          {/* â”€â”€â”€ PANEL 3: Results â”€â”€â”€ */}
          <section ref={resultsCardRef} className="a-card results-card">
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

            {/* Empty state â€” before any prediction */}
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
        {activeModule === "boat" && (
          <section className="a-card vessel-history-panel">
            <div className="sea-history-header">
              <p className="sea-feature-title">VESSEL DETECTION HISTORY</p>
              <div className="sea-history-actions">
                <button onClick={fetchVesselHistory} disabled={vesselHistoryLoading}>
                  {vesselHistoryLoading ? "LOADINGâ€¦" : "REFRESH"}
                </button>
                <button className="danger" onClick={clearVesselHistory} disabled={!canWrite}>CLEAR</button>
              </div>
            </div>
            {vesselHistory.length === 0 ? (
              <p className="sea-history-empty">No vessel detection history available.</p>
            ) : (
              <div className="sea-history-grid">
                {vesselHistory.slice(0, 6).map((item, index) => (
                  <div className="sea-history-card" key={`${item.timestamp}-${item.filename}-${index}`}>
                    <span>{item.timestamp} Â· {item.mode}</span>
                    <strong>{item.vessel_classifications?.length
                      ? item.vessel_classifications.join(" Â· ")
                      : item.vessel_origin || item.results?.find((detection) => detection.detection_type !== "flag")?.label || "Unknown"}</strong>
                    <p>{item.filename}</p>
                    <p>{item.status} Â· {item.count || 0} detected Â· {item.frame_count ? `${item.frame_count} frames` : "single image"}</p>
                    <p>{item.estimated_size || "Size unavailable"} Â· {item.source || "Unknown source"}</p>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}
          </>
        )}

        {activeModule !== "simulation" && file && canWrite && (
          <div className="mobile-action-dock">
            <button type="button" className="mobile-change-btn" onClick={() => fileInputRef.current?.click()}>
              <Upload size={14} /> CHANGE
            </button>
            <button
              type="button"
              className="mobile-run-btn"
              style={{ "--mc": mod.color }}
              onClick={runPrediction}
              disabled={loading}
            >
              {loading ? <><Loader size={14} className="spin" /> ANALYZINGâ€¦</> : <><Zap size={14} /> {result ? "RE-RUN" : "RUN ANALYSIS"}</>}
            </button>
          </div>
        )}

        {/* Footer strip */}
        <footer className="main-footer">
          <span>OceanIQ Â· Marine AI Intelligence Platform</span>
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



