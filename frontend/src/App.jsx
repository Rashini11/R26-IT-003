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

import { Fragment, useState, useEffect, useRef, useCallback, useMemo } from "react";
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
  Film,
  ChevronRight,
  Network,
  Globe,
  Layers,
  Download,
  Navigation,
  UserRoundCog,
  LockKeyhole,
  X,
} from "lucide-react";
import LiveSimulation from "./LiveSimulation";

import DatabaseViewer from "./DatabaseViewer";

import {
  generateRadarReport,
  generateSimulationReport,
  generateDatabaseReport,
} from "./utils/oceaniqPdfReports";


import { RadarDatabaseHistory } from "./DatabaseHistory";
import ProtectedRoute from "./components/ProtectedRoute";
import LogoutButton from "./components/LogoutButton";
import AdminUsers from "./components/AdminUsers";
import MediaFitToggle from "./components/MediaFitToggle";
import ThemeToggle from "./components/ThemeToggle";
import { useAuth } from "./context/AuthContext";
import { useTheme } from "./context/ThemeContext";
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
    lightColor: "#007fa5",
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
    description: "Identify vessel types and read the vessel name, port, and IMO number from separate images.",
    icon: Ship,
    color: "#a78bfa",
    lightColor: "#6d4cc7",
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
    tag: "SAR · AIS · GRU · CPA",
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

function parseVesselClassifications(value = "") {
  return value
    .split(/\s*(?:,|\bor\b)\s*/i)
    .map((label) => label.trim())
    .filter(Boolean);
}

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
        const label = detection.label;
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

/* ══════════════════════════════════════════════════════════
   MAIN APP COMPONENT
   ══════════════════════════════════════════════════════════ */
function AppContent() {
  const { user, canWrite, isAdmin, accessLevel } = useAuth();
  const { theme } = useTheme();

  /* ── Application state ── */
  const [activeModule, setActiveModule] = useState("hull");
  const [file, setFile] = useState(null);
  const [vesselVideoFile, setVesselVideoFile] = useState(null);
  const [vesselVideoPreview, setVesselVideoPreview] = useState(null);
  const [vesselInfoFile, setVesselInfoFile] = useState(null);
  const [vesselInfoPreview, setVesselInfoPreview] = useState(null);
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
  const [vesselHistoryError, setVesselHistoryError] = useState("");
  const [vesselHistoryStorage, setVesselHistoryStorage] = useState("unknown");
  const [expandedHistoryImage, setExpandedHistoryImage] = useState(null);
  const [vesselRecordId, setVesselRecordId] = useState(null);
  const [shipAnalysisDone, setShipAnalysisDone] = useState(false);
  const [videoAnalysisDone, setVideoAnalysisDone] = useState(false);
  const [vesselInfoAnalysisDone, setVesselInfoAnalysisDone] = useState(false);
  const [vesselDraft, setVesselDraft] = useState({
    vessel_name: "",
    port: "",
    imo_number: "",
    vessel_classifications: "",
    notes: "",
  });
  const [editingVesselRecordId, setEditingVesselRecordId] = useState(null);
  const [editingVesselDraft, setEditingVesselDraft] = useState(null);
  const [vesselDraftDirty, setVesselDraftDirty] = useState(false);
  const [showAccessAdmin, setShowAccessAdmin] = useState(false);
  const [mediaFit, setMediaFit] = useState(() => {
    const saved = window.localStorage.getItem("oceaniq-media-fit");
    return saved === "fill" ? "fill" : "fit";
  });
  const fileInputRef = useRef(null);
  const vesselInfoFileInputRef = useRef(null);
  const resultsCardRef = useRef(null);

  useEffect(() => {
    window.localStorage.setItem("oceaniq-media-fit", mediaFit);
  }, [mediaFit]);
  useEffect(() => () => {
    if (vesselInfoPreview) URL.revokeObjectURL(vesselInfoPreview);
  }, [vesselInfoPreview]);
  useEffect(() => () => {
    if (vesselVideoPreview) URL.revokeObjectURL(vesselVideoPreview);
  }, [vesselVideoPreview]);
  useEffect(() => {
    if (!expandedHistoryImage) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setExpandedHistoryImage(null);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [expandedHistoryImage]);
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

  const fetchVesselHistory = useCallback(async () => {
    try {
      setVesselHistoryLoading(true);
      setVesselHistoryError("");
      const { data } = await axios.get(`${API_BASE_URL}/vessel-detection-history`);
      setVesselHistory(data.history || []);
      setVesselHistoryStorage(data.storage || "unknown");
    } catch (error) {
      console.error("Failed to load vessel detection history:", error);
      setVesselHistoryError("Could not load saved vessel records.");
    } finally {
      setVesselHistoryLoading(false);
    }
  }, []);

  /* ── File selection handler ── */
  const processFile = useCallback((f) => {
    if (!canWrite || !f) return;
    const fileExtension = f.name.split(".").pop()?.toLowerCase();
    const isVideo = activeModule === "boat" && (
      f.type.startsWith("video/") ||
      ["mp4", "mov", "avi", "mkv", "webm"].includes(fileExtension)
    );
    if (isVideo) {
      setFile(null);
      setPreview(null);
      setVesselVideoFile(f);
      setVesselVideoPreview(URL.createObjectURL(f));
      setShipAnalysisDone(false);
      setVideoAnalysisDone(false);
      setVesselInfoAnalysisDone(false);
      setVesselRecordId(null);
      setVesselDraft({
        vessel_name: "",
        port: "",
        imo_number: "",
        vessel_classifications: "",
        notes: "",
      });
      setResult(null);
      return;
    }
    if (!f.type.startsWith("image/")) return;
    setFile(f);
    if (activeModule === "boat") {
      setVesselVideoFile(null);
      setVesselVideoPreview(null);
      setVideoAnalysisDone(false);
      setVesselRecordId(null);
      setShipAnalysisDone(false);
      setVesselDraft((current) => ({ ...current, vessel_classifications: "" }));
      setResult((current) => current?.video_analysis_performed ? null : current ? {
        ...current,
        image: null,
        results: [],
        count: 0,
        confidence: 0,
        vessel_detection_performed: false,
      } : null);
    }
    if (activeModule !== "boat") setResult(null);
    setShowGradcam(false);
    setPreview(URL.createObjectURL(f));
  }, [activeModule, canWrite]);

  const handleFileChange = (e) => processFile(e.target.files[0]);
  const handleVesselInfoFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!canWrite || !selectedFile?.type.startsWith("image/")) return;
    setVesselInfoFile(selectedFile);
    setVesselInfoPreview(URL.createObjectURL(selectedFile));
    setVesselInfoAnalysisDone(false);
    setVesselDraft({
      vessel_name: "",
      port: "",
      imo_number: "",
      vessel_classifications: "",
      notes: "",
    });
    setVesselRecordId(null);
    setResult((current) => current ? {
      ...current,
      vessel_info_image: null,
      vessel_info: null,
      vessel_info_detections: [],
      vessel_info_performed: false,
    } : null);
  };
  const clearVesselImage = () => {
    setFile(null);
    setPreview(null);
    setVesselRecordId(null);
    setShipAnalysisDone(false);
    setResult((current) => (
      current?.vessel_info_performed
        ? {
            ...current,
            vessel_detection_performed: false,
            image: null,
            results: [],
            count: 0,
            confidence: 0,
          }
        : null
    ));
  };
  const clearVesselVideo = () => {
    setVesselVideoFile(null);
    setVesselVideoPreview(null);
    setVideoAnalysisDone(false);
    setShipAnalysisDone(false);
    setVesselInfoAnalysisDone(false);
    setVesselRecordId(null);
    setVesselDraft({
      vessel_name: "",
      port: "",
      imo_number: "",
      vessel_classifications: "",
      notes: "",
    });
    setResult(null);
  };
  const clearVesselInfoImage = () => {
    setVesselInfoFile(null);
    setVesselInfoPreview(null);
    setVesselInfoAnalysisDone(false);
    setVesselRecordId(null);
    setResult((current) => current ? {
      ...current,
      vessel_info_performed: false,
      vessel_info_image: null,
      vessel_info: null,
      vessel_info_detections: [],
    } : null);
  };

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
    setVesselVideoFile(null);
    setVesselVideoPreview(null);
    setVesselInfoFile(null);
    setVesselInfoPreview(null);
    setVesselRecordId(null);
    setShipAnalysisDone(false);
    setVideoAnalysisDone(false);
    setVesselInfoAnalysisDone(false);
    setPreview(null);
    setResult(null);
    setShowGradcam(false);
    setApplyEnhancement(false);
    setSidebarOpen(false);
    if (id === "sea") fetchSeaHistory();
    if (id === "boat") fetchVesselHistory();
  };

  /* ══════════════════════════════════════════════
     BACKEND CALL
     axios POST to the selected module endpoint.
     FormData carries the image file.
     All original endpoints are preserved.
     ══════════════════════════════════════════════ */
  const runPrediction = async () => {
    if (activeModule === "boat") return runVesselAnalysis("ship");
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

  const runVesselAnalysis = async (stage) => {
    const sourceFile = stage === "video"
      ? vesselVideoFile
      : stage === "ship" ? file : vesselInfoFile;
    if (!sourceFile || loading || !canWrite) return;
    const form = new FormData();
    form.append(
      stage === "video" ? "video_file" : stage === "ship" ? "file" : "vessel_info_file",
      sourceFile,
    );
    try {
      setLoading(true);
      const endpoint = stage === "video"
        ? `${API_BASE_URL}/predict-boat-video`
        : MODULES.boat.endpoint;
      const { data } = await axios.post(endpoint, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      if (data?.error) {
        setResult({ __error: true, message: data.error, details: data.validation });
        return;
      }
      if (stage === "video") {
        setResult(data);
        setVideoAnalysisDone(true);
        setShipAnalysisDone(false);
        setVesselInfoAnalysisDone(false);
        setVesselRecordId(null);
        setVesselDraft((current) => ({
          ...current,
          vessel_classifications: [...new Set(data.vessel_classifications || [])].join(" or "),
        }));
        return;
      }
      setResult((current) => ({
        ...current,
        ...data,
        image: stage === "ship" ? data.image : current?.image || null,
        results: stage === "ship" ? data.results || [] : current?.results || [],
        count: stage === "ship" ? data.count || 0 : current?.count || 0,
        confidence: stage === "ship" ? data.confidence || 0 : current?.confidence || 0,
        status: stage === "ship" ? data.status : current?.status || data.status,
        vessel_detection_performed: stage === "ship" || Boolean(current?.vessel_detection_performed),
        vessel_info_image: stage === "info" ? data.vessel_info_image : current?.vessel_info_image || null,
        vessel_info: stage === "info" ? data.vessel_info : current?.vessel_info || null,
        vessel_info_detections: stage === "info"
          ? data.vessel_info_detections || []
          : current?.vessel_info_detections || [],
        vessel_info_performed: stage === "info" || Boolean(current?.vessel_info_performed),
      }));
      if (stage === "ship") {
        setShipAnalysisDone(true);
        setVesselRecordId(null);
        setVesselDraft((current) => ({
          ...current,
          vessel_classifications: [...new Set(data.vessel_classifications || [])].join(" or "),
        }));
      } else {
        setVesselInfoAnalysisDone(true);
        setVesselDraft((current) => ({
          ...current,
          vessel_name: data.vessel_info?.vessel_name || "",
          port: data.vessel_info?.port || "",
          imo_number: data.vessel_info?.imo_number || "",
        }));
        setVesselRecordId(null);
      }
      setVesselDraftDirty(false);
    } catch (err) {
      console.error("Vessel analysis error:", err);
      setResult({ __error: true, message: "Vessel analysis failed. Check the backend and try again." });
    } finally {
      setLoading(false);
    }
  };

  const submitVesselRecord = async () => {
    if (vesselVideoFile || videoAnalysisDone || (!shipAnalysisDone && !vesselInfoAnalysisDone) || loading || !canWrite) return;
    const record = {
      ...(result || {}),
      image: shipAnalysisDone ? file?.name : null,
      vessel_info_image: vesselInfoAnalysisDone ? vesselInfoFile?.name : null,
      vessel_detection_performed: shipAnalysisDone,
      vessel_info_performed: vesselInfoAnalysisDone,
      vessel_info: vesselInfoAnalysisDone ? {
        vessel_name: (vesselDraft.vessel_name || "").trim(),
        port: (vesselDraft.port || "").trim(),
        imo_number: (vesselDraft.imo_number || "").trim(),
      } : null,
      notes: (vesselDraft.notes || "").trim(),
      results: shipAnalysisDone
        ? result?.results || []
        : [],
      vessel_classifications: shipAnalysisDone
        ? parseVesselClassifications(vesselDraft.vessel_classifications)
        : [],
      vessel_kinds: shipAnalysisDone ? result?.vessel_kinds || [] : [],
      count: shipAnalysisDone ? result?.count || 0 : 0,
      status: shipAnalysisDone ? result?.status || "Analyzed" : "Vessel information analyzed",
    };
    const form = new FormData();
    form.append("record_json", JSON.stringify(record));
    if (shipAnalysisDone && file) form.append("ship_image", file);
    if (vesselInfoAnalysisDone && vesselInfoFile) {
      form.append("vessel_info_image", vesselInfoFile);
    }
    try {
      setLoading(true);
      const { data } = await axios.post(
        `${API_BASE_URL}/vessel-detection-records`,
        form,
        { headers: { "Content-Type": "multipart/form-data" } },
      );
      if (data?.error) {
        setResult((current) => ({ ...current, __saveError: data.error }));
        return;
      }
      setVesselRecordId(data.record.record_id);
      setVesselHistoryStorage(data.storage || vesselHistoryStorage);
      setVesselDraftDirty(false);
      setResult((current) => ({
        ...current,
        record_id: data.record.record_id,
        cloudinary_images: data.record.cloudinary_images,
        __saveSuccess: `Record saved to ${data.storage}.`,
      }));
      await fetchVesselHistory();
    } catch (error) {
      console.error("Failed to save vessel record:", error);
      setResult((current) => ({
        ...current,
        __saveError: "Could not save the record and upload its images. Check Cloudinary configuration.",
      }));
    } finally {
      setLoading(false);
    }
  };

  const updateVesselRecord = async (recordId, draft) => {
    const classifications = parseVesselClassifications(draft.vessel_classifications || "");
    const updates = {
      vessel_info: {
        vessel_name: (draft.vessel_name || "").trim() || null,
        port: (draft.port || "").trim() || null,
        imo_number: (draft.imo_number || "").trim() || null,
      },
      vessel_classifications: classifications,
      results: draft.results || [],
      notes: (draft.notes || "").trim(),
    };
    const { data } = await axios.put(
      `${API_BASE_URL}/vessel-detection-history/${recordId}`,
      updates,
    );
    if (data?.error) throw new Error(data.error);
    return data.record;
  };

  const saveVesselRecordEdits = async () => {
    if (!vesselRecordId || loading) return;
    try {
      setLoading(true);
      await updateVesselRecord(vesselRecordId, {
        ...vesselDraft,
        results: result?.results || [],
      });
      setVesselDraftDirty(false);
      setResult((current) => ({ ...current, __saveSuccess: "Saved vessel record updated." }));
      await fetchVesselHistory();
    } catch (error) {
      console.error("Failed to update vessel record:", error);
      setResult((current) => ({ ...current, __saveError: "Could not update the saved vessel record." }));
    } finally {
      setLoading(false);
    }
  };

  const startEditingVesselRecord = (item) => {
    setEditingVesselRecordId(item.record_id);
    setEditingVesselDraft({
      vessel_name: item.vessel_info?.vessel_name || "",
      port: item.vessel_info?.port || "",
      imo_number: item.vessel_info?.imo_number || "",
      vessel_classifications: (item.vessel_classifications || []).join(" or "),
      notes: item.notes || "",
      results: item.results || [],
    });
  };

  const saveEditedVesselHistoryRecord = async () => {
    try {
      await updateVesselRecord(editingVesselRecordId, editingVesselDraft);
      setEditingVesselRecordId(null);
      setEditingVesselDraft(null);
      await fetchVesselHistory();
    } catch (error) {
      console.error("Failed to update vessel history record:", error);
      setVesselHistoryError("Could not update the selected vessel record.");
    }
  };

  const deleteVesselHistoryRecord = async (recordId) => {
    if (!canWrite || !window.confirm("Delete this saved vessel record?")) return;
    try {
      const { data } = await axios.delete(
        `${API_BASE_URL}/vessel-detection-history/${recordId}`,
      );
      if (data?.error) throw new Error(data.error);
      setVesselHistory((current) => current.filter((item) => item.record_id !== recordId));
      if (vesselRecordId === recordId) setVesselRecordId(null);
    } catch (error) {
      console.error("Failed to delete vessel history record:", error);
      setVesselHistoryError(error.response?.data?.error || "Could not delete the vessel record.");
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
      const label = detection.label;
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

  /* ── Sea-state PDF report from the Sea-State branch ── */
const generateSeaStatePDF = () => {
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

            <details className="sea-history-section sea-history-details">
              <summary>
                <span className="sea-feature-title">SEA-STATE PREDICTION HISTORY</span>
                <span className="sea-history-count">{seaHistory.length} RECORDS · OPEN</span>
              </summary>
              <div className="sea-history-header sea-history-header--controls">
                <p className="sea-support-text">Recent operational classifications are kept here so the live result remains the primary focus.</p>
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
            </details>
          </div>
        )}

        {/* ══════════════════════════
            BOAT DETECTION results
            ══════════════════════════ */}
        {activeModule === "boat" && (
          <div className="result-body">
            {result.video_analysis_performed && (
              <section className="vessel-info-panel vessel-video-results" style={{ borderColor: mod.colorMid }}>
                <div className="vessel-info-heading">
                  <h3>VIDEO VESSEL ANALYSIS · {result.video || vesselVideoFile?.name}</h3>
                  <span className="vessel-model-chip">VIDEO · NOT SAVED</span>
                </div>
                <p className="vessel-video-note">
                  Video is analyzed locally, one frame per second. Neither the video nor its frames are sent to Cloudinary or saved in the database.
                </p>
                <div className="boat-analysis-summary" style={{ borderColor: mod.colorMid }}>
                  <div>
                    <span>VIDEO LENGTH</span>
                    <strong>{result.video_duration_seconds}s</strong>
                  </div>
                  <div>
                    <span>FRAMES ANALYZED</span>
                    <strong>{result.frames_analyzed}</strong>
                  </div>
                  <div>
                    <span>VESSEL OBJECTS</span>
                    <strong>{result.count}</strong>
                  </div>
                </div>
                {result.vessel_classifications?.length ? (
                  <div className="vessel-detected-types">
                    {result.vessel_classifications.map((type) => (
                      <span className="vessel-detected-type" key={type}>
                        <Ship size={19} strokeWidth={1.7} />
                        {type}
                      </span>
                    ))}
                  </div>
                ) : (
                  <div className="empty-result">
                    <Ship size={34} opacity={0.25} />
                    <p>No vessels were detected in the sampled frames.</p>
                  </div>
                )}
                {result.frames?.length > 0 && (
                  <div className="vessel-video-frame-list">
                    {result.frames.map((frame) => (
                      <div className="vessel-video-frame" key={frame.time_seconds}>
                        <span className="vessel-video-time">
                          <Film size={13} /> {frame.time_seconds}s
                        </span>
                        <span>
                          {frame.detections?.length
                            ? [...new Set(frame.detections.map((detection) => detection.label))].join(" or ")
                            : "No vessel detected"}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            )}
            {result.vessel_info && (
              <section className="vessel-info-panel" style={{ borderColor: mod.colorMid }}>
                <div className="vessel-info-heading">
                  <h3>VESSEL IDENTITY · {result.vessel_info_image || "BACKSIDE IMAGE"}</h3>
                  {result.vessel_info_performed && <span className="vessel-model-chip">OCR MODEL</span>}
                </div>
                {vesselInfoPreview && (
                  <img
                    className="vessel-result-thumb"
                    src={vesselInfoPreview}
                    alt="Uploaded vessel backside"
                  />
                )}
                <div className="vessel-info-grid">
                  {[
                    ["VESSEL NAME", result.vessel_info.vessel_name],
                    ["PORT", result.vessel_info.port],
                    ["IMO NUMBER", result.vessel_info.imo_number],
                  ].map(([label, value]) => (
                    <div className="vessel-info-field" key={label}>
                      <span>{label}</span>
                      <strong>{value || "Not read"}</strong>
                    </div>
                  ))}
                </div>
                {result.vessel_info_detections?.length > 0 && (
                  <div className="vessel-ocr-detections">
                    {result.vessel_info_detections.map((item, index) => (
                      <span className="vessel-ocr-chip" key={`${item.field}-${index}`}>
                        {item.field.replaceAll("_", " ")} · {item.text || "Text not read"}
                      </span>
                    ))}
                  </div>
                )}
              </section>
            )}
            {!result.video_analysis_performed && result.vessel_detection_performed && cleanVesselDetections.length > 0 ? (
              <>
                <section className="vessel-info-panel vessel-type-heading" style={{ borderColor: mod.colorMid }}>
                  <div className="vessel-info-heading">
                    <h3>VESSEL TYPE MODEL · {result.image || "UPLOADED IMAGE"}</h3>
                    <span className="vessel-model-chip">SHIP-DETECTION</span>
                  </div>
                  {preview && (
                    <img className="vessel-result-thumb" src={preview} alt="Analyzed vessel image" />
                  )}
                  <div className="vessel-detected-types">
                    {[...new Set(cleanVesselDetections.map((item) => item.label))].map((type) => (
                      <span className="vessel-detected-type" key={type}>
                        <Ship size={19} strokeWidth={1.7} />
                        {type}
                      </span>
                    ))}
                  </div>
                </section>
                <div className="boat-analysis-summary" style={{ borderColor: mod.colorMid }}>
                  <div>
                    <span>OBJECTS FOUND</span>
                    <strong>{cleanVesselDetections.length}</strong>
                  </div>
                  <div>
                    <span>VESSEL TYPES</span>
                    <strong>{new Set(cleanVesselDetections.map((item) => item.label)).size}</strong>
                  </div>
                  <div>
                    <span>TOTAL OBJECTS</span>
                    <strong>{result.count ?? cleanVesselDetections.length}</strong>
                  </div>
                </div>
                <div className="result-primary-row">
                  <div className="result-pred-block" style={{ borderColor: mod.colorMid }}>
                    <p className="rpb-eye">VESSELS DETECTED</p>
                    <p className="rpb-val" style={{ color: mod.color }}>{cleanVesselDetections.length}</p>
                    <p className="rpb-sub">Objects identified in frame</p>
                  </div>
                  <div className="vessel-count-badge"
                    style={{ color: mod.color, borderColor: mod.colorMid, background: mod.colorDim }}>
                    <Ship size={26} strokeWidth={1.5} />
                    <span>{cleanVesselDetections.length} FOUND</span>
                  </div>
                </div>

                <div className="vessel-list">
                  {cleanVesselDetections.map((det, i) => (
                    <div key={i} className="vessel-item" style={{ borderColor: mod.colorMid }}>
                      <div className="vessel-index"
                        style={{ background: mod.colorDim, color: mod.color }}>
                        <Crosshair size={12} /> {i + 1}
                      </div>
                      <span className="vessel-kind-chip">{det.vessel_kind || "Vessel"}</span>
                      <span className="vessel-label">{det.label}</span>
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
            ) : !result.video_analysis_performed && result.vessel_detection_performed ? (
              <div className="empty-result">
                <Ship size={38} opacity={0.2} />
                <p>No vessels detected in this frame.</p>
              </div>
            ) : null}

            {!result.video_analysis_performed && !vesselVideoFile && (shipAnalysisDone || vesselInfoAnalysisDone) && (
              <section className="vessel-review-panel" style={{ borderColor: mod.colorMid }}>
                <div className="vessel-info-heading">
                  <h3>FINAL VESSEL RECORD · REVIEW & EDIT</h3>
                  <span className="vessel-model-chip">DRAFT</span>
                </div>
                <p className="vessel-review-hint">
                  Check or correct the detected values before submitting. Images are sent to Cloudinary only when you submit this record.
                </p>
                <div className="vessel-edit-grid">
                  <label>
                    <span>VESSEL NAME</span>
                    <input
                      value={vesselDraft.vessel_name}
                      onChange={(event) => {
                        setVesselDraft((current) => ({ ...current, vessel_name: event.target.value }));
                        setVesselDraftDirty(true);
                      }}
                    />
                  </label>
                  <label>
                    <span>PORT</span>
                    <input
                      value={vesselDraft.port}
                      onChange={(event) => {
                        setVesselDraft((current) => ({ ...current, port: event.target.value }));
                        setVesselDraftDirty(true);
                      }}
                    />
                  </label>
                  <label>
                    <span>IMO NUMBER</span>
                    <input
                      value={vesselDraft.imo_number}
                      onChange={(event) => {
                        setVesselDraft((current) => ({ ...current, imo_number: event.target.value }));
                        setVesselDraftDirty(true);
                      }}
                      placeholder="IMO 9074729"
                    />
                  </label>
                  <label>
                    <span>DETECTED VESSEL TYPES</span>
                    <input
                      value={vesselDraft.vessel_classifications}
                      onChange={(event) => {
                        setVesselDraft((current) => ({ ...current, vessel_classifications: event.target.value }));
                        setVesselDraftDirty(true);
                      }}
                      placeholder="Merchant Ship or Fishing Boat"
                      disabled={!shipAnalysisDone}
                    />
                  </label>
                  <label className="vessel-notes-field">
                    <span>NOTES</span>
                    <textarea
                      value={vesselDraft.notes}
                      onChange={(event) => {
                        setVesselDraft((current) => ({ ...current, notes: event.target.value }));
                        setVesselDraftDirty(true);
                      }}
                      rows={3}
                    />
                  </label>
                </div>
                {result.__saveError && <p className="vessel-save-message vessel-save-message--error">{result.__saveError}</p>}
                {result.__saveSuccess && <p className="vessel-save-message">{result.__saveSuccess}</p>}
                {vesselRecordId ? (
                  <button
                    type="button"
                    className="vessel-submit-record"
                    onClick={saveVesselRecordEdits}
                    disabled={!vesselDraftDirty || loading || !canWrite}
                  >
                    {loading ? "SAVING…" : "UPDATE SAVED RECORD"}
                  </button>
                ) : (
                  <button
                    type="button"
                    className="vessel-submit-record"
                    onClick={submitVesselRecord}
                    disabled={loading || !canWrite}
                  >
                    {loading ? "UPLOADING IMAGES & SAVING…" : "SUBMIT IMAGES & DETAILS"}
                  </button>
                )}
              </section>
            )}

          </div>
        )}

        {/* ══════════════════════════
            RADAR results
            ══════════════════════════ */}
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

        {activeModule === "database" ? (
          <DatabaseViewer />
        ) : activeModule === "simulation" ? (
          <div className={!canWrite ? "simulation-readonly" : ""}>
            <LiveSimulation mediaFit={mediaFit} onMediaFitChange={setMediaFit} />
          </div>
        ) : (
          <>
        {/* ── 3-panel analysis grid ── */}
        <div className="analysis-grid">

          {/* ─── PANEL 1: Upload ─── */}
          <section className="a-card upload-card">
            <div className="a-card-header">
              <span className="a-card-label"><Upload size={11} /> {activeModule === "boat" ? "VESSEL IMAGE OR VIDEO" : "INPUT IMAGE"}</span>
              <span className="a-card-tag" style={{ color: mod.color }}>{mod.statusLabel}</span>
            </div>

            {/* Drag-and-drop zone */}
            <label
              className={`drop-zone ${dragOver ? "dz--over" : ""} ${(file || vesselVideoFile) ? "dz--filled" : ""} ${!canWrite ? "dz--readonly" : ""}`}
              style={dragOver ? { borderColor: mod.color } : {}}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept={activeModule === "boat" ? "image/*,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm" : "image/*"}
                onClick={(e) => { e.currentTarget.value = ""; }}
                onChange={handleFileChange}
                disabled={!canWrite}
              />

              {vesselVideoFile ? (
                <div className="dz-loaded">
                  <Film size={24} style={{ color: mod.color }} />
                  <p className="dz-fname">{vesselVideoFile.name}</p>
                  <p className="dz-change">Video is analyzed temporarily and is never saved to the database</p>
                </div>
              ) : file ? (
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
                  <p className="dz-main">Drop {activeModule === "boat" ? "vessel image or video" : "image"} here</p>
                  <p className="dz-sub">or click to browse</p>
                  <p className="dz-fmt">{activeModule === "boat" ? "IMAGE · MP4 · MOV · AVI · MKV · WEBM · MAX 100 MB / 5 MIN" : "PNG · JPG · JPEG · BMP · WEBP"}</p>
                </div>
              )}
            </label>
            {activeModule === "boat" && (
              <div className="vessel-stage-actions">
                <button
                  type="button"
                  className="run-btn vessel-stage-btn"
                  style={{ "--mc": mod.color }}
                  onClick={() => runVesselAnalysis(vesselVideoFile ? "video" : "ship")}
                  disabled={(!file && !vesselVideoFile) || loading || !canWrite}
                >
                  {loading && (file || vesselVideoFile) ? <Loader size={14} className="spin" /> : <ScanEye size={14} />}
                  {vesselVideoFile
                    ? videoAnalysisDone ? "RE-ANALYZE VIDEO" : "1 · ANALYZE VIDEO"
                    : shipAnalysisDone ? "RE-ANALYZE VESSEL TYPE" : "1 · ANALYZE VESSEL TYPE"}
                </button>
                {shipAnalysisDone && <span className="vessel-stage-complete">TYPE ANALYSIS READY</span>}
                {videoAnalysisDone && <span className="vessel-stage-complete">VIDEO ANALYSIS READY · NOT SAVED</span>}
                {vesselVideoFile && (
                  <button type="button" className="vessel-clear-input" onClick={clearVesselVideo}>
                    <X size={13} /> REMOVE VIDEO
                  </button>
                )}
                {file && (
                  <button type="button" className="vessel-clear-input" onClick={clearVesselImage}>
                    <X size={13} /> REMOVE VESSEL IMAGE
                  </button>
                )}
              </div>
            )}

            {activeModule === "boat" && !vesselVideoFile && (
              <div className="vessel-info-upload">
                <p className="vessel-info-upload-title">BACKSIDE PHOTO · NAME / PORT / IMO · OPTIONAL</p>
                <label className={`drop-zone ${vesselInfoFile ? "dz--filled" : ""} ${!canWrite ? "dz--readonly" : ""}`}>
                  <input
                    ref={vesselInfoFileInputRef}
                    type="file"
                    accept="image/*"
                    onClick={(e) => { e.currentTarget.value = ""; }}
                    onChange={handleVesselInfoFileChange}
                    disabled={!canWrite}
                  />
                  {vesselInfoFile ? (
                    <div className="dz-loaded vessel-info-upload-loaded">
                      {vesselInfoPreview && <img src={vesselInfoPreview} alt="Vessel backside preview" />}
                      <p className="dz-fname">{vesselInfoFile.name}</p>
                      <p className="dz-change">Click to change backside photo</p>
                    </div>
                  ) : (
                    <div className="dz-empty">
                      <div className="dz-upload-icon"><Upload size={20} style={{ color: mod.color }} /></div>
                      <p className="dz-main">Upload clear backside image</p>
                      <p className="dz-sub">Used separately for vessel text detection and OCR</p>
                      <p className="dz-fmt">PNG · JPG · JPEG · BMP · WEBP</p>
                    </div>
                  )}
                </label>
                <div className="vessel-stage-actions">
                  <button
                    type="button"
                    className="run-btn vessel-stage-btn"
                    style={{ "--mc": mod.color }}
                    onClick={() => runVesselAnalysis("info")}
                    disabled={!vesselInfoFile || loading || !canWrite}
                  >
                    {loading && vesselInfoFile ? <Loader size={14} className="spin" /> : <ScanEye size={14} />}
                    {vesselInfoAnalysisDone ? "RE-ANALYZE VESSEL DETAILS" : "2 · READ NAME / PORT / IMO"}
                  </button>
                  {vesselInfoAnalysisDone && <span className="vessel-stage-complete">DETAIL ANALYSIS READY</span>}
                  {vesselInfoFile && (
                  <button type="button" className="vessel-clear-input" onClick={clearVesselInfoImage}>
                    <X size={13} /> REMOVE BACKSIDE PHOTO
                  </button>
                  )}
                </div>
              </div>
            )}

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

            {activeModule !== "boat" && <button
              className="run-btn"
              style={{ "--mc": mod.color, "--mc-dim": mod.colorDim }}
              onClick={runPrediction}
              disabled={!file || loading || !canWrite}
            >
              {loading
                ? <><Loader size={15} className="spin" /> ANALYZING…</>
                : <><Zap size={15} /> RUN ANALYSIS</>}
            </button>}
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
              {activeModule === "boat" && vesselVideoPreview ? (
                <video
                  src={vesselVideoPreview}
                  className={`preview-img preview-img--${mediaFit}`}
                  controls
                  muted
                  playsInline
                  preload="metadata"
                />
              ) : preview ? (
                <div className="preview-img-wrap">
                  <img
                    src={preview}
                    alt="Preview"
                    className={`preview-img preview-img--${mediaFit}`}
                  />

                  {activeModule === "boat" && !loading && (
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
        {activeModule === "boat" && (
          <section className="a-card vessel-history-panel">
            <div className="sea-history-header">
              <div>
                <p className="sea-feature-title">VESSEL DETECTION RECORDS</p>
                <p className="vessel-storage-status">STORAGE · {vesselHistoryStorage.toUpperCase()}</p>
              </div>
              <div className="sea-history-actions">
                <button onClick={fetchVesselHistory} disabled={vesselHistoryLoading}>
                  {vesselHistoryLoading ? "LOADING…" : "REFRESH"}
                </button>
              </div>
            </div>
            {vesselHistoryError && <p className="vessel-save-message vessel-save-message--error">{vesselHistoryError}</p>}
            {vesselHistory.length === 0 ? (
              <p className="sea-history-empty">No submitted vessel records are available.</p>
            ) : (
              <>
                <div className="vessel-table-scroll">
                  <table className="vessel-record-table">
                  <thead>
                    <tr>
                      <th>VESSEL / TYPE</th>
                      <th>IDENTITY DETAILS</th>
                      <th>IMAGES</th>
                      <th>STATUS</th>
                      <th>SUBMITTED</th>
                      <th>ACTIONS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {vesselHistory.map((item, index) => (
                      <Fragment key={`${item.timestamp}-${item.record_id || index}`}>
                        <tr>
                          <td>
                            <strong className="vessel-table-primary">
                              {item.vessel_classifications?.length
                                ? item.vessel_classifications.join(" or ")
                                : item.vessel_info?.vessel_name || "Vessel record"}
                            </strong>
                            <span className="vessel-table-secondary">
                              {item.vessel_info?.vessel_name || "Name not identified"}
                            </span>
                          </td>
                          <td>
                            <span>{item.vessel_info?.port || "Port not identified"}</span>
                            <span className="vessel-table-secondary">{item.vessel_info?.imo_number || "IMO not identified"}</span>
                            {item.notes && <span className="vessel-table-secondary">{item.notes}</span>}
                          </td>
                          <td>
                            <div className="vessel-record-images">
                              {item.image_url && (
                                <img
                                  src={item.image_url}
                                  alt="Vessel"
                                  role="button"
                                  tabIndex={0}
                                  onClick={() => setExpandedHistoryImage({ url: item.image_url, alt: "Vessel" })}
                                  onKeyDown={(event) => {
                                    if (event.key === "Enter" || event.key === " ") {
                                      event.preventDefault();
                                      setExpandedHistoryImage({ url: item.image_url, alt: "Vessel" });
                                    }
                                  }}
                                />
                              )}
                              {item.vessel_info_image_url && (
                                <img
                                  src={item.vessel_info_image_url}
                                  alt="Vessel backside"
                                  role="button"
                                  tabIndex={0}
                                  onClick={() => setExpandedHistoryImage({ url: item.vessel_info_image_url, alt: "Vessel backside" })}
                                  onKeyDown={(event) => {
                                    if (event.key === "Enter" || event.key === " ") {
                                      event.preventDefault();
                                      setExpandedHistoryImage({ url: item.vessel_info_image_url, alt: "Vessel backside" });
                                    }
                                  }}
                                />
                              )}
                              {!item.image_url && !item.vessel_info_image_url && <span>No image</span>}
                            </div>
                          </td>
                          <td><span className="vessel-status-pill">{item.status || "Saved"}</span></td>
                          <td>{item.timestamp || "—"}</td>
                          <td>
                            <div className="vessel-history-controls">
                              <button onClick={() => startEditingVesselRecord(item)} disabled={!canWrite || !item.record_id}>EDIT</button>
                              <button className="danger" onClick={() => deleteVesselHistoryRecord(item.record_id)} disabled={!canWrite || !item.record_id}>DELETE</button>
                            </div>
                          </td>
                        </tr>
                        {editingVesselRecordId === item.record_id && editingVesselDraft && (
                          <tr className="vessel-table-edit-row">
                            <td colSpan={6}>
                              <div className="vessel-history-edit">
                                {[
                                  ["vessel_name", "VESSEL NAME"],
                                  ["port", "PORT"],
                                  ["imo_number", "IMO NUMBER"],
                                  ["vessel_classifications", "VESSEL TYPES"],
                                ].map(([field, label]) => (
                                  <label key={field}>
                                    <span>{label}</span>
                                    <input
                                      value={editingVesselDraft[field]}
                                      onChange={(event) => setEditingVesselDraft((current) => ({
                                        ...current,
                                        [field]: event.target.value,
                                      }))}
                                    />
                                  </label>
                                ))}
                                <label className="vessel-table-notes">
                                  <span>NOTES</span>
                                  <textarea
                                    rows={2}
                                    value={editingVesselDraft.notes}
                                    onChange={(event) => setEditingVesselDraft((current) => ({
                                      ...current,
                                      notes: event.target.value,
                                    }))}
                                  />
                                </label>
                                <div className="vessel-history-controls">
                                  <button onClick={saveEditedVesselHistoryRecord}>SAVE CHANGES</button>
                                  <button onClick={() => setEditingVesselRecordId(null)}>CANCEL</button>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    ))}
                  </tbody>
                  </table>
                </div>
              </>
            )}
          </section>
        )}
          </>
        )}

        {expandedHistoryImage && (
          <div
            className="vessel-image-lightbox"
            role="presentation"
            onClick={() => setExpandedHistoryImage(null)}
          >
            <div
              className="vessel-image-lightbox__content"
              role="dialog"
              aria-modal="true"
              aria-label="Expanded vessel image"
              onClick={(event) => event.stopPropagation()}
            >
              <button
                type="button"
                className="vessel-image-lightbox__close"
                onClick={() => setExpandedHistoryImage(null)}
                aria-label="Close expanded image"
              >
                <X size={16} /> CLOSE
              </button>
              <img src={expandedHistoryImage.url} alt={expandedHistoryImage.alt} />
              <span>{expandedHistoryImage.alt} · CLICK OUTSIDE OR PRESS ESC TO CLOSE</span>
            </div>
          </div>
        )}

        {activeModule !== "simulation" && activeModule !== "boat" && file && canWrite && (
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
              {loading ? <><Loader size={14} className="spin" /> ANALYZING…</> : <><Zap size={14} /> {result ? "RE-RUN" : "RUN ANALYSIS"}</>}
            </button>
          </div>
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
