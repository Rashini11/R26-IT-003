import { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";
import {
  Upload,
  X,
  Loader,
  Send,
  Trash2,
  Image as ImageIcon,
  AlertTriangle,
} from "lucide-react";
import { API_BASE_URL, useAuth } from "./context/AuthContext";
import "./ChatAssistant.css";

const ANALYSIS_OPTIONS = [
  {
    id: "radar",
    label: "Radar objects",
    detail: "Classify radar screenshots as bird, ship, or unknown.",
  },
  {
    id: "vessel",
    label: "Vessel detection",
    detail: "Locate and classify vessels in drone imagery.",
  },
  {
    id: "sea",
    label: "Sea state",
    detail: "Read sea-surface conditions from marine imagery.",
  },
  {
    id: "hull",
    label: "Hull condition",
    detail: "Assess underwater hull condition with Grad-CAM.",
  },
];

function makeId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function ProbabilityChart({ chart }) {
  if (!chart?.items?.length) return null;
  const max = Math.max(...chart.items.map((item) => Number(item.value) || 0), 1);

  return (
    <div className="chat-chart" role="img" aria-label={chart.title || "Probability chart"}>
      <p className="chat-chart-title">{chart.title}</p>
      <ul className="chat-chart-list">
        {chart.items.map((item) => {
          const value = Number(item.value) || 0;
          const width = Math.max(4, Math.round((value / max) * 100));
          return (
            <li key={item.label} className="chat-chart-row">
              <span className="chat-chart-label">{item.label}</span>
              <div className="chat-chart-track" aria-hidden="true">
                <div className="chat-chart-fill" style={{ width: `${width}%` }} />
              </div>
              <span className="chat-chart-value">
                {value}
                {chart.unit === "percent" ? "%" : ""}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function ResultVisuals({ visualisation }) {
  if (!visualisation) return null;
  const gradcam = visualisation.gradcam;
  const annotated = visualisation.annotated_image;
  const chart = visualisation.probability_chart;

  return (
    <div className="chat-visuals">
      {chart && <ProbabilityChart chart={chart} />}
      {gradcam?.base64 && (
        <figure className="chat-media-fig">
          <img
            src={`data:${gradcam.mime_type || "image/jpeg"};base64,${gradcam.base64}`}
            alt="Grad-CAM explanation heatmap"
          />
          <figcaption>{gradcam.description}</figcaption>
        </figure>
      )}
      {annotated?.base64 && (
        <figure className="chat-media-fig">
          <img
            src={`data:${annotated.mime_type || "image/jpeg"};base64,${annotated.base64}`}
            alt="Annotated vessel detections"
          />
          <figcaption>{annotated.description}</figcaption>
        </figure>
      )}
      {visualisation.kind_note && <p className="chat-note">{visualisation.kind_note}</p>}
    </div>
  );
}

export default function ChatAssistant() {
  const { canWrite } = useAuth();
  const [messages, setMessages] = useState([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [busy, setBusy] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [statusText, setStatusText] = useState("");
  const [pinnedResultId, setPinnedResultId] = useState(null);
  const [resultCount, setResultCount] = useState(0);
  const listRef = useRef(null);
  const stickToBottomRef = useRef(true);
  const fileInputRef = useRef(null);

  const appendMessage = useCallback((message) => {
    setMessages((prev) => [...prev, { id: makeId(), at: new Date().toISOString(), ...message }]);
  }, []);

  const loadWelcome = useCallback(async () => {
    try {
      const { data } = await axios.get(`${API_BASE_URL}/chat/welcome`);
      appendMessage({
        role: "assistant",
        kind: "welcome",
        text: data.summary,
        analysisChoices: data.analysis_choices,
        actions: data.follow_up_actions,
        limitations: data.limitations,
      });
      setStatusText("Chat assistant ready.");
    } catch (error) {
      appendMessage({
        role: "assistant",
        kind: "error",
        text:
          error?.response?.status === 401 || error?.response?.status === 403
            ? "Authentication is required for the chat assistant."
            : "Unable to load the chat welcome message. Check that the backend is running.",
      });
      setStatusText("Chat welcome failed.");
    }
  }, [appendMessage]);

  useEffect(() => {
    loadWelcome();
  }, [loadWelcome]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    const node = listRef.current;
    if (!node || !stickToBottomRef.current) return;
    node.scrollTop = node.scrollHeight;
  }, [messages, busy]);

  const onScroll = () => {
    const node = listRef.current;
    if (!node) return;
    const distance = node.scrollHeight - node.scrollTop - node.clientHeight;
    stickToBottomRef.current = distance < 80;
  };

  const setImageFile = (nextFile) => {
    if (!nextFile) return;
    if (!nextFile.type.startsWith("image/")) {
      appendMessage({
        role: "assistant",
        kind: "error",
        text: "Please upload an image file (JPEG, PNG, BMP, or WEBP).",
      });
      setStatusText("Rejected non-image upload.");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const url = URL.createObjectURL(nextFile);
    setFile(nextFile);
    setPreviewUrl(url);
    appendMessage({
      role: "user",
      kind: "upload",
      text: `Image ready: ${nextFile.name}`,
      previewUrl: url,
      fileName: nextFile.name,
    });
    if (!selectedAnalysis) {
      appendMessage({
        role: "assistant",
        kind: "needs_selection",
        text: "Which analysis should run on this image? Automatic image-type recognition is not available in this version.",
        analysisChoices: ANALYSIS_OPTIONS,
      });
    }
    setStatusText(`Image selected: ${nextFile.name}`);
  };

  const clearImage = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    setStatusText("Image removed.");
  };

  const clearConversation = () => {
    setMessages([]);
    setPinnedResultId(null);
    setResultCount(0);
    clearImage();
    setSelectedAnalysis(null);
    setTextInput("");
    stickToBottomRef.current = true;
    loadWelcome();
    setStatusText("Conversation cleared.");
  };

  const runAnalysis = async (analysisOverride = null) => {
    const analysis = analysisOverride || selectedAnalysis;
    if (!canWrite) {
      appendMessage({
        role: "assistant",
        kind: "error",
        text: "Read-only accounts cannot run chat analyses.",
      });
      return;
    }
    if (busy) return;
    if (!file) {
      appendMessage({
        role: "assistant",
        kind: "info",
        text: "Upload an image before running analysis.",
      });
      return;
    }
    if (!analysis) {
      appendMessage({
        role: "assistant",
        kind: "needs_selection",
        text: "Select an analysis type before submission. Automatic routing is unavailable.",
        analysisChoices: ANALYSIS_OPTIONS,
      });
      return;
    }

    setBusy(true);
    setStatusText("Running analysis…");
    appendMessage({
      role: "assistant",
      kind: "processing",
      text: `Running ${ANALYSIS_OPTIONS.find((item) => item.id === analysis)?.label || analysis}…`,
    });

    const form = new FormData();
    form.append("file", file);
    form.append("analysis", analysis);
    if (analysis === "sea") form.append("apply_enhancement", "false");

    try {
      const { data } = await axios.post(`${API_BASE_URL}/chat/analyze`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      appendMessage({
        role: "assistant",
        kind: "result",
        text: data.summary,
        payload: data,
        resultId: data.result_id,
        analysisId: data.analysis_id,
        previewUrl,
        fileName: file.name,
      });
      setPinnedResultId(data.result_id);
      if (data.status === "success") setResultCount((count) => count + 1);
      setStatusText(
        data.status === "success"
          ? "Analysis complete."
          : `Analysis finished with status: ${data.status}.`
      );
    } catch (error) {
      const detail =
        error?.response?.data?.detail ||
        error?.response?.data?.summary ||
        error?.response?.data?.error;
      appendMessage({
        role: "assistant",
        kind: "error",
        text:
          error?.response?.status === 401 || error?.response?.status === 403
            ? "Your session expired or lacks permission. Please sign in again."
            : detail || "Analysis request failed.",
      });
      setStatusText("Analysis request failed.");
    } finally {
      setBusy(false);
    }
  };

  const findResultPayload = (resultId) => {
    const match = [...messages]
      .reverse()
      .find((message) => message.kind === "result" && message.resultId === (resultId || pinnedResultId));
    return match?.payload || null;
  };

  const runFollowUp = async (action, resultId = null) => {
    if (ANALYSIS_OPTIONS.some((item) => item.id === action)) {
      setSelectedAnalysis(action);
      const label = ANALYSIS_OPTIONS.find((item) => item.id === action).label;
      appendMessage({
        role: "user",
        kind: "choice",
        text: `Selected analysis: ${label}`,
      });
      appendMessage({
        role: "assistant",
        kind: "info",
        text: `Analysis set to ${label}. Upload an image if needed, then press Analyse image.`,
        actions: [{ id: "run_analysis", label: "Analyse current image" }],
      });
      setStatusText(`Analysis type: ${action}`);
      return;
    }

    if (action === "run_analysis") {
      await runAnalysis();
      return;
    }

    if (action === "analyse_another") {
      clearImage();
      appendMessage({
        role: "assistant",
        kind: "info",
        text: "Upload another image when ready. Previous results stay in this conversation until you clear it.",
      });
      return;
    }

    if (action === "clear_conversation") {
      clearConversation();
      return;
    }

    setBusy(true);
    try {
      const payload = findResultPayload(resultId);
      const { data } = await axios.post(`${API_BASE_URL}/chat/follow-up`, {
        action,
        result_payload: payload || {},
      });
      appendMessage({
        role: "assistant",
        kind: data.status === "unsupported" ? "info" : "followup",
        text: data.summary,
        details: data.details,
        actions: data.follow_up_actions,
        relatedResultId: payload?.result_id || resultId || pinnedResultId,
      });
      setStatusText("Follow-up ready.");
    } catch (error) {
      appendMessage({
        role: "assistant",
        kind: "error",
        text:
          error?.response?.status === 401 || error?.response?.status === 403
            ? "Your session expired or lacks permission. Please sign in again."
            : "Follow-up request failed.",
      });
      setStatusText("Follow-up failed.");
    } finally {
      setBusy(false);
    }
  };

  const sendTextCommand = async () => {
    const text = textInput.trim();
    if (!text || busy) return;
    appendMessage({ role: "user", kind: "text", text });
    setTextInput("");
    setBusy(true);
    try {
      const { data } = await axios.post(`${API_BASE_URL}/chat/intent`, { text });
      if (data.matched && data.intent) {
        const payload = findResultPayload(pinnedResultId);
        const follow = await axios.post(`${API_BASE_URL}/chat/follow-up`, {
          action: data.intent,
          result_payload: payload || {},
          text,
        });
        appendMessage({
          role: "assistant",
          kind: "followup",
          text: follow.data.summary,
          details: follow.data.details,
          actions: follow.data.follow_up_actions,
          relatedResultId: payload?.result_id || pinnedResultId,
        });
      } else {
        appendMessage({
          role: "assistant",
          kind: "info",
          text: data.message,
          actions: data.follow_up_actions,
        });
      }
      setStatusText("Command processed.");
    } catch {
      appendMessage({
        role: "assistant",
        kind: "error",
        text: "Could not process that command.",
      });
      setStatusText("Command failed.");
    } finally {
      setBusy(false);
    }
  };

  const selectedMeta = ANALYSIS_OPTIONS.find((item) => item.id === selectedAnalysis);
  const showEmptyState = messages.length <= 1 && !file && !busy;

  return (
    <div className="chat-shell">
      <section className="chat-hero">
        <div className="chat-hero-copy">
          <div className="eyebrow">Guided analysis</div>
          <h2>
            Analyse a maritime image with <span>OceanIQ.</span>
          </h2>
          <p>
            Choose an analysis lens, add your image, and receive a focused readout from the real
            hull, sea-state, vessel or radar model. Automatic image-type routing is not available
            in this version.
          </p>
        </div>
        <div className="chat-hero-card">
          <div className="eyebrow">Current session</div>
          <b>{resultCount}</b>
          <small>
            successful analyses in this browser session
            {selectedMeta ? ` · ${selectedMeta.label} selected` : " · no lens selected"}
          </small>
        </div>
      </section>

      <div className="sr-only" aria-live="polite">
        {statusText}
      </div>

      <div className="chat-layout">
        <section className="chat-thread-panel">
          <div className="chat-card-head">
            <h3>Analysis workspace</h3>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span className={`chat-pill ${busy ? "" : "live"}`}>
                {busy ? "Processing" : file ? "Image ready" : "Awaiting input"}
              </span>
              <button type="button" className="chat-btn ghost" onClick={clearConversation} disabled={busy}>
                <Trash2 size={14} aria-hidden="true" /> Clear
              </button>
            </div>
          </div>

          <div
            className="chat-thread"
            ref={listRef}
            onScroll={onScroll}
            tabIndex={0}
            aria-label="Chat messages"
          >
            {showEmptyState && (
              <div className="chat-empty">
                <div>
                  <div className="chat-radar" aria-hidden="true" />
                  <h3>Start with an image</h3>
                  <p>
                    Upload a maritime frame and select one analysis lens. Results appear here as a
                    concise operational brief. Radar, sea-surface, hull and vessel imagery are
                    supported.
                  </p>
                  <div className="chat-empty-actions">
                    <button
                      type="button"
                      className="chat-btn primary"
                      style={{ width: "auto" }}
                      disabled={!canWrite || busy}
                      onClick={() => fileInputRef.current?.click()}
                    >
                      Upload image
                    </button>
                    <button
                      type="button"
                      className="chat-btn"
                      disabled={busy}
                      onClick={() => runFollowUp("change_analysis")}
                    >
                      Explore a lens
                    </button>
                  </div>
                </div>
              </div>
            )}

            {messages.map((message) => (
              <article
                key={message.id}
                className={`chat-bubble chat-bubble--${message.role} chat-bubble--${message.kind || "text"}`}
              >
                <p className="chat-bubble-text">{message.text}</p>

                {message.previewUrl && (
                  <img
                    className="chat-inline-preview"
                    src={message.previewUrl}
                    alt={message.fileName || "Uploaded image"}
                  />
                )}

                {message.kind === "result" && message.payload && (
                  <div className="chat-result-card">
                    <div className="chat-result-meta">
                      <span>{message.payload.component || message.analysisId}</span>
                      <span className={`chat-status chat-status--${message.payload.status}`}>
                        {message.payload.status}
                      </span>
                    </div>
                    {message.payload.model_version && (
                      <p className="chat-note">Model: {message.payload.model_version}</p>
                    )}
                    <ResultVisuals visualisation={message.payload.visualisation} />
                    {!!message.payload.limitations?.length && (
                      <ul className="chat-limitations">
                        {message.payload.limitations.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    )}
                    <div className="chat-actions">
                      {(message.payload.follow_up_actions || []).map((action) => (
                        <button
                          key={`${message.id}-${action.id}`}
                          type="button"
                          className="chat-chip"
                          disabled={busy}
                          onClick={() => runFollowUp(action.id, message.resultId)}
                        >
                          {action.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {message.analysisChoices && (
                  <div className="chat-actions">
                    {message.analysisChoices.map((choice) => (
                      <button
                        key={choice.id}
                        type="button"
                        className={`chat-chip ${selectedAnalysis === choice.id ? "chat-chip--active" : ""}`}
                        disabled={busy}
                        onClick={() => runFollowUp(choice.id)}
                      >
                        {choice.label}
                      </button>
                    ))}
                  </div>
                )}

                {message.actions && message.kind !== "result" && (
                  <div className="chat-actions">
                    {message.actions.map((action) => (
                      <button
                        key={`${message.id}-${action.id}`}
                        type="button"
                        className="chat-chip"
                        disabled={busy}
                        onClick={() => runFollowUp(action.id, message.relatedResultId)}
                      >
                        {action.label}
                      </button>
                    ))}
                  </div>
                )}

                {message.details?.probability_chart && (
                  <ProbabilityChart chart={message.details.probability_chart} />
                )}
              </article>
            ))}

            {busy && (
              <div className="chat-busy" role="status">
                <Loader size={14} className="spin" aria-hidden="true" /> Processing…
              </div>
            )}
          </div>

          <div className="chat-composer">
            <label className="chat-text-label" htmlFor="chat-command-input">
              Predefined command
            </label>
            <div className="chat-composer-row">
              <input
                id="chat-command-input"
                type="text"
                value={textInput}
                disabled={busy}
                placeholder="Try: show model confidence…"
                onChange={(event) => setTextInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    sendTextCommand();
                  }
                }}
              />
              <button
                type="button"
                className="chat-btn"
                onClick={sendTextCommand}
                disabled={busy || !textInput.trim()}
              >
                <Send size={14} aria-hidden="true" /> Send
              </button>
            </div>
            <p className="chat-composer-hint">
              Free-form natural language is not supported. Use buttons or short predefined commands.
            </p>
          </div>
        </section>

        <aside className="chat-side-panel">
          <section className="chat-side-card">
            <div className="chat-card-head">
              <h3>Choose a lens</h3>
              <span className="eyebrow">Required</span>
            </div>
            <div className="chat-card-body">
              <div className="chat-analysis-grid">
                {ANALYSIS_OPTIONS.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    className={`chat-analysis-btn ${selectedAnalysis === option.id ? "is-active" : ""}`}
                    onClick={() => setSelectedAnalysis(option.id)}
                    disabled={busy}
                  >
                    <b>{option.label}</b>
                    <small>{option.detail}</small>
                  </button>
                ))}
              </div>
              <p className="chat-hint-note">
                {selectedMeta
                  ? `${selectedMeta.label} selected. Add an image, then run analysis.`
                  : "Select a lens to prepare the analysis pipeline. Automatic routing is unavailable."}
              </p>
            </div>
          </section>

          <section className="chat-side-card">
            <div className="chat-card-head">
              <h3>Image intake</h3>
              <span className="eyebrow">PNG · JPG · BMP · WEBP</span>
            </div>
            <div className="chat-card-body">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                disabled={!canWrite || busy}
                className="sr-only"
                onChange={(event) => setImageFile(event.target.files?.[0])}
              />
              {previewUrl ? (
                <div className="chat-preview-wrap">
                  <img src={previewUrl} alt="Selected upload preview" />
                  <div className="chat-preview-actions">
                    <button
                      type="button"
                      className="chat-btn ghost"
                      onClick={clearImage}
                      disabled={busy || !canWrite}
                    >
                      <X size={14} aria-hidden="true" /> Remove
                    </button>
                    <button
                      type="button"
                      className="chat-btn ghost"
                      disabled={busy || !canWrite}
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <ImageIcon size={14} aria-hidden="true" /> Replace
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  type="button"
                  className={`chat-drop ${!canWrite ? "is-disabled" : ""}`}
                  disabled={!canWrite || busy}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={18} aria-hidden="true" />
                  <span>Drop a frame here or browse files</span>
                </button>
              )}

              <button
                type="button"
                className="chat-btn primary"
                disabled={!canWrite || busy || !file || !selectedAnalysis}
                onClick={() => runAnalysis()}
              >
                {busy ? <Loader size={14} className="spin" aria-hidden="true" /> : <Send size={14} aria-hidden="true" />}
                Analyse image
              </button>

              <div className="chat-metrics" aria-label="Session notes">
                <div className="chat-metric">
                  <small>Components</small>
                  <b>4</b>
                </div>
                <div className="chat-metric">
                  <small>Mode</small>
                  <b>Det.</b>
                </div>
                <div className="chat-metric">
                  <small>Routing</small>
                  <b>Manual</b>
                </div>
              </div>

              {!canWrite && (
                <p className="chat-note">
                  <AlertTriangle size={12} aria-hidden="true" />
                  Read-only access cannot run analyses.
                </p>
              )}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
