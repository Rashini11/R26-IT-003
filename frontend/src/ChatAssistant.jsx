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
          const width = Math.max(0, Math.min(100, chart.unit === "percent" ? value : (value / max) * 100));
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
  // OCEANIQ_IMAGE_ROUTER_V1
  const [routingSuggestion, setRoutingSuggestion] = useState(null);
  const [routingBusy, setRoutingBusy] = useState(false);

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
  // OCEANIQ_CHAT_UI_V2
  const manualSelectionRef = useRef(false);
  const uploadRevisionRef = useRef(0);
  const imageUrlsRef = useRef(new Set());
  const analysisLockRef = useRef(false);

  const chooseAnalysis = (component) => {
    manualSelectionRef.current = true;
    setSelectedAnalysis(component);
  };

  const appendMessage = useCallback((message) => {
    setMessages((prev) => [...prev, { id: makeId(), at: new Date().toISOString(), ...message }]);
  }, []);

  const loadWelcome = useCallback(async () => {
    try {
      const { data } = await axios.get(`${API_BASE_URL}/chat/welcome`);
      appendMessage({
        role: "assistant",
        kind: "welcome",
        text: "Welcome to OceanIQ. Upload an image for automatic component selection, then press Analyse. Manual selection is always available.",
        analysisChoices: data.analysis_choices,
        actions: data.follow_up_actions,
        limitations: ["Experimental image router; unsupported-image detection is limited. No LLM is used."],
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
    const urls = imageUrlsRef.current;
    return () => {
      urls.forEach((url) => URL.revokeObjectURL(url));
      urls.clear();
    };
  }, []);

  // The page owns scrolling. New responses do not move a reader away
  // from older messages; no nested conversation scrollbar is used.

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
    if (busy) return;
    if (nextFile.size > 10 * 1024 * 1024) {
      appendMessage({ role: "assistant", kind: "error",
        text: "Please choose an image under 10 MB." });
      return;
    }
    uploadRevisionRef.current += 1;
    manualSelectionRef.current = false;
    setSelectedAnalysis(null);
    setRoutingSuggestion(null);
    const url = URL.createObjectURL(nextFile);
    imageUrlsRef.current.add(url);
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
        text: "Checking the image type. The matching component will be selected automatically. You can override it manually.",
        analysisChoices: ANALYSIS_OPTIONS,
      });
    }
    setStatusText(`Image selected: ${nextFile.name}`);
  };

  const clearImage = () => {
    uploadRevisionRef.current += 1;
    manualSelectionRef.current = false;
    setSelectedAnalysis(null);
    setRoutingSuggestion(null);
    setRoutingBusy(false);
    setFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    setStatusText("Image removed.");
  };

  const clearConversation = () => {
    if (busy) return;
    imageUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    imageUrlsRef.current.clear();
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


  // Obtain a suggestion for each new upload; component inference still
  // requires an explicit user action. Ignore obsolete upload responses.
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const revision = uploadRevisionRef.current;
    setRoutingSuggestion(null);

    if (!file || !canWrite) {
      setRoutingBusy(false);
      return () => {
        active = false;
        controller.abort();
      };
    }

    setRoutingBusy(true);
    const form = new FormData();
    form.append("file", file);

    axios.post(`${API_BASE_URL}/chat/route`, form, {
      signal: controller.signal,
    }).then(({ data }) => {
      if (!active || revision !== uploadRevisionRef.current) return;
      const validComponent = ANALYSIS_OPTIONS.some(
        (option) => option.id === data.suggested_component
      );
      const detected = validComponent && data.status === "needs_confirmation";
      if (detected && !manualSelectionRef.current) {
        setSelectedAnalysis(data.suggested_component);
      }
      setRoutingSuggestion({
        ...data, sourceFile: file,
        message: detected
          ? `Detected image type: ${data.label}. The matching component is selected automatically. Press Analyse image to run it, or choose a different component.`
          : data.message,
      });
      setStatusText(detected ? `Detected ${data.label}. Ready to analyse.` : data.message);
    }).catch((error) => {
      if (!active || revision !== uploadRevisionRef.current || error?.code === "ERR_CANCELED") return;
      setRoutingSuggestion({
        status: "unavailable",
        sourceFile: file,
        message: "Suggestion unavailable. Choose an analysis manually.",
      });
    }).finally(() => {
      if (active && revision === uploadRevisionRef.current) setRoutingBusy(false);
    });

    return () => {
      active = false;
      controller.abort();
    };
  }, [file, canWrite]);

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
    if (busy || analysisLockRef.current) return;
    if (routingBusy) {
      setStatusText("Checking image type. Please wait.");
      return;
    }
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
        text: "Wait for image detection or choose a component manually before analysis.",
        analysisChoices: ANALYSIS_OPTIONS,
      });
      return;
    }

    analysisLockRef.current = true;
    setBusy(true);
    const processingId = makeId();
    setStatusText("Running analysis…");
    appendMessage({
      id: processingId,
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
      setMessages((previous) => previous.filter((message) => message.id !== processingId));
      analysisLockRef.current = false;
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
      chooseAnalysis(action);
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
            Upload an image and OceanIQ will detect its type and select the matching component.
            Press Analyse to view the result, or change the component manually.
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
                    Upload a maritime image to detect its component. Results appear here as a
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

                {message.previewUrl && message.kind === "upload" && (
                  <details className="chat-source-image">
                    <summary>View uploaded image</summary>
                    <img className="chat-inline-preview" src={message.previewUrl}
                      alt={message.fileName || "Uploaded image"} />
                  </details>
                )}

                {message.kind === "result" && message.payload && (
                  <div className="chat-result-card">
                    <div className="chat-result-meta">
                      <span>{message.payload.component || message.analysisId}</span>
                      <span className={`chat-status chat-status--${message.payload.status}`}>
                        {message.payload.status?.replaceAll("_", " ")}
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
            <section className="chat-side-card" aria-label="Image-type suggestion">
              <div className="chat-card-head">
                <h3>Image-type detection</h3>
                <span className="eyebrow">Experimental</span>
              </div>
              <div className="chat-card-body" aria-live="polite">
                {routingBusy ? (
                  <p>Checking image type…</p>
                ) : routingSuggestion?.sourceFile === file ? (
                  <>
                    <p>{routingSuggestion.message}</p>
                    {typeof routingSuggestion.score === "number" && (
                      <p className="chat-note">
                        Image-type model score: {routingSuggestion.score}%.
                        This is not a guarantee of correctness.
                      </p>
                    )}
                    <p className="chat-note">
                      You can choose any analysis manually below.
                      Unsupported-image detection is limited.
                    </p>
                  </>
                ) : (
                  <p>Upload an image to receive an analysis suggestion.</p>
                )}
              </div>
            </section>

          <section className="chat-side-card">
            <div className="chat-card-head">
              <h3>Analysis component</h3>
              <span className="eyebrow">Manual override</span>
            </div>
            <div className="chat-card-body">
              <div className="chat-analysis-grid">
                {ANALYSIS_OPTIONS.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    className={`chat-analysis-btn ${selectedAnalysis === option.id ? "is-active" : ""}`}
                    onClick={() => chooseAnalysis(option.id)}
                    disabled={busy}
                  >
                    <b>{option.label}</b>
                    <small>{option.detail}</small>
                  </button>
                ))}
              </div>
              <p className="chat-hint-note">
                {selectedMeta
                  ? `${selectedMeta.label} selected${manualSelectionRef.current ? " manually" : " automatically"}. Press Analyse when ready.`
                  : "Upload an image for a suggestion, or select an analysis manually."}
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
                  <span>Browse for a maritime image</span>
                </button>
              )}

              <button
                type="button"
                className="chat-btn primary"
                disabled={!canWrite || busy || routingBusy || !file || !selectedAnalysis}
                onClick={() => runAnalysis()}
              >
                {busy ? <Loader size={14} className="spin" aria-hidden="true" /> : <Send size={14} aria-hidden="true" />}
                {routingBusy ? "Detecting image type…" : selectedMeta ? `Analyse ${selectedMeta.label}` : "Analyse image"}
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
                  <b>{manualSelectionRef.current ? "Manual" : "Auto"}</b>
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
