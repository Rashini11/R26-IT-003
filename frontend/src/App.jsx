import { useState } from "react";
import axios from "axios";
import { jsPDF } from "jspdf";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

const modules = {
  hull: {
    title: "Hull Defect Detection",
    endpoint: `${API_BASE_URL}/predict-hull-defect`,
    description: "Detects hull defects such as corrosion, cracks, and biofouling.",
  },
  sea: {
    title: "Sea State Classification",
    endpoint: `${API_BASE_URL}/predict-sea-state`,
    description: "Analyzes sea images and classifies sea state conditions.",
  },
  boat: {
    title: "Boat Detection",
    endpoint: `${API_BASE_URL}/predict-boat-detection`,
    description: "Detects boats or vessels from maritime images.",
  },
  radar: {
    title: "Radar Object Classification",
    endpoint: `${API_BASE_URL}/predict-radar-object`,
    description: "Classifies radar objects as bird, ship, or unknown using YOLO + CNN.",
  },
};

function App() {
  const [mode, setMode] = useState("hull");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showGradcam, setShowGradcam] = useState(false);

  const [applyEnhancement, setApplyEnhancement] = useState(false);
  const [seaHistory, setSeaHistory] = useState([]);

  const selectedModule = modules[mode];

  const fetchSeaHistory = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/sea-state-history`);
      const historyData = response.data.history || [];
      setSeaHistory([...historyData].reverse());
    } catch (error) {
      console.error(error);
    }
  };

  const clearSeaHistory = async () => {
    try {
      await axios.delete(`${API_BASE_URL}/sea-state-history`);
      setSeaHistory([]);
    } catch (error) {
      console.error(error);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    setFile(selectedFile);
    setResult(null);
    setShowGradcam(false);

    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    } else {
      setPreview(null);
    }
  };

  const handleModeChange = (newMode) => {
    setMode(newMode);
    setFile(null);
    setPreview(null);
    setResult(null);
    setShowGradcam(false);
    setApplyEnhancement(false);

    if (newMode === "sea") {
      fetchSeaHistory();
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    if (mode === "sea") {
      formData.append("apply_enhancement", applyEnhancement);
    }

    try {
      setLoading(true);

      const response = await axios.post(selectedModule.endpoint, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(response.data);

      if (mode === "sea") {
        fetchSeaHistory();
      }
    } catch (error) {
      console.error(error);
      alert("Prediction failed. Please check whether the backend is running.");
    } finally {
      setLoading(false);
    }
  };

const generatePDF = () => {
  if (!result) return;

  const doc = new jsPDF();

  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();

  const margin = 20;
  const contentWidth = pageWidth - margin * 2;

  let y = 20;

  // =====================================================
  // PAGE / POSITION HELPERS
  // =====================================================

  const checkPageSpace = (requiredSpace = 10) => {
    if (y + requiredSpace > pageHeight - 25) {
      addFooter();
      doc.addPage();
      y = 20;
    }
  };

  const addFooter = () => {
    const pageNumber = doc.internal.getNumberOfPages();

    doc.setFontSize(8);
    doc.setTextColor(100, 100, 100);

    doc.text(
      "OceanIQ - Marine AI Inspection System",
      margin,
      pageHeight - 10
    );

    doc.text(
      `Page ${pageNumber}`,
      pageWidth - margin,
      pageHeight - 10,
      { align: "right" }
    );
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

    const lines = doc.splitTextToSize(
      String(text),
      contentWidth - indent
    );

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

    const lines = doc.splitTextToSize(
      String(value ?? "N/A"),
      contentWidth - labelWidth
    );

    doc.text(
      lines[0],
      margin + labelWidth,
      y
    );

    y += 5.5;

    // If the value wraps onto multiple lines
    for (let i = 1; i < lines.length; i++) {
      checkPageSpace(6);

      doc.text(
        lines[i],
        margin + labelWidth,
        y
      );

      y += 5.5;
    }

    y += 1;
  };

  const addBullet = (text) => {
    const lines = doc.splitTextToSize(
      String(text),
      contentWidth - 8
    );

    lines.forEach((line, index) => {
      checkPageSpace(6);

      doc.text(
        index === 0 ? `• ${line}` : `  ${line}`,
        margin + 3,
        y
      );

      y += 5.5;
    });

    y += 1;
  };


  // =====================================================
  // HEADER
  // =====================================================

  doc.setFont("helvetica", "bold");
  doc.setFontSize(20);
  doc.setTextColor(20, 20, 20);

  doc.text(
    "Marine AI Inspection System",
    pageWidth / 2,
    y,
    { align: "center" }
  );

  y += 9;

  doc.setFontSize(15);
  doc.setTextColor(0, 120, 180);

  doc.text(
    "Sea State Classification Report",
    pageWidth / 2,
    y,
    { align: "center" }
  );

  y += 12;

  doc.setDrawColor(0, 160, 200);
  doc.line(margin, y, pageWidth - margin, y);

  y += 10;


  // =====================================================
  // GENERAL INFORMATION
  // =====================================================

  addTitle("Prediction Information");

  addKeyValue(
    "File",
    result.filename || "N/A"
  );

  addKeyValue(
    "Timestamp",
    result.timestamp || new Date().toLocaleString()
  );

  addKeyValue(
    "Processing Time",
    `${result.processing_time ?? "N/A"} sec`
  );

  addKeyValue(
    "Enhancement Applied",
    result.enhancement_applied ? "Yes" : "No"
  );

  y += 3;


  // =====================================================
  // SEA STATE RESULT
  // =====================================================

  addTitle("Sea State Classification");

  checkPageSpace(20);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(17);
  doc.setTextColor(20, 20, 20);

  doc.text(
    `Predicted Sea State: ${String(
      result.predicted_sea_state || "N/A"
    ).toUpperCase()}`,
    margin,
    y
  );

  y += 9;

  doc.setFontSize(11);
  doc.setTextColor(40, 40, 40);

  doc.text(
    `AI Confidence: ${result.confidence ?? 0}%`,
    margin,
    y
  );

  y += 8;


  // =====================================================
  // OVERALL RISK
  // =====================================================

  addTitle("Overall Risk");

  const risk = result.risk_indicator;

  addKeyValue(
    "Risk Level",
    risk?.level || "N/A"
  );

  addKeyValue(
    "Risk Score",
    `${risk?.score ?? "N/A"}/100`
  );

  if (risk?.reasons && risk.reasons.length > 0) {
    addKeyValue(
      "Risk Factors",
      risk.reasons.join(", ")
    );
  } else {
    addKeyValue(
      "Risk Factors",
      "None"
    );
  }

  y += 3;


  // =====================================================
  // CLASS PROBABILITIES
  // =====================================================

  addTitle("Class Probabilities");

  const probabilities = result.probabilities || {};

  Object.entries(probabilities).forEach(([label, probability]) => {
    addKeyValue(
      label,
      `${probability}%`
    );
  });

  y += 3;


  // =====================================================
  // IMAGE QUALITY
  // =====================================================

  addTitle("Image Quality Analysis");

  const quality = result.image_quality;

  if (quality) {
    addKeyValue(
      "Brightness",
      `${quality.brightness_status} (${quality.brightness_value})`
    );

    addKeyValue(
      "Contrast",
      `${quality.contrast_status} (${quality.contrast_value})`
    );

    addKeyValue(
      "Sharpness",
      `${quality.sharpness_status} (${quality.sharpness_value})`
    );

    addKeyValue(
      "Visibility",
      quality.visibility_status
    );
  }

  addKeyValue(
    "Enhancement Applied",
    result.enhancement_applied ? "Yes" : "No"
  );

  y += 3;


  // =====================================================
  // DECISION SUPPORT
  // =====================================================

  addTitle("Decision Support Recommendation");

  const recommendation = result.recommendation;

  if (recommendation) {
    addKeyValue(
      "Risk Level",
      recommendation.risk_level
    );

    addText(
      recommendation.message
    );
  }


  // =====================================================
  // WEATHER SUITABILITY
  // =====================================================

  addTitle("Weather Suitability");

  const weather = result.weather_suitability;

  if (weather) {

    addKeyValue(
      "Condition",
      weather.condition
    );

    addKeyValue(
      "Suitability Score",
      `${weather.score}/100`
    );

    if (
      weather.operations &&
      weather.operations.length > 0
    ) {
      checkPageSpace(8);

      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.text(
        "Suitable Operations:",
        margin,
        y
      );

      y += 6;

      weather.operations.forEach((operation) => {
        addBullet(operation);
      });
    }

    if (weather.reason) {
      addKeyValue(
        "Reason",
        weather.reason
      );
    }
  }


  // =====================================================
  // SYSTEM WARNINGS
  // =====================================================

  addTitle("System Warnings");

  if (
    result.warnings &&
    result.warnings.length > 0
  ) {
    result.warnings.forEach((warning) => {
      addBullet(warning);
    });
  } else {
    addText("No system warnings.");
  }


  // =====================================================
  // FINAL FOOTER
  // =====================================================

  addFooter();


  // =====================================================
  // SAVE PDF
  // =====================================================

  const filename = result.filename
    ? `Sea_State_Report_${result.filename
        .replace(/\.[^/.]+$/, "")
        .replace(/[^a-zA-Z0-9_-]/g, "_")}.pdf`
    : "Sea_State_Prediction_Report.pdf";

  doc.save(filename);
};

  const renderResult = () => {
    if (!result) return null;

    if (result.error) {
      return (
        <section className="result-card">
          <h2>{selectedModule.title} Result</h2>
          <p className="error-message">{result.error}</p>
        </section>
      );
    }

    return (
      <section className="result-card">
        <div className="result-header">
          <div>
            <p className="section-label">Analysis Result</p>
            <h2>{selectedModule.title}</h2>
          </div>
        </div>

        {mode === "hull" && (
          <>
            <ResultItem label="Prediction" value={result.prediction} />
            <ResultItem
              label="Confidence"
              value={`${(result.confidence * 100).toFixed(2)}%`}
            />

            <ResultItem label="Recommendation" value={result.recommendation} />
            <ResultItem label="Warning" value={result.warning} />

            {result.gradcam && (
              <>
                <button
                  className="secondary-button"
                  onClick={() => setShowGradcam(!showGradcam)}
                >
                  {showGradcam ? "Hide Grad-CAM" : "Show Grad-CAM"}
                </button>

                {showGradcam && (
                  <div className="gradcam-box">
                    <h3>Grad-CAM Visualization</h3>
                    <img
                      src={`data:image/jpeg;base64,${result.gradcam}`}
                      alt="Grad-CAM"
                    />
                  </div>
                )}
              </>
            )}
          </>
        )}

        {mode === "sea" && (
          <>
            <div className="prediction-left">

                <span>Predicted Sea State</span>

                <strong className="prediction-state">
                    {result.predicted_sea_state}
                </strong>

                <div className="prediction-confidence">

                <div className="prediction-score">
                    AI Confidence • {result.confidence}%
                </div>

                    <div
                        className="prediction-label"
                        style={{
                            color: getConfidenceColor(result.confidence)
                        }}
                    >
                        {getConfidenceLabel(result.confidence)}
                    </div>

                </div>

            </div>

            {result.processing_time !== undefined && (
              <div className="mini-card">
                <span>Processing Time</span>
                <strong>{result.processing_time} sec</strong>
              </div>
            )}

            {result.recommendation && (
              <div
              className={`sea-risk-box risk-${result.recommendation.risk_level
                ?.toLowerCase()
                .replace(/\s+/g, "-")}`}
              >
              <div className="risk-header">
                <span>Overall Risk</span>

                <div className="risk-badge">
                  {result.recommendation.risk_level}
                </div>
              </div>

              <div className="risk-meter">
                <div
                  className="risk-fill"
                  style={{
                    width: `${result.weather_suitability?.score || result.confidence}%`,
                  }}
                ></div>
              </div>

              <div className="risk-footer">
                <span>Low</span>
                <span>Moderate</span>
                <span>High</span>
                <span>Extreme</span>
              </div>
            </div>
          )}

          <button
            className="pdf-button"
            onClick={generatePDF}
          >
            📄 Download Prediction Report
          </button>

            {result.probabilities && (
              <div className="probability-list">
                <h3>Class Probabilities</h3>

                {Object.entries(result.probabilities).map(([key, value]) => (
                  <div className="sea-probability-row" key={key}>
                    <span>{key}</span>
                    <div className="sea-probability-bar">
                      <div
                        className="sea-probability-fill"
                        style={{ width: `${value}%` }}
                      ></div>
                    </div>
                    <strong>{value}%</strong>
                  </div>
                ))}
              </div>
            )}

            {result.image_quality && (
              <div className="sea-extra-section">
                <h3>Image Quality Analysis</h3>

                <ResultItem
                  label="Brightness"
                  value={`${result.image_quality.brightness_status} (${result.image_quality.brightness_value})`}
                />
                <ResultItem
                  label="Contrast"
                  value={`${result.image_quality.contrast_status} (${result.image_quality.contrast_value})`}
                />
                <ResultItem
                  label="Sharpness"
                  value={`${result.image_quality.sharpness_status} (${result.image_quality.sharpness_value})`}
                />
                <ResultItem
                  label="Visibility"
                  value={result.image_quality.visibility_status}
                />
                <ResultItem
                  label="Enhancement Applied"
                  value={result.enhancement_applied ? "Yes" : "No"}
                />
              </div>
            )}

            {result.recommendation && (
              <div className="sea-extra-section sea-recommendation">
                <h3>Decision Support Recommendation</h3>
                <p>{result.recommendation.message}</p>
              </div>
            )}

            {result.weather_suitability && (
              <div className="sea-extra-section sea-weather">
                <h3>Weather Suitability</h3>

                <ResultItem
                  label="Weather Condition"
                  value={result.weather_suitability.condition}
                />

                <div className="probability-list">
                  <h4>Suitability Score</h4>

                  <div className="sea-probability-row">
                    <span>Score</span>

                    <div className="sea-probability-bar">
                      <div
                        className="sea-probability-fill"
                        style={{
                          width: `${result.weather_suitability.score}%`,
                        }}
                      ></div>
                    </div>

                    <strong>{result.weather_suitability.score}/100</strong>
                  </div>
                </div>

                <div className="weather-operations">
                  <h4>Suitable Operations</h4>

                  <div className="weather-operation-badges">
                    {result.weather_suitability.operations.map((operation, index) => (
                      <div className="operation-badge" key={index}>
                        <span className="badge-dot"></span>
                        <span>{operation}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="sea-reason-box">
                    <h4>Reason</h4>
                    <p>{result.weather_suitability.reason}</p>
                </div>
              </div>
            )}

            {result.warnings && (
              <div className="sea-extra-section sea-warning">
                <h3>System Warnings</h3>
                {result.warnings.map((warning, index) => (
                  <p key={index}>⚠️ {warning}</p>
                ))}
              </div>
            )}

            <div className="sea-history-section">
              <div className="sea-history-header">
                <h3>Sea State Prediction History</h3>
                <div>
                  <button className="secondary-button" onClick={fetchSeaHistory}>
                    Refresh
                  </button>
                  <button
                    className="secondary-button sea-clear-button"
                    onClick={clearSeaHistory}
                  >
                    Clear
                  </button>
                </div>
              </div>

              {seaHistory.length === 0 ? (
                <p className="empty-text">
                  No sea-state prediction history available.
                </p>
              ) : (
                <div className="sea-history-grid">
                  {seaHistory.slice(0, 6).map((item, index) => (
                    <div className="sea-history-card" key={index}>
                      <p>
                        <strong>Time:</strong> {item.timestamp}
                      </p>
                      <p>
                        <strong>File:</strong> {item.filename}
                      </p>
                      <p>
                        <strong>Prediction:</strong> {item.predicted_sea_state}
                      </p>
                      <p>
                        <strong>Confidence:</strong> {item.confidence}%
                      </p>
                      <p>
                        <strong>Risk:</strong>{" "}
                        {item.recommendation?.risk_level}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        )}

        {mode === "boat" && (
          <>
            {result.results && result.results.length > 0 ? (
              <>
                <ResultItem label="Detected Objects" value={result.count} />

                <div className="detection-grid">
                  {result.results.map((detection, index) => (
                    <div className="mini-card" key={index}>
                      <span>{detection.label}</span>
                      <strong>{detection.confidence}%</strong>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="empty-text">No boats detected in this image.</p>
            )}
          </>
        )}

        {mode === "radar" && (
          <>
            <div className="final-prediction">
              <span>Final Prediction</span>
              <strong>{result.final_prediction}</strong>
            </div>

            <ResultItem label="Decision Status" value={result.decision_status} />

            <div className="model-output-grid">
              <div className="mini-card">
                <span>YOLO Model</span>
                <p>{result.yolo_prediction}</p>
                <strong>{result.yolo_confidence}%</strong>
              </div>

              <div className="mini-card">
                <span>CNN Verification</span>
                <p>{result.cnn_prediction}</p>
                <strong>{result.cnn_confidence}%</strong>
              </div>
            </div>
          </>
        )}
      </section>
    );
  };

  return (
    <div className="app">
      <div className="ocean-bg"></div>

      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">OQ</div>
          <div>
            <h1>OceanIQ</h1>
            <p>Marine AI Inspection System</p>
          </div>
        </div>

        <div className="status-pill">
          <span></span>
          AI System Online
        </div>
      </header>

      <section className="hero">
        <p className="eyebrow">Research Project R26-IT-003</p>
        <h2>Intelligent Ocean Monitoring & Marine Inspection</h2>
        <p>
          OceanIQ integrates multiple AI modules for hull defect detection, sea
          state classification, boat detection, and radar object classification.
        </p>
      </section>

      <main className="dashboard">
        <section className="module-panel">
          <p className="section-label">Select AI Module</p>

          <div className="module-grid">
            {Object.entries(modules).map(([key, module]) => (
              <button
                key={key}
                className={`module-card ${mode === key ? "active" : ""}`}
                onClick={() => handleModeChange(key)}
              >
                <span>{module.title}</span>
                <small>{module.description}</small>
              </button>
            ))}
          </div>
        </section>

        <section className="upload-panel">
          <div className="panel-heading">
            <p className="section-label">Input Image</p>
            <h2>{selectedModule.title}</h2>
            <p>{selectedModule.description}</p>
          </div>

          <label className="upload-box">
            <input type="file" accept="image/*" onChange={handleFileChange} />
            <div>
              <strong>{file ? file.name : "Upload image file"}</strong>
              <span>PNG, JPG, JPEG, BMP supported</span>
            </div>
          </label>

          {mode === "sea" && (
            <label className="enhancement-toggle">
              <input
                type="checkbox"
                checked={applyEnhancement}
                onChange={(e) => setApplyEnhancement(e.target.checked)}
              />
              <span>Apply image enhancement before prediction</span>
            </label>
          )}

          <button
            className="primary-button"
            onClick={handleUpload}
            disabled={loading}
          >
            {loading ? "Analyzing..." : "Run AI Prediction"}
          </button>
        </section>

        <section className="preview-panel">
          <p className="section-label">Preview</p>

          {preview ? (
            <img src={preview} alt="Uploaded preview" className="preview-image" />
          ) : (
            <div className="empty-preview">
              <strong>No image selected</strong>
              <span>Upload an image to preview it here</span>
            </div>
          )}
        </section>

        {renderResult()}
      </main>
    </div>
  );
}

function getConfidenceLabel(confidence) {

  if (confidence >= 90)
    return "Excellent Prediction Confidence";

  if (confidence >= 75)
    return "High Prediction Confidence";

  if (confidence >= 60)
    return "Moderate Prediction Confidence";

  if (confidence >= 40)
    return "Low Prediction Confidence";

  return "Very Low Prediction Confidence";
}

function getConfidenceColor(confidence) {

  if (confidence >= 90)
    return "#22c55e";

  if (confidence >= 75)
    return "#3b82f6";

  if (confidence >= 60)
    return "#eab308";

  if (confidence >= 40)
    return "#f97316";

  return "#ef4444";
}

function ResultItem({ label, value }) {
  return (
    <div className="result-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;