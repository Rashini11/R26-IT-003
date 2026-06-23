import { useState } from "react";
import axios from "axios";
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
            <div className="final-prediction sea-final">
              <span>Predicted Sea State</span>
              <strong>{result.predicted_sea_state}</strong>
            </div>

            <ResultItem label="Confidence" value={`${result.confidence}%`} />

            {result.recommendation && (
              <div
                className={`sea-risk-box risk-${result.recommendation.risk_level
                  ?.toLowerCase()
                  .replace(" ", "-")}`}
              >
                <span>Risk Level</span>
                <strong>{result.recommendation.risk_level}</strong>
              </div>
            )}

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

function ResultItem({ label, value }) {
  return (
    <div className="result-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;