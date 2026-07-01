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

  const selectedModule = modules[mode];

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
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      const response = await axios.post(selectedModule.endpoint, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(response.data);
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
            <ResultItem label="Prediction" value={result.predicted_sea_state} />
            <ResultItem label="Confidence" value={`${result.confidence}%`} />

            {result.probabilities && (
              <div className="probability-list">
                <h3>Class Probabilities</h3>
                {Object.entries(result.probabilities).map(([key, value]) => (
                  <ResultItem key={key} label={key} value={`${value}%`} />
                ))}
              </div>
            )}
          </>
        )}

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