import { useState } from "react";
import axios from "axios";

function App() {
  const [mode, setMode] = useState("hull");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showGradcam, setShowGradcam] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setShowGradcam(false);
  };

  const handleModeChange = (e) => {
    setMode(e.target.value);
    setResult(null);
    setShowGradcam(false);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select an image");

    const endpoint =
      mode === "hull"
        ? "http://127.0.0.1:8000/predict-hull-defect"
        : mode === "sea"
        ? "http://127.0.0.1:8000/predict-sea-state"
        : "http://127.0.0.1:8000/predict-boat-detection";

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      const res = await axios.post(endpoint, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(res.data);
      setShowGradcam(false);
    } catch (err) {
      console.log(err);
      alert("Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Marine AI Inspection System</h1>

      <select value={mode} onChange={handleModeChange}>
        <option value="hull">Hull Defect Detection</option>
        <option value="sea">Sea State Classification</option>
        <option value="boat">Boat Detection</option>
      </select>

      <br /><br />

      <input type="file" onChange={handleFileChange} />

      <br /><br />

      <button onClick={handleUpload}>
        {loading ? "Predicting..." : "Predict"}
      </button>

      {file && (
        <div style={{ marginTop: "20px" }}>
          <h3>Uploaded Image</h3>
          <img src={URL.createObjectURL(file)} width="300" alt="uploaded" />
        </div>
      )}

      {result && mode === "hull" && (
        <div style={{ marginTop: "20px", display: "inline-block", textAlign: "center" }}>
          <h2>Hull Defect Result</h2>
          <p><b>Prediction:</b> {result.prediction}</p>
          <p><b>Confidence:</b> {(result.confidence * 100).toFixed(2)}%</p>
          <p><b>Recommendation:</b> {result.recommendation}</p>
          <p><b>Warning:</b> {result.warning}</p>

          <button onClick={() => setShowGradcam(!showGradcam)}>
            {showGradcam ? "Hide Grad-CAM" : "Show Grad-CAM"}
          </button>

          {showGradcam && (
            <div style={{ marginTop: "20px" }}>
              <h3>Grad-CAM Visualization</h3>
              <img
                src={`data:image/jpeg;base64,${result.gradcam}`}
                width="300"
                alt="gradcam"
              />
            </div>
          )}
        </div>
      )}

      {result && mode === "sea" && (
        <div style={{ marginTop: "20px", display: "inline-block", textAlign: "center" }}>
          <h2>Sea State Result</h2>
          <p><b>Prediction:</b> {result.predicted_sea_state}</p>
          <p><b>Confidence:</b> {result.confidence}%</p>

          <h3>Probabilities</h3>
          {Object.entries(result.probabilities).map(([key, value]) => (
            <p key={key}>
              <b>{key}:</b> {value}%
            </p>
          ))}
        </div>
      )}

      {result && mode === "boat" && (
        <div style={{ marginTop: "20px", display: "inline-block", textAlign: "center" }}>
          <h2>Boat Detection Result</h2>
          {result.results && result.results.length > 0 ? (
            <div>
              <p><b>Found {result.count} object(s):</b></p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'center' }}>
                {result.results.map((detection, idx) => (
                  <div key={idx} style={{
                    background: '#000000',
                    padding: '8px 12px',
                    borderRadius: '4px',
                    border: '1px solid #b2ebf2',
                    margin: '30px',
                  }}>
                    {detection.label}: {detection.confidence}%
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p>No boats detected in this image.</p>
          )}
        </div>
      )}
    </div>
  );
}

export default App;