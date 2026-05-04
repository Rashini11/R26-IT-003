import { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showGradcam, setShowGradcam] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null); 
    setShowGradcam(false); // reset Grad-CAM visibility when new image selected
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select an image");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      const res = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      setResult(res.data);
    } catch (err) {
      console.log(err);
      alert("Backend not connected");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Hull Defect Detection AI</h1>

      <input type="file" onChange={handleFileChange} />

      <br /><br />

      <button onClick={handleUpload}>
        {loading ? "Predicting..." : "Predict"}
      </button>

      {/* 🔹 Show uploaded image */}
      {file && (
        <div style={{ marginTop: "20px" }}>
          <h3>Uploaded Image</h3>
          <img src={URL.createObjectURL(file)} width="300" />
        </div>
      )}

      {/* 🔹 Show results */}
      {result && (
        <div style={{ marginTop: "20px" }}>
          <h2>Result</h2>
          <p><b>Prediction:</b> {result.prediction}</p>
          <p><b>Confidence:</b> {result.confidence}</p>
          <p><b>Recommendation:</b> {result.recommendation}</p>
          <p><b>Warning:</b> {result.warning}</p>

          {/* 🔥 Grad-CAM Image */}
          <button onClick={() => setShowGradcam(!showGradcam)}>
            {showGradcam ? "Hide Grad-CAM" : "Show Grad-CAM"}
          </button>

          {showGradcam && (
            <div style={{ marginTop: "20px" }}>
              <h3>Grad-CAM Visualization</h3>
              <img
                src={`data:image/jpeg;base64,${result.gradcam}`}
                width="300"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;