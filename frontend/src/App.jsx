import { useState } from "react";
import axios from "axios";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleUpload = async () => {
    if (!image) {
      alert("Select an image first");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData
      );

      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Prediction failed");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "40px" }}>
      <h1>🌊 Sea State Classification</h1>

      <input type="file" onChange={handleChange} />

      {preview && (
        <div>
          <img
            src={preview}
            alt="preview"
            style={{ width: "300px", marginTop: "20px" }}
          />
        </div>
      )}

      <br />
      <button onClick={handleUpload} style={{ marginTop: "20px" }}>
        Predict
      </button>

      {result && (
        <div style={{ marginTop: "30px" }}>
          <h2>Prediction: {result.predicted_sea_state}</h2>
          <h3>Confidence: {result.confidence}%</h3>

          <h4>Probabilities:</h4>
          {Object.entries(result.probabilities).map(([key, value]) => (
            <p key={key}>
              {key}: {value}%
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;