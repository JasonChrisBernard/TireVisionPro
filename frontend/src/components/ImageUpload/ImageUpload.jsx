import React, { useState } from "react";
import axios from "axios";
import "./ImageUpload.css";

const ImageUpload = ({ setDefectDetails, setGradcamUrl, setLimeUrl, setShapUrl  }) => {
  const [image, setImage] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleImageChange = async (event) => {
    const file = event.target.files[0];
    if (!file) {
      setError("❌ Please select a valid image file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setError("");
    setLoading(true);

    try {
      // 🔥 Send image to Flask backend
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (response.data.error) {
        throw new Error(response.data.error);
      }

      console.log("✅ Server Response:", response.data);
      setImage(URL.createObjectURL(file)); // ✅ Show uploaded image

      // ✅ Set image URLs
      setGradcamUrl(response.data.gradcam || null);
      setLimeUrl(response.data.lime || null);
      setShapUrl(response.data.shap || null);  // ✅ Add this line

      // ✅ Set explanation text properly
      setDefectDetails({
        gradcam: response.data.explanation || "No Grad-CAM explanation available.",
        lime: response.data.lime_explanation || "No LIME explanation available.",
        shap: response.data.shap_explanation || "No SHAP explanation available.", // ✅ Add this line
        defect_status: response.data.defect_status,
        probability: response.data.probability
      });
      

    } catch (err) {
      console.error("❌ Upload Error:", err);
      setError(`❌ Error: ${err.response?.data?.error || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <h2>Upload Tire Image</h2>
      <input type="file" accept="image/*" onChange={handleImageChange} />

      {loading && <p>📡 Uploading and Analyzing...</p>}

      {image && !loading && (
        <div className="image-preview">
          <img src={image} alt="Uploaded Tire" className="preview-image" />
        </div>
      )}

      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default ImageUpload;
