import React from "react";
import "./DefectAnalysis.css";

const DefectAnalysis = ({ gradcamUrl, limeUrl, shapUrl }) => {
  return (
    <div className="analysis-container">
      <h2>Defect Analysis</h2>

      {/* ✅ Display Grad-CAM Image Correctly */}
      {gradcamUrl ? (
        <>
          <h3>Grad-CAM Heatmap</h3>
          <img src={`data:image/jpeg;base64,${gradcamUrl}`} alt="Grad-CAM Heatmap" className="heatmap-image" />
        </>
      ) : (
        <p className="no-analysis">No Grad-CAM analysis available yet.</p>
      )}

      {/* ✅ Display LIME Explanation Correctly */}
      {limeUrl ? (
        <>
          <h3>LIME Explanation</h3>
          <img src={`data:image/jpeg;base64,${limeUrl}`} alt="LIME Explanation" className="heatmap-image" />
        </>
      ) : (
        <p className="no-analysis">No LIME explanation available yet.</p>
      )}
      
      {shapUrl ? (
        <>
          <h3>SHAP Explanation</h3>
          <img src={`data:image/jpeg;base64,${shapUrl}`} alt="SHAP Explanation" className="heatmap-image" />
        </>
      ) : (
        <p className="no-analysis">No SHAP explanation available yet.</p>
      )}


    </div>
  );
};

export default DefectAnalysis;
