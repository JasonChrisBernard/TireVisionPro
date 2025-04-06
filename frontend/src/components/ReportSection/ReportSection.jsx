import React from "react";
import "./ReportSection.css";

const ReportSection = ({ defectDetails }) => {
  return (
    <div className="report-container">
      <h2>AI-Generated Report</h2>

      {defectDetails ? (
        <>


          <h3>Grad-CAM Explanation</h3>
          <p className="report-text">{defectDetails.gradcam}</p>

          <h3>LIME Explanation</h3>
          <p className="report-text">{defectDetails.lime}</p>

          <h3>SHAP Explanation</h3>
          <p className="report-text">{defectDetails.shap}</p>
        </>
      ) : (
        <p className="no-report">Waiting for defect analysis...</p>
      )}
    </div>
  );
};

export default ReportSection;
