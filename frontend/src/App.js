import React, { useState } from "react";
import ImageUpload from "./components/ImageUpload/ImageUpload";
import DefectAnalysis from "./components/DefectAnalysis/DefectAnalysis";
import ReportSection from "./components/ReportSection/ReportSection";
import "./styles.css";

const App = () => {
  const [defectDetails, setDefectDetails] = useState("");
  const [gradcamUrl, setGradcamUrl] = useState(null);
  const [limeUrl, setLimeUrl] = useState(null);
  const [shapUrl, setShapUrl] = useState(null); // ✅ New state


  return (
    <div className="app-container">
      <h1> ----𖥕 TIRE VISION PRO 𖥕----</h1>

      <ImageUpload 
        setDefectDetails={setDefectDetails} 
        setGradcamUrl={setGradcamUrl} 
        setLimeUrl={setLimeUrl} 
        setShapUrl={setShapUrl} // ✅ Pass it
      />

      <DefectAnalysis gradcamUrl={gradcamUrl} limeUrl={limeUrl} shapUrl={shapUrl} />
      <ReportSection defectDetails={defectDetails} />
    </div>
  );
};

export default App;
