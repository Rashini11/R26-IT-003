import { Loader, ScanEye, Upload, X } from "lucide-react";

// Show the actions for each vessel analysis step.
export default function VesselDetectionStageActions({
  color,
  file,
  vesselVideoFile,
  loading,
  canWrite,
  shipAnalysisDone,
  videoAnalysisDone,
  vesselInfoFile,
  vesselInfoPreview,
  vesselInfoAnalysisDone,
  onAnalyzeVessel,
  onAnalyzeVideo,
  onAnalyzeInfo,
  onInfoFileChange,
  onRemoveImage,
  onRemoveVideo,
  onRemoveInfoImage,
}) {
  return (
    <>
      <div className="vessel-stage-actions">
        <button
          type="button"
          className="run-btn vessel-stage-btn"
          style={{ "--mc": color }}
          onClick={vesselVideoFile ? onAnalyzeVideo : onAnalyzeVessel}
          disabled={(!file && !vesselVideoFile) || loading || !canWrite}
        >
          {loading && (file || vesselVideoFile)
            ? <Loader size={14} className="spin" />
            : <ScanEye size={14} />}
          {vesselVideoFile
            ? videoAnalysisDone ? "RE-ANALYZE VIDEO" : "1 · ANALYZE VIDEO"
            : shipAnalysisDone ? "RE-ANALYZE VESSEL TYPE" : "1 · ANALYZE VESSEL TYPE"}
        </button>
        {shipAnalysisDone && <span className="vessel-stage-complete">TYPE ANALYSIS READY</span>}
        {videoAnalysisDone && <span className="vessel-stage-complete">VIDEO ANALYSIS READY · NOT SAVED</span>}
        {vesselVideoFile && (
          <button type="button" className="vessel-clear-input" onClick={onRemoveVideo}>
            <X size={13} /> REMOVE VIDEO
          </button>
        )}
        {file && (
          <button type="button" className="vessel-clear-input" onClick={onRemoveImage}>
            <X size={13} /> REMOVE VESSEL IMAGE
          </button>
        )}
      </div>
      {!vesselVideoFile && (
        <div className="vessel-info-upload">
          <p className="vessel-info-upload-title">BACKSIDE PHOTO · NAME / PORT / IMO · OPTIONAL</p>
          <label className={`drop-zone ${vesselInfoFile ? "dz--filled" : ""} ${!canWrite ? "dz--readonly" : ""}`}>
            <input
              type="file"
              accept="image/*"
              onClick={(event) => { event.currentTarget.value = ""; }}
              onChange={onInfoFileChange}
              disabled={!canWrite}
            />
            {vesselInfoFile ? (
              <div className="dz-loaded vessel-info-upload-loaded">
                {vesselInfoPreview && <img src={vesselInfoPreview} alt="Vessel backside preview" />}
                <p className="dz-fname">{vesselInfoFile.name}</p>
                <p className="dz-change">Click to change backside photo</p>
              </div>
            ) : (
              <div className="dz-empty">
                <div className="dz-upload-icon"><Upload size={20} style={{ color }} /></div>
                <p className="dz-main">Upload clear backside image</p>
                <p className="dz-sub">Used separately for vessel text detection and OCR</p>
                <p className="dz-fmt">PNG · JPG · JPEG · BMP · WEBP</p>
              </div>
            )}
          </label>
          <div className="vessel-stage-actions">
            <button
              type="button"
              className="run-btn vessel-stage-btn"
              style={{ "--mc": color }}
              onClick={onAnalyzeInfo}
              disabled={!vesselInfoFile || loading || !canWrite}
            >
              {loading && vesselInfoFile
                ? <Loader size={14} className="spin" />
                : <ScanEye size={14} />}
              {vesselInfoAnalysisDone ? "RE-ANALYZE VESSEL DETAILS" : "2 · READ NAME / PORT / IMO"}
            </button>
            {vesselInfoAnalysisDone && <span className="vessel-stage-complete">DETAIL ANALYSIS READY</span>}
            {vesselInfoFile && (
              <button type="button" className="vessel-clear-input" onClick={onRemoveInfoImage}>
                <X size={13} /> REMOVE BACKSIDE PHOTO
              </button>
            )}
          </div>
        </div>
      )}
    </>
  );
}
