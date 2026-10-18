import { useMemo } from "react";
import { Crosshair, Download, Film, Ship } from "lucide-react";
import PortCountryLookup from "./PortCountryLookup";
import {
  formatVesselConfidence,
  getVesselDisplayConfidence,
} from "../utils/vesselDetectionUtils";

// Display image and video analysis results and the editable vessel record.
export default function VesselDetectionResults({
  activeModule,
  result,
  mod,
  vesselVideoFile,
  vesselInfoPreview,
  cleanVesselDetections,
  preview,
  downloadBoatAnalysis,
  vesselDraft,
  setVesselDraft,
  setVesselDraftDirty,
  shipAnalysisDone,
  vesselInfoAnalysisDone,
  vesselRecordId,
  vesselDraftDirty,
  loading,
  canWrite,
  saveVesselRecordEdits,
  submitVesselRecord,
  DataRow,
}) {
  const vesselInfoDetections = useMemo(
    () => (result.vessel_info_detections || []).map((item) => ({
      ...item,
      displayConfidence: getVesselDisplayConfidence(item.confidence),
    })),
    [result.vessel_info_detections],
  );

  return (
    <>
        {activeModule === "boat" && (
          <div className="result-body">
            {result.video_analysis_performed && (
              <section className="vessel-info-panel vessel-video-results" style={{ borderColor: mod.colorMid }}>
                <div className="vessel-info-heading">
                  <h3>VIDEO VESSEL ANALYSIS · {result.video || vesselVideoFile?.name}</h3>
                  <span className="vessel-model-chip">VIDEO · NOT SAVED</span>
                </div>
                <p className="vessel-video-note">
                  Video is analyzed locally, one frame per second. Neither the video nor its frames are sent to Cloudinary or saved in the database.
                </p>
                <div className="boat-analysis-summary" style={{ borderColor: mod.colorMid }}>
                  <div>
                    <span>VIDEO LENGTH</span>
                    <strong>{result.video_duration_seconds}s</strong>
                  </div>
                  <div>
                    <span>FRAMES ANALYZED</span>
                    <strong>{result.frames_analyzed}</strong>
                  </div>
                  <div>
                    <span>VESSEL OBJECTS</span>
                    <strong>{result.count}</strong>
                  </div>
                </div>
                {result.vessel_classifications?.length ? (
                  <div className="vessel-detected-types">
                    {result.vessel_classifications.map((type) => (
                      <span className="vessel-detected-type" key={type}>
                        <Ship size={19} strokeWidth={1.7} />
                        {type}
                      </span>
                    ))}
                  </div>
                ) : (
                  <div className="empty-result">
                    <Ship size={34} opacity={0.25} />
                    <p>No vessels were detected in the sampled frames.</p>
                  </div>
                )}
                {result.frames?.length > 0 && (
                  <div className="vessel-video-frame-list">
                    {result.frames.map((frame) => (
                      <div className="vessel-video-frame" key={frame.time_seconds}>
                        <span className="vessel-video-time">
                          <Film size={13} /> {frame.time_seconds}s
                        </span>
                        <span>
                          {frame.detections?.length
                            ? [...new Set(frame.detections.map((detection) => detection.label))].join(" or ")
                            : "No vessel detected"}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            )}
            {result.vessel_info && (
              <section className="vessel-info-panel" style={{ borderColor: mod.colorMid }}>
                <div className="vessel-info-heading">
                  <h3>VESSEL IDENTITY · {result.vessel_info_image || "BACKSIDE IMAGE"}</h3>
                  {result.vessel_info_performed && <span className="vessel-model-chip">OCR MODEL</span>}
                </div>
                {vesselInfoPreview && (
                  <img
                    className="vessel-result-thumb"
                    src={vesselInfoPreview}
                    alt="Uploaded vessel backside"
                  />
                )}
                <div className="vessel-info-grid">
                  {[
                    ["VESSEL NAME", result.vessel_info.vessel_name],
                    ["PORT", result.vessel_info.port],
                    ["PORT COUNTRY", result.vessel_info.port_country],
                    ["PORT TYPE", result.vessel_info.port_type],
                    ["OCR PORT TEXT", result.vessel_info.port_ocr],
                    ["PORT MATCH", result.vessel_info.port_match_status],
                    ["IMO NUMBER", result.vessel_info.imo_number],
                  ].map(([label, value]) => (
                    <div className="vessel-info-field" key={label}>
                      <span>{label}</span>
                      <strong>{value || "Not read"}</strong>
                    </div>
                  ))}
                </div>
                {vesselInfoDetections.length > 0 && (
                  <div className="vessel-ocr-detections">
                    {vesselInfoDetections.map((item, index) => (
                      <div className="vessel-ocr-chip" key={`${item.field}-${index}`}>
                        <span>
                          {item.field.replaceAll("_", " ")} · {item.matched_text || item.text || "Text not read"}
                        </span>
                        {formatVesselConfidence(item.displayConfidence) && (
                          <span>Detector confidence: {formatVesselConfidence(item.displayConfidence)}</span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </section>
            )}
            {!result.video_analysis_performed && result.vessel_detection_performed && cleanVesselDetections.length > 0 ? (
              <>
                <section className="vessel-info-panel vessel-type-heading" style={{ borderColor: mod.colorMid }}>
                  <div className="vessel-info-heading">
                    <h3>VESSEL TYPE MODEL · {result.image || "UPLOADED IMAGE"}</h3>
                    <span className="vessel-model-chip">SHIP-DETECTION</span>
                  </div>
                  {preview && (
                    <img className="vessel-result-thumb" src={preview} alt="Analyzed vessel image" />
                  )}
                  <div className="vessel-detected-types">
                    {[...new Set(cleanVesselDetections.map((item) => item.label))].map((type) => (
                      <span className="vessel-detected-type" key={type}>
                        <Ship size={19} strokeWidth={1.7} />
                        {type}
                      </span>
                    ))}
                  </div>
                </section>
                <div className="boat-analysis-summary" style={{ borderColor: mod.colorMid }}>
                  <div>
                    <span>OBJECTS FOUND</span>
                    <strong>{cleanVesselDetections.length}</strong>
                  </div>
                  <div>
                    <span>VESSEL TYPES</span>
                    <strong>{new Set(cleanVesselDetections.map((item) => item.label)).size}</strong>
                  </div>
                  <div>
                    <span>TOTAL OBJECTS</span>
                    <strong>{result.count ?? cleanVesselDetections.length}</strong>
                  </div>
                </div>
                <div className="result-primary-row">
                  <div className="result-pred-block" style={{ borderColor: mod.colorMid }}>
                    <p className="rpb-eye">VESSELS DETECTED</p>
                    <p className="rpb-val" style={{ color: mod.color }}>{cleanVesselDetections.length}</p>
                    <p className="rpb-sub">Objects identified in frame</p>
                  </div>
                  <div className="vessel-count-badge"
                    style={{ color: mod.color, borderColor: mod.colorMid, background: mod.colorDim }}>
                    <Ship size={26} strokeWidth={1.5} />
                    <span>{cleanVesselDetections.length} FOUND</span>
                  </div>
                </div>

                <div className="vessel-list">
                  {cleanVesselDetections.map((det, i) => (
                    <div key={i} className="vessel-item" style={{ borderColor: mod.colorMid }}>
                      <div className="vessel-index"
                        style={{ background: mod.colorDim, color: mod.color }}>
                        <Crosshair size={12} /> {i + 1}
                      </div>
                      <span className="vessel-kind-chip">{det.vessel_kind || "Vessel"}</span>
                      <span className="vessel-label">{det.label}</span>
                      <span className="vessel-confidence-score">
                        Confidence: {formatVesselConfidence(det.confidence) || "N/A"}
                      </span>
                    </div>
                  ))}
                </div>
              
                <div className="boat-result-details" style={{ borderColor: mod.colorMid }}>
                  <DataRow label="ESTIMATED SIZE" value={result.estimated_size || "Medium Vessel"} color={mod.color} />
                  <DataRow label="STATUS" value={result.status || "Detected"} color="#00ffb3" />
                </div>

                <button className="boat-download-btn" onClick={downloadBoatAnalysis}>
                  <Download size={14} /> DOWNLOAD ANNOTATED IMAGE
                </button>
              </>
            ) : !result.video_analysis_performed && result.vessel_detection_performed ? (
              <div className="empty-result">
                <Ship size={38} opacity={0.2} />
                <p>No vessels detected in this frame.</p>
              </div>
            ) : null}

            {!result.video_analysis_performed && !vesselVideoFile && (shipAnalysisDone || vesselInfoAnalysisDone) && (
              <section className="vessel-review-panel" style={{ borderColor: mod.colorMid }}>
                <div className="vessel-info-heading">
                  <h3>FINAL VESSEL RECORD · REVIEW & EDIT</h3>
                  <span className="vessel-model-chip">DRAFT</span>
                </div>
                <p className="vessel-review-hint">
                  Check or correct the detected values before submitting. Images are sent to Cloudinary only when you submit this record.
                </p>
                <div className="vessel-edit-grid">
                  <label>
                    <span>VESSEL NAME</span>
                    <input
                      value={vesselDraft.vessel_name}
                      onChange={(event) => {
                        setVesselDraft((current) => ({ ...current, vessel_name: event.target.value }));
                        setVesselDraftDirty(true);
                      }}
                    />
                  </label>
                  <label>
                    <span>PORT</span>
                    <input
                      value={vesselDraft.port}
                      onChange={(event) => {
                        setVesselDraft((current) => ({
                          ...current,
                          port: event.target.value,
                          port_country: "",
                          port_country_code: "",
                          port_type: "Foreign",
                        }));
                        setVesselDraftDirty(true);
                      }}
                    />
                  </label>
                  <PortCountryLookup
                    port={vesselDraft.port}
                    country={vesselDraft.port_country}
                    onCountryFound={(portCountry) => {
                      setVesselDraft((current) => ({ ...current, ...portCountry }));
                      setVesselDraftDirty(true);
                    }}
                  />
                  <label>
                    <span>PORT TYPE</span>
                    <select
                      value={vesselDraft.port_type || "Foreign"}
                      onChange={(event) => {
                        setVesselDraft((current) => ({
                          ...current,
                          port_type: event.target.value,
                        }));
                        setVesselDraftDirty(true);
                      }}
                    >
                      <option value="Local">Local</option>
                      <option value="Foreign">Foreign</option>
                    </select>
                  </label>
                  <label>
                    <span>IMO NUMBER</span>
                    <input
                      value={vesselDraft.imo_number}
                      onChange={(event) => {
                        setVesselDraft((current) => ({ ...current, imo_number: event.target.value }));
                        setVesselDraftDirty(true);
                      }}
                      placeholder="IMO 9074729"
                    />
                  </label>
                  <label>
                    <span>DETECTED VESSEL TYPES</span>
                    <input
                      value={vesselDraft.vessel_classifications}
                      onChange={(event) => {
                        setVesselDraft((current) => ({ ...current, vessel_classifications: event.target.value }));
                        setVesselDraftDirty(true);
                      }}
                      placeholder="Merchant Ship or Fishing Boat"
                      disabled={!shipAnalysisDone}
                    />
                  </label>
                  <label className="vessel-notes-field">
                    <span>NOTES</span>
                    <textarea
                      value={vesselDraft.notes}
                      onChange={(event) => {
                        setVesselDraft((current) => ({ ...current, notes: event.target.value }));
                        setVesselDraftDirty(true);
                      }}
                      rows={3}
                    />
                  </label>
                </div>
                {result.__saveError && <p className="vessel-save-message vessel-save-message--error">{result.__saveError}</p>}
                {result.__saveSuccess && <p className="vessel-save-message">{result.__saveSuccess}</p>}
                {vesselRecordId ? (
                  <button
                    type="button"
                    className="vessel-submit-record"
                    onClick={saveVesselRecordEdits}
                    disabled={!vesselDraftDirty || loading || !canWrite}
                  >
                    {loading ? "SAVING…" : "UPDATE SAVED RECORD"}
                  </button>
                ) : (
                  <button
                    type="button"
                    className="vessel-submit-record"
                    onClick={submitVesselRecord}
                    disabled={loading || !canWrite}
                  >
                    {loading ? "UPLOADING IMAGES & SAVING…" : "SUBMIT IMAGES & DETAILS"}
                  </button>
                )}
              </section>
            )}

          </div>
        )}
    </>
  );
}
