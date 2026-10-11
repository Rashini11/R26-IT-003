import { useMemo, useRef, useState } from "react";
import axios from "axios";
import {
  AlertTriangle,
  CheckCircle2,
  ClipboardCheck,
  Loader,
  SearchCheck,
} from "lucide-react";
import { API_BASE_URL } from "./context/AuthContext";
import "./VesselInspectionPanel.css";

const VESSEL_FIELDS = [
  ["vessel_name", "VESSEL NAME"],
  ["vessel_type", "VESSEL TYPE"],
  ["imo_number", "IMO NUMBER"],
  ["mmsi_number", "MMSI NUMBER"],
  ["flag_country", "FLAG / COUNTRY"],
  ["destination", "DESTINATION"],
  ["reported_at", "REPORTED DATE / TIME"],
  ["speed", "SPEED"],
  ["heading", "HEADING"],
];

const emptyFields = () =>
  Object.fromEntries(VESSEL_FIELDS.map(([key]) => [key, ""]));

const localDateTime = () => {
  const now = new Date();
  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 16);
};

function extractedImo(prediction) {
  const readings = prediction?.id_text?.readings || [];
  for (const reading of readings) {
    const match = String(reading?.text || "").match(/\b(?:IMO\s*)?(\d{7})\b/i);
    if (match) return match[1];
  }
  return "";
}

function classificationFields(prediction) {
  const classification = [
    ...(prediction?.results || []).map((item) => item?.label || ""),
    prediction?.vessel_origin || "",
  ].join(" ");
  const vesselType = /\bship\b/i.test(classification)
    ? "Ship"
    : /\bboat\b/i.test(classification)
      ? "Boat"
      : "";
  return {
    vessel_type: vesselType,
    flag_country: /\blocal\b/i.test(classification)
      ? "Local"
      : /\bforeign\b/i.test(classification)
        ? "Foreign"
        : "",
  };
}

function detailMessage(error) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join(" · ");
  return "Unable to complete the vessel inspection request.";
}

export default function VesselInspectionPanel({
  file,
  isVideo,
  prediction,
  canWrite,
}) {
  const [rawAis, setRawAis] = useState("");
  const [aisFields, setAisFields] = useState(emptyFields);
  const [verifiedState, setVerifiedState] = useState(null);
  const [matchState, setMatchState] = useState(null);
  const [manualCandidates, setManualCandidates] = useState([]);
  const [candidateSearch, setCandidateSearch] = useState("");
  const [candidateSearchBusy, setCandidateSearchBusy] = useState(false);
  const [candidateSearchError, setCandidateSearchError] = useState("");
  const [manualConfirm, setManualConfirm] = useState(false);
  const [selectedVesselId, setSelectedVesselId] = useState("");
  const [inspectionAt, setInspectionAt] = useState(localDateTime);
  const [note, setNote] = useState("");
  const [threatStatus, setThreatStatus] = useState("");
  const [busy, setBusy] = useState("");
  const [messageState, setMessageState] = useState(null);
  const [savedAnalysisKey, setSavedAnalysisKey] = useState("");
  const submissionKey = useRef({ analysisKey: "", value: "" });

  const analysisKey = file
    ? `${file.name}:${file.lastModified}:${file.size}:${prediction?.timestamp || (prediction ? "complete" : "")}`
    : "";
  const verified =
    verifiedState?.analysisKey === analysisKey
      ? verifiedState.fields
      : emptyFields();
  const match =
    matchState?.analysisKey === analysisKey ? matchState.data : null;
  const message =
    messageState?.analysisKey === analysisKey ? messageState.message : null;
  const savedInspection = Boolean(analysisKey && savedAnalysisKey === analysisKey);
  const isImageAnalysis = Boolean(file && !isVideo && !prediction?.__error);
  const candidateVessels = useMemo(
    () =>
      [match?.vessel, match?.candidate_vessel].filter(
        (vessel, index, values) =>
          vessel && values.findIndex((item) => item?.id === vessel.id) === index,
      ),
    [match],
  );

  const setCurrentMessage = (nextMessage) =>
    setMessageState(nextMessage ? { analysisKey, message: nextMessage } : null);

  const parseAndIdentify = async () => {
    setBusy("identify");
    setCurrentMessage(null);
    setManualCandidates([]);
    setCandidateSearchError("");
    try {
      const { data } = await axios.post(`${API_BASE_URL}/vessel-inspections/identify`, {
        raw_ais_text: rawAis,
        vessel_fields: aisFields,
        verified_data: verified,
        ai_results: {
          prediction,
          extracted_id_text: prediction?.id_text || null,
        },
      });
      const parsed = { ...emptyFields(), ...data.ais_fields };
      setAisFields(parsed);
      setVerifiedState((current) => {
        const currentFields =
          current?.analysisKey === analysisKey ? current.fields : emptyFields();
        const inferred = classificationFields(prediction);
        const next = { ...currentFields };
        for (const [key] of VESSEL_FIELDS) {
          if (!next[key] && parsed[key]) next[key] = parsed[key];
        }
        if (!next.vessel_type && inferred.vessel_type) {
          next.vessel_type = inferred.vessel_type;
        }
        if (!next.flag_country && inferred.flag_country) {
          next.flag_country = inferred.flag_country;
        }
        if (!next.imo_number) next.imo_number = extractedImo(prediction);
        return { analysisKey, fields: next };
      });
      setMatchState({ analysisKey, data });
      setSelectedVesselId(
        data.status === "existing"
          ? data.vessel?.id || ""
          : data.status === "manual_review_required"
            ? ""
            : "",
      );
      setManualConfirm(false);
      if (data.status === "existing") {
        setCurrentMessage({
          kind: "success",
          text: `This vessel already exists${data.matched_by ? ` (matched by ${data.matched_by.toUpperCase()})` : ""}. You can continue with this vessel's new inspection.`,
        });
      } else if (data.status === "manual_review_required") {
        setCurrentMessage({
          kind: "warning",
          text: data.reason || "The vessel identifiers are inconsistent. Select and manually verify the correct vessel.",
        });
      } else if (data.status === "manual_confirmation_required") {
        setCurrentMessage({
          kind: "warning",
          text: data.reason,
        });
      } else {
        setCurrentMessage({
          kind: "success",
          text: "No vessel with these identifiers exists. A vessel record can be created after review.",
        });
      }
    } catch (error) {
      setMatchState(null);
      setCurrentMessage({ kind: "error", text: detailMessage(error) });
    } finally {
      setBusy("");
    }
  };

  const searchExistingVessels = async () => {
    const search = candidateSearch.trim() || verified.vessel_name?.trim() || "";
    if (search.length < 2) {
      setCandidateSearchError("Enter at least two characters to search existing vessels.");
      return;
    }
    setCandidateSearchBusy(true);
    setCandidateSearchError("");
    try {
      const { data } = await axios.get(
        `${API_BASE_URL}/vessel-inspections/vessels/search`,
        { params: { search } },
      );
      setManualCandidates(data.candidates || []);
      setSelectedVesselId("");
      setManualConfirm(false);
      if (!data.candidates?.length) {
        setCandidateSearchError("No matching vessels found. Confirm manually if this is a new vessel.");
      }
    } catch (error) {
      setCandidateSearchError(detailMessage(error));
    } finally {
      setCandidateSearchBusy(false);
    }
  };

  const saveInspection = async (event) => {
    event.preventDefault();
    if (!canWrite || busy || !match || !isImageAnalysis || savedInspection) return;
    if (!threatStatus) {
      setCurrentMessage({ kind: "error", text: "Select and confirm a threat assessment." });
      return;
    }
    if (
      (match.status === "manual_confirmation_required" ||
        match.status === "manual_review_required") &&
      !manualConfirm
    ) {
      setCurrentMessage({
        kind: "error",
        text: "Confirm the vessel identity manually before saving.",
      });
      return;
    }

    setBusy("save");
    setCurrentMessage(null);
    const form = new FormData();
    if (submissionKey.current.analysisKey !== analysisKey || !submissionKey.current.value) {
      submissionKey.current = {
        analysisKey,
        value: window.crypto.randomUUID(),
      };
    }
    const inspectionDate = new Date(inspectionAt);
    const payload = {
      submission_key: submissionKey.current.value,
      raw_ais_text: rawAis,
      ais_fields: aisFields,
      ai_results: {
        prediction,
        extracted_id_text: prediction?.id_text || null,
      },
      verified_data: verified,
      vessel_id: selectedVesselId || null,
      manual_identity_confirmed: manualConfirm,
      inspection_at: inspectionDate.toISOString(),
      inspector_note: note,
      threat_status: threatStatus,
    };
    form.append("payload", JSON.stringify(payload));
    form.append("file", file);

    try {
      const { data } = await axios.post(`${API_BASE_URL}/vessel-inspections`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setCurrentMessage({
        kind: "success",
        text: data.vessel_created
          ? "New vessel and inspection saved successfully."
          : data.duplicate_submission
            ? "This inspection was already saved."
            : "New vessel inspection saved successfully.",
      });
      setSavedAnalysisKey(analysisKey);
    } catch (error) {
      setCurrentMessage({ kind: "error", text: detailMessage(error) });
    } finally {
      setBusy("");
    }
  };

  const renderFields = (values, onChange, disabled = false) => (
    <div className="inspection-fields">
      {VESSEL_FIELDS.map(([key, label]) => (
        <label key={key}>
          <span>{label}</span>
          <input
            value={values[key] || ""}
            onChange={(event) => onChange(key, event.target.value)}
            disabled={disabled}
            maxLength={key === "vessel_name" ? 120 : 100}
          />
        </label>
      ))}
    </div>
  );

  return (
    <section className="a-card vessel-inspection-panel">
      <div className="a-card-header">
        <span className="a-card-label">
          <ClipboardCheck size={12} /> VESSEL VERIFICATION &amp; INSPECTION
        </span>
        <span className="a-card-tag">AIS · AI · OPERATOR</span>
      </div>

      {!isImageAnalysis && (
        <p className="inspection-hint">
          Run vessel image analysis first. Inspections require the analysed image.
          Video analyses are not yet supported for inspection image storage.
        </p>
      )}

      <div className="inspection-source-grid">
        <div className="inspection-source-card">
          <h3>ORIGINAL AIS DATA</h3>
          <label className="inspection-ais-input">
            <span>PASTE AIS TEXT</span>
            <textarea
              value={rawAis}
              onChange={(event) => {
                setRawAis(event.target.value);
                setMatchState(null);
                setCurrentMessage(null);
              }}
              rows={5}
              maxLength={20_000}
              placeholder="Paste available AIS fields as received. Missing fields are left blank."
              disabled={!canWrite}
            />
          </label>
          <p className="inspection-source-caption">
            Structured fields parsed from the supplied AIS text; source values are preserved for audit.
          </p>
          {renderFields(aisFields, () => {}, true)}
          <button
            className="inspection-button inspection-button--secondary"
            type="button"
            onClick={parseAndIdentify}
            disabled={!canWrite || busy === "identify"}
          >
            {busy === "identify" ? <Loader size={13} className="spin" /> : <SearchCheck size={13} />}
            PARSE AIS &amp; CHECK VESSEL
          </button>
        </div>

        <div className="inspection-source-card">
          <h3>AI PREDICTION / OCR EVIDENCE</h3>
          {prediction && !prediction.__error ? (
            <>
              <p className="inspection-evidence-line">
                <strong>Classification:</strong>{" "}
                {[
                  ...(prediction.results || []).map((item) => item?.label),
                  prediction.vessel_origin,
                ].filter(Boolean).join(" · ") || "No vessel classification returned"}
              </p>
              <p className="inspection-evidence-line">
                <strong>OCR:</strong>{" "}
                {(prediction.id_text?.readings || []).map((reading) => reading.text).filter(Boolean).join(" · ") ||
                  prediction.id_text?.message ||
                  "No text extracted"}
              </p>
              <p className="inspection-source-caption">
                Model evidence is retained as returned; it does not set the threat assessment.
              </p>
            </>
          ) : (
            <p className="inspection-hint">No image prediction is available yet.</p>
          )}
        </div>
      </div>

      {match && (
        <div className={`inspection-match inspection-match--${match.status}`}>
          {match.status === "existing" ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}
          <span>{message?.text || match.reason || "Vessel identity checked."}</span>
        </div>
      )}

      <form className="inspection-review-form" onSubmit={saveInspection}>
        <div className="inspection-verified-section">
          <div className="inspection-section-heading">
            <h3>FINAL OPERATOR-VERIFIED DATA</h3>
            <span>Review and correct before saving</span>
          </div>
          {renderFields(verified, (key, value) => {
            setVerifiedState((current) => {
              const fields =
                current?.analysisKey === analysisKey ? current.fields : emptyFields();
              return { analysisKey, fields: { ...fields, [key]: value } };
            });
            setMatchState(null);
            setCurrentMessage(null);
          }, !canWrite)}
        </div>

        {match?.status === "manual_review_required" && candidateVessels.length > 0 && (
          <fieldset className="inspection-identity-options">
            <legend>MANUALLY VERIFY THE EXISTING VESSEL</legend>
            {candidateVessels.map((vessel) => (
              <label key={vessel.id}>
                <input
                  type="radio"
                  name="matched-vessel"
                  value={vessel.id}
                  checked={selectedVesselId === vessel.id}
                  onChange={() => setSelectedVesselId(vessel.id)}
                />
                <span>
                  {vessel.verified_data?.vessel_name ||
                    vessel.vessel_name ||
                    "Unnamed vessel"}{" "}
                  · IMO {vessel.imo_number || "—"} · MMSI {vessel.mmsi_number || "—"}
                </span>
              </label>
            ))}
          </fieldset>
        )}

        {match?.status === "manual_confirmation_required" && (
          <div className="inspection-manual-vessel-search">
            <p>Search the vessel registry for a possible existing vessel. A name match is only a candidate and will never be selected automatically.</p>
            <div>
              <input
                value={candidateSearch}
                onChange={(event) => {
                  setCandidateSearch(event.target.value);
                  setManualCandidates([]);
                  setSelectedVesselId("");
                  setManualConfirm(false);
                  setCandidateSearchError("");
                }}
                maxLength={120}
                placeholder="Search by vessel name, IMO or MMSI"
                disabled={!canWrite || candidateSearchBusy}
              />
              <button
                className="inspection-button inspection-button--secondary"
                type="button"
                onClick={searchExistingVessels}
                disabled={!canWrite || candidateSearchBusy}
              >
                {candidateSearchBusy ? <Loader size={13} className="spin" /> : <SearchCheck size={13} />}
                SEARCH REGISTRY
              </button>
            </div>
            {candidateSearchError && <p className="inspection-message inspection-message--warning">{candidateSearchError}</p>}
            {manualCandidates.length > 0 && (
              <fieldset className="inspection-identity-options">
                <legend>MANUALLY SELECT A POSSIBLE EXISTING VESSEL</legend>
                {manualCandidates.map((vessel) => (
                  <label key={vessel.id}>
                    <input
                      type="radio"
                      name="manual-vessel"
                      value={vessel.id}
                      checked={selectedVesselId === vessel.id}
                      onChange={() => {
                        setSelectedVesselId(vessel.id);
                        setManualConfirm(false);
                      }}
                    />
                    <span>
                      {vessel.verified_data?.vessel_name || vessel.vessel_name || "Unnamed vessel"}{" "}
                      · IMO {vessel.imo_number || "—"} · MMSI {vessel.mmsi_number || "—"}
                    </span>
                  </label>
                ))}
              </fieldset>
            )}
          </div>
        )}

        {match?.status === "manual_confirmation_required" && (
          <label className="inspection-confirmation">
            <input
              type="checkbox"
              checked={manualConfirm}
              onChange={(event) => setManualConfirm(event.target.checked)}
            />
            {selectedVesselId
              ? "I manually verified that the selected registry vessel is the vessel in this inspection."
              : "I manually verified this as a new vessel; no valid IMO or MMSI was available."}
          </label>
        )}
        {match?.status === "manual_review_required" && candidateVessels.length > 0 && (
          <label className="inspection-confirmation">
            <input
              type="checkbox"
              checked={manualConfirm}
              onChange={(event) => setManualConfirm(event.target.checked)}
            />
            I verified that the selected existing vessel is the vessel in this inspection.
          </label>
        )}

        <div className="inspection-form-grid">
          <label>
            <span>INSPECTION DATE &amp; TIME</span>
            <input
              type="datetime-local"
              value={inspectionAt}
              onChange={(event) => setInspectionAt(event.target.value)}
              required
              disabled={!canWrite}
            />
          </label>
          <label className="inspection-note-field">
            <span>INSPECTOR&apos;S NOTE</span>
            <textarea
              value={note}
              onChange={(event) => setNote(event.target.value)}
              maxLength={5_000}
              rows={3}
              placeholder="Add observations from the inspection."
              disabled={!canWrite}
            />
          </label>
        </div>

        <fieldset className="inspection-threat-choice">
          <legend>THREAT ASSESSMENT — OPERATOR CONFIRMATION REQUIRED</legend>
          <label>
            <input
              type="radio"
              name="threat-status"
              value="threat"
              checked={threatStatus === "threat"}
              onChange={(event) => setThreatStatus(event.target.value)}
              required
              disabled={!canWrite}
            />
            <span>Threat</span>
          </label>
          <label>
            <input
              type="radio"
              name="threat-status"
              value="not_threat"
              checked={threatStatus === "not_threat"}
              onChange={(event) => setThreatStatus(event.target.value)}
              disabled={!canWrite}
            />
            <span>Not a threat</span>
          </label>
        </fieldset>

        {message && (
          <p className={`inspection-message inspection-message--${message.kind}`} role="status">
            {message.text}
          </p>
        )}

        <button
          className="inspection-button inspection-button--save"
          type="submit"
          disabled={!canWrite || !isImageAnalysis || !match || Boolean(busy) || savedInspection}
        >
          {busy === "save" ? <Loader size={14} className="spin" /> : <ClipboardCheck size={14} />}
          SAVE VESSEL INSPECTION
        </button>
        {savedInspection && (
          <button
            className="inspection-button inspection-button--secondary"
            type="button"
            onClick={() => {
              submissionKey.current = { analysisKey, value: "" };
              setInspectionAt(localDateTime());
              setNote("");
              setThreatStatus("");
              setManualConfirm(false);
              setCurrentMessage(null);
              setSavedAnalysisKey("");
            }}
          >
            START ANOTHER INSPECTION
          </button>
        )}
      </form>
    </section>
  );
}
