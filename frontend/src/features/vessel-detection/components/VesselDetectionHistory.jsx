import { Fragment } from "react";
import { VESSEL_HISTORY_PAGE_SIZE } from "../utils/vesselDetectionUtils";
import PortCountryLookup from "./PortCountryLookup";

// Show saved vessel records and controls for editing or deleting them.
export default function VesselDetectionHistory({
  activeModule,
  vesselHistoryStorage,
  fetchVesselHistory,
  vesselHistoryLoading,
  vesselHistoryError,
  vesselHistory,
  canWrite,
  startEditingVesselRecord,
  deleteVesselHistoryRecord,
  currentVesselHistoryRecords,
  vesselHistoryStartIndex,
  editingVesselRecordId,
  editingVesselDraft,
  setEditingVesselDraft,
  saveEditedVesselHistoryRecord,
  setEditingVesselRecordId,
  vesselHistoryPageCount,
  setVesselHistoryPage,
  currentVesselHistoryPage,
  setExpandedHistoryImage,
}) {
  return (
    <>
        {activeModule === "boat" && (
          <section className="a-card vessel-history-panel">
            <div className="sea-history-header">
              <div>
                <p className="sea-feature-title">VESSEL DETECTION RECORDS</p>
                <p className="vessel-storage-status">STORAGE · {vesselHistoryStorage.toUpperCase()}</p>
              </div>
              <div className="sea-history-actions">
                <button onClick={fetchVesselHistory} disabled={vesselHistoryLoading}>
                  {vesselHistoryLoading ? "LOADING…" : "REFRESH"}
                </button>
              </div>
            </div>
            {vesselHistoryError && <p className="vessel-save-message vessel-save-message--error">{vesselHistoryError}</p>}
            {vesselHistory.length === 0 ? (
              <p className="sea-history-empty">No submitted vessel records are available.</p>
            ) : (
              <>
                <div className="vessel-table-scroll">
                  <table className="vessel-record-table">
                  <thead>
                    <tr>
                      <th>VESSEL / TYPE</th>
                      <th>IDENTITY DETAILS</th>
                      <th>IMAGES</th>
                      <th>STATUS</th>
                      <th>SUBMITTED</th>
                      <th>ACTIONS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentVesselHistoryRecords.map((item, index) => (
                      <Fragment key={`${item.timestamp}-${item.record_id || vesselHistoryStartIndex + index}`}>
                        <tr>
                          <td>
                            <strong className="vessel-table-primary">
                              {item.vessel_classifications?.length
                                ? item.vessel_classifications.join(" or ")
                                : item.vessel_info?.vessel_name || "Vessel record"}
                            </strong>
                            <span className="vessel-table-secondary">
                              {item.vessel_info?.vessel_name || "Name not identified"}
                            </span>
                          </td>
                          <td>
                            <span>{item.vessel_info?.port || "Port not identified"}</span>
                            {item.vessel_info?.port_country && (
                              <span className="vessel-table-secondary">
                                Port country: {item.vessel_info.port_country}
                              </span>
                            )}
                            <span className="vessel-table-secondary">
                              {item.vessel_info?.port_type || "Foreign"}
                            </span>
                            <span className="vessel-table-secondary">{item.vessel_info?.imo_number || "IMO not identified"}</span>
                            {item.notes && <span className="vessel-table-secondary">{item.notes}</span>}
                          </td>
                          <td>
                            <div className="vessel-record-images">
                              {item.image_url && (
                                <img
                                  src={item.image_url}
                                  alt="Vessel"
                                  role="button"
                                  tabIndex={0}
                                  onClick={() => setExpandedHistoryImage({ url: item.image_url, alt: "Vessel" })}
                                  onKeyDown={(event) => {
                                    if (event.key === "Enter" || event.key === " ") {
                                      event.preventDefault();
                                      setExpandedHistoryImage({ url: item.image_url, alt: "Vessel" });
                                    }
                                  }}
                                />
                              )}
                              {item.vessel_info_image_url && (
                                <img
                                  src={item.vessel_info_image_url}
                                  alt="Vessel backside"
                                  role="button"
                                  tabIndex={0}
                                  onClick={() => setExpandedHistoryImage({ url: item.vessel_info_image_url, alt: "Vessel backside" })}
                                  onKeyDown={(event) => {
                                    if (event.key === "Enter" || event.key === " ") {
                                      event.preventDefault();
                                      setExpandedHistoryImage({ url: item.vessel_info_image_url, alt: "Vessel backside" });
                                    }
                                  }}
                                />
                              )}
                              {!item.image_url && !item.vessel_info_image_url && <span>No image</span>}
                            </div>
                          </td>
                          <td><span className="vessel-status-pill">{item.status || "Saved"}</span></td>
                          <td>{item.timestamp || "—"}</td>
                          <td>
                            <div className="vessel-history-controls">
                              <button onClick={() => startEditingVesselRecord(item)} disabled={!canWrite || !item.record_id}>EDIT</button>
                              <button className="danger" onClick={() => deleteVesselHistoryRecord(item.record_id)} disabled={!canWrite || !item.record_id}>DELETE</button>
                            </div>
                          </td>
                        </tr>
                        {editingVesselRecordId === item.record_id && editingVesselDraft && (
                          <tr className="vessel-table-edit-row">
                            <td colSpan={6}>
                              <div className="vessel-history-edit">
                                {[
                                  ["vessel_name", "VESSEL NAME"],
                                  ["port", "PORT"],
                                  ["imo_number", "IMO NUMBER"],
                                  ["vessel_classifications", "VESSEL TYPES"],
                                ].map(([field, label]) => (
                                  <label key={field}>
                                    <span>{label}</span>
                                    <input
                                      value={editingVesselDraft[field]}
                                      onChange={(event) => setEditingVesselDraft((current) => ({
                                        ...current,
                                        [field]: event.target.value,
                                        ...(field === "port" ? {
                                          port_country: "",
                                          port_country_code: "",
                                          port_type: "Foreign",
                                        } : {}),
                                      }))}
                                    />
                                  </label>
                                ))}
                                <PortCountryLookup
                                  port={editingVesselDraft.port}
                                  country={editingVesselDraft.port_country}
                                  onCountryFound={(portCountry) => setEditingVesselDraft((current) => ({
                                    ...current,
                                    ...portCountry,
                                  }))}
                                />
                                <label>
                                  <span>PORT TYPE</span>
                                  <select
                                    value={editingVesselDraft.port_type || "Foreign"}
                                    onChange={(event) => setEditingVesselDraft((current) => ({
                                      ...current,
                                      port_type: event.target.value,
                                    }))}
                                  >
                                    <option value="Local">Local</option>
                                    <option value="Foreign">Foreign</option>
                                  </select>
                                </label>
                                <label className="vessel-table-notes">
                                  <span>NOTES</span>
                                  <textarea
                                    rows={2}
                                    value={editingVesselDraft.notes}
                                    onChange={(event) => setEditingVesselDraft((current) => ({
                                      ...current,
                                      notes: event.target.value,
                                    }))}
                                  />
                                </label>
                                <div className="vessel-history-controls">
                                  <button onClick={saveEditedVesselHistoryRecord}>SAVE CHANGES</button>
                                  <button onClick={() => setEditingVesselRecordId(null)}>CANCEL</button>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    ))}
                  </tbody>
                  </table>
                </div>
                {vesselHistoryPageCount > 1 && (
                  <nav className="vessel-history-pagination" aria-label="Vessel history pages">
                    <span>
                      Showing {vesselHistoryStartIndex + 1}–
                      {Math.min(vesselHistoryStartIndex + VESSEL_HISTORY_PAGE_SIZE, vesselHistory.length)}
                      {" "}of {vesselHistory.length}
                    </span>
                    <div className="vessel-history-pagination-controls">
                      <button
                        type="button"
                        onClick={() => setVesselHistoryPage((page) => Math.max(1, page - 1))}
                        disabled={currentVesselHistoryPage === 1}
                      >
                        PREVIOUS
                      </button>
                      <span>PAGE {currentVesselHistoryPage} OF {vesselHistoryPageCount}</span>
                      <button
                        type="button"
                        onClick={() => setVesselHistoryPage((page) => Math.min(vesselHistoryPageCount, page + 1))}
                        disabled={currentVesselHistoryPage === vesselHistoryPageCount}
                      >
                        NEXT
                      </button>
                    </div>
                  </nav>
                )}
              </>
            )}
          </section>
        )}
    </>
  );
}
