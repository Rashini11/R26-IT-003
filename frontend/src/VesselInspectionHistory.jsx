import { useEffect, useState } from "react";
import axios from "axios";
import {
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  FileSearch,
  Loader,
  RefreshCw,
  Search,
  ShieldCheck,
} from "lucide-react";
import { API_BASE_URL } from "./context/AuthContext";
import "./VesselInspectionHistory.css";

const prettyDate = (value) =>
  value ? new Date(value).toLocaleString() : "Not recorded";

const errorText = (error) => {
  const detail = error?.response?.data?.detail;
  return typeof detail === "string" ? detail : "Unable to load vessel inspection records.";
};

function DetailList({ title, value }) {
  const entries = Object.entries(value || {});
  return (
    <section className="inspection-detail-block">
      <h3>{title}</h3>
      {entries.length ? (
        <dl>
          {entries.map(([key, item]) => (
            <div key={key}>
              <dt>{key.replaceAll("_", " ")}</dt>
              <dd>{item && typeof item === "object" ? JSON.stringify(item) : item || "—"}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p>No data recorded.</p>
      )}
    </section>
  );
}

export default function VesselInspectionHistory() {
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [threatStatus, setThreatStatus] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize] = useState(10);
  const [records, setRecords] = useState([]);
  const [total, setTotal] = useState(0);
  const [pages, setPages] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState("");
  const activeDetail =
    detail?.inspection?.id === selectedId ? detail : null;
  const loadRecords = async () => {
    setLoading(true);
    setError("");
    try {
      const params = { page, page_size: pageSize };
      if (search) params.search = search;
      if (threatStatus) params.threat_status = threatStatus;
      if (dateFrom) params.date_from = new Date(`${dateFrom}T00:00:00`).toISOString();
      if (dateTo) params.date_to = new Date(`${dateTo}T23:59:59.999`).toISOString();
      const { data } = await axios.get(`${API_BASE_URL}/vessel-inspections`, { params });
      setRecords(data.records || []);
      setTotal(data.total || 0);
      setPages(data.pages || 0);
      if (selectedId && !(data.records || []).some((record) => record.id === selectedId)) {
        setSelectedId("");
      }
    } catch (requestError) {
      setError(errorText(requestError));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = window.setTimeout(() => {
      loadRecords();
    }, 180);
    return () => window.clearTimeout(timer);
    // Search and date fields drive the current server-side query.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search, threatStatus, dateFrom, dateTo, page, pageSize]);

  useEffect(() => {
    if (!selectedId) return undefined;
    let cancelled = false;
    let objectUrl = "";
    const timer = window.setTimeout(() => {
      setDetailLoading(true);
      setImageUrl("");
      axios
        .get(`${API_BASE_URL}/vessel-inspections/${selectedId}`, { withCredentials: true })
        .then(({ data }) => {
          if (cancelled) return;
          setDetail(data);
          const imagePath = data.inspection?.image?.image_url;
          if (imagePath) {
            return axios
              .get(`${API_BASE_URL}${imagePath}`, {
                responseType: "blob",
                withCredentials: true,
              })
              .then(({ data: image }) => {
                if (!cancelled) {
                  objectUrl = URL.createObjectURL(image);
                  setImageUrl(objectUrl);
                }
              });
          }
          return undefined;
        })
        .catch((requestError) => {
          if (!cancelled) setError(errorText(requestError));
        })
        .finally(() => {
          if (!cancelled) setDetailLoading(false);
        });
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [selectedId]);

  const applySearch = (event) => {
    event.preventDefault();
    setPage(1);
    setSearch(searchInput.trim());
  };

  const clearFilters = () => {
    setSearchInput("");
    setSearch("");
    setThreatStatus("");
    setDateFrom("");
    setDateTo("");
    setPage(1);
    setSelectedId("");
  };

  return (
    <div className="vessel-inspection-history">
      <section className="a-card inspection-history-controls">
        <div className="a-card-header">
          <span className="a-card-label"><FileSearch size={12} /> VESSEL INSPECTION HISTORY</span>
          <span className="a-card-tag">{total} RECORDS</span>
        </div>
        <form className="inspection-history-filters" onSubmit={applySearch}>
          <label className="inspection-history-search">
            <span>VESSEL NAME / IMO / MMSI</span>
            <div>
              <Search size={14} />
              <input
                value={searchInput}
                onChange={(event) => setSearchInput(event.target.value)}
                placeholder="Search saved inspections"
              />
            </div>
          </label>
          <label>
            <span>THREAT STATUS</span>
            <select value={threatStatus} onChange={(event) => { setPage(1); setThreatStatus(event.target.value); }}>
              <option value="">All assessments</option>
              <option value="threat">Threat</option>
              <option value="not_threat">Not a threat</option>
            </select>
          </label>
          <label>
            <span>FROM DATE</span>
            <input type="date" value={dateFrom} onChange={(event) => { setPage(1); setDateFrom(event.target.value); }} />
          </label>
          <label>
            <span>TO DATE</span>
            <input type="date" value={dateTo} onChange={(event) => { setPage(1); setDateTo(event.target.value); }} />
          </label>
          <button type="submit" className="inspection-history-action">
            <Search size={13} /> SEARCH
          </button>
          <button type="button" className="inspection-history-action" onClick={clearFilters}>
            CLEAR FILTERS
          </button>
        </form>
        <div className="inspection-history-toolbar">
          <span>{loading ? "Loading inspection records…" : `${total} inspection${total === 1 ? "" : "s"} found`}</span>
          <button type="button" onClick={loadRecords} disabled={loading}>
            {loading ? <Loader size={13} className="spin" /> : <RefreshCw size={13} />} REFRESH
          </button>
        </div>
      </section>

      {error && (
        <div className="inspection-history-state inspection-history-state--error" role="alert">
          <AlertTriangle size={16} /> {error}
        </div>
      )}

      {loading ? (
        <div className="inspection-history-state"><Loader size={18} className="spin" /> Loading records…</div>
      ) : records.length === 0 ? (
        <div className="inspection-history-state inspection-history-state--empty">
          <FileSearch size={23} />
          <strong>No vessel inspections found</strong>
          <span>Saved inspection records will appear here.</span>
        </div>
      ) : (
        <div className="inspection-history-layout">
          <section className="inspection-history-list" aria-label="Vessel inspection records">
            {records.map((record) => {
              const vessel = record.vessel || {};
              const fields = record.verified_data || {};
              const threat = record.threat_status === "threat";
              return (
                <button
                  type="button"
                  className={`a-card inspection-history-record ${record.id === selectedId ? "inspection-history-record--selected" : ""}`}
                  key={record.id}
                  onClick={() => setSelectedId(record.id)}
                >
                  <div className={`inspection-threat-badge ${threat ? "is-threat" : "is-safe"}`}>
                    {threat ? <AlertTriangle size={12} /> : <ShieldCheck size={12} />}
                    {threat ? "THREAT" : "NOT A THREAT"}
                  </div>
                  <strong>{fields.vessel_name || vessel.vessel_name || "Unnamed vessel"}</strong>
                  <span>{fields.vessel_type || "Type unavailable"} · {fields.flag_country || "Flag unavailable"}</span>
                  <span>IMO {fields.imo_number || vessel.imo_number || "—"} · MMSI {fields.mmsi_number || vessel.mmsi_number || "—"}</span>
                  <time>{prettyDate(record.inspection_at)}</time>
                </button>
              );
            })}
            <div className="inspection-history-pagination">
              <span>Page {page} of {Math.max(pages, 1)}</span>
              <button type="button" onClick={() => setPage((current) => Math.max(1, current - 1))} disabled={page <= 1 || loading}>
                <ChevronLeft size={14} /> PREVIOUS
              </button>
              <button type="button" onClick={() => setPage((current) => Math.min(pages, current + 1))} disabled={page >= pages || loading}>
                NEXT <ChevronRight size={14} />
              </button>
            </div>
          </section>

          <section className="a-card inspection-history-detail">
            {!selectedId ? (
              <div className="inspection-history-state inspection-history-state--empty">
                <FileSearch size={22} />
                <span>Select an inspection to view its full details.</span>
              </div>
            ) : detailLoading ? (
              <div className="inspection-history-state"><Loader size={17} className="spin" /> Loading inspection…</div>
            ) : activeDetail ? (
              <>
                <div className="inspection-detail-heading">
                  <div>
                    <span>INSPECTION RECORD</span>
                    <h2>{activeDetail.inspection?.verified_data?.vessel_name || "Unnamed vessel"}</h2>
                  </div>
                  <button type="button" onClick={() => setSelectedId("")}>CLOSE</button>
                </div>
                <div className={`inspection-threat-badge ${activeDetail.inspection?.threat_status === "threat" ? "is-threat" : "is-safe"}`}>
                  {activeDetail.inspection?.threat_status === "threat" ? <AlertTriangle size={12} /> : <ShieldCheck size={12} />}
                  {activeDetail.inspection?.threat_status === "threat" ? "THREAT" : "NOT A THREAT"}
                </div>
                <dl className="inspection-summary-details">
                  <div><dt>INSPECTION DATE</dt><dd>{prettyDate(activeDetail.inspection?.inspection_at)}</dd></div>
                  <div><dt>VESSEL REFERENCE</dt><dd>{activeDetail.inspection?.vessel_id || "—"}</dd></div>
                  <div><dt>INSPECTOR NOTE</dt><dd>{activeDetail.inspection?.inspector_note || "No note recorded"}</dd></div>
                </dl>
                {imageUrl && <img className="inspection-stored-image" src={imageUrl} alt="Analysed vessel inspection" />}
                {!imageUrl && activeDetail.inspection?.image && (
                  <p className="inspection-image-pending">Image unavailable or inaccessible with current Drive configuration.</p>
                )}
                <DetailList title="ORIGINAL AIS DATA" value={activeDetail.inspection?.ais_snapshot?.parsed} />
                {activeDetail.inspection?.ais_snapshot?.raw_text && (
                  <pre className="inspection-raw-evidence">{activeDetail.inspection.ais_snapshot.raw_text}</pre>
                )}
                <DetailList title="FINAL VERIFIED DATA" value={activeDetail.inspection?.verified_data} />
                <DetailList title="RECORDED CORRECTIONS" value={activeDetail.inspection?.correction_history} />
                <section className="inspection-detail-block">
                  <h3>AI PREDICTIONS &amp; OCR</h3>
                  <pre className="inspection-raw-evidence">
                    {JSON.stringify(activeDetail.inspection?.ai_results || {}, null, 2)}
                  </pre>
                </section>
              </>
            ) : (
              <div className="inspection-history-state inspection-history-state--error">Inspection details are unavailable.</div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
