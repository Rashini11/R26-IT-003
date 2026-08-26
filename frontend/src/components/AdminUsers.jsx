import { useCallback, useEffect, useState } from "react";
import axios from "axios";
import {
  CheckCircle2,
  Eye,
  PencilLine,
  RefreshCw,
  ShieldCheck,
  UserRoundCog,
  X,
  XCircle,
} from "lucide-react";
import { API_BASE_URL } from "../context/AuthContext";

const accessLabel = (user) => {
  if (user.role === "admin") return "ADMIN";
  if (user.approval_status !== "approved") return String(user.approval_status || "pending").toUpperCase();
  return user.access_level === "read_write" ? "READ / WRITE" : "READ ONLY";
};

export default function AdminUsers({ onClose }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState("");
  const [error, setError] = useState("");

  const loadUsers = useCallback(async () => {
    try {
      setLoading(true);
      setError("");
      const { data } = await axios.get(`${API_BASE_URL}/auth/admin/users`);
      setUsers(data.users || []);
    } catch (err) {
      setError(err?.response?.data?.detail || "Unable to load user profiles.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  const updateUser = async (userId, payload) => {
    try {
      setBusyId(userId);
      setError("");
      await axios.patch(`${API_BASE_URL}/auth/admin/users/${userId}`, payload);
      await loadUsers();
    } catch (err) {
      setError(err?.response?.data?.detail || "Unable to update user access.");
    } finally {
      setBusyId("");
    }
  };

  return (
    <div className="access-modal-backdrop" onMouseDown={onClose}>
      <section className="access-modal" onMouseDown={(event) => event.stopPropagation()}>
        <header className="access-modal-head">
          <div>
            <span><ShieldCheck size={13} /> ADMIN ACCESS CONTROL</span>
            <h2>User Profiles & Permissions</h2>
            <p>Approve new profiles and assign operational access levels.</p>
          </div>
          <button onClick={onClose} aria-label="Close"><X size={18} /></button>
        </header>

        <div className="access-toolbar">
          <div className="access-legend">
            <span><Eye size={12} /> READ ONLY = view GET data</span>
            <span><PencilLine size={12} /> READ / WRITE = run analyses and controls</span>
          </div>
          <button onClick={loadUsers} disabled={loading}><RefreshCw size={13} /> REFRESH</button>
        </div>

        {error && <div className="access-error">{error}</div>}

        <div className="access-table-wrap">
          <table className="access-table">
            <thead>
              <tr>
                <th>PROFILE</th>
                <th>STATUS</th>
                <th>ACCESS</th>
                <th>ACCOUNT</th>
                <th>ACTIONS</th>
              </tr>
            </thead>
            <tbody>
              {users.map((user) => {
                const busy = busyId === user.id;
                const isAdmin = user.role === "admin";
                return (
                  <tr key={user.id}>
                    <td>
                      <div className="access-profile">
                        <UserRoundCog size={15} />
                        <div>
                          <strong>{user.full_name || user.username}</strong>
                          <span>@{user.username} · {user.email || "no email"}</span>
                        </div>
                      </div>
                    </td>
                    <td><span className={`access-pill access-pill--${user.approval_status || "approved"}`}>{String(user.approval_status || "approved").toUpperCase()}</span></td>
                    <td><span className="access-level">{accessLabel(user)}</span></td>
                    <td>{user.is_active ? <span className="access-active"><CheckCircle2 size={12} /> ACTIVE</span> : <span className="access-disabled"><XCircle size={12} /> DISABLED</span>}</td>
                    <td>
                      {isAdmin ? (
                        <span className="access-admin-lock">ADMIN ACCOUNT</span>
                      ) : (
                        <div className="access-actions">
                          <button disabled={busy} onClick={() => updateUser(user.id, { approval_status: "approved", access_level: "read_only", is_active: true })}>READ ONLY</button>
                          <button disabled={busy} onClick={() => updateUser(user.id, { approval_status: "approved", access_level: "read_write", is_active: true })}>READ / WRITE</button>
                          <button disabled={busy} className="warn" onClick={() => updateUser(user.id, { approval_status: "rejected" })}>REJECT</button>
                          <button disabled={busy} className="danger" onClick={() => updateUser(user.id, { is_active: !user.is_active })}>{user.is_active ? "DISABLE" : "ENABLE"}</button>
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
              {!loading && users.length === 0 && (
                <tr><td colSpan="5" className="access-empty">No user profiles found.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
