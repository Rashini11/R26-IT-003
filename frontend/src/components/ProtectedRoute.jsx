import { Clock3, RefreshCw, ShieldX } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import Login from "./Login";
import "./Login.css";

function AccessStatus() {
  const { user, isPending, isRejected, refreshSession, logout } = useAuth();

  return (
    <div className="auth-page">
      <div className="auth-grid" />
      <div className="auth-panel-wrap" style={{ gridColumn: "1 / -1" }}>
        <div className="auth-panel auth-status-card">
          <div className="auth-status-icon">
            {isRejected ? <ShieldX size={25} /> : <Clock3 size={25} />}
          </div>
          <span className="auth-kicker">OCEANIQ ACCESS CONTROL</span>
          <h2>{isRejected ? "Access Request Rejected" : "Approval Pending"}</h2>
          <p>
            {isRejected
              ? "This profile is authenticated, but operational dashboard access has not been approved. Contact an OceanIQ administrator if this should be reviewed."
              : "Your profile was created successfully. An OceanIQ administrator must assign READ ONLY or READ / WRITE access before you can enter the dashboard."}
          </p>

          <div className="auth-status-meta">
            <div><span>OPERATOR</span><b>{user?.full_name || user?.username || "—"}</b></div>
            <div><span>USERNAME</span><b>{user?.username || "—"}</b></div>
            <div><span>STATUS</span><b>{String(user?.approval_status || "pending").toUpperCase()}</b></div>
            <div><span>ACCESS</span><b>{String(user?.access_level || "none").replaceAll("_", " ").toUpperCase()}</b></div>
          </div>

          <div className="auth-status-actions">
            {!isRejected && (
              <button className="auth-secondary" onClick={() => refreshSession().catch(() => {})}>
                <RefreshCw size={12} /> REFRESH STATUS
              </button>
            )}
            <button className="auth-secondary" onClick={logout}>SIGN OUT</button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ProtectedRoute({ children }) {
  const { isAuthenticated, loading, isPending, isRejected, canRead } = useAuth();

  if (loading) {
    return (
      <div className="auth-page">
        <div className="auth-grid" />
        <div className="auth-panel-wrap" style={{ gridColumn: "1 / -1" }}>
          <div className="auth-panel auth-status-card">
            <div className="auth-status-icon"><RefreshCw size={25} /></div>
            <span className="auth-kicker">SECURE SESSION</span>
            <h2>Verifying Access</h2>
            <p>Checking your authenticated backend session and assigned OceanIQ permissions.</p>
          </div>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) return <Login />;
  if (isPending || isRejected || !canRead) return <AccessStatus />;
  return children;
}

export default ProtectedRoute;
