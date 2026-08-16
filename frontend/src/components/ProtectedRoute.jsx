import { useAuth } from "../context/AuthContext";
import Login from "./Login";

function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div className="login-page">
        <div className="login-bg"></div>
        <div className="login-card">
          <div className="login-brand">
            <div className="brand-icon">OQ</div>
            <div>
              <h1>OceanIQ</h1>
              <p>Marine AI Inspection System</p>
            </div>
          </div>
          <h2>Securing Session</h2>
          <p className="login-subtitle">Verifying your authenticated backend session...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Login />;
  }

  return children;
}

export default ProtectedRoute;
