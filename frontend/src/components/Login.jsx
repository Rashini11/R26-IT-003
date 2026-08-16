import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import "./Login.css";

function Login() {
  const { login, error } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    login(username, password);
  };

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

        <h2>Login</h2>
        <p className="login-subtitle">Sign in to access the inspection dashboard</p>

        <form onSubmit={handleSubmit}>
          <label>
            Username
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username"
              autoComplete="username"
              required
            />
          </label>

          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
              autoComplete="current-password"
              required
            />
          </label>

          {error && <p className="login-error">{error}</p>}

          <button type="submit">Sign In</button>
        </form>
      </div>
    </div>
  );
}

export default Login;