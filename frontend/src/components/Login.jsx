import { useState } from "react";
import {
  Anchor,
  LockKeyhole,
  ShieldCheck,
  UserPlus,
  Waves,
  Radar,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import ThemeToggle from "./ThemeToggle";
import "./Login.css";

function Login() {
  const { login, signup, error, clearError } = useAuth();
  const [mode, setMode] = useState("login");
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState("");
  const [form, setForm] = useState({
    fullName: "",
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const update = (key) => (event) => {
    setForm((current) => ({ ...current, [key]: event.target.value }));
    clearError();
    setSuccess("");
  };

  const switchMode = (nextMode) => {
    setMode(nextMode);
    clearError();
    setSuccess("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (submitting) return;

    if (mode === "signup" && form.password !== form.confirmPassword) {
      setSuccess("");
      return;
    }

    setSubmitting(true);
    try {
      if (mode === "login") {
        await login(form.username, form.password);
        return;
      }

      const result = await signup({
        fullName: form.fullName,
        username: form.username,
        email: form.email,
        password: form.password,
      });

      if (result.ok) {
        setSuccess(result.data?.message || "Profile created successfully.");
        setMode("login");
        setForm((current) => ({
          ...current,
          password: "",
          confirmPassword: "",
        }));
      }
    } finally {
      setSubmitting(false);
    }
  };

  const passwordMismatch =
    mode === "signup" &&
    form.confirmPassword.length > 0 &&
    form.password !== form.confirmPassword;

  return (
    <div className="auth-page">
      <div className="auth-theme-control"><ThemeToggle /></div>
      <div className="auth-grid" />
      <div className="auth-scanline" />

      <section className="auth-intro">
        <div className="auth-brand-row">
          <div className="auth-logo"><Anchor size={25} /></div>
          <div>
            <h1>OceanIQ</h1>
            <p>MARINE AI INTELLIGENCE PLATFORM</p>
          </div>
        </div>

        <div className="auth-intro-copy">
          <span className="auth-kicker">R26-IT-003 · SECURE ACCESS GATEWAY</span>
          <h2>Maritime intelligence.<br />Controlled access.</h2>
          <p>
            Authenticate to access AI-assisted hull inspection, sea-state analysis,
            vessel detection, radar classification and live maritime simulation.
          </p>
        </div>

        <div className="auth-system-list">
          <div><Waves size={15} /><span>SEA STATE</span><b>ONLINE</b></div>
          <div><Radar size={15} /><span>RADAR INTELLIGENCE</span><b>ONLINE</b></div>
          <div><ShieldCheck size={15} /><span>ACCESS CONTROL</span><b>ENFORCED</b></div>
        </div>
      </section>

      <section className="auth-panel-wrap">
        <div className="auth-panel">
          <div className="auth-panel-topline">
            <LockKeyhole size={15} />
            <span>SECURE OPERATOR PORTAL</span>
          </div>

          <div className="auth-tabs">
            <button
              type="button"
              className={mode === "login" ? "active" : ""}
              onClick={() => switchMode("login")}
            >
              SIGN IN
            </button>
            <button
              type="button"
              className={mode === "signup" ? "active" : ""}
              onClick={() => switchMode("signup")}
            >
              CREATE PROFILE
            </button>
          </div>

          <div className="auth-panel-heading">
            <span>{mode === "login" ? "AUTHENTICATE" : "REQUEST ACCESS"}</span>
            <h3>{mode === "login" ? "Operator Login" : "Create OceanIQ Profile"}</h3>
            <p>
              {mode === "login"
                ? "Enter your secured OceanIQ account credentials."
                : "New profiles require administrator approval before dashboard access is granted."}
            </p>
          </div>

          <form className="auth-form" onSubmit={handleSubmit}>
            {mode === "signup" && (
              <label>
                <span>FULL NAME</span>
                <input
                  value={form.fullName}
                  onChange={update("fullName")}
                  placeholder="Operator full name"
                  autoComplete="name"
                  minLength={2}
                  required
                  disabled={submitting}
                />
              </label>
            )}

            <label>
              <span>USERNAME</span>
              <input
                value={form.username}
                onChange={update("username")}
                placeholder="Enter username"
                autoComplete="username"
                minLength={mode === "signup" ? 3 : 1}
                required
                disabled={submitting}
              />
            </label>

            {mode === "signup" && (
              <label>
                <span>EMAIL</span>
                <input
                  type="email"
                  value={form.email}
                  onChange={update("email")}
                  placeholder="operator@example.com"
                  autoComplete="email"
                  required
                  disabled={submitting}
                />
              </label>
            )}

            <label>
              <span>PASSWORD</span>
              <input
                type="password"
                value={form.password}
                onChange={update("password")}
                placeholder={mode === "signup" ? "Minimum 10 characters" : "Enter password"}
                autoComplete={mode === "signup" ? "new-password" : "current-password"}
                minLength={mode === "signup" ? 10 : 1}
                required
                disabled={submitting}
              />
            </label>

            {mode === "signup" && (
              <label>
                <span>CONFIRM PASSWORD</span>
                <input
                  type="password"
                  value={form.confirmPassword}
                  onChange={update("confirmPassword")}
                  placeholder="Re-enter password"
                  autoComplete="new-password"
                  minLength={10}
                  required
                  disabled={submitting}
                />
              </label>
            )}

            {passwordMismatch && (
              <p className="auth-message auth-message--error">Passwords do not match.</p>
            )}
            {error && <p className="auth-message auth-message--error">{error}</p>}
            {success && <p className="auth-message auth-message--success">{success}</p>}

            <button
              className="auth-primary"
              type="submit"
              disabled={submitting || passwordMismatch}
            >
              {mode === "login" ? <LockKeyhole size={15} /> : <UserPlus size={15} />}
              {submitting
                ? mode === "login" ? "AUTHENTICATING..." : "CREATING PROFILE..."
                : mode === "login" ? "SIGN IN TO OCEANIQ" : "CREATE PROFILE"}
            </button>
          </form>

          {mode === "signup" && (
            <div className="auth-approval-note">
              <ShieldCheck size={14} />
              <p>
                Registration does not automatically grant operational access. An administrator
                assigns either <strong>READ ONLY</strong> or <strong>READ / WRITE</strong> access.
              </p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export default Login;
