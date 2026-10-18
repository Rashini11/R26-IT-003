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
          <span className="auth-kicker">R26-IT-003 · Research access</span>
          <h2>Maritime intelligence.<br />Controlled access.</h2>
          <p>
            Sign in to use hull condition, sea-state, vessel detection, radar classification,
            the guided chat assistant and live simulation.
          </p>
        </div>

        <div className="auth-system-list">
          <div><Waves size={15} aria-hidden="true" /><span>Sea state analysis</span><b>Available after sign-in</b></div>
          <div><Radar size={15} aria-hidden="true" /><span>Radar & vessel modules</span><b>Available after sign-in</b></div>
          <div><ShieldCheck size={15} aria-hidden="true" /><span>Access control</span><b>Session required</b></div>
        </div>
      </section>

      <section className="auth-panel-wrap">
        <div className="auth-panel">
          <div className="auth-panel-topline">
            <LockKeyhole size={15} aria-hidden="true" />
            <span>OceanIQ portal</span>
          </div>

          <div className="auth-tabs">
            <button
              type="button"
              className={mode === "login" ? "active" : ""}
              onClick={() => switchMode("login")}
            >
              Sign in
            </button>
            <button
              type="button"
              className={mode === "signup" ? "active" : ""}
              onClick={() => switchMode("signup")}
            >
              Create profile
            </button>
          </div>

          <div className="auth-panel-heading">
            <span>{mode === "login" ? "Authenticate" : "Request access"}</span>
            <h3>{mode === "login" ? "Sign in" : "Create OceanIQ profile"}</h3>
            <p>
              {mode === "login"
                ? "Enter your OceanIQ account credentials."
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
              <p className="auth-error">Passwords do not match.</p>
            )}
            {error && <p className="auth-error">{error}</p>}
            {success && <p className="auth-success">{success}</p>}

            <button
              className="auth-submit"
              type="submit"
              disabled={submitting || passwordMismatch}
            >
              {mode === "login" ? <LockKeyhole size={15} aria-hidden="true" /> : <UserPlus size={15} aria-hidden="true" />}
              {submitting
                ? mode === "login" ? "Signing in…" : "Creating profile…"
                : mode === "login" ? "Sign in to OceanIQ" : "Create profile"}
            </button>
          </form>

          {mode === "signup" && (
            <div className="auth-hint">
              <p>
                Registration does not automatically grant operational access. An administrator
                assigns either <strong>read only</strong> or <strong>read / write</strong> access.
              </p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export default Login;
