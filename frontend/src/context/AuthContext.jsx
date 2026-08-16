import { createContext, useContext, useEffect, useState } from "react";
import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const AuthContext = createContext(null);

// Every OceanIQ API request should include the HttpOnly session cookie.
axios.defaults.withCredentials = true;

export function AuthProvider({ children }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [csrfToken, setCsrfToken] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const applySession = (data) => {
    setIsAuthenticated(Boolean(data?.authenticated));
    setUser(data?.user || null);
    setCsrfToken(data?.csrf_token || "");
    setError("");
  };

  const clearSession = () => {
    setIsAuthenticated(false);
    setUser(null);
    setCsrfToken("");
  };

  useEffect(() => {
    if (csrfToken) {
      axios.defaults.headers.common["X-CSRF-Token"] = csrfToken;
    } else {
      delete axios.defaults.headers.common["X-CSRF-Token"];
    }
  }, [csrfToken]);

  useEffect(() => {
    let mounted = true;

    const restoreSession = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/auth/me`, {
          withCredentials: true,
        });
        if (mounted) applySession(response.data);
      } catch {
        if (mounted) clearSession();
      } finally {
        if (mounted) setLoading(false);
      }
    };

    restoreSession();
    return () => {
      mounted = false;
    };
  }, []);

  const login = async (username, password) => {
    setError("");
    try {
      const response = await axios.post(
        `${API_BASE_URL}/auth/login`,
        { username, password },
        { withCredentials: true }
      );
      applySession(response.data);
      return true;
    } catch (err) {
      clearSession();
      const detail = err?.response?.data?.detail;
      setError(detail || "Unable to sign in. Please try again.");
      return false;
    }
  };

  const logout = async () => {
    try {
      await axios.post(
        `${API_BASE_URL}/auth/logout`,
        {},
        {
          withCredentials: true,
          headers: csrfToken ? { "X-CSRF-Token": csrfToken } : {},
        }
      );
    } finally {
      clearSession();
    }
  };

  const value = {
    isAuthenticated,
    user,
    loading,
    error,
    csrfToken,
    login,
    logout,
    clearError: () => setError(""),
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
