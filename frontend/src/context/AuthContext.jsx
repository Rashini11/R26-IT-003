import { createContext, useContext, useEffect, useMemo, useState } from "react";
import axios from "axios";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? "" : "http://localhost:8000");

const AuthContext = createContext(null);

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

  const refreshSession = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/auth/me`, {
        withCredentials: true,
      });
      applySession(response.data);
      return response.data;
    } catch (err) {
      if (err?.response?.status === 401) clearSession();
      throw err;
    }
  };

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
      return { ok: true, data: response.data };
    } catch (err) {
      clearSession();
      const detail = err?.response?.data?.detail;
      setError(detail || "Unable to sign in. Please try again.");
      return { ok: false, error: detail || "Unable to sign in." };
    }
  };

  const signup = async ({ fullName, username, email, password }) => {
    setError("");
    try {
      const response = await axios.post(`${API_BASE_URL}/auth/signup`, {
        full_name: fullName,
        username,
        email,
        password,
      });
      return { ok: true, data: response.data };
    } catch (err) {
      const detail = err?.response?.data?.detail;
      const message = Array.isArray(detail)
        ? detail.map((item) => item.msg).join(" · ")
        : detail || "Unable to create profile. Please try again.";
      setError(message);
      return { ok: false, error: message };
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

  const permissions = useMemo(() => {
    const isAdmin = user?.role === "admin";
    const approved = user?.approval_status === "approved";
    const accessLevel = user?.access_level || "none";

    return {
      isAdmin,
      isPending: user?.approval_status === "pending",
      isRejected: user?.approval_status === "rejected",
      canRead: isAdmin || (approved && ["read_only", "read_write"].includes(accessLevel)),
      canWrite: isAdmin || (approved && accessLevel === "read_write"),
      accessLevel: isAdmin ? "read_write" : accessLevel,
    };
  }, [user]);

  const value = {
    isAuthenticated,
    user,
    loading,
    error,
    csrfToken,
    ...permissions,
    login,
    signup,
    logout,
    refreshSession,
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
