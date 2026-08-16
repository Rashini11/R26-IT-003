import { createContext, useContext, useState } from "react";

// ---------------------------------------------------------
// Hardcoded login for this research project.
// Change these two values to whatever you want the team to use.
// ---------------------------------------------------------
const ADMIN_USERNAME = "admin";
const ADMIN_PASSWORD = "admin1"; // Change this to a secure password for your research project

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  // sessionStorage so it clears when the browser tab is closed;
  // switch to localStorage if you want it to stay logged in across sessions
  const [isAuthenticated, setIsAuthenticated] = useState(
    sessionStorage.getItem("oceaniq_auth") === "true"
  );
  const [error, setError] = useState("");

  const login = (username, password) => {
    if (username === ADMIN_USERNAME && password === ADMIN_PASSWORD) {
      sessionStorage.setItem("oceaniq_auth", "true");
      setIsAuthenticated(true);
      setError("");
      return true;
    }

    setError("Invalid username or password");
    return false;
  };

  const logout = () => {
    sessionStorage.removeItem("oceaniq_auth");
    setIsAuthenticated(false);
  };

  const value = { isAuthenticated, login, logout, error };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }

  return context;
}