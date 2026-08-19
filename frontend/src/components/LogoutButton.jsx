import { useState } from "react";
import { useAuth } from "../context/AuthContext";

function LogoutButton() {
  const { logout, user } = useAuth();
  const [busy, setBusy] = useState(false);

  const handleLogout = async () => {
    if (busy) return;
    setBusy(true);
    try {
      await logout();
    } finally {
      setBusy(false);
    }
  };

  return (
    <button
      type="button"
      onClick={handleLogout}
      disabled={busy}
      title={user?.username ? `Signed in as ${user.username}` : "Sign out"}
      style={{
        border: "1px solid rgba(0, 212, 255, 0.28)",
        background: "rgba(0, 212, 255, 0.08)",
        color: "#9be8ff",
        borderRadius: "6px",
        padding: "7px 10px",
        fontSize: "11px",
        letterSpacing: "0.06em",
        cursor: busy ? "wait" : "pointer",
        opacity: busy ? 0.65 : 1,
      }}
    >
      {busy ? "SIGNING OUT" : "LOGOUT"}
    </button>
  );
}

export default LogoutButton;
