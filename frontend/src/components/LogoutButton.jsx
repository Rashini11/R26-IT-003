import { useState } from "react";
import { LogOut } from "lucide-react";
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
      className="logout-btn"
    >
      <LogOut size={13} />
      <span>{busy ? "SIGNING OUT" : "LOGOUT"}</span>
    </button>
  );
}

export default LogoutButton;
