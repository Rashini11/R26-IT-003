const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const pythonPath =
  process.platform === "win32"
    ? path.resolve("venv", "Scripts", "python.exe")
    : path.resolve("venv", "bin", "python");

if (!fs.existsSync(pythonPath)) {
  console.error(`[OceanIQ] Python virtual environment not found: ${pythonPath}`);
  process.exit(1);
}

console.log(`[OceanIQ] Platform: ${process.platform}`);
console.log(`[OceanIQ] Starting backend using: ${pythonPath}`);

const backend = spawn(
  pythonPath,
  [
    "-m",
    "uvicorn",
    "backend.main:app",
    "--reload",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
  ],
  {
    cwd: process.cwd(),
    stdio: "inherit",
  }
);

backend.on("error", (error) => {
  console.error("[OceanIQ] Failed to start backend:", error.message);
  process.exit(1);
});

backend.on("exit", (code, signal) => {
  if (signal) {
    console.log(`[OceanIQ] Backend stopped: ${signal}`);
  } else {
    console.log(`[OceanIQ] Backend exited with code: ${code}`);
  }

  process.exit(code ?? 0);
});