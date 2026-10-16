const { spawn } = require("child_process");
const path = require("path");

const pythonPath =
  process.platform === "win32"
    ? path.join("venv", "Scripts", "python.exe")
    : path.join("venv", "bin", "python");

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
  console.error(
    "[OceanIQ] Failed to start backend:",
    error.message
  );
  process.exit(1);
});

backend.on("exit", (code, signal) => {
  if (signal) {
    console.log(`[OceanIQ] Backend stopped by signal: ${signal}`);
  } else {
    console.log(`[OceanIQ] Backend exited with code: ${code}`);
  }

  process.exit(code ?? 0);
});