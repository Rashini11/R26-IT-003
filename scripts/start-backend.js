const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const projectRoot = path.resolve(__dirname, "..");
const pythonExecutable =
  process.platform === "win32" ? "python.exe" : "python";
const pythonRelativePath =
  process.platform === "win32"
    ? path.join("Scripts", pythonExecutable)
    : path.join("bin", pythonExecutable);
const pythonPath = [".venv", "venv"]
  .map((environment) => path.join(projectRoot, environment, pythonRelativePath))
  .find((candidate) => fs.existsSync(candidate));

if (!pythonPath) {
  console.error(
    `[OceanIQ] Python virtual environment not found. Checked .venv and venv under ${projectRoot}.`
  );
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
    cwd: projectRoot,
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