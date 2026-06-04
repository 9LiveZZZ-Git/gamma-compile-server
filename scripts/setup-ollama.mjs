#!/usr/bin/env node
// Post-install Ollama CORS setup.
//
// Runs automatically after `npm install` via the package.json
// "postinstall" hook. Idempotent: detects whether OLLAMA_ORIGINS is
// already configured with the editor's origins and exits silently if
// so. Otherwise prints what it's doing, sets the env var via the
// platform-appropriate mechanism (launchctl on macOS, systemd override
// on Linux, setx on Windows), and prints restart instructions.
//
// Never installs Ollama itself (that's an upstream decision; we just
// link to ollama.com/download in the not-installed branch). Never
// crashes the npm install on failure -- prints a hint and exits 0.

import { spawnSync, execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// Editor origins that need CORS access to the local Ollama daemon.
// Mirrors the DEFAULT_ALLOWED_ORIGINS list in src/server.js so the same
// dev-loop URLs work for both the compile server's /compile route and
// for Ollama's /api/* routes.
const EDITOR_ORIGINS = [
  "https://9livezzz-git.github.io",
  "http://localhost:8000",
  "http://localhost:8080",
  "http://localhost:5173",
  "http://localhost:3000",
  "http://127.0.0.1:8000",
  "http://127.0.0.1:8080",
  "http://127.0.0.1:5173",
  "http://127.0.0.1:3000",
];
const TARGET_ORIGINS = EDITOR_ORIGINS.join(",");

// One-line console-friendly check: is the value we'd set already
// effectively present (either an exact match, or "*", or at least
// includes the GitHub Pages origin which is the load-bearing one)?
function originsAlreadyConfigured(current) {
  if (!current) return false;
  if (current.trim() === "*") return true;
  return current.includes("https://9livezzz-git.github.io");
}

function ollamaInstalled() {
  try {
    const r = spawnSync("ollama", ["--version"], { stdio: "pipe", timeout: 2000 });
    return r.status === 0;
  } catch (_) {
    return false;
  }
}

function getCurrentOriginsMac() {
  try {
    const r = spawnSync("launchctl", ["getenv", "OLLAMA_ORIGINS"], { stdio: "pipe", timeout: 2000 });
    if (r.status === 0) return (r.stdout || Buffer.alloc(0)).toString().trim();
  } catch (_) {}
  return "";
}

function getCurrentOriginsLinux() {
  const candidates = [
    "/etc/systemd/system/ollama.service.d/override.conf",
    "/etc/systemd/system/ollama.service.d/gamma-compile-server.conf",
  ];
  for (const path of candidates) {
    try {
      if (!existsSync(path)) continue;
      const txt = readFileSync(path, "utf8");
      const m = txt.match(/Environment="OLLAMA_ORIGINS=([^"]+)"/);
      if (m) return m[1];
    } catch (_) {}
  }
  return "";
}

function getCurrentOriginsWin() {
  try {
    const r = spawnSync(
      "powershell",
      ["-NoProfile", "-Command", "[Environment]::GetEnvironmentVariable('OLLAMA_ORIGINS', 'User')"],
      { stdio: "pipe", timeout: 4000 }
    );
    if (r.status === 0) return (r.stdout || Buffer.alloc(0)).toString().trim();
  } catch (_) {}
  return "";
}

function setOriginsMac() {
  try {
    spawnSync("launchctl", ["setenv", "OLLAMA_ORIGINS", TARGET_ORIGINS], { stdio: "ignore", timeout: 4000 });
    console.log("    ✓ Set OLLAMA_ORIGINS via launchctl.");
    console.log("");
    console.log("    ⚠ Restart Ollama for it to pick up the new origins:");
    console.log("        1. Menu bar llama icon → Quit Ollama");
    console.log("        2. Open Ollama from Applications");
    console.log("");
    console.log("    To make this persistent across reboots, append to ~/.zprofile:");
    console.log(`        launchctl setenv OLLAMA_ORIGINS "${TARGET_ORIGINS}"`);
    return true;
  } catch (e) {
    console.log("    ✗ launchctl setenv failed: " + (e && e.message || e));
    return false;
  }
}

function setOriginsLinux() {
  const dropInDir = "/etc/systemd/system/ollama.service.d";
  const overridePath = join(dropInDir, "gamma-compile-server.conf");
  const content =
    "# Written by @9livezzz/gamma-compile-server postinstall.\n" +
    "# Adds the editor's origins to Ollama's CORS allowlist so the\n" +
    "# browser editor can call /api/* directly. Delete this file to\n" +
    "# revert; re-run `npm install` to regenerate.\n" +
    "[Service]\n" +
    `Environment="OLLAMA_ORIGINS=${TARGET_ORIGINS}"\n`;
  try {
    if (!existsSync(dropInDir)) mkdirSync(dropInDir, { recursive: true });
    writeFileSync(overridePath, content);
    console.log("    ✓ Wrote " + overridePath);
    console.log("");
    console.log("    Apply the change with:");
    console.log("        sudo systemctl daemon-reload");
    console.log("        sudo systemctl restart ollama");
    return true;
  } catch (e) {
    console.log("    ✗ Could not write " + overridePath + " (need root):");
    console.log("      " + (e && e.message || e));
    console.log("");
    console.log("    Run manually:");
    console.log("        sudo mkdir -p " + dropInDir);
    console.log(
      "        sudo tee " + overridePath + " <<'EOF'\n" +
      content +
      "EOF"
    );
    console.log("        sudo systemctl daemon-reload && sudo systemctl restart ollama");
    return false;
  }
}

function setOriginsWin() {
  try {
    const r = spawnSync(
      "setx",
      ["OLLAMA_ORIGINS", TARGET_ORIGINS],
      { stdio: "ignore", timeout: 4000 }
    );
    if (r.status !== 0) throw new Error("setx exit " + r.status);
    console.log("    ✓ Set OLLAMA_ORIGINS in User environment (setx).");
    console.log("");
    console.log("    ⚠ Restart Ollama for it to pick up the new origins:");
    console.log("        1. System tray (^ chevron) → llama icon → Quit");
    console.log("        2. Start menu → search 'Ollama' → relaunch");
    return true;
  } catch (e) {
    console.log("    ✗ setx failed: " + (e && e.message || e));
    return false;
  }
}

function main() {
  console.log("");
  console.log("── @9livezzz/gamma-compile-server :: Ollama setup ─────────────");

  if (!ollamaInstalled()) {
    console.log("    (Ollama not installed — skipping CORS setup)");
    console.log("");
    console.log("    To enable LLM nodes in the Gamma Node editor:");
    console.log("        1. Install Ollama: https://ollama.com/download");
    console.log("        2. Pull a model:  ollama pull llama3.2");
    console.log("        3. Re-run:        npm install");
    console.log("           (This script reruns and configures CORS for the editor.)");
    console.log("───────────────────────────────────────────────────────────────");
    console.log("");
    return;
  }

  let current = "";
  if      (process.platform === "darwin") current = getCurrentOriginsMac();
  else if (process.platform === "linux")  current = getCurrentOriginsLinux();
  else if (process.platform === "win32")  current = getCurrentOriginsWin();

  if (originsAlreadyConfigured(current)) {
    console.log("    ✓ Ollama CORS already configured for the editor — no changes.");
    if (current.length <= 80) console.log("      OLLAMA_ORIGINS = " + current);
    else                      console.log("      OLLAMA_ORIGINS = " + current.slice(0, 80) + "…");
    console.log("───────────────────────────────────────────────────────────────");
    console.log("");
    return;
  }

  console.log("    Configuring Ollama to accept requests from the editor…");
  console.log("");
  console.log("    Target origins:");
  console.log("        " + TARGET_ORIGINS);
  console.log("");

  let ok = false;
  if      (process.platform === "darwin") ok = setOriginsMac();
  else if (process.platform === "linux")  ok = setOriginsLinux();
  else if (process.platform === "win32")  ok = setOriginsWin();
  else {
    console.log("    ⚠ Unknown platform: " + process.platform);
    console.log("      Set OLLAMA_ORIGINS manually to the value above.");
  }

  if (ok) console.log("    ✓ Setup complete (after the restart above).");

  console.log("───────────────────────────────────────────────────────────────");
  console.log("");
}

// Never let this crash the npm install. Always exit 0 + log any error.
try {
  main();
} catch (e) {
  console.log("");
  console.log("── @9livezzz/gamma-compile-server :: Ollama setup ─────────────");
  console.log("    ✗ Setup script crashed (continuing without configuring):");
  console.log("      " + (e && e.message || e));
  console.log("    You can configure CORS manually — see README.md.");
  console.log("───────────────────────────────────────────────────────────────");
  console.log("");
  process.exit(0);
}
