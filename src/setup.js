// First-run toolchain setup. Clones emsdk + Gamma, activates emsdk.
// Cached under cacheDir; subsequent runs no-op if everything is present.

import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";
import { promises as fs } from "node:fs";
import { join } from "node:path";

const exec = promisify(execFile);

const EMSDK_REPO = "https://github.com/emscripten-core/emsdk.git";
const GAMMA_REPO = "https://github.com/AlloSphere-Research-Group/Gamma.git";
const EMSDK_VERSION = "latest";

export async function ensureToolchain({ cacheDir }) {
  await fs.mkdir(cacheDir, { recursive: true });

  const emsdkDir = process.env.GAMMA_COMPILE_EMSDK || join(cacheDir, "emsdk");
  const gammaDir = process.env.GAMMA_COMPILE_GAMMA || join(cacheDir, "Gamma");

  await ensureGit();

  if (!(await dirExists(emsdkDir))) {
    console.log("→ Cloning emsdk → " + emsdkDir);
    await run("git", ["clone", "--depth", "1", EMSDK_REPO, emsdkDir]);
  } else {
    console.log("✓ emsdk already cloned");
  }

  // emsdk install + activate is idempotent (skips if already installed).
  console.log("→ Installing emsdk " + EMSDK_VERSION + " (~700 MB on first run)…");
  const emsdkBin = process.platform === "win32"
    ? join(emsdkDir, "emsdk.bat")
    : join(emsdkDir, "emsdk");
  await runStreaming(emsdkBin, ["install", EMSDK_VERSION], { cwd: emsdkDir });
  await runStreaming(emsdkBin, ["activate", EMSDK_VERSION], { cwd: emsdkDir });

  if (!(await dirExists(gammaDir))) {
    console.log("→ Cloning Gamma → " + gammaDir);
    await run("git", ["clone", "--depth", "1", GAMMA_REPO, gammaDir]);
  } else {
    console.log("✓ Gamma already cloned");
  }

  // Sanity check: the headers we expect should be there.
  const oscHeader = join(gammaDir, "Gamma", "Oscillator.h");
  if (!(await fileExists(oscHeader))) {
    throw new Error(
      "Gamma checkout looks incomplete — Oscillator.h missing at " + oscHeader +
      ". Try deleting " + gammaDir + " and re-running."
    );
  }

  console.log("✓ Toolchain ready");
  return { emsdkDir, gammaDir };
}

async function ensureGit() {
  try {
    await exec("git", ["--version"]);
  } catch (e) {
    throw new Error(
      "git not found on PATH. Install Git from https://git-scm.com/downloads " +
      "and re-run gamma-compile-server."
    );
  }
}

async function dirExists(p) {
  try { const s = await fs.stat(p); return s.isDirectory(); } catch (_) { return false; }
}
async function fileExists(p) {
  try { const s = await fs.stat(p); return s.isFile(); } catch (_) { return false; }
}

async function run(cmd, args, opts = {}) {
  const { stdout, stderr } = await exec(cmd, args, { ...opts, maxBuffer: 50 * 1024 * 1024 });
  if (stdout) process.stdout.write(stdout);
  if (stderr) process.stderr.write(stderr);
}

// Streaming variant for long-running commands (emsdk install) so the
// user sees progress instead of staring at a frozen terminal.
function runStreaming(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { ...opts, stdio: ["ignore", "inherit", "inherit"], shell: process.platform === "win32" });
    p.on("error", reject);
    p.on("close", code => {
      if (code === 0) resolve();
      else reject(new Error(cmd + " exited " + code));
    });
  });
}
