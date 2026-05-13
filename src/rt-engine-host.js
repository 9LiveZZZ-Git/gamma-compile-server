// rt-engine-host.js -- Node-side host integration for the
// gamma-rt-engine Rust binary. Discovers the engine, probes its
// capabilities + proxies WebSocket traffic between the browser
// editor and the engine process.
//
// Sprint 7.5.6.a part 1: discovery + probe + spawn. Frame-streaming
// proxy lands in part 2 when the engine has real frames to send.

import { spawn, execFile } from "node:child_process";
import { promisify } from "node:util";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { platform, arch } from "node:os";

const execFileAsync = promisify(execFile);

const ENGINE_BINARY_NAMES = {
  "win32:x64":     "gamma-rt-engine-windows-x86_64.exe",
  "linux:x64":     "gamma-rt-engine-linux-x86_64",
  "darwin:arm64":  "gamma-rt-engine-macos-arm64",
  "darwin:x64":    "gamma-rt-engine-macos-x86_64"
};

function platformKey() {
  return platform() + ":" + arch();
}

/* Locate the engine binary. Search order:
 *   1. GAMMA_RT_ENGINE env var (explicit override)
 *   2. <cacheDir>/rt-engine/<binary>   (the auto-fetch location)
 *   3. PATH (if the user installed it globally)
 *   4. Local dev: rt-engine/target/release/gamma-rt-engine (running
 *      from a checked-out monorepo with `cargo build` already run) */
export function findEngineBinary(cacheDir) {
  if (process.env.GAMMA_RT_ENGINE && existsSync(process.env.GAMMA_RT_ENGINE)) {
    return process.env.GAMMA_RT_ENGINE;
  }
  const expectedName = ENGINE_BINARY_NAMES[platformKey()];
  if (expectedName) {
    const cached = join(cacheDir, "rt-engine", expectedName);
    if (existsSync(cached)) return cached;
  }
  // Local dev fallback: monorepo checkout with a built debug binary.
  const localBin = platform() === "win32" ? "gamma-rt-engine.exe" : "gamma-rt-engine";
  const devPath = join(process.cwd(), "rt-engine", "target", "release", localBin);
  if (existsSync(devPath)) return devPath;
  const devPathDebug = join(process.cwd(), "rt-engine", "target", "debug", localBin);
  if (existsSync(devPathDebug)) return devPathDebug;
  return null;
}

/* Run the engine in --probe mode + parse its JSON capabilities
 * report. Returns the parsed object, or null if probe fails or the
 * binary isn't found. The Node side runs this on startup so the
 * /health endpoint can surface RT availability to the editor. */
export async function probeEngine(cacheDir, logger = console) {
  const bin = findEngineBinary(cacheDir);
  if (!bin) {
    logger.log("[rt-engine] not found on this machine. " +
      "Install via `cargo install` or by downloading the binary release " +
      "from the gamma-compile-server GitHub releases.");
    return null;
  }
  try {
    const { stdout } = await execFileAsync(bin, ["--probe"], { timeout: 5000 });
    const caps = JSON.parse(stdout);
    logger.log("[rt-engine] probed:",
      `vulkan_rt=${caps.vulkan_rt}`,
      `metal_rt_hardware=${caps.metal_rt_hardware}`,
      `os=${caps.os}`);
    return { binary: bin, capabilities: caps };
  } catch (e) {
    logger.warn("[rt-engine] probe failed:", e && e.message);
    return null;
  }
}

/* Spawn the engine in long-lived server mode. Returns a handle with
 * { child, port, wsUrl, stop() }. The Node side opens its own WS to
 * the engine + proxies between the editor's /rt endpoint and the
 * engine's local socket.
 *
 * Sprint 7.5.6.a part 1: spawn + log lifecycle. The browser-side
 * /rt proxy lands in part 2. */
export function spawnEngine(binary, opts = {}) {
  const port = opts.port || 9100;
  const host = opts.host || "127.0.0.1";
  const backend = opts.backend || "auto";
  const logger = opts.logger || console;

  const child = spawn(binary, [
    "--port", String(port),
    "--host", host,
    "--backend", backend
  ], { stdio: ["ignore", "inherit", "inherit"] });

  child.on("error", (e) => logger.warn("[rt-engine] spawn error:", e.message));
  child.on("exit", (code, signal) =>
    logger.log("[rt-engine] exited code=" + code + " signal=" + signal));

  logger.log("[rt-engine] spawned pid=" + child.pid + " on ws://" + host + ":" + port);

  return {
    child,
    port,
    wsUrl: "ws://" + host + ":" + port + "/",
    stop() {
      try { child.kill(); } catch (_) {}
    }
  };
}
