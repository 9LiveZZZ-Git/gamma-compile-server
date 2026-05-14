// rt-engine-host.js -- Node-side host integration for the
// gamma-rt-engine Rust binary. Discovers the engine, probes its
// capabilities, auto-spawns it as a child process so the user
// doesn't need a second terminal.
//
// Sprint 7.5.6.a part 1: discovery + probe.
// Sprint 7.5.6.a part 2h: auto-spawn + lifecycle + port-conflict
//   detection (don't trample a manual `gamma-rt-engine` instance).

import { spawn, execFile } from "node:child_process";
import { promisify } from "node:util";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { platform, arch } from "node:os";
import net from "node:net";

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
 *   3. PATH (if the user installed it globally)  -- not implemented yet
 *   4. Local dev: rt-engine/target/release/gamma-rt-engine (running
 *      from a checked-out monorepo with `cargo build --release` run) */
export function findEngineBinary(cacheDir) {
  if (process.env.GAMMA_RT_ENGINE && existsSync(process.env.GAMMA_RT_ENGINE)) {
    return process.env.GAMMA_RT_ENGINE;
  }
  const expectedName = ENGINE_BINARY_NAMES[platformKey()];
  if (expectedName) {
    const cached = join(cacheDir, "rt-engine", expectedName);
    if (existsSync(cached)) return cached;
  }
  // Local dev fallback: monorepo checkout with a built binary.
  const localBin = platform() === "win32" ? "gamma-rt-engine.exe" : "gamma-rt-engine";
  const devPath = join(process.cwd(), "rt-engine", "target", "release", localBin);
  if (existsSync(devPath)) return devPath;
  const devPathDebug = join(process.cwd(), "rt-engine", "target", "debug", localBin);
  if (existsSync(devPathDebug)) return devPathDebug;
  return null;
}

/* Run the engine in --probe mode + parse its JSON capabilities
 * report. Returns { binary, capabilities } or null on failure.
 * Cheap (~30ms) -- the engine binary exits immediately after writing
 * the JSON; doesn't bind a port. */
export async function probeEngine(cacheDir, logger = console) {
  const bin = findEngineBinary(cacheDir);
  if (!bin) {
    logger.log("[rt-engine] binary not found on this machine.");
    logger.log("[rt-engine] Build it: cd rt-engine && cargo build --release");
    logger.log("[rt-engine] Or set GAMMA_RT_ENGINE=/path/to/gamma-rt-engine to override.");
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

/* Check whether something is already listening on the given local
 * port. Used before auto-spawn so we don't trample a user's manual
 * `gamma-rt-engine` instance running in a separate terminal. Returns
 * true if the port is busy, false if free. Fast (~50ms max). */
export function isPortInUse(port, host = "127.0.0.1") {
  return new Promise((resolve) => {
    const sock = new net.Socket();
    let resolved = false;
    const done = (v) => { if (resolved) return; resolved = true; sock.destroy(); resolve(v); };
    sock.setTimeout(500);
    sock.once("connect", () => done(true));
    sock.once("timeout", () => done(false));
    sock.once("error", () => done(false));
    sock.connect(port, host);
  });
}

/* Spawn the engine in long-lived server mode. Pipes stdout/stderr
 * through `logger.log` with an `[rt-engine]` prefix so the user sees
 * the engine's INFO logs interleaved with the compile-server's own
 * without needing a second terminal.
 *
 * Returns a handle with:
 *   { child, port, wsUrl, stop() }
 * `stop()` sends SIGTERM and waits up to 2s; called by the server's
 * SIGINT/SIGTERM handler so Ctrl-C cleans up the engine cleanly.
 *
 * No auto-restart on crash -- if the engine panics, the user should
 * see the stack trace in the log and restart the compile-server. An
 * auto-restart loop would mask issues during early development. */
export function spawnEngine(binary, opts = {}) {
  const port = opts.port || 9100;
  const host = opts.host || "127.0.0.1";
  const backend = opts.backend || "auto";
  const logger = opts.logger || console;

  const child = spawn(binary, [
    "--port", String(port),
    "--host", host,
    "--backend", backend
  ], { stdio: ["ignore", "pipe", "pipe"] });

  // Forward stdout + stderr line-by-line with a clear prefix. Engine
  // uses env_logger which writes [INFO]/[WARN]/[ERROR] to stderr;
  // anything on stdout would be unstructured output (rare).
  const pipeLines = (stream, label) => {
    let buf = "";
    stream.on("data", (chunk) => {
      buf += chunk.toString();
      let nl;
      while ((nl = buf.indexOf("\n")) !== -1) {
        const line = buf.slice(0, nl).trimEnd();
        buf = buf.slice(nl + 1);
        if (line) logger.log("[rt-engine] " + line);
      }
    });
    stream.on("end", () => {
      if (buf.trim()) logger.log("[rt-engine] " + buf.trimEnd());
    });
  };
  pipeLines(child.stdout, "stdout");
  pipeLines(child.stderr, "stderr");

  child.on("error", (e) => {
    logger.warn("[rt-engine] spawn error: " + e.message);
  });
  child.on("exit", (code, signal) => {
    if (code === 0 || signal === "SIGTERM") {
      logger.log("[rt-engine] exited cleanly");
    } else {
      logger.warn("[rt-engine] exited unexpectedly code=" + code + " signal=" + signal);
      logger.warn("[rt-engine] not auto-restarting -- restart the compile-server to retry");
    }
  });

  logger.log("[rt-engine] spawned pid=" + child.pid + " on ws://" + host + ":" + port);

  return {
    child,
    port,
    wsUrl: "ws://" + host + ":" + port + "/",
    async stop() {
      return new Promise((resolve) => {
        if (child.exitCode !== null) { resolve(); return; }
        const onExit = () => resolve();
        child.once("exit", onExit);
        try { child.kill("SIGTERM"); } catch (_) {}
        // SIGKILL as fallback if SIGTERM is ignored for 2s.
        setTimeout(() => {
          if (child.exitCode === null) {
            logger.warn("[rt-engine] SIGTERM ignored; sending SIGKILL");
            try { child.kill("SIGKILL"); } catch (_) {}
          }
        }, 2000);
      });
    }
  };
}
