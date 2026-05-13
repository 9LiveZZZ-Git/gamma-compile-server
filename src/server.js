// HTTP server that exposes a single /compile endpoint plus /health.
// Default deployment binds to loopback and accepts requests from the
// GitHub Pages editor + a handful of localhost dev origins. The CLI
// can extend either dimension (--host 0.0.0.0, --allowOrigin <url>) to
// support LAN setups (e.g. patch on an iPad against a Mac daemon).

import express from "express";
import cors from "cors";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { compile } from "./compile.js";
import { startOscBridge } from "./osc-bridge.js";
import { probeEngine } from "./rt-engine-host.js";
import { attachRtProxy } from "./rt-proxy.js";

// Read package version once at startup so /health reports the actual
// running version. Better than hardcoding a string that drifts on bumps.
const __dirname = dirname(fileURLToPath(import.meta.url));
const PKG_VERSION = (() => {
  try {
    return JSON.parse(readFileSync(join(__dirname, "..", "package.json"), "utf8")).version || "0.0.0";
  } catch (_) { return "0.0.0"; }
})();

const DEFAULT_ALLOWED_ORIGINS = [
  "https://9livezzz-git.github.io",
  "http://localhost:8080",   // for local dev of the editor
  "http://localhost:5173",
  "http://localhost:3000",
  "http://127.0.0.1:8080",
  "http://127.0.0.1:5173",
  "http://127.0.0.1:3000",
];

export async function startServer({
  port, host, extraOrigins, toolchain, cacheDir,
  // OSC bridge options (all optional; bridge is on by default).
  osc = true, oscInPort = 9000, oscOutHost = "127.0.0.1", oscOutPort = 9001
}) {
  const bindHost = host || "127.0.0.1";
  // Normalize extra origins (strip trailing slash, drop blanks).
  const cleanedExtras = (extraOrigins || [])
    .map(o => String(o).trim().replace(/\/+$/, ""))
    .filter(Boolean);
  const allowAny = cleanedExtras.includes("*");
  const allowedOrigins = new Set([
    ...DEFAULT_ALLOWED_ORIGINS,
    ...cleanedExtras.filter(o => o !== "*")
  ]);
  const app = express();

  // Chrome's Private Network Access policy (Chrome 130+) blocks
  // fetches from public HTTPS origins to loopback / private network
  // addresses unless the server explicitly opts in via this header
  // on preflight responses. Without it, the browser rejects the
  // request with "Permission was denied for this request to access
  // the `loopback` address space" before any of our cors middleware
  // gets to weigh in. Set it on every response — harmless when not a
  // PNA preflight, required when it is.
  app.use((req, res, next) => {
    res.setHeader("Access-Control-Allow-Private-Network", "true");
    next();
  });

  app.use(cors({
    origin: (origin, cb) => {
      // No origin = curl / direct hit. Allow.
      if (!origin) return cb(null, true);
      if (allowAny) return cb(null, true);
      if (allowedOrigins.has(origin)) return cb(null, true);
      // Anything else is rejected. Use --allowOrigin <url> on the CLI
      // (or `*` for any origin) when self-hosting the editor.
      cb(new Error("CORS: origin not allowed: " + origin));
    },
    methods: ["GET", "POST", "OPTIONS"],
    // Standard PNA + preflight headers we want to be explicit about so
    // Chrome doesn't reject on missing Allow-Headers etc.
    allowedHeaders: ["Content-Type", "Access-Control-Allow-Private-Network"]
  }));

  app.use(express.json({ limit: "2mb" }));

  // Sprint 7.5.6.a part 1 -- probe the rt-engine binary at server
  // start. The result is reported back via /health so the editor
  // knows whether to enable the RayTracedScene node. Falls through
  // silently if the engine isn't installed (it's optional).
  let rtEngineInfo = null;
  try {
    rtEngineInfo = await probeEngine(cacheDir);
  } catch (e) {
    console.warn("[rt-engine] probe threw:", e && e.message);
  }

  app.get("/health", (req, res) => {
    res.json({
      ok: true,
      service: "gamma-compile-server",
      version: PKG_VERSION,
      toolchain: { emsdkDir: toolchain.emsdkDir, gammaDir: toolchain.gammaDir },
      // OSC capability surface so the editor can probe whether the
      // running daemon has the bridge enabled before attempting to
      // open the WebSocket. Editor uses this to decide whether to
      // light up the OSC connection UI.
      osc: osc
        ? { enabled: true, wsPath: "/osc", inPort: oscInPort,
            defaultOut: { host: oscOutHost, port: oscOutPort } }
        : { enabled: false },
      // Sprint 7.5.6.a part 1 -- RT engine availability. The
      // editor's RayTracedScene node reads this to decide whether
      // to offer RT rendering or fall back to raster Scene.
      // proxyReady flipped true in part 2d -- the /rt WebSocket
      // proxy is now attached + can forward to the engine. (User
      // still has to start the engine binary manually for now;
      // auto-spawn is a follow-up.)
      rtEngine: rtEngineInfo
        ? { available: true,
            capabilities: rtEngineInfo.capabilities,
            wsPath: "/rt",
            proxyReady: true,
            enginePort: 9100 }
        : { available: false, proxyReady: true, enginePort: 9100 }
    });
  });

  app.post("/compile", async (req, res) => {
    const t0 = Date.now();
    const { wrappedSrc, optLevel } = req.body || {};
    if (typeof wrappedSrc !== "string" || !wrappedSrc.length) {
      return res.status(400).json({ error: "Missing wrappedSrc string in body" });
    }
    try {
      const result = await compile({ wrappedSrc, toolchain, optLevel });
      const ms = Date.now() - t0;
      if (result.error) {
        return res.status(422).json({
          error: result.error,
          stderr: result.stderr || "",
          elapsedMs: ms
        });
      }
      res.set("Content-Type", "application/wasm");
      res.set("X-Compile-Stderr", encodeURIComponent(result.stderr || "").slice(0, 8000));
      res.set("X-Compile-Elapsed-Ms", String(ms));
      res.send(result.wasm);
    } catch (err) {
      res.status(500).json({
        error: err && err.message || String(err),
        stack: err && err.stack || ""
      });
    }
  });

  const httpServer = app.listen(port, bindHost, async () => {
    const isLanBound = bindHost === "0.0.0.0" || bindHost === "::";
    const displayHost = isLanBound ? "<LAN>" : bindHost;
    const fmtLine = (s) => "│ " + s.padEnd(60) + " │";
    console.log("");
    console.log("┌" + "─".repeat(62) + "┐");
    console.log(fmtLine("gamma-compile-server listening on http://" + displayHost + ":" + port));
    console.log(fmtLine(""));
    if (isLanBound) {
      console.log(fmtLine("⚠ Bound to all interfaces — reachable from your LAN."));
      console.log(fmtLine("  Only do this on a trusted network."));
      console.log(fmtLine(""));
    }
    console.log(fmtLine("Open the editor:"));
    console.log(fmtLine("  https://9livezzz-git.github.io/Gamma-Node/"));
    console.log(fmtLine(""));
    console.log(fmtLine("Click ▶. The editor auto-detects this daemon and"));
    console.log(fmtLine("routes compile requests here."));
    console.log(fmtLine(""));

    // Sprint 7.5.6.a part 2d -- /rt WebSocket proxy. Forwards browser
    // editor traffic to the gamma-rt-engine running on its own local
    // port (9100 default). User must start the engine manually for
    // now; the proxy reports a clean error to the editor if the
    // engine isn't reachable. Same origin policy as the OSC bridge.
    try {
      attachRtProxy({
        httpServer,
        engineHost: "127.0.0.1",
        enginePort: 9100,
        allowedOrigins,
        allowAnyOrigin: allowAny
      });
      console.log(fmtLine("RT engine proxy:"));
      console.log(fmtLine("  ws:    ws://" + displayHost + ":" + port + "/rt"));
      console.log(fmtLine("  → engine at ws://127.0.0.1:9100/"));
      console.log(fmtLine(""));
    } catch (e) {
      console.log(fmtLine("⚠ RT proxy attach failed: " + (e && e.message || e)));
      console.log(fmtLine(""));
    }

    // OSC bridge — UDP listener + WebSocket fan-out. Same host as the
    // HTTP server; WS upgrade attaches to the already-listening
    // httpServer so /osc lives on the same port as /compile. The UDP
    // port is separate so external apps (TouchOSC, Reaper, Max) can
    // target it directly with their conventional OSC client.
    if (osc) {
      try {
        await startOscBridge({
          httpServer,
          oscInPort,
          oscOutHost,
          oscOutPort,
          bindHost,
          allowedOrigins,
          allowAnyOrigin: allowAny
        });
        console.log(fmtLine("OSC bridge:"));
        console.log(fmtLine("  inbound:  udp://" + (isLanBound ? "<LAN>" : bindHost) + ":" + oscInPort));
        console.log(fmtLine("  outbound: udp://" + oscOutHost + ":" + oscOutPort + " (default)"));
        console.log(fmtLine("  ws:       ws://" + displayHost + ":" + port + "/osc"));
        console.log(fmtLine(""));
      } catch (e) {
        console.log(fmtLine("⚠ OSC bridge failed to start:"));
        console.log(fmtLine("  " + (e && e.message || String(e))));
        console.log(fmtLine("  (compile path still works; use --noOsc to silence)"));
        console.log(fmtLine(""));
      }
    } else {
      console.log(fmtLine("OSC bridge: disabled (--noOsc)"));
      console.log(fmtLine(""));
    }

    if (cleanedExtras.length) {
      console.log(fmtLine("Extra allowed origins:"));
      for (const o of cleanedExtras) console.log(fmtLine("  " + o));
      console.log(fmtLine(""));
    }
    console.log(fmtLine("Cache: " + cacheDir.slice(0, 52)));
    console.log(fmtLine(""));
    console.log(fmtLine("Stop with Ctrl-C."));
    console.log("└" + "─".repeat(62) + "┘");
  });

  return httpServer;
}
