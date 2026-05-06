// HTTP server that exposes a single /compile endpoint plus /health.
// Designed to be hit ONLY from the editor running on
// 9livezzz-git.github.io. CORS opens for that origin.

import express from "express";
import cors from "cors";
import { compile } from "./compile.js";

const ALLOWED_ORIGINS = [
  "https://9livezzz-git.github.io",
  "http://localhost:8080",   // for local dev of the editor
  "http://localhost:5173",
  "http://localhost:3000",
  "http://127.0.0.1:8080",
  "http://127.0.0.1:5173",
  "http://127.0.0.1:3000",
];

export async function startServer({ port, toolchain, cacheDir }) {
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
      if (ALLOWED_ORIGINS.includes(origin)) return cb(null, true);
      // Anything else is rejected. Add to ALLOWED_ORIGINS if you self-host.
      cb(new Error("CORS: origin not allowed: " + origin));
    },
    methods: ["GET", "POST", "OPTIONS"],
    // Standard PNA + preflight headers we want to be explicit about so
    // Chrome doesn't reject on missing Allow-Headers etc.
    allowedHeaders: ["Content-Type", "Access-Control-Allow-Private-Network"]
  }));

  app.use(express.json({ limit: "2mb" }));

  app.get("/health", (req, res) => {
    res.json({
      ok: true,
      service: "gamma-compile-server",
      version: "0.1.0",
      toolchain: { emsdkDir: toolchain.emsdkDir, gammaDir: toolchain.gammaDir }
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

  app.listen(port, "127.0.0.1", () => {
    console.log("");
    console.log("┌──────────────────────────────────────────────────────────────┐");
    console.log("│ gamma-compile-server listening on http://127.0.0.1:" + String(port).padEnd(8) + " │");
    console.log("│                                                              │");
    console.log("│ Open the editor:                                             │");
    console.log("│   https://9livezzz-git.github.io/Gamma-Node/                 │");
    console.log("│                                                              │");
    console.log("│ Click ▶. The editor auto-detects this daemon and routes     │");
    console.log("│ compile requests here. Compile time should drop to seconds.  │");
    console.log("│                                                              │");
    console.log("│ Cache: " + cacheDir.padEnd(54) + "│");
    console.log("│                                                              │");
    console.log("│ Stop with Ctrl-C.                                            │");
    console.log("└──────────────────────────────────────────────────────────────┘");
  });
}
