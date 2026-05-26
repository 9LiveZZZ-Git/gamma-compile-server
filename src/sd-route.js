// gamma-compile-server -- /sprite-gen route + persistent SD worker
// management. Spawns the Python worker on first /sprite-gen request,
// keeps it warm across subsequent calls. Talks via Unix socket on
// Mac/Linux, TCP fallback on Windows.
//
// Adding more models later (Flux.2, etc.):
//   - extend MODELS map with the model name + default sampling defaults
//   - run scripts/install-sd.sh with the new model arg to fetch weights
//   - SpriteCreator's "model" dropdown reads the same name and sends it
//     in the /sprite-gen body
//
// Lifecycle:
//   - Worker is NOT spawned on server start (preload cost = 30-60s)
//   - First /sprite-gen request triggers spawnWorker(model)
//   - Worker stays alive until server shutdown
//   - Switching models = killing current worker + spawning a new one

import { spawn } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import net from "node:net";
import os from "node:os";

const __dirname  = dirname(fileURLToPath(import.meta.url));
const SERVER_DIR = join(__dirname, "..");
// Honor GAMMA_MODELS_DIR so model weights can live off the server
// checkout (e.g. server in OneDrive, models on a different drive).
const MODELS_DIR = process.env.GAMMA_MODELS_DIR
  ? process.env.GAMMA_MODELS_DIR
  : join(SERVER_DIR, "models");
const VENV_DIR   = join(MODELS_DIR, "sd-venv");
const WORKER_PY  = join(__dirname, "sd-worker.py");
if (process.env.GAMMA_MODELS_DIR) {
  console.log("[sd-route] GAMMA_MODELS_DIR=" + MODELS_DIR);
}

// Per-OS socket path. Unix socket on Mac/Linux (~1µs RTT); TCP on
// Windows (Node's Unix-socket support there is iffy, plus most Win SD
// installs use WSL2 anyway).
const IS_WIN = process.platform === "win32";
const DEFAULT_SOCKET = IS_WIN ? "tcp:127.0.0.1:8766" : "/tmp/gamma-sd-worker.sock";

const MODELS = {
  "z-image-turbo": {
    label: "Z-Image-Turbo",
    defaultSteps: 9,
    defaultGuidance: 4.0,
    defaultNative: 768,
    defaultLora: "pixel-art-xl.safetensors",
    defaultLoraStrength: 1.0
  },
  "sdxl": {
    label: "SDXL + Pixel Art XL LoRA",
    defaultSteps: 20,
    defaultGuidance: 7.0,
    defaultNative: 1024,
    defaultLora: "pixel-art-xl.safetensors",
    defaultLoraStrength: 1.2
  },
  "flux2-klein": {
    label: "FLUX.2 klein (no LoRA yet)",
    defaultSteps: 8,
    defaultGuidance: 3.5,
    defaultNative: 768,
    defaultLora: null,
    defaultLoraStrength: 0
  }
};

// Per-server state. Single worker for now; the model + socket are
// recreated on the (rare) request that picks a different model.
const state = {
  worker: null,        // ChildProcess
  model: null,
  socket: DEFAULT_SOCKET,
  ready: false,        // worker has finished its preload (sees "listening" log)
  startingPromise: null,
};

function pythonBin() {
  // venv path is platform-specific
  const exe = IS_WIN ? "python.exe" : "python";
  const sub = IS_WIN ? "Scripts" : "bin";
  return join(VENV_DIR, sub, exe);
}

function ensureWorker(model) {
  if (!MODELS[model]) {
    throw new Error("unknown SD model: " + model + " (known: " + Object.keys(MODELS).join(", ") + ")");
  }
  if (state.worker && state.model === model) {
    return state.ready
      ? Promise.resolve(state.worker)
      : state.startingPromise;
  }
  if (state.worker && state.model !== model) {
    console.log("[sd-route] swapping model: " + state.model + " → " + model);
    try { state.worker.kill("SIGTERM"); } catch (_) {}
    state.worker = null;
    state.ready  = false;
  }
  const py = pythonBin();
  if (!existsSync(py)) {
    throw new Error(
      "Python venv missing at " + py + "\n" +
      "  run gamma-compile-server/scripts/install-sd.sh first"
    );
  }
  const modelDir = join(MODELS_DIR, model);
  if (!existsSync(modelDir)) {
    throw new Error(
      "Model not downloaded: " + modelDir + "\n" +
      "  run scripts/install-sd.sh " + model
    );
  }
  console.log("[sd-route] spawning worker model=" + model + " socket=" + state.socket);
  const proc = spawn(py, [WORKER_PY, "--socket", state.socket, "--model", model], {
    cwd: SERVER_DIR,
    stdio: ["ignore", "pipe", "pipe"]
  });
  state.worker = proc;
  state.model  = model;
  state.ready  = false;

  proc.stdout.on("data", (chunk) => {
    const text = chunk.toString();
    process.stdout.write("[sd-worker] " + text);
    if (text.includes("listening on")) {
      state.ready = true;
    }
  });
  proc.stderr.on("data", (chunk) => {
    process.stderr.write("[sd-worker:err] " + chunk.toString());
  });
  proc.on("exit", (code, sig) => {
    console.log("[sd-route] worker exited code=" + code + " sig=" + sig);
    if (state.worker === proc) {
      state.worker = null;
      state.ready  = false;
    }
  });

  state.startingPromise = new Promise((resolve, reject) => {
    const start = Date.now();
    const tick = setInterval(() => {
      if (state.ready) {
        clearInterval(tick);
        resolve(proc);
      } else if (!state.worker) {
        clearInterval(tick);
        reject(new Error("worker exited during startup"));
      } else if (Date.now() - start > 180000) {  // 3 min
        clearInterval(tick);
        reject(new Error("worker did not become ready within 3 minutes"));
      }
    }, 200);
  });
  return state.startingPromise;
}

function callWorker(reqObj) {
  return new Promise((resolve, reject) => {
    const isUnix = !state.socket.startsWith("tcp:");
    let client;
    if (isUnix) {
      client = net.connect(state.socket);
    } else {
      const [, host, port] = state.socket.match(/^tcp:([^:]+):(\d+)$/);
      client = net.connect(parseInt(port, 10), host);
    }
    const chunks = [];
    let timeout = setTimeout(() => {
      try { client.destroy(); } catch (_) {}
      reject(new Error("worker timeout after 180s"));
    }, 180000);
    client.on("connect", () => {
      client.write(JSON.stringify(reqObj) + "\n");
    });
    client.on("data", (b) => chunks.push(b));
    client.on("end", () => {
      clearTimeout(timeout);
      try {
        const text = Buffer.concat(chunks).toString("utf-8").trim();
        const resp = JSON.parse(text);
        resolve(resp);
      } catch (e) {
        reject(new Error("worker returned malformed JSON: " + e.message));
      }
    });
    client.on("error", (e) => {
      clearTimeout(timeout);
      reject(e);
    });
  });
}

export function attachSdRoute(app) {
  // List installed models + their defaults so the editor can show a
  // dropdown that's always in sync with what's actually on disk.
  app.get("/sprite-gen/info", (req, res) => {
    const installed = {};
    for (const name of Object.keys(MODELS)) {
      const modelDir = join(MODELS_DIR, name);
      installed[name] = {
        ...MODELS[name],
        installed: existsSync(join(modelDir, ".installed")) || existsSync(modelDir)
      };
    }
    res.json({
      ok: true,
      pythonVenv: existsSync(pythonBin()),
      models: installed,
      currentWorker: state.worker ? { model: state.model, ready: state.ready } : null,
      socket: state.socket
    });
  });

  // POST /sprite-gen  body: {model, prompt, negative?, width, height, steps?, guidance?, seed?, lora?, lora_strength?}
  // Returns: 200 image/png on success, 4xx/5xx JSON on error.
  app.post("/sprite-gen", async (req, res) => {
    const body = req.body || {};
    const model = body.model || "z-image-turbo";
    const cfg = MODELS[model] || {};
    if (typeof body.prompt !== "string" || !body.prompt.length) {
      return res.status(400).json({ error: "missing prompt string" });
    }
    try {
      await ensureWorker(model);
    } catch (e) {
      return res.status(503).json({ error: "worker unavailable: " + e.message });
    }
    const reqObj = {
      prompt:   body.prompt,
      negative: body.negative || "blurry, smooth, anti-aliased, low quality, watermark, jpeg artifacts",
      width:    Number(body.width)  || cfg.defaultNative || 512,
      height:   Number(body.height) || cfg.defaultNative || 512,
      steps:    Number(body.steps)  || cfg.defaultSteps  || 20,
      guidance: Number(body.guidance) || cfg.defaultGuidance || 7.0,
      seed:     (typeof body.seed === "number") ? body.seed : null,
      lora:     (body.lora === undefined) ? cfg.defaultLora : body.lora,
      lora_strength: (typeof body.lora_strength === "number")
        ? body.lora_strength : (cfg.defaultLoraStrength || 1.0)
    };
    try {
      const resp = await callWorker(reqObj);
      if (!resp.ok) {
        return res.status(500).json({ error: resp.error, trace: resp.trace });
      }
      // Return raw PNG so the browser can decode directly without a base64 hop.
      const buf = Buffer.from(resp.png_b64, "base64");
      res.set("Content-Type", "image/png");
      res.set("X-SD-Elapsed-Ms", String(resp.elapsed_ms));
      res.set("X-SD-Model", resp.model || model);
      res.set("X-SD-Device", resp.device || "?");
      res.send(buf);
    } catch (e) {
      res.status(500).json({ error: "sprite-gen failed: " + e.message });
    }
  });

  // Graceful shutdown -- kill the worker so the socket file is cleaned up.
  const shutdown = () => {
    if (state.worker) {
      try { state.worker.kill("SIGTERM"); } catch (_) {}
    }
  };
  process.once("beforeExit", shutdown);
  process.once("SIGINT",  () => { shutdown(); });
  process.once("SIGTERM", () => { shutdown(); });
}
