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
} else if (!existsSync(MODELS_DIR)) {
  // Warn at startup -- without the env var, /sprite-gen will 503 with
  // "venv missing" on the first request. Better to nag now.
  console.warn("[sd-route] no models dir at " + MODELS_DIR + " and GAMMA_MODELS_DIR unset.");
  console.warn("[sd-route]   if you installed elsewhere, set the env var before npm start:");
  console.warn("[sd-route]     export GAMMA_MODELS_DIR=/path/to/gamma-models    (mac/linux)");
  console.warn("[sd-route]     $env:GAMMA_MODELS_DIR = 'D:\\gamma-models'        (windows ps)");
  console.warn("[sd-route]   /sprite-gen will return 503 until the venv + model are found.");
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
    // Turbo distillation -- model is trained for 4-9 inference
    // steps; past that the scheduler extrapolates and the output
    // diverges (NaN / flat color) instead of getting "more detailed".
    // Server-side clamp protects users who carry over a 20-step
    // SDXL config when switching models.
    maxSteps: 9,
    defaultGuidance: 4.0,
    // 1024 is the model's actual training resolution. We were running
    // at 512 to cut M4 gen time in half, but the output looked muddy
    // and the subject filled only a tiny fraction of the post-downsample
    // sprite. 1024 takes ~4× longer per step but the quality jump is
    // dramatic -- pixel-art sprites need the LoRA + prompt to operate
    // at native res to land cleanly after nearest-neighbor downsample.
    defaultNative: 1024,
    defaultLora: null,
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
  // Generation progress, updated by parsing the worker's stdout
  // [progress] step=N total=M elapsed_ms=... lines and surfaced via
  // GET /sprite-gen/progress so the editor can drive a progress bar.
  progress: {
    busy: false,
    step: 0,
    total: 0,
    elapsedMs: 0,
    startedAt: 0,
    lastUpdate: 0,
  },
};

function _resetProgress(total) {
  state.progress.busy = true;
  state.progress.step = 0;
  state.progress.total = total | 0;
  state.progress.elapsedMs = 0;
  state.progress.startedAt = Date.now();
  state.progress.lastUpdate = Date.now();
}

function _finishProgress() {
  state.progress.busy = false;
  state.progress.lastUpdate = Date.now();
}

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
    // Parse [progress] step=N total=M elapsed_ms=K lines emitted by the
    // worker's callback_on_step_end. There may be several per chunk so
    // we walk all matches and take the latest values.
    const re = /\[progress\]\s+step=(\d+)\s+total=(\d+)\s+elapsed_ms=(\d+)/g;
    let m, last = null;
    while ((m = re.exec(text)) !== null) last = m;
    if (last) {
      state.progress.step      = parseInt(last[1], 10);
      state.progress.total     = parseInt(last[2], 10);
      state.progress.elapsedMs = parseInt(last[3], 10);
      state.progress.lastUpdate = Date.now();
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
    // Clamp dimensions to the model's native resolution. Diffusion
    // models fail catastrophically below ~256 px (output collapses to
    // a flat color); the editor's downsample step turns 768→32 cleanly,
    // so we always generate at native res regardless of what the
    // browser sent for the final sprite size.
    const native = cfg.defaultNative || 512;
    const reqW = Number(body.width)  || native;
    const reqH = Number(body.height) || native;
    const genW = reqW < 256 ? native : reqW;
    const genH = reqH < 256 ? native : reqH;
    let stepsReq = Number(body.steps) || cfg.defaultSteps || 20;
    if (cfg.maxSteps && stepsReq > cfg.maxSteps) {
      console.warn("[sd-route] " + model + ": clamping steps " + stepsReq
        + " -> " + cfg.maxSteps + " (turbo model diverges past max)");
      stepsReq = cfg.maxSteps;
    }
    const reqObj = {
      prompt:   body.prompt,
      negative: body.negative || "blurry, smooth, anti-aliased, low quality, watermark, jpeg artifacts",
      width:    genW,
      height:   genH,
      steps:    stepsReq,
      guidance: Number(body.guidance) || cfg.defaultGuidance || 7.0,
      seed:     (typeof body.seed === "number") ? body.seed : null,
      lora:     (body.lora === undefined) ? cfg.defaultLora : body.lora,
      lora_strength: (typeof body.lora_strength === "number")
        ? body.lora_strength : (cfg.defaultLoraStrength || 1.0)
    };
    _resetProgress(reqObj.steps);
    try {
      const resp = await callWorker(reqObj);
      if (!resp.ok) {
        _finishProgress();
        return res.status(500).json({ error: resp.error, trace: resp.trace });
      }
      // Return raw PNG so the browser can decode directly without a base64 hop.
      const buf = Buffer.from(resp.png_b64, "base64");
      res.set("Content-Type", "image/png");
      res.set("X-SD-Elapsed-Ms", String(resp.elapsed_ms));
      res.set("X-SD-Model", resp.model || model);
      res.set("X-SD-Device", resp.device || "?");
      _finishProgress();
      res.send(buf);
    } catch (e) {
      _finishProgress();
      res.status(500).json({ error: "sprite-gen failed: " + e.message });
    }
  });

  // Lightweight progress poll for the browser. Returns the latest values
  // the worker has reported via its callback_on_step_end stdout lines.
  // The route consumer polls this ~2-3 Hz during a generation; in
  // between steps the worker is silent so step won't advance, but
  // browser-side elapsed timers keep moving.
  app.get("/sprite-gen/progress", (req, res) => {
    res.json({
      busy:      state.progress.busy,
      step:      state.progress.step,
      total:     state.progress.total,
      elapsedMs: state.progress.elapsedMs,
      // Server-clock elapsed since the POST landed -- useful for
      // pre-step latency (warm-up, LoRA swap) where the worker hasn't
      // yet fired its first callback.
      serverElapsedMs: state.progress.startedAt
        ? (Date.now() - state.progress.startedAt) : 0,
      model: state.model,
      ready: state.ready,
    });
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
