# gamma-compile-server

A tiny local Emscripten compile daemon for the [Gamma Node Editor](https://9livezzz-git.github.io/Gamma-Node/)'s real-time audio preview. Runs on `localhost:8765`. The editor auto-detects it and routes compile requests here instead of using the in-browser Wasmer clang (which OOMs on Gamma's templates).

**Why this exists:** in-browser C++ compilation via `@wasmer/sdk` is fundamentally too memory-constrained for Gamma's template-heavy headers — a single-pass compile + link of the demo patch hits ~4 GB and dies. Native Emscripten on a dev machine handles the same source in seconds.

**What it ships:** real Emscripten + the actual Gamma source. The wasm output is byte-identical-ish to what AlloLib Studio Online produces — full production fidelity.

**Plus, optionally:** a sibling **`gamma-rt-engine`** binary in `rt-engine/` (Rust + Metal-RT) that powers the editor's `RayTracedScene` node — hardware-accelerated path tracing with MetalFX denoise + upscale, streamed back to the editor over a separate WebSocket. Opt-in; the daemon spawns it on demand when the editor probes `/health` and the binary is present. M-series Macs only for now; Vulkan-RT for PC GPUs is in the roadmap. See "Hardware ray tracing" below.

## Quick start

Requires **Node 20+** and **git** on your PATH.

```bash
npx @9livezzz/gamma-compile-server
```

First run downloads ~700 MB (Emscripten SDK + Gamma source) into a cache directory. Subsequent runs start in seconds.

Then open the editor at https://9livezzz-git.github.io/Gamma-Node/ and click ▶. Status pill should read **`local-cli detected`** instead of falling back to the Wasmer path. Compile time per patch ≈ 5–15 seconds.

## CLI flags

```
gamma-compile-server [--port 8765] [--host 127.0.0.1]
                     [--allowOrigin <url>]... [--cacheDir <path>]
                     [--skipSetup] [--setupOnly]

  --port         HTTP port (default 8765).
  --host         Network interface to bind to (default 127.0.0.1, i.e.
                 loopback only). Use 0.0.0.0 to accept connections from
                 other devices on your LAN — see "LAN setup" below.
                 ⚠ Only do this on a trusted network: /compile compiles
                 whatever C++ you send it.
  --allowOrigin  Extra CORS origin (repeatable). Default whitelist is
                 the GitHub Pages editor + localhost dev ports. Pass
                 the URL you're serving the editor from when self-
                 hosting (e.g. http://192.168.1.42:8000), or "*" to
                 allow any origin.
  --cacheDir     Where to keep emsdk + Gamma. Defaults are:
                   Windows  %LOCALAPPDATA%\gamma-compile
                   macOS    ~/Library/Caches/gamma-compile
                   Linux    ~/.cache/gamma-compile
  --skipSetup    Skip toolchain check (point at pre-installed emsdk
                 via GAMMA_COMPILE_EMSDK env var).
  --setupOnly    Download + install the toolchain and exit, without
                 starting the server. Useful for installer scripts.
```

## LAN setup (e.g. patch on iPad → daemon on Mac)

The daemon binds to loopback by default — fine when the editor and the
daemon run on the same machine. To use the daemon from a phone or
tablet on the same network:

1. **On the host machine** (the one with the toolchain — typically a
   Mac or PC), bind the daemon to all interfaces and whitelist the URL
   you'll serve the editor from:

   ```bash
   gamma-compile-server --host 0.0.0.0 \
       --allowOrigin http://192.168.1.42:8000
   ```

2. **Serve the editor over plain HTTP** from the same host. The GitHub
   Pages copy is served over HTTPS, and browsers block fetches from
   HTTPS pages to non-localhost HTTP URLs (mixed content). Easiest
   workaround: clone the editor repo and run

   ```bash
   cd Gamma-Node
   python -m http.server 8000
   ```

3. **On the client device** (iPad, phone, second laptop), open
   `http://192.168.1.42:8000/gamma-node-editor.html`. Open ⚙ Settings,
   set **Compile server URL** to `http://192.168.1.42:8765`, hit
   **Test connection**, then **Save**.

Replace `192.168.1.42` with your host's actual LAN IP. The daemon's
startup banner shows when it's bound to all interfaces.

## Sprite generation (optional)

The `/sprite-gen` POST route runs a local Stable Diffusion model and
returns a PNG, used by the editor's **SpriteCreator** node (Sprite
Studio modal → backend = `Compile-server SD`). Bundled, open-source,
no third-party API keys.

**Default model:** [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
(Apache 2.0) + the
[Pixel Art XL LoRA](https://huggingface.co/nerijs/pixel-art-xl) for
sprite-friendly output. Other models slot in via `MODEL_REGISTRY` in
`src/sd-worker.py` — currently supports `z-image-turbo`, `sdxl`, and
`flux2-klein` (Flux is wired but the install script doesn't pull it
by default; pass `flux2-klein` as the first arg to install-sd.sh).

**Architecture:**

- `src/sd-worker.py` is a long-lived Python process that loads the model
  ONCE (~30–60s on Apple Silicon) then serves generation requests over
  a Unix socket (TCP on Windows). Subsequent generations skip the
  model-load cost.
- `src/sd-route.js` is the Express route. Spawns the worker on the
  first `/sprite-gen` request, keeps it warm. Returns raw PNG bytes.
- `scripts/install-sd.sh` is the one-shot installer: creates a Python
  venv at `models/sd-venv/`, installs `torch + diffusers + mlx`,
  downloads the model + LoRA from HuggingFace, runs a self-test.

**Common prerequisites (any OS):**

- Python 3.10+
- 15 GB free disk (model + venv)
- An HTTPS connection (~6–12 GB initial download)

**`GAMMA_MODELS_DIR` env var (recommended):** point the install scripts +
server at a custom directory for the model cache. Useful when the
compile-server checkout lives in iCloud / OneDrive / Dropbox (you
don't want 6+ GB of model weights syncing) or on a small system drive
(point at a roomier secondary drive). Honored by `install-sd.sh`,
`install-sd.ps1`, `src/sd-route.js`, and `src/sd-worker.py`:

```bash
# Linux / macOS
export GAMMA_MODELS_DIR=/Volumes/Workdrive/gamma-models
./scripts/install-sd.sh
npm start                                 # server reads same env var

# Windows PowerShell
$env:GAMMA_MODELS_DIR = "D:\gamma-models"
.\scripts\install-sd.ps1
npm start
```

### macOS (Apple Silicon recommended)

```bash
cd gamma-compile-server
chmod +x scripts/install-sd.sh
./scripts/install-sd.sh                    # default: Z-Image-Turbo
# or pick a specific model:
./scripts/install-sd.sh sdxl               # SDXL + Pixel Art XL LoRA
./scripts/install-sd.sh flux2-klein        # Flux.2 klein (no LoRA)
```

Apple Silicon (M1/M2/M3/M4) uses the Metal Performance Shaders (MPS)
backend via PyTorch — auto-detected, no extra setup. Z-Image-Turbo at
768×768 takes about 30–60 s on M4 Air 32 GB (model-load on first gen
is the slow part; subsequent gens are fast).

If you don't have `python3` on PATH, install via [python.org](https://www.python.org/downloads/macos/) or `brew install python@3.12`.

### Linux

Same as macOS:

```bash
cd gamma-compile-server
chmod +x scripts/install-sd.sh
./scripts/install-sd.sh
```

NVIDIA + CUDA is auto-detected and dramatically faster than CPU
(typically <10 s/sprite). On AMD GPUs install ROCm-flavored PyTorch
into the venv before running the script.

### Windows (PowerShell)

Native install via PowerShell (no WSL2 needed):

```powershell
# Prerequisites (install once):
#   - Python 3.10+ from python.org (check "Add Python to PATH")
#   - Git for Windows from git-scm.com

cd gamma-compile-server
.\scripts\install-sd.ps1                   # default: z-image-turbo
.\scripts\install-sd.ps1 sdxl              # alternative
.\scripts\install-sd.ps1 flux2-klein
```

NVIDIA GPUs are auto-detected and used. AMD / Intel iGPU falls back to
CPU (SDXL on CPU is impractical → pick z-image-turbo).

If PowerShell blocks the script, run as Administrator and:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### WSL2 fallback (Windows users with AMD/Intel GPU)

If the native Windows path isn't fast enough, run the Linux script
inside Ubuntu via WSL2:

```bash
# inside Ubuntu / WSL2:
cd /mnt/c/Users/<you>/.../gamma-compile-server
chmod +x scripts/install-sd.sh
./scripts/install-sd.sh
```

The Node.js compile-server side stays running on Windows; only the
Python SD worker lives in WSL. The worker's TCP-socket fallback
handles the cross-OS bridge.

**Status check (all OSes):**

```bash
curl http://127.0.0.1:8765/sprite-gen/info
# or in PowerShell:
Invoke-RestMethod http://127.0.0.1:8765/sprite-gen/info
```

Reports installed models, current worker state, socket path.

**Adding a new model (e.g. when the Flux.2 ecosystem matures):**

1. Add an entry to `MODEL_REGISTRY` in `src/sd-worker.py` with the
   local cache dir + a pipeline-factory function.
2. Add a matching entry to `MODELS` in `src/sd-route.js` with
   sampling defaults (steps, guidance, native size, LoRA name).
3. Add a `pull_model()` case in `scripts/install-sd.sh` with the
   HuggingFace repo name.
4. Run `./scripts/install-sd.sh <new-model-name>` to download weights.

## Hardware ray tracing (optional)

The `rt-engine/` sibling directory ships a separate Rust binary —
`gamma-rt-engine` — that handles the editor's `RayTracedScene` node.
True path tracing (glass, mirrors, area lights with soft shadows,
multi-bounce GI), denoised + upscaled by Apple's
`MTLFXTemporalDenoisedScaler`. Independent from the audio compile
path; you can run one without the other.

**Hardware support (current):**

| GPU                 | RT path                                                  | Status                                     |
|---------------------|----------------------------------------------------------|--------------------------------------------|
| Apple M3 / M4 / M5  | Metal-RT hardware traversal + MetalFX denoise+upscale    | Production-grade, 60 fps at `preview` 720p |
| Apple M1 / M2       | Metal-RT software traversal (MPS) + MetalFX              | Preview-quality at `draft` preset only     |
| Everything else     | RT node falls back to raster `Scene` in the editor       | Engine doesn't run                         |

PC Vulkan-RT (NVIDIA / AMD / Intel Arc) is on the roadmap but not
yet shipped — track `gamma-compile-server`'s `rt-engine/` sources
and the editor's `docs/RAYTRACING.md` phase plan.

### Building the engine

Requires **Rust 1.78+** and **Xcode Command Line Tools** (for the
Metal SDK headers).

```bash
cargo build --release --manifest-path rt-engine/Cargo.toml
```

The build embeds the Metal kernel (`triangle.metal`) via
`include_str!`, so any shader change requires a `cargo build`. The
binary lands at `rt-engine/target/release/gamma-rt-engine`.

### Running the engine

Two patterns work:

**A) Auto-spawn (simplest).** Just start the compile daemon — it
probes for the engine binary at the path above and spawns it as a
child process on first `/health` probe from the editor. Engine
lifetime = daemon lifetime. Logs come out mixed with the daemon's.

```bash
node bin/gamma-compile-server.js
# (or `npx @9livezzz/gamma-compile-server` once published)
```

**B) Run separately (better for iterating on engine code).** Start
the engine in its own terminal so its logs are clean; the daemon
detects port 9100 already in use and uses the existing instance.
You can Ctrl-C + restart the engine without touching the daemon.

```bash
# Terminal 1: engine
cargo run --release --manifest-path rt-engine/Cargo.toml

# Terminal 2: daemon
node bin/gamma-compile-server.js
```

Engine listens on `ws://127.0.0.1:9100/`. The editor probes
`/health` on the daemon (port 8765) to discover the engine's port,
then connects **directly** to the engine WebSocket. The daemon
isn't in the per-frame RT path at all — it only does the initial
spawn + capability advertisement.

> ⚠ **After a `cargo build` you must restart the engine process.**
> The daemon has no way to know your binary on disk is newer than
> the running process, so if you forget this step the daemon's
> "port already in use; assuming external instance" log line is
> the symptom — kill the orphan (`pkill gamma-rt-engine` or
> `lsof -ti :9100 | xargs kill`) and start the new binary.

### What the editor sees

The editor's `RayTracedScene` node has the same input shape as the
raster `Scene` (4 meshes / camera / 4 lights / clear color) plus
quality knobs: `quality` preset (draft / preview / final → 1/4/16
spp + 2/4/8 bounces), `displaySize` (480p–1080p), and `renderScale`
(native / quality / balanced / performance / ultra — DLSS/FSR
convention; the kernel shades at that fraction and MetalFX upscales
to display dims). When the engine is unreachable, the node renders
status-coded fallback colors so the user knows what's wrong without
opening dev tools — see the editor's README "Hardware ray tracing"
section for the color key.

### Engine-side state

Each connected editor session gets its own `MetalRenderer` with
private G-buffer textures, TDS scaler, and acceleration structure.
The renderer is rebuilt on `configure` (dims / scale change) or
on dropped WS. Path-tracing accumulation resets on any scene,
camera, light, material, or quality change so TDS history doesn't
mix pre-/post-change samples.

## How the editor finds it

On first Play click, the editor does:

```js
const probe = await fetch("http://localhost:8765/health", { signal: AbortSignal.timeout(200) }).catch(() => null);
if (probe && probe.ok) usingLocalCli = true;
```

If the daemon is running, the editor POSTs the wrapped patch C++ to `http://localhost:8765/compile` and gets back the compiled WASM bytes. If not, it falls back to the in-browser Wasmer path (or the JS reimpl when that lands).

## Cache directory layout

```
<cacheDir>/
  emsdk/                       Cloned from emscripten-core/emsdk
  Gamma/                       Cloned from AlloSphere-Research-Group/Gamma
  Gamma/.libgamma-cache/       Per-session cached libgamma.a
                               (rebuilt on first compile each time
                               the server is restarted)
```

To wipe and re-download: delete the cache directory and re-run.

## Architecture

`bin/gamma-compile-server.js` parses CLI flags and either runs setup (download emsdk + Gamma) or skips and starts the server.

`src/setup.js` clones repos, runs `emsdk install latest && emsdk activate latest`. All idempotent.

`src/compile.js` wraps `em++`. On first request it builds `libgamma.a` from the 11 web-buildable Gamma sources (same set AlloLib Studio Online uses). Subsequent requests link the new patch against the cached library — quick.

`src/server.js` is an Express app with two routes:
- `GET /health` — liveness probe + version + toolchain paths + RT engine port advertisement (if running).
- `POST /compile` — body `{ wrappedSrc, optLevel }`, returns `application/wasm` bytes plus stderr in headers, or JSON error with stderr inline.

`rt-engine/` is the optional native ray-tracer (Rust + Metal-RT). The daemon spawns it on demand and surfaces its port in `/health` so the editor can connect directly. Engine sources live under `rt-engine/src/`; the entry point is `rt-engine/src/main.rs`.

CORS allows the editor's origin (`9livezzz-git.github.io`) plus common localhost ports for local development.

## Troubleshooting

**"git not found"** — install Git from https://git-scm.com/downloads, restart terminal.

**Port 8765 in use** — pass `--port 9000` or kill whatever's listening (`netstat -ano | findstr 8765` on Windows).

**emsdk install fails on Windows with permission errors** — run the terminal as Administrator just for the first run. Subsequent starts don't need admin.

**Editor says "local-cli detected" but compiles fail** — check the daemon's terminal output; emcc errors are printed inline.

**RT engine startup says "port 9100 already in use; assuming external instance"** — there's an orphaned `gamma-rt-engine` from a previous run. Kill it (`pkill gamma-rt-engine` or `lsof -ti :9100 | xargs kill` on macOS) and restart the daemon so it spawns the freshly-built binary.

**`RayTracedScene` node viewport is solid red / crimson / amber** — engine-side problem. See the editor's README "Troubleshooting" subsection of "Hardware ray tracing" — each color maps to a specific failure mode.

## License

MIT.
