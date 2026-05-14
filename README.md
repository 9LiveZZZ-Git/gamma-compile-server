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
