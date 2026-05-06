# gamma-compile-server

A tiny local Emscripten compile daemon for the [Gamma Node Editor](https://9livezzz-git.github.io/Gamma-Node/)'s real-time audio preview. Runs on `localhost:8765`. The editor auto-detects it and routes compile requests here instead of using the in-browser Wasmer clang (which OOMs on Gamma's templates).

**Why this exists:** in-browser C++ compilation via `@wasmer/sdk` is fundamentally too memory-constrained for Gamma's template-heavy headers — a single-pass compile + link of the demo patch hits ~4 GB and dies. Native Emscripten on a dev machine handles the same source in seconds.

**What it ships:** real Emscripten + the actual Gamma source. The wasm output is byte-identical-ish to what AlloLib Studio Online produces — full production fidelity.

## Quick start

Requires **Node 20+** and **git** on your PATH.

```bash
npx @9livezzz/gamma-compile-server
```

First run downloads ~700 MB (Emscripten SDK + Gamma source) into a cache directory. Subsequent runs start in seconds.

Then open the editor at https://9livezzz-git.github.io/Gamma-Node/ and click ▶. Status pill should read **`local-cli detected`** instead of falling back to the Wasmer path. Compile time per patch ≈ 5–15 seconds.

## CLI flags

```
gamma-compile-server [--port 8765] [--cacheDir <path>]
                     [--skipSetup] [--setupOnly]

  --port       HTTP port (default 8765).
  --cacheDir   Where to keep emsdk + Gamma. Defaults are:
                 Windows  %LOCALAPPDATA%\gamma-compile
                 macOS    ~/Library/Caches/gamma-compile
                 Linux    ~/.cache/gamma-compile
  --skipSetup  Skip toolchain check (point at pre-installed emsdk
               via GAMMA_COMPILE_EMSDK env var).
  --setupOnly  Download + install the toolchain and exit, without
               starting the server. Useful for installer scripts.
```

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
- `GET /health` — liveness probe + version + toolchain paths
- `POST /compile` — body `{ wrappedSrc, optLevel }`, returns `application/wasm` bytes plus stderr in headers, or JSON error with stderr inline.

CORS allows the editor's origin (`9livezzz-git.github.io`) plus common localhost ports for local development.

## Troubleshooting

**"git not found"** — install Git from https://git-scm.com/downloads, restart terminal.

**Port 8765 in use** — pass `--port 9000` or kill whatever's listening (`netstat -ano | findstr 8765` on Windows).

**emsdk install fails on Windows with permission errors** — run the terminal as Administrator just for the first run. Subsequent starts don't need admin.

**Editor says "local-cli detected" but compiles fail** — check the daemon's terminal output; emcc errors are printed inline.

## License

MIT.
