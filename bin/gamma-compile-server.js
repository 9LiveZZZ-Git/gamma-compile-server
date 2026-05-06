#!/usr/bin/env node
// Entry point. Parses CLI flags, runs first-run setup, starts the server.

import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ensureToolchain } from "../src/setup.js";
import { startServer } from "../src/server.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

const { values: opts } = parseArgs({
  options: {
    port:        { type: "string",  default: "8765" },
    cacheDir:    { type: "string",  default: "" },         // default resolved per-OS
    skipSetup:   { type: "boolean", default: false },      // assume toolchain present
    setupOnly:   { type: "boolean", default: false },      // download & exit
    help:        { type: "boolean", short: "h", default: false },
    version:     { type: "boolean", short: "v", default: false }
  },
  allowPositionals: false
});

if (opts.help) {
  console.log(`gamma-compile-server — local Emscripten compile daemon for the
Gamma Node Editor's real-time audio preview.

Usage: gamma-compile-server [--port 8765] [--cacheDir <path>]
                            [--skipSetup] [--setupOnly]

  --port       HTTP port to listen on (default 8765).
  --cacheDir   Where to cache emsdk + Gamma source.
               Default: %LOCALAPPDATA%\\gamma-compile (Windows) or
               ~/.cache/gamma-compile (macOS/Linux).
  --skipSetup  Skip the first-run toolchain check (use if you've
               manually pointed the daemon at a pre-installed emsdk
               via GAMMA_COMPILE_EMSDK env var).
  --setupOnly  Download + verify the toolchain and exit, without
               starting the server. Useful for installer scripts.
  --help       Show this message.
  --version    Print version and exit.

The editor (https://9livezzz-git.github.io/Gamma-Node/) detects this
daemon by polling localhost:<port>/health on first Play click. If
present, the editor routes compile requests here instead of using
the in-browser Wasmer clang. Compile times are ~5–15 s per patch
(vs the in-browser path's many minutes / OOM).

First run downloads ~700 MB (emsdk + Gamma source) into the cache
directory. Subsequent starts are instant.`);
  process.exit(0);
}

if (opts.version) {
  // Read version from package.json
  const pkg = JSON.parse(
    await (await import("node:fs/promises")).readFile(
      join(__dirname, "..", "package.json"),
      "utf8"
    )
  );
  console.log(pkg.version);
  process.exit(0);
}

const port = parseInt(opts.port, 10);
if (!Number.isFinite(port) || port < 1 || port > 65535) {
  console.error("✗ Invalid --port:", opts.port);
  process.exit(1);
}

const cacheDir = opts.cacheDir || defaultCacheDir();
console.log("→ Cache directory:", cacheDir);

let toolchain;
if (!opts.skipSetup) {
  try {
    toolchain = await ensureToolchain({ cacheDir });
  } catch (err) {
    console.error("✗ Toolchain setup failed:", err.message);
    process.exit(2);
  }
} else {
  // Trust the caller; expect emsdk + Gamma at the standard cache paths.
  toolchain = {
    emsdkDir: process.env.GAMMA_COMPILE_EMSDK || join(cacheDir, "emsdk"),
    gammaDir: process.env.GAMMA_COMPILE_GAMMA || join(cacheDir, "Gamma")
  };
}

if (opts.setupOnly) {
  console.log("✓ Toolchain ready at:", cacheDir);
  process.exit(0);
}

await startServer({ port, toolchain, cacheDir });

function defaultCacheDir() {
  if (process.platform === "win32") {
    const local = process.env.LOCALAPPDATA || join(process.env.USERPROFILE || "", "AppData", "Local");
    return join(local, "gamma-compile");
  }
  if (process.platform === "darwin") {
    return join(process.env.HOME || "", "Library", "Caches", "gamma-compile");
  }
  return join(process.env.XDG_CACHE_HOME || join(process.env.HOME || "", ".cache"), "gamma-compile");
}
