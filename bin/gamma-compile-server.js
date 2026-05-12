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
    host:        { type: "string",  default: "127.0.0.1" }, // bind interface
    allowOrigin: { type: "string",  multiple: true, default: [] }, // extra CORS origins
    cacheDir:    { type: "string",  default: "" },         // default resolved per-OS
    skipSetup:   { type: "boolean", default: false },      // assume toolchain present
    setupOnly:   { type: "boolean", default: false },      // download & exit
    // OSC bridge flags. Bridge is ON by default; --noOsc disables it
    // for setups that don't want a UDP port bound (e.g. shared dev
    // boxes where port 9000 is taken).
    noOsc:       { type: "boolean", default: false },
    oscInPort:   { type: "string",  default: "9000" },     // udp listen port
    oscOutHost:  { type: "string",  default: "127.0.0.1" },// default udp send target host
    oscOutPort:  { type: "string",  default: "9001" },     // default udp send target port
    help:        { type: "boolean", short: "h", default: false },
    version:     { type: "boolean", short: "v", default: false }
  },
  allowPositionals: false
});

if (opts.help) {
  console.log(`gamma-compile-server — local Emscripten compile daemon for the
Gamma Node Editor's real-time audio preview.

Usage: gamma-compile-server [--port 8765] [--host 127.0.0.1]
                            [--allowOrigin <url>]... [--cacheDir <path>]
                            [--skipSetup] [--setupOnly]

  --port         HTTP port to listen on (default 8765).
  --host         Network interface to bind to (default 127.0.0.1, i.e.
                 loopback only). Use 0.0.0.0 to accept connections from
                 other devices on your LAN — e.g. patch on an iPad
                 against a Mac running this daemon. ONLY do this on a
                 trusted network; the /compile endpoint runs Emscripten
                 on whatever C++ it receives.
  --allowOrigin  Extra CORS origin to allow (repeatable). Default
                 whitelist is the GitHub Pages editor + a handful of
                 localhost dev URLs. Add your editor's origin if you're
                 self-hosting it — e.g. http://192.168.1.42:8000. Pass
                 "*" to allow any origin (matches "host=0.0.0.0" risk
                 profile; only use on trusted networks).
  --cacheDir     Where to cache emsdk + Gamma source.
                 Default: %LOCALAPPDATA%\\gamma-compile (Windows) or
                 ~/.cache/gamma-compile (macOS/Linux).
  --skipSetup    Skip the first-run toolchain check (use if you've
                 manually pointed the daemon at a pre-installed emsdk
                 via GAMMA_COMPILE_EMSDK env var).
  --setupOnly    Download + verify the toolchain and exit, without
                 starting the server. Useful for installer scripts.
  --noOsc        Disable the OSC bridge (UDP listener + WebSocket
                 fan-out). The bridge lets external OSC apps
                 (TouchOSC / Reaper / Max / Pure Data / hardware
                 controllers) drive editor parameters via the OscIn
                 node, and lets the editor send OSC out via OscOut.
                 On by default; disable if UDP port 9000 is taken or
                 you don't need OSC routing.
  --oscInPort    UDP port to listen on for inbound OSC (default 9000).
                 Point your OSC clients at this port.
  --oscOutHost   Default destination host for outbound OSC (default
                 127.0.0.1). Editor's OscOut node can override per
                 message.
  --oscOutPort   Default destination port for outbound OSC (default
                 9001). Editor's OscOut node can override per message.
  --help         Show this message.
  --version      Print version and exit.

The editor (https://9livezzz-git.github.io/Gamma-Node/) detects this
daemon by polling localhost:<port>/health on first Play click. If
present, the editor routes compile requests here instead of using
the in-browser Wasmer clang. Compile times are ~5–15 s per patch
(vs the in-browser path's many minutes / OOM).

LAN setup (e.g. patch on iPad → daemon on Mac):
  $ gamma-compile-server --host 0.0.0.0 \\
      --allowOrigin http://192.168.1.42:8000
  Then in the editor's ⚙ Settings → Compile server URL, set
  http://<mac-lan-ip>:8765, and serve the editor over plain http://
  from the same Mac (e.g. python -m http.server 8000) so the iPad
  doesn't hit mixed-content blocking.

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

// Validate OSC port flags. Allow 0 only when --noOsc is set (sentinel
// that means "no UDP at all"); otherwise require a real port number.
const oscInPort  = parseInt(opts.oscInPort,  10);
const oscOutPort = parseInt(opts.oscOutPort, 10);
if (!opts.noOsc) {
  if (!Number.isFinite(oscInPort)  || oscInPort  < 1 || oscInPort  > 65535) {
    console.error("✗ Invalid --oscInPort:",  opts.oscInPort);
    process.exit(1);
  }
  if (!Number.isFinite(oscOutPort) || oscOutPort < 1 || oscOutPort > 65535) {
    console.error("✗ Invalid --oscOutPort:", opts.oscOutPort);
    process.exit(1);
  }
}

await startServer({
  port,
  host: opts.host,
  extraOrigins: opts.allowOrigin || [],
  toolchain,
  cacheDir,
  osc:         !opts.noOsc,
  oscInPort,
  oscOutHost:  opts.oscOutHost,
  oscOutPort
});

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
