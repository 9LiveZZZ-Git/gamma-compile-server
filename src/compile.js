// The actual compile pipeline. Takes the wrapped patch C++ source and
// returns either { wasm: Buffer, stderr: string } or { error, stderr }.
//
// Architecture: write source to a temp dir, invoke emcc with Gamma's
// 11 web-buildable .cpp files alongside the patch, read the wasm back.
// Each request is fully isolated (its own temp dir) so concurrent
// compiles don't stomp each other.

import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomBytes } from "node:crypto";

// Same 11 sources we've been using everywhere — Gamma compiled cleanly
// against Emscripten in the AlloLib Studio Online build, so this is
// the proven set.
const GAMMA_SOURCES = [
  "Conversion.cpp", "DFT.cpp", "Domain.cpp", "FFT_fftpack.cpp",
  "Print.cpp", "Scheduler.cpp", "Timer.cpp", "arr.cpp",
  "fftpack++1.cpp", "fftpack++2.cpp", "scl.cpp"
];

// Cached compiled Gamma library — built once on first compile, reused
// for all subsequent compiles in this server session. Drops per-patch
// compile from "all 11 Gamma sources + patch" (~30 s) to "patch + link"
// (~3-5 s).
let cachedLibPath = null;
let buildLibPromise = null;

export async function compile({ wrappedSrc, toolchain, optLevel = "O1" }) {
  if (typeof wrappedSrc !== "string" || !wrappedSrc.length) {
    return { error: "Empty source", stderr: "" };
  }

  // Build (or reuse) the cached libgamma.a once per session.
  if (!cachedLibPath) {
    if (!buildLibPromise) {
      buildLibPromise = buildLibgamma({ toolchain });
    }
    cachedLibPath = await buildLibPromise;
  }

  const workDir = await fs.mkdtemp(join(tmpdir(), "gamma-compile-"));
  try {
    const srcPath = join(workDir, "patch.cpp");
    const wasmPath = join(workDir, "patch.wasm");
    await fs.writeFile(srcPath, wrappedSrc, "utf8");

    const args = [
      "-std=c++17",
      "-" + optLevel,
      "-fno-exceptions",
      "-Wno-deprecated-declarations",
      "-Wno-pragma-once-outside-header",
      "-I", toolchain.gammaDir,
      "-sSTANDALONE_WASM=1",
      "-sEXPORTED_FUNCTIONS=_preview_init,_preview_tick,_preview_set,_preview_setter_count,_preview_set_sr,_malloc,_free",
      "-Wl,--no-entry",
      srcPath,
      cachedLibPath,
      "-o", wasmPath
    ];

    const { code, stderr } = await runEmcc(toolchain, args);

    let wasm = null;
    try { wasm = await fs.readFile(wasmPath); } catch (_) {}

    if (!wasm) {
      return { error: "emcc produced no output (exit " + code + ")", stderr };
    }
    return { wasm, stderr };
  } finally {
    // Cleanup workspace; best-effort.
    fs.rm(workDir, { recursive: true, force: true }).catch(() => {});
  }
}

async function buildLibgamma({ toolchain }) {
  console.log("→ Building libgamma.a (one-time, ~30 s)…");
  const libDir = join(toolchain.gammaDir, ".libgamma-cache");
  const libPath = join(libDir, "libgamma.a");
  await fs.mkdir(libDir, { recursive: true });

  // Compile each Gamma source to a .o file.
  const objs = [];
  for (const src of GAMMA_SOURCES) {
    const srcPath = join(toolchain.gammaDir, "src", src);
    const objPath = join(libDir, src.replace(".cpp", ".o"));
    objs.push(objPath);
    const { code, stderr } = await runEmcc(toolchain, [
      "-std=c++17",
      "-O2",
      "-fno-exceptions",
      "-Wno-deprecated-declarations",
      "-I", toolchain.gammaDir,
      "-c", srcPath,
      "-o", objPath
    ]);
    if (code !== 0) {
      throw new Error("Failed to compile Gamma source " + src + ":\n" + stderr);
    }
  }

  // Archive into a single .a using emar (emsdk ships it).
  const { code: arCode, stderr: arErr } = await runEmar(toolchain, [
    "rcs", libPath, ...objs
  ]);
  if (arCode !== 0) {
    throw new Error("Failed to archive libgamma.a:\n" + arErr);
  }

  console.log("✓ libgamma.a cached at " + libPath);
  return libPath;
}

function runEmcc(toolchain, args) {
  return runEmsdk(toolchain, "em++", args);
}
function runEmar(toolchain, args) {
  return runEmsdk(toolchain, "emar", args);
}

// Run an emsdk binary by invoking the emsdk env script first so PATH
// + EMSDK_NODE etc. are in scope. Cross-platform: emsdk_env.bat on
// Windows, emsdk_env.sh elsewhere.
function runEmsdk(toolchain, tool, args) {
  return new Promise((resolve) => {
    let stderr = "";
    let cmd, cmdArgs;
    if (process.platform === "win32") {
      // Windows: invoke through the emsdk batch wrapper.
      const toolBat = join(toolchain.emsdkDir, "upstream", "emscripten", tool + ".bat");
      cmd = toolBat;
      cmdArgs = args;
    } else {
      // Unix: source emsdk_env.sh first, then run.
      cmd = "bash";
      cmdArgs = [
        "-c",
        `. "${join(toolchain.emsdkDir, "emsdk_env.sh")}" >/dev/null 2>&1 && ` +
        `${tool} ${args.map(a => `"${a.replace(/"/g, '\\"')}"`).join(" ")}`
      ];
    }
    const p = spawn(cmd, cmdArgs, {
      shell: process.platform === "win32",
      stdio: ["ignore", "pipe", "pipe"]
    });
    p.stdout.on("data", d => { stderr += d.toString(); });
    p.stderr.on("data", d => { stderr += d.toString(); });
    p.on("close", code => resolve({ code, stderr }));
    p.on("error", err => resolve({ code: -1, stderr: err.message }));
  });
}
