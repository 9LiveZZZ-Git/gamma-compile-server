// Asset hosting + streaming for the Gamma Node Editor (Phase 8.B.15 /
// §8.F bootstrap — "Blob's Adventure"). Houses environment meshes
// (GLB), PBR textures, and HDRIs on disk and serves them to the editor
// with HTTP range streaming, so big files don't choke the browser and
// CORS-blocked sources (m3-org, GitHub, Poly Haven) are reachable.
//
// Attached the same way as the SD / RT modules: attachAssetRoutes(app).
//
// On disk:
//   gamma-compile-server/
//     assets/                 (gitignored — the local art cache)
//       meshes/   *.glb
//       textures/ *.{png,jpg,webp}
//       hdris/    *.{hdr,exr}
//       audio/    *.{ogg,wav,mp3}
//       manifest.json
//   src/asset-seed.json        (curated m3-org + SpectraStudios list,
//                               auto-fetched on first launch)

import express from "express";
import {
  createReadStream, createWriteStream, statSync, existsSync,
  mkdirSync, readFileSync, writeFileSync
} from "node:fs";
import { join, dirname, extname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import https from "node:https";
import http from "node:http";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ASSETS_DIR = join(__dirname, "..", "assets");
const MANIFEST_PATH = join(ASSETS_DIR, "manifest.json");
const SEED_PATH = join(__dirname, "asset-seed.json");

const TYPE_DIR = { mesh: "meshes", texture: "textures", hdri: "hdris", audio: "audio" };

// Source-URL whitelist for /assets/fetch + the seed auto-fetch. Only
// these hosts may be downloaded server-side (prevents the endpoint
// becoming an open proxy).
const FETCH_HOST_WHITELIST = [
  "m3-org.github.io",
  "raw.githubusercontent.com",
  "github.com",
  "githubusercontent.com",
  "objects.githubusercontent.com",
  "codeload.github.com",
  "media.githubusercontent.com",
  "polyhaven.com",
  "dl.polyhaven.org",
  "api.polyhaven.com"
];

const CONTENT_TYPE = {
  ".glb": "model/gltf-binary", ".gltf": "model/gltf+json",
  ".hdr": "image/vnd.radiance", ".exr": "image/x-exr",
  ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".ogg": "audio/ogg", ".wav": "audio/wav", ".mp3": "audio/mpeg", ".opus": "audio/opus"
};

function typeForExt(ext) {
  ext = ext.toLowerCase();
  if (ext === ".glb" || ext === ".gltf") return "mesh";
  if (ext === ".hdr" || ext === ".exr") return "hdri";
  if (ext === ".png" || ext === ".jpg" || ext === ".jpeg" || ext === ".webp") return "texture";
  if (ext === ".ogg" || ext === ".wav" || ext === ".mp3" || ext === ".opus") return "audio";
  return "file";
}

// Deterministic id from a filename: lowercase, non-alnum → underscore.
function slugify(name) {
  return basename(name, extname(name)).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function ensureDirs() {
  for (const d of [ASSETS_DIR, ...Object.values(TYPE_DIR).map(t => join(ASSETS_DIR, t))]) {
    if (!existsSync(d)) mkdirSync(d, { recursive: true });
  }
}

function loadManifest() {
  try {
    if (existsSync(MANIFEST_PATH)) {
      const m = JSON.parse(readFileSync(MANIFEST_PATH, "utf8"));
      if (Array.isArray(m)) return m;
    }
  } catch (_) {}
  return [];
}
function saveManifest(m) {
  try { writeFileSync(MANIFEST_PATH, JSON.stringify(m, null, 2)); } catch (e) {
    console.warn("[assets] manifest save failed:", e && e.message);
  }
}

let _manifest = [];
const _fetchStatus = { running: false, total: 0, done: 0, current: null, errors: [] };

function manifestById(id) { return _manifest.find(e => e && e.id === id); }

function addEntry(entry) {
  const i = _manifest.findIndex(e => e && e.id === entry.id);
  if (i >= 0) _manifest[i] = { ..._manifest[i], ...entry };
  else _manifest.push(entry);
  saveManifest(_manifest);
}

// Whitelist a URL by host suffix.
function hostAllowed(url) {
  let h;
  try { h = new URL(url).hostname; } catch (_) { return false; }
  return FETCH_HOST_WHITELIST.some(w => h === w || h.endsWith("." + w));
}

// Download a URL to a destination path, following redirects (GitHub raw
// 302s to objects.githubusercontent.com). Resolves to the byte size.
function download(url, destPath, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 6) return reject(new Error("too many redirects"));
    const lib = url.startsWith("https") ? https : http;
    const req = lib.get(url, { headers: { "User-Agent": "gamma-compile-server" } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        const next = new URL(res.headers.location, url).href;
        return resolve(download(next, destPath, redirects + 1));
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error("HTTP " + res.statusCode + " for " + url));
      }
      mkdirSync(dirname(destPath), { recursive: true });
      const ws = createWriteStream(destPath);
      res.pipe(ws);
      ws.on("finish", () => ws.close(() => {
        let size = 0; try { size = statSync(destPath).size; } catch (_) {}
        resolve(size);
      }));
      ws.on("error", reject);
    });
    req.on("error", reject);
    req.setTimeout(120000, () => req.destroy(new Error("download timeout")));
  });
}

// Fetch one source asset (url → assets/<typeDir>/<name>) + manifest it.
async function fetchAsset({ url, name, source, type }) {
  if (!hostAllowed(url)) throw new Error("source host not whitelisted: " + url);
  const fname = name || basename(new URL(url).pathname);
  const t = type || typeForExt(extname(fname));
  const rel = join(TYPE_DIR[t] || ".", fname).replace(/\\/g, "/");
  const dest = join(ASSETS_DIR, rel);
  const size = await download(url, dest);
  const entry = {
    id: slugify(fname), type: t, name: basename(fname, extname(fname)),
    file: rel, size, source: source || "fetch", url, addedAt: Date.now()
  };
  addEntry(entry);
  return entry;
}

// Background seed: pull any curated asset-seed.json entries not already
// present. Sequential so we don't hammer GitHub; tolerant of 404s.
async function runSeedFetch() {
  let seed = [];
  try { if (existsSync(SEED_PATH)) seed = JSON.parse(readFileSync(SEED_PATH, "utf8")); } catch (_) {}
  const missing = (Array.isArray(seed) ? seed : []).filter(s => {
    const id = slugify(s.name || basename(new URL(s.url).pathname));
    return !manifestById(id);
  });
  if (!missing.length) return;
  _fetchStatus.running = true;
  _fetchStatus.total = missing.length;
  _fetchStatus.done = 0;
  _fetchStatus.errors = [];
  console.log("[assets] seed fetch: " + missing.length + " asset(s) to download");
  for (const s of missing) {
    _fetchStatus.current = s.name || s.url;
    try {
      const e = await fetchAsset(s);
      console.log("[assets]   ✓ " + e.id + " (" + Math.round(e.size / 1024) + " KB)");
    } catch (err) {
      console.warn("[assets]   ✗ " + (s.name || s.url) + ": " + (err && err.message));
      _fetchStatus.errors.push({ asset: s.name || s.url, error: err && err.message });
    }
    _fetchStatus.done++;
  }
  _fetchStatus.current = null;
  _fetchStatus.running = false;
  console.log("[assets] seed fetch complete (" + (_fetchStatus.total - _fetchStatus.errors.length) +
    "/" + _fetchStatus.total + " ok)");
}

export function attachAssetRoutes(app, opts = {}) {
  ensureDirs();
  _manifest = loadManifest();

  // GET /assets — the manifest (optionally filtered by ?type=mesh).
  app.get("/assets", (req, res) => {
    const t = req.query.type;
    const list = t ? _manifest.filter(e => e.type === t) : _manifest;
    res.json({ assets: list, count: list.length, fetching: _fetchStatus.running });
  });

  // GET /assets/fetch-status — seed/download progress for the editor.
  app.get("/assets/fetch-status", (req, res) => res.json({ ..._fetchStatus }));

  // GET /assets/:id — stream the asset bytes with HTTP range support.
  app.get("/assets/:id", (req, res) => {
    const entry = manifestById(req.params.id);
    if (!entry) return res.status(404).json({ error: "asset not found: " + req.params.id });
    const filePath = join(ASSETS_DIR, entry.file);
    if (!existsSync(filePath)) return res.status(404).json({ error: "asset file missing on disk" });
    let stat; try { stat = statSync(filePath); } catch (_) { return res.status(500).end(); }
    const ct = CONTENT_TYPE[extname(filePath).toLowerCase()] || "application/octet-stream";
    const range = req.headers.range;
    if (range) {
      const m = /bytes=(\d*)-(\d*)/.exec(range);
      let start = m && m[1] ? parseInt(m[1], 10) : 0;
      let end = m && m[2] ? parseInt(m[2], 10) : stat.size - 1;
      if (isNaN(start)) start = 0;
      if (isNaN(end) || end >= stat.size) end = stat.size - 1;
      if (start > end || start >= stat.size) {
        return res.status(416).set("Content-Range", "bytes */" + stat.size).end();
      }
      res.status(206).set({
        "Content-Range": "bytes " + start + "-" + end + "/" + stat.size,
        "Accept-Ranges": "bytes",
        "Content-Length": end - start + 1,
        "Content-Type": ct
      });
      createReadStream(filePath, { start, end }).pipe(res);
    } else {
      res.status(200).set({
        "Content-Length": stat.size, "Accept-Ranges": "bytes", "Content-Type": ct
      });
      createReadStream(filePath).pipe(res);
    }
  });

  // POST /assets/fetch {source, url, name?, type?} — server-side
  // download from a whitelisted source into the cache + manifest.
  app.post("/assets/fetch", async (req, res) => {
    const { url, name, source, type } = req.body || {};
    if (typeof url !== "string" || !url) return res.status(400).json({ error: "missing url" });
    if (!hostAllowed(url)) return res.status(403).json({ error: "host not whitelisted" });
    try {
      const entry = await fetchAsset({ url, name, source, type });
      res.json({ ok: true, asset: entry });
    } catch (err) {
      res.status(502).json({ error: err && err.message || String(err) });
    }
  });

  // POST /assets/reseed — re-run the seed fetch (idempotent; skips
  // already-cached entries). Returns immediately; poll fetch-status.
  app.post("/assets/reseed", (req, res) => {
    if (_fetchStatus.running) return res.json({ ok: true, alreadyRunning: true });
    runSeedFetch().catch(e => console.warn("[assets] reseed error:", e && e.message));
    res.json({ ok: true, started: true });
  });

  // PUT /assets/import/:name — raw-body upload (the Poly Haven
  // drag-drop path). Body is the file bytes; type inferred from the
  // name's extension. express.raw handles up to 256 MB here.
  app.put("/assets/import/:name",
    express.raw({ type: "*/*", limit: "256mb" }),
    (req, res) => {
      const name = req.params.name;
      const buf = req.body;
      if (!Buffer.isBuffer(buf) || !buf.length) return res.status(400).json({ error: "empty body" });
      const t = typeForExt(extname(name));
      const rel = join(TYPE_DIR[t] || ".", name).replace(/\\/g, "/");
      const dest = join(ASSETS_DIR, rel);
      try {
        mkdirSync(dirname(dest), { recursive: true });
        writeFileSync(dest, buf);
        const entry = {
          id: slugify(name), type: t, name: basename(name, extname(name)),
          file: rel, size: buf.length, source: "import", addedAt: Date.now()
        };
        addEntry(entry);
        res.json({ ok: true, asset: entry });
      } catch (err) {
        res.status(500).json({ error: err && err.message || String(err) });
      }
    });

  // DELETE /assets/:id — drop from manifest (leaves the file on disk;
  // a future GC could prune orphans).
  app.delete("/assets/:id", (req, res) => {
    const i = _manifest.findIndex(e => e && e.id === req.params.id);
    if (i < 0) return res.status(404).json({ error: "not found" });
    _manifest.splice(i, 1);
    saveManifest(_manifest);
    res.json({ ok: true });
  });

  // Kick off the curated seed fetch in the background (non-blocking).
  if (opts.seed !== false) {
    runSeedFetch().catch(e => console.warn("[assets] seed fetch error:", e && e.message));
  }

  console.log("[assets] route attached — cache: " + ASSETS_DIR + " (" + _manifest.length + " cached)");
}
