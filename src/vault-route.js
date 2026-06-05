// Tektite vault hosting — server-side companion to the editor's
// Tektite tab (Gamma-Node sprint 10y client + 10z server).
//
// The editor's vault is normally browser-side IndexedDB.  When a user
// drops a file larger than ~25 MB and a compile-server URL is set,
// the editor offers to upload it here instead of cramming it into the
// IDB quota.  This module hosts:
//
//   PUT    /vault/file/<path>   -- raw body upload (creates dirs)
//   GET    /vault/file/<path>   -- file body with Content-Type +
//                                  HTTP range streaming for big media
//   DELETE /vault/file/<path>   -- removes the file
//   GET    /vault/list?prefix=  -- walk the prefix subtree + return
//                                  metadata only (no payload)
//
// On disk:
//   gamma-compile-server/
//     vault/                    (gitignored; created on first PUT)
//       <id-or-subpath>/...     (filenames straight from the editor)
//
// The editor prefixes every upload with "vault/" (constant
// TEKTITE_VAULT_REMOTE_FOLDER in src/tektite/storage.js), but the
// server doesn't enforce that prefix -- it just refuses paths that
// climb out of VAULT_DIR via "..".

import express from "express";
import {
  createReadStream, existsSync, mkdirSync, readdirSync,
  statSync, unlinkSync, writeFileSync
} from "node:fs";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const VAULT_DIR = join(__dirname, "..", "vault");

// Per-file upload cap.  Bigger than the asset-import 256 MB because
// users dropping 4K source videos or large LLM training corpora is
// the explicit reason this route exists.  Past 2 GB the editor side
// will refuse anyway (browser memory limits on Blob -> ArrayBuffer
// during the PUT).
const VAULT_LIMIT_MB = 2048;

const CONTENT_TYPE = {
  ".png": "image/png",  ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
  ".gif": "image/gif",  ".svg": "image/svg+xml", ".bmp": "image/bmp",
  ".webp": "image/webp", ".avif": "image/avif",
  ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
  ".ogg": "audio/ogg",  ".flac": "audio/flac", ".3gp": "audio/3gpp",
  ".mp4": "video/mp4",  ".webm": "video/webm", ".mov": "video/quicktime",
  ".mkv": "video/x-matroska", ".ogv": "video/ogg",
  ".pdf": "application/pdf",
  ".doc": "application/msword",
  ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ".rtf": "application/rtf",
  ".odt": "application/vnd.oasis.opendocument.text",
  ".csv": "text/csv",   ".tsv": "text/tab-separated-values",
  ".json": "application/json", ".jsonl": "application/x-ndjson",
  ".parquet": "application/octet-stream",
  ".yaml": "application/yaml", ".yml": "application/yaml",
  ".xml": "application/xml",
  ".txt": "text/plain", ".log": "text/plain",
  ".md": "text/markdown", ".markdown": "text/markdown",
  ".zip": "application/zip", ".tar": "application/x-tar",
  ".gz": "application/gzip", ".7z": "application/x-7z-compressed",
  ".tgz": "application/gzip",
  ".gpatch": "application/json", ".gdsp": "text/x-c++src",
  ".html": "text/html", ".htm": "text/html",
  ".css": "text/css",
  ".py": "text/x-python", ".rs": "text/x-rust",
  ".c": "text/x-c", ".h": "text/x-c", ".cpp": "text/x-c++src",
  ".cxx": "text/x-c++src", ".cc": "text/x-c++src", ".hpp": "text/x-c++hdr",
  ".js": "text/javascript", ".mjs": "text/javascript",
  ".ts": "text/x-typescript", ".tsx": "text/x-typescript",
  ".jsx": "text/javascript",
  ".go": "text/x-go", ".rb": "text/x-ruby", ".php": "text/x-php",
  ".lua": "text/x-lua", ".sh": "text/x-shellscript",
  ".sql": "text/x-sql", ".java": "text/x-java",
  ".raku": "text/x-raku", ".pl": "text/x-perl",
  ".wgsl": "text/plain", ".glsl": "text/plain"
};

function safePath(reqPath) {
  // Reject blank, absolute, or traversal paths.
  if (!reqPath) throw new Error("empty path");
  const norm = String(reqPath).replace(/\\/g, "/").replace(/^\/+/, "");
  if (!norm || norm.indexOf("..") >= 0 || norm.indexOf("\0") >= 0) {
    throw new Error("invalid path");
  }
  return join(VAULT_DIR, norm);
}

function urlFor(req, sub) {
  // Build the canonical URL the editor stores in IDB.  Honor
  // X-Forwarded-Proto / Host so the same daemon reverse-proxied
  // through Caddy / nginx returns the public URL the browser used.
  const proto = req.get("x-forwarded-proto") || req.protocol;
  const host  = req.get("x-forwarded-host")  || req.get("host");
  return proto + "://" + host + "/vault/file/" + sub;
}

export function attachVaultRoutes(app) {
  if (!existsSync(VAULT_DIR)) mkdirSync(VAULT_DIR, { recursive: true });

  // PUT /vault/file/<path>  -- raw body upload.  Path can be a
  // nested subpath like "vault/photo.png" or "vault/videos/clip.mp4";
  // intermediate directories are created on demand.
  app.put("/vault/file/*",
    express.raw({ type: "*/*", limit: VAULT_LIMIT_MB + "mb" }),
    (req, res) => {
      const sub = req.params[0];
      let dest;
      try { dest = safePath(sub); }
      catch (e) { return res.status(400).json({ error: e.message }); }
      const buf = req.body;
      if (!Buffer.isBuffer(buf) || !buf.length) {
        return res.status(400).json({ error: "empty body" });
      }
      try {
        mkdirSync(dirname(dest), { recursive: true });
        writeFileSync(dest, buf);
        res.json({ ok: true, url: urlFor(req, sub), size: buf.length });
      } catch (err) {
        res.status(500).json({ error: err && err.message || String(err) });
      }
    });

  // GET /vault/file/<path>  -- streamed read with range support so
  // the editor can paint big videos / PDFs progressively.
  app.get("/vault/file/*", (req, res) => {
    const sub = req.params[0];
    let fp;
    try { fp = safePath(sub); }
    catch (e) { return res.status(400).json({ error: e.message }); }
    if (!existsSync(fp)) return res.status(404).json({ error: "not found" });
    let stat; try { stat = statSync(fp); }
    catch (_) { return res.status(500).end(); }
    if (!stat.isFile()) return res.status(400).json({ error: "not a file" });
    const ct = CONTENT_TYPE[extname(fp).toLowerCase()] || "application/octet-stream";
    const range = req.headers.range;
    if (range) {
      const m = /bytes=(\d*)-(\d*)/.exec(range);
      let start = m && m[1] ? parseInt(m[1], 10) : 0;
      let end   = m && m[2] ? parseInt(m[2], 10) : stat.size - 1;
      if (isNaN(start)) start = 0;
      if (isNaN(end) || end >= stat.size) end = stat.size - 1;
      if (start > end || start >= stat.size) {
        return res.status(416).set("Content-Range", "bytes */" + stat.size).end();
      }
      res.status(206).set({
        "Content-Range":  "bytes " + start + "-" + end + "/" + stat.size,
        "Accept-Ranges":  "bytes",
        "Content-Length": end - start + 1,
        "Content-Type":   ct
      });
      createReadStream(fp, { start, end }).pipe(res);
    } else {
      res.status(200).set({
        "Content-Length": stat.size,
        "Accept-Ranges":  "bytes",
        "Content-Type":   ct
      });
      createReadStream(fp).pipe(res);
    }
  });

  // DELETE /vault/file/<path>
  app.delete("/vault/file/*", (req, res) => {
    const sub = req.params[0];
    let fp;
    try { fp = safePath(sub); }
    catch (e) { return res.status(400).json({ error: e.message }); }
    if (!existsSync(fp)) return res.status(404).json({ error: "not found" });
    try { unlinkSync(fp); res.status(204).end(); }
    catch (err) { res.status(500).json({ error: err && err.message || String(err) }); }
  });

  // GET /vault/list?prefix=foo  -- recursive metadata listing.
  app.get("/vault/list", (req, res) => {
    const prefix = String(req.query.prefix || "").replace(/\\/g, "/").replace(/^\/+/, "");
    if (prefix.indexOf("..") >= 0) return res.status(400).json({ error: "invalid prefix" });
    const root = prefix ? join(VAULT_DIR, prefix) : VAULT_DIR;
    function walk(dir, rel) {
      if (!existsSync(dir)) return [];
      const out = [];
      let entries;
      try { entries = readdirSync(dir); } catch (_) { return []; }
      for (const name of entries) {
        const fp = join(dir, name);
        const sub = (rel ? rel + "/" : "") + name;
        let stat; try { stat = statSync(fp); } catch (_) { continue; }
        if (stat.isDirectory()) out.push(...walk(fp, sub));
        else if (stat.isFile()) {
          out.push({
            id:         (prefix ? prefix + "/" : "") + sub,
            size:       stat.size,
            modifiedAt: stat.mtimeMs,
            createdAt:  stat.birthtimeMs || stat.mtimeMs
          });
        }
      }
      return out;
    }
    res.json({ files: walk(root, "") });
  });

  // GET /vault/stats  -- total bytes + file count, surfaced in the
  // editor's DevTools menu so the user can see how much they've
  // pushed to the server.
  app.get("/vault/stats", (req, res) => {
    function walk(dir) {
      if (!existsSync(dir)) return { count: 0, bytes: 0 };
      let count = 0, bytes = 0;
      let entries;
      try { entries = readdirSync(dir); } catch (_) { return { count, bytes }; }
      for (const name of entries) {
        const fp = join(dir, name);
        let stat; try { stat = statSync(fp); } catch (_) { continue; }
        if (stat.isDirectory()) { const s = walk(fp); count += s.count; bytes += s.bytes; }
        else if (stat.isFile()) { count++; bytes += stat.size; }
      }
      return { count, bytes };
    }
    res.json(walk(VAULT_DIR));
  });
}
