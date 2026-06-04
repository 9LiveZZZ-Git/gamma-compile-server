// Polls a local Ollama daemon (default http://127.0.0.1:11434) and
// caches { present, version, models, baseUrl } so /health responses
// stay sub-ms even when Ollama is up + slow or down + timing out.
//
// Phase B sprint 3 of the editor's docs/LLM-KNOWLEDGE-PHASE.md §4
// roadmap. The editor probes Ollama directly (Chrome's loopback mixed-
// content exception covers it) and that direct probe is the
// authoritative source for the model badge. This server-side advertise
// is the SECONDARY signal for two cases:
//
//   1. Cold start of the editor against a LAN-deployed daemon — the
//      compile-server already knows whether its host machine has Ollama
//      running, so the editor can light up the badge from /health on
//      first /health probe instead of waiting on its own 30 s direct
//      probe.
//
//   2. Mixed-content blocking — when the editor loads from
//      https://9livezzz-git.github.io and the user points the
//      compile-server at a non-loopback http:// LAN host, the direct
//      probe is blocked but /health (which goes through the compile-
//      server) still resolves. The editor can fall back to
//      health.ollama for at least basic availability info.
//
// Probe runs at server startup, logs the first result, then re-polls
// every 60 s. setInterval is unref'd so it doesn't keep the event
// loop alive if everything else exits cleanly. Failures are swallowed
// silently after the first startup line (no log spam when the user
// stops + restarts Ollama).

const POLL_INTERVAL_MS = 60_000;
const DEFAULT_URL = "http://127.0.0.1:11434";
const PROBE_TIMEOUT_MS = 2_000;

let _snapshot = {
  present:   false,
  version:   null,
  models:    [],
  baseUrl:   DEFAULT_URL,
  fetchedAt: 0,
  error:     null
};
let _interval = null;
let _baseUrl  = DEFAULT_URL;

async function _probeOnce(baseUrl) {
  const url = baseUrl || DEFAULT_URL;
  const out = {
    present:   false,
    version:   null,
    models:    [],
    baseUrl:   url,
    fetchedAt: Date.now(),
    error:     null
  };
  try {
    const vres = await fetch(url + "/api/version", {
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS)
    });
    if (!vres.ok) throw new Error("HTTP " + vres.status + " on /api/version");
    const v = await vres.json();
    out.present = true;
    out.version = (v && typeof v.version === "string") ? v.version : "unknown";
    // Tags is best-effort; some daemons take longer to enumerate. A
    // failure here doesn't flip `present` -- the daemon clearly exists.
    try {
      const tres = await fetch(url + "/api/tags", {
        signal: AbortSignal.timeout(PROBE_TIMEOUT_MS)
      });
      if (tres.ok) {
        const t = await tres.json();
        if (Array.isArray(t.models)) {
          // Trim the payload to the fields the editor needs. Each
          // /api/tags response carries a chunky modelfile / parameters
          // blob per model that's irrelevant for the badge.
          out.models = t.models.map(m => ({
            name: m.name,
            size: typeof m.size === "number" ? m.size : null,
            modified_at: m.modified_at || null
          }));
        }
      }
    } catch (_) { /* tags optional */ }
  } catch (e) {
    out.error = (e && e.message) || String(e);
  }
  _snapshot = out;
  return out;
}

/* Start the polling probe. Logs the first result, then re-polls every
 * 60 s in the background. Idempotent -- calling twice replaces the
 * interval rather than running two. */
export async function startOllamaProbe(baseUrl) {
  _baseUrl = (baseUrl && String(baseUrl).trim().replace(/\/+$/, "")) || DEFAULT_URL;
  const first = await _probeOnce(_baseUrl);
  if (first.present) {
    console.log("[ollama] daemon detected at " + first.baseUrl + " v" + first.version +
      " (" + first.models.length + " model" + (first.models.length === 1 ? "" : "s") + ")");
  } else {
    console.log("[ollama] no daemon at " + first.baseUrl +
      " — install from ollama.com/download to enable LLM nodes in the editor");
  }
  if (_interval) clearInterval(_interval);
  _interval = setInterval(() => {
    _probeOnce(_baseUrl).catch(() => { /* silent on routine probe failures */ });
  }, POLL_INTERVAL_MS);
  if (_interval.unref) _interval.unref();
}

/* Read the cached snapshot. Cheap, synchronous, safe to call from any
 * HTTP handler. Returns a shallow clone so callers can't mutate the
 * cache directly. */
export function getOllamaSnapshot() {
  return Object.assign({}, _snapshot);
}

/* Stop the polling probe. Used by tests + clean-shutdown paths. */
export function stopOllamaProbe() {
  if (_interval) {
    clearInterval(_interval);
    _interval = null;
  }
}
