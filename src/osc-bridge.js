// OSC bridge — UDP listener + WebSocket fan-out to browsers.
//
// Browsers can't open raw UDP sockets, so this daemon plays the role
// of a bidirectional translator:
//
//   external app  ──UDP/OSC──>  this bridge  ──WS JSON──>  editor
//   editor        ──WS JSON──>  this bridge  ──UDP/OSC──>  external app
//
// Wire protocol (JSON over WebSocket).
//
//   server → client on connect:
//     { type: "hello", oscInPort, version }
//
//   server → client on inbound OSC:
//     { type: "in", address, args, from }       // single message
//     { type: "in-bundle", messages: [...] }    // OSC bundle (rare)
//
//   client → server, send outbound OSC:
//     { type: "send", address, args, host?, port? }
//   (host/port default to the bridge's configured outbound target)
//
//   client → server, change outbound default:
//     { type: "target", host, port }
//
//   server → client, on send / config error:
//     { type: "error", message, where }
//
// Subscription model: by default the bridge forwards EVERY inbound
// message to every connected client. Filtering happens client-side
// (the OscIn node only fires when address matches its pattern). This
// keeps the bridge stateless + the client's subscription set is just
// metadata for diagnostics; ~95% of patches care about <10 addresses
// and the per-message JSON encode is cheap.

import { createSocket } from "node:dgram";
import { WebSocketServer } from "ws";
import { decodeOsc, encodeOsc, encodeBundle } from "./osc-codec.js";

const PKG_VERSION = "0.3.1";

/* Start the OSC bridge attached to an existing HTTP server (so the
 * WebSocket upgrade lives on the same port as /compile and /health).
 * Listens on UDP `oscInPort` for inbound OSC. Returns a handle with
 * { udp, wss, stop() } the caller can use to shut down. */
export function startOscBridge({
  httpServer,
  oscInPort = 9000,
  oscOutHost = "127.0.0.1",
  oscOutPort = 9001,
  bindHost = "127.0.0.1",
  allowedOrigins = null,        // Set | null. null = same as HTTP (delegated)
  allowAnyOrigin = false,
  logger = console
}) {
  // UDP socket. ipv4 only for now -- TouchOSC + most controllers
  // talk ipv4. ipv6 setups still work over a router that exposes a
  // v4 address; full v6 support is a follow-up.
  const udp = createSocket({ type: "udp4", reuseAddr: true });

  // Per-connection state -- bookkeeping only. Subscriptions are
  // tracked but not enforced (the bridge fans-out unconditionally;
  // see the comment at the top of the file). Logged on /diag.
  const clients = new Set();
  let defaultOutHost = oscOutHost;
  let defaultOutPort = oscOutPort;

  // ── UDP receive path ─────────────────────────────────────────────
  udp.on("message", (buf, rinfo) => {
    if (!clients.size) return;
    let packet;
    try {
      packet = decodeOsc(buf);
    } catch (e) {
      logger.warn("[osc] decode failed from " + rinfo.address + ":" + rinfo.port + " — " + e.message);
      return;
    }
    const from = rinfo.address + ":" + rinfo.port;
    broadcast(packet, from);
  });

  function broadcast(packet, from) {
    const json = packet.bundle
      ? JSON.stringify({
          type: "in-bundle",
          from,
          // Flatten nested bundles into a single array of messages
          // for the client -- bundles within bundles are rare in
          // practice and the client API is simpler this way.
          messages: flattenBundle(packet)
        })
      : JSON.stringify({
          type: "in",
          from,
          address: packet.address,
          args: serializeArgs(packet.args)
        });
    for (const c of clients) {
      if (c.readyState === 1 /* OPEN */) c.send(json);
    }
  }

  function flattenBundle(b) {
    const out = [];
    for (const m of b.messages) {
      if (m.bundle) out.push(...flattenBundle(m));
      else out.push({ address: m.address, args: serializeArgs(m.args) });
    }
    return out;
  }

  // BigInt + Buffer aren't JSON-safe. Convert to JSON-friendly forms
  // (string for BigInt, base64 for Buffer) so the wire stays clean.
  function serializeArgs(args) {
    return args.map(a => {
      if (typeof a === "bigint") return { _bigint: a.toString() };
      if (Buffer.isBuffer(a))    return { _blob:   a.toString("base64") };
      return a;
    });
  }

  function deserializeArg(a) {
    if (a && typeof a === "object" && !Array.isArray(a)) {
      if ("_bigint" in a) return BigInt(a._bigint);
      if ("_blob"   in a) return Buffer.from(a._blob, "base64");
    }
    return a;
  }

  // ── WebSocket server (attached to existing HTTP server) ──────────
  const wss = new WebSocketServer({
    server: httpServer,
    path: "/osc",
    verifyClient: (info, cb) => {
      // Re-use the HTTP server's origin policy. WebSocket from a
      // public origin to a loopback bridge requires the Origin be
      // explicitly allow-listed -- the existing CORS allow-list
      // covers GitHub Pages + localhost dev ports.
      if (allowAnyOrigin) return cb(true);
      const origin = info.origin || info.req.headers.origin;
      if (!origin) return cb(true);                       // direct ws client (no Origin), e.g. curl
      if (allowedOrigins && allowedOrigins.has(origin)) return cb(true);
      cb(false, 403, "Origin not allowed: " + origin);
    }
  });

  wss.on("connection", (ws, req) => {
    clients.add(ws);
    const peer = req.socket.remoteAddress + ":" + req.socket.remotePort;
    ws.send(JSON.stringify({
      type: "hello",
      oscInPort,
      defaultOut: { host: defaultOutHost, port: defaultOutPort },
      version: PKG_VERSION
    }));
    logger.log("[osc] ws client " + peer + " connected  (" + clients.size + " open)");

    ws.on("message", (data) => {
      let msg;
      try { msg = JSON.parse(data.toString()); }
      catch (e) {
        return wsError(ws, "json-parse", e.message);
      }
      if (!msg || typeof msg !== "object") return;

      if (msg.type === "send") {
        const addr = msg.address;
        const args = Array.isArray(msg.args) ? msg.args.map(deserializeArg) : [];
        const host = msg.host || defaultOutHost;
        const port = Number(msg.port) || defaultOutPort;
        if (typeof addr !== "string" || !addr.startsWith("/")) {
          return wsError(ws, "send", "address must start with '/'");
        }
        let packet;
        try { packet = encodeOsc(addr, args); }
        catch (e) { return wsError(ws, "send-encode", e.message); }
        udp.send(packet, port, host, (err) => {
          if (err) wsError(ws, "send-udp", err.message);
        });
      } else if (msg.type === "send-bundle") {
        // Outbound OSC bundle -- multiple messages packed into a
        // single UDP datagram with an "immediate" timetag (the
        // editor's _tickOscOut uses this when it has >1 OscOut
        // value-change in a single frame, so the receiving app sees
        // them as atomic).
        if (!Array.isArray(msg.messages) || msg.messages.length === 0) {
          return wsError(ws, "send-bundle", "messages array empty");
        }
        const cleaned = msg.messages.map(m => ({
          address: m && m.address,
          args: Array.isArray(m && m.args) ? m.args.map(deserializeArg) : []
        }));
        // Validate addresses up-front so a single bad entry doesn't
        // sneak through and surface as a generic encode error.
        for (const m of cleaned) {
          if (typeof m.address !== "string" || !m.address.startsWith("/")) {
            return wsError(ws, "send-bundle", "every bundle element needs an address starting with '/'");
          }
        }
        const host = msg.host || defaultOutHost;
        const port = Number(msg.port) || defaultOutPort;
        let packet;
        try { packet = encodeBundle(cleaned); }
        catch (e) { return wsError(ws, "send-bundle-encode", e.message); }
        udp.send(packet, port, host, (err) => {
          if (err) wsError(ws, "send-bundle-udp", err.message);
        });
      } else if (msg.type === "target") {
        if (typeof msg.host === "string") defaultOutHost = msg.host;
        if (Number.isFinite(Number(msg.port))) defaultOutPort = Number(msg.port);
        logger.log("[osc] outbound target -> " + defaultOutHost + ":" + defaultOutPort);
      } else if (msg.type === "subscribe" || msg.type === "unsubscribe") {
        // Tracked but not enforced -- see file header. Useful for
        // diagnostics. Client typically sends one subscribe per
        // OscIn node listed in the patch.
        ws._oscSubs = ws._oscSubs || new Set();
        const patterns = Array.isArray(msg.patterns) ? msg.patterns : [];
        for (const p of patterns) {
          if (msg.type === "subscribe") ws._oscSubs.add(p);
          else ws._oscSubs.delete(p);
        }
      }
    });

    ws.on("close", () => {
      clients.delete(ws);
      logger.log("[osc] ws client " + peer + " closed     (" + clients.size + " open)");
    });

    ws.on("error", (e) => {
      logger.warn("[osc] ws error from " + peer + ": " + e.message);
    });
  });

  function wsError(ws, where, message) {
    if (ws.readyState !== 1) return;
    ws.send(JSON.stringify({ type: "error", where, message }));
  }

  // ── Bind UDP listener ────────────────────────────────────────────
  return new Promise((resolve, reject) => {
    udp.once("error", reject);
    udp.bind(oscInPort, bindHost, () => {
      udp.removeListener("error", reject);
      const addr = udp.address();
      logger.log("[osc] listening on udp://" + addr.address + ":" + addr.port +
                 "  ws on path /osc  outbound -> " + defaultOutHost + ":" + defaultOutPort);
      resolve({
        udp,
        wss,
        stop() {
          for (const c of clients) try { c.close(); } catch (_) {}
          try { wss.close(); } catch (_) {}
          try { udp.close(); } catch (_) {}
        }
      });
    });
  });
}
