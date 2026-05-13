// rt-proxy.js -- WebSocket reverse-proxy with split-handshake.
// Sprint 7.5.6.a part 2d, attempt 3.
//
// History:
//   v1: ws.WebSocketServer both sides. Worked for handshake but
//       frame reframing produced "Invalid frame header" in Chrome.
//   v2: raw TCP forward both sides. Engine handshake completed
//       cleanly when bytes arrived, but Chrome closes the browser
//       end at +3ms because no 101 ever arrives -- the proxy was
//       waiting for the engine's 101 to forward back. Browser is
//       impatient on localhost and aborts.
//   v3 (this): proxy answers the browser's WS upgrade WITH ITS
//       OWN 101 immediately (computed from Sec-WebSocket-Key);
//       the browser sees onopen synchronously. In parallel, the
//       proxy opens a TCP socket to the engine, forwards the
//       original upgrade request, then strips the engine's 101
//       response and pipes only the WS frame bytes through
//       afterwards.
//
// Result: one logical WS session (browser ↔ engine in spirit), two
// handshakes that complete independently, no frame reframing. The
// proxy only stripping the engine's 101 prefix once -- the rest
// is raw byte forwarding.
//
// Extension handling: we DO NOT forward Sec-WebSocket-Extensions
// to the engine. The proxy advertises NO extensions in its 101 to
// the browser, so the browser uses uncompressed frames. The engine
// also gets a request with no extensions, so it responds with no
// extensions. Both sides agree -- uncompressed frames flow through
// the byte pipe.
//
// User must start the engine manually for now:
//   ./rt-engine/target/release/gamma-rt-engine --port 9100
// Auto-spawn lands in a follow-up.

import net from "node:net";
import crypto from "node:crypto";

const WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

function wsAccept(secWebSocketKey) {
  return crypto.createHash("sha1")
    .update(secWebSocketKey + WS_GUID)
    .digest("base64");
}

export function attachRtProxy({
  httpServer,
  engineHost = "127.0.0.1",
  enginePort = 9100,
  allowedOrigins = null,
  allowAnyOrigin = false,
  logger = console
}) {
  httpServer.on("upgrade", (req, clientSocket, head) => {
    if (req.url !== "/rt") return;

    const peer = clientSocket.remoteAddress + ":" + clientSocket.remotePort;
    const t0 = Date.now();
    const log = (msg) => logger.log("[rt-proxy] +" + (Date.now() - t0) + "ms " + peer + " " + msg);

    // Origin check.
    const origin = req.headers.origin || "";
    const originOk =
      allowAnyOrigin ||
      !origin ||
      (allowedOrigins && allowedOrigins.has(origin));
    if (!originOk) {
      log("rejected origin: " + origin);
      clientSocket.write(
        "HTTP/1.1 403 Forbidden\r\n" +
        "Content-Type: text/plain\r\n" +
        "Connection: close\r\n" +
        "\r\n" +
        "Origin not allowed: " + origin
      );
      clientSocket.destroy();
      return;
    }

    // Validate the WS upgrade. We need a Sec-WebSocket-Key to compute
    // the Accept hash. Missing key = malformed request; refuse.
    const upgrade = (req.headers.upgrade || "").toLowerCase();
    const wsKey = req.headers["sec-websocket-key"];
    const wsVersion = req.headers["sec-websocket-version"];
    if (upgrade !== "websocket" || !wsKey || wsVersion !== "13") {
      log("bad upgrade headers: upgrade=" + upgrade + " key=" + !!wsKey + " ver=" + wsVersion);
      clientSocket.write(
        "HTTP/1.1 400 Bad Request\r\n" +
        "Content-Type: text/plain\r\n" +
        "Connection: close\r\n" +
        "\r\n" +
        "Bad WebSocket upgrade request"
      );
      clientSocket.destroy();
      return;
    }

    log("upgrading; bridging to " + engineHost + ":" + enginePort);

    // Match the socket-setup that ws.WebSocketServer.completeUpgrade
    // does before writing the 101: disable inactivity timeout +
    // disable Nagle. Without setNoDelay our 101 (a small write)
    // can get coalesced with later writes in a way that confuses
    // Chrome's WS-frame parser right after onopen.
    try {
      clientSocket.setTimeout(0);
      clientSocket.setNoDelay(true);
    } catch (e) {
      log("setTimeout/setNoDelay threw: " + e.message);
    }

    // ── STEP 1: answer the browser with our own 101 right now.
    // This is what Chrome was waiting for. Without it Chrome
    // bails at ~+3ms.
    const accept = wsAccept(wsKey);
    // Include Access-Control-Allow-Private-Network for Chrome's PNA
    // policy (public origin → private network). The compile-server's
    // Express middleware sets this on HTTP responses; the upgrade
    // handler bypasses Express so we have to set it ourselves here.
    const proxy101 =
      "HTTP/1.1 101 Switching Protocols\r\n" +
      "Upgrade: websocket\r\n" +
      "Connection: Upgrade\r\n" +
      "Sec-WebSocket-Accept: " + accept + "\r\n" +
      "Access-Control-Allow-Private-Network: true\r\n" +
      "\r\n";
    try {
      clientSocket.write(proxy101);
    } catch (e) {
      log("could not write proxy 101: " + e.message);
      try { clientSocket.destroy(); } catch (_) {}
      return;
    }
    log("sent proxy 101 to browser (+ PNA header + setNoDelay)");

    // Belt-and-suspenders: immediately follow with a server-sent
    // WS PING frame (opcode 0x9, no mask, no payload). This is
    // 2 bytes (0x89 0x00) and is a known-valid frame. It proves
    // to Chrome's parser that the post-handshake channel is
    // delivering correctly-framed data, in case Chrome's WS
    // implementation has some "first-frame timeout" heuristic.
    // Browser will respond with a PONG which we just ignore
    // (forwarding the bytes to the engine before its handshake
    // finishes would confuse tungstenite -- so we drop client
    // bytes that are 2-byte PONG frames before engine handshake
    // is stripped; see browserBuffer handling below).
    try {
      clientSocket.write(Buffer.from([0x89, 0x00]));
      log("sent server PING after 101");
    } catch (e) {
      log("could not write server PING: " + e.message);
    }

    // ── STEP 2: open the engine TCP and start its own handshake
    // in parallel. The engine will send its OWN 101 back through
    // this socket; we strip it before forwarding any subsequent
    // bytes to the browser.
    const engineSocket = net.connect({ host: engineHost, port: enginePort });

    let engineHandshakeStripped = false;
    let engineStripBuf = Buffer.alloc(0);
    let browserBuffer = [];   // browser frames received before engine handshake completes

    // Build the upgrade request to send to the engine. We strip
    // Sec-WebSocket-Extensions and Sec-WebSocket-Protocol so the
    // engine doesn't negotiate extensions that the browser (via
    // our 101) thinks are off.
    let engineUpgradeRequest = "GET " + req.url + " HTTP/1.1\r\n";
    for (const [key, value] of Object.entries(req.headers)) {
      if (key === "sec-websocket-extensions" || key === "sec-websocket-protocol") continue;
      if (Array.isArray(value)) {
        for (const v of value) engineUpgradeRequest += key + ": " + v + "\r\n";
      } else {
        engineUpgradeRequest += key + ": " + value + "\r\n";
      }
    }
    engineUpgradeRequest += "\r\n";

    engineSocket.on("connect", () => {
      log("engineSocket connected; writing " + engineUpgradeRequest.length + "B upgrade");
      try {
        engineSocket.write(engineUpgradeRequest);
        if (head && head.length) engineSocket.write(head);
      } catch (e) {
        log("engine write threw: " + e.message);
        closeBoth();
      }
    });

    // Browser → engine. Browser starts sending WS frames as soon as
    // its onopen fires (immediately after our 101). We buffer them
    // until the engine has finished its own handshake; otherwise the
    // engine's tokio-tungstenite parser sees WS frame bytes mid-HTTP
    // parse and bombs with "Handshake not finished".
    clientSocket.on("data", (chunk) => {
      log("client→ " + chunk.length + "B (engineHandshakeStripped=" + engineHandshakeStripped + ")");
      if (engineHandshakeStripped) {
        try { engineSocket.write(chunk); } catch (_) {}
      } else {
        browserBuffer.push(chunk);
      }
    });

    // Engine → browser. First N bytes are the engine's 101 response;
    // we discard everything up to and including the first \r\n\r\n.
    // After that all bytes are WS frames -- forward verbatim.
    engineSocket.on("data", (chunk) => {
      log("engine→ " + chunk.length + "B");
      if (engineHandshakeStripped) {
        try { clientSocket.write(chunk); } catch (_) {}
        return;
      }
      // Accumulate until we see the end of the engine's HTTP headers.
      engineStripBuf = Buffer.concat([engineStripBuf, chunk]);
      const idx = engineStripBuf.indexOf("\r\n\r\n");
      if (idx === -1) {
        if (engineStripBuf.length > 16384) {
          log("engine handshake response > 16KiB; bailing");
          closeBoth();
        }
        return;
      }
      // Dump the response preamble for diagnostics. Should be
      // "HTTP/1.1 101 Switching Protocols ..."; if it's anything
      // else the engine refused our upgrade and we want to know why.
      const headers = engineStripBuf.slice(0, idx).toString("utf8");
      log("engine handshake response: " + headers.split("\r\n")[0]);
      engineHandshakeStripped = true;
      // Anything after \r\n\r\n is the first WS frame from the engine.
      const tail = engineStripBuf.slice(idx + 4);
      engineStripBuf = Buffer.alloc(0);
      if (tail.length > 0) {
        try { clientSocket.write(tail); } catch (_) {}
      }
      // Now flush buffered browser frames to the engine.
      if (browserBuffer.length > 0) {
        let total = 0;
        for (const c of browserBuffer) total += c.length;
        log("flushing " + browserBuffer.length + " buffered browser chunks (" + total + "B) to engine");
        for (const c of browserBuffer) {
          try { engineSocket.write(c); } catch (_) {}
        }
        browserBuffer = [];
      }
    });

    const closeBoth = () => {
      try { engineSocket.destroy(); } catch (_) {}
      try { clientSocket.destroy(); } catch (_) {}
    };
    clientSocket.on("close", (hadError) => {
      log("clientSocket 'close' (hadError=" + !!hadError + ")");
      closeBoth();
    });
    clientSocket.on("error", (e) => {
      log("clientSocket 'error': " + e.message);
      closeBoth();
    });
    engineSocket.on("close", (hadError) => {
      log("engineSocket 'close' (hadError=" + !!hadError + ")");
      closeBoth();
    });
    engineSocket.on("error", (e) => {
      log("engineSocket 'error': " + e.message);
      closeBoth();
    });
  });

  return null;
}
