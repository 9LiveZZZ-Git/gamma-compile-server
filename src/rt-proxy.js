// rt-proxy.js -- WebSocket proxy between the browser editor + the
// local gamma-rt-engine binary. Sprint 7.5.6.a part 2d.
//
// The browser editor opens a WebSocket to this compile-server's
// /rt path (typically wss://9livezzz-git.github.io ↔ ws://localhost:8765/rt).
// The proxy opens its own WebSocket to the engine on its local port
// (default 127.0.0.1:9100) + forwards text + binary in both
// directions. Either side disconnecting closes the other.
//
// The proxy itself is stateless -- it doesn't care about the
// JSON message shapes or binary frame contents. That keeps the
// engine protocol changeable without redeploying the Node side.
//
// User must start the engine manually for now:
//   ./rt-engine/target/release/gamma-rt-engine --port 9100
// Auto-spawn lands in a follow-up; the cleanest UX is for the
// compile-server to start the engine on first /rt connection,
// but that needs binary-discovery polish + lifecycle management.

import { WebSocketServer, WebSocket } from "ws";

export function attachRtProxy({
  httpServer,
  engineHost = "127.0.0.1",
  enginePort = 9100,
  allowedOrigins = null,
  allowAnyOrigin = false,
  logger = console
}) {
  const wss = new WebSocketServer({
    server: httpServer,
    path: "/rt",
    // Disable per-message deflate on browser ↔ proxy too. Raw RGBA
    // pixel data is essentially incompressible (already noise-like
    // for ray-traced scenes); deflate would burn CPU for ~0% gain
    // and adds a CPU hop on every 1.9 MB frame.
    perMessageDeflate: false,
    verifyClient: (info, cb) => {
      // Same origin policy as the OSC bridge -- GH Pages + localhost
      // dev ports allowed by default.
      if (allowAnyOrigin) return cb(true);
      const origin = info.origin || info.req.headers.origin;
      if (!origin) return cb(true);
      if (allowedOrigins && allowedOrigins.has(origin)) return cb(true);
      cb(false, 403, "Origin not allowed: " + origin);
    }
  });

  wss.on("connection", (clientWs, req) => {
    const peer = req.socket.remoteAddress + ":" + req.socket.remotePort;
    logger.log("[rt-proxy] client " + peer + " connected; bridging to " +
               "ws://" + engineHost + ":" + enginePort);

    let engineWs;
    try {
      // perMessageDeflate: false -- Node `ws` defaults to negotiating
      // permessage-deflate with `client_max_window_bits` (no value)
      // in the upgrade request. Some versions of tokio-tungstenite
      // (Rust engine side) parse that as a malformed extension header
      // and reject the handshake outright. We don't need compression
      // for loopback traffic anyway; raw RGBA at 30fps is ~57 Mbps
      // which fits in a Unix socket no problem. Same flag also passed
      // to the WebSocketServer above so the browser ↔ proxy hop
      // doesn't waste cycles re-deflating either.
      engineWs = new WebSocket("ws://" + engineHost + ":" + enginePort + "/",
                                { perMessageDeflate: false });
    } catch (e) {
      logger.warn("[rt-proxy] could not open engine WS: " + e.message);
      _sendErr(clientWs, "proxy-init", e.message);
      try { clientWs.close(); } catch (_) {}
      return;
    }

    // Track whether we've forwarded a "hello" so a client connecting
    // before the engine ws is open doesn't lose the message.
    const pendingFromClient = [];
    let engineReady = false;

    engineWs.on("open", () => {
      engineReady = true;
      logger.log("[rt-proxy] engine connected for " + peer);
      // Flush any queued client → engine messages.
      while (pendingFromClient.length) {
        const { data, isBinary } = pendingFromClient.shift();
        try { engineWs.send(data, { binary: isBinary }); } catch (_) {}
      }
    });

    engineWs.on("error", (e) => {
      logger.warn("[rt-proxy] engine ws error for " + peer + ": " + e.message);
      _sendErr(clientWs, "proxy-engine", e.message +
        " — is the rt-engine running? Start it with: " +
        "./rt-engine/target/release/gamma-rt-engine --port " + enginePort);
    });

    // Forward client → engine.
    clientWs.on("message", (data, isBinary) => {
      if (engineReady && engineWs.readyState === WebSocket.OPEN) {
        try { engineWs.send(data, { binary: isBinary }); } catch (_) {}
      } else {
        // Queue until engine is ready (short window during handshake).
        if (pendingFromClient.length < 64) pendingFromClient.push({ data, isBinary });
      }
    });

    // Forward engine → client. Binary frames are the big ones (raw
    // RGBA pixel data); text frames are control messages
    // (hello/frame-config/error).
    engineWs.on("message", (data, isBinary) => {
      if (clientWs.readyState === WebSocket.OPEN) {
        try { clientWs.send(data, { binary: isBinary }); } catch (_) {}
      }
    });

    // Cleanup: either side disconnects closes the other.
    const closeBoth = () => {
      try { engineWs.close(); } catch (_) {}
      try { clientWs.close(); } catch (_) {}
    };
    clientWs.on("close", () => {
      logger.log("[rt-proxy] client " + peer + " disconnected");
      closeBoth();
    });
    engineWs.on("close", () => {
      logger.log("[rt-proxy] engine disconnected for " + peer);
      closeBoth();
    });
  });

  return wss;
}

function _sendErr(ws, where, message) {
  if (ws.readyState !== WebSocket.OPEN) return;
  try {
    ws.send(JSON.stringify({ type: "error", where, message }));
  } catch (_) {}
}
