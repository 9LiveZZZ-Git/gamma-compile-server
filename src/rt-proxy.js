// rt-proxy.js -- TCP-level forward between the browser editor + the
// local gamma-rt-engine binary. Sprint 7.5.6.a part 2d.
//
// The browser editor opens a WebSocket to this compile-server's
// /rt path (typically https://9livezzz-git.github.io ↔ ws://localhost:8765/rt).
// We intercept the raw HTTP upgrade request, open a plain TCP socket
// to the engine's WebSocket port (default 127.0.0.1:9100), replay
// the original GET-Upgrade request bytes to it, then bidirectionally
// pipe the two sockets together. The WS handshake happens between
// the BROWSER and the ENGINE directly -- the proxy only shuffles
// bytes.
//
// Why not use `ws` library both sides? The original design did that
// (browser ↔ ws.WebSocketServer on the proxy, ws.WebSocket on the
// proxy → engine.tokio-tungstenite). Two WS endpoints in series
// means TWO handshakes + reframing in the middle. Chrome rejected
// the resulting frames with "Invalid frame header", and the
// proxy → engine handshake itself failed with tungstenite's
// "Handshake not finished" -- proxy was closing the half-open
// engine WS while the browser was still in handshake flux.
//
// TCP-level forward sidesteps all of that. One handshake (browser
// ↔ engine). Identical framing both sides. The proxy is invisible
// to the WS protocol.
//
// User must start the engine manually for now:
//   ./rt-engine/target/release/gamma-rt-engine --port 9100
// Auto-spawn lands in a follow-up; the cleanest UX is for the
// compile-server to start the engine on first /rt connection,
// but that needs binary-discovery polish + lifecycle management.

import net from "node:net";

export function attachRtProxy({
  httpServer,
  engineHost = "127.0.0.1",
  enginePort = 9100,
  allowedOrigins = null,
  allowAnyOrigin = false,
  logger = console
}) {
  // We attach to the underlying Node HTTP server's `upgrade` event
  // directly instead of wrapping the socket in a ws.WebSocketServer.
  // Multiple `upgrade` listeners can coexist on the same httpServer
  // (the OSC bridge attaches its own ws.WebSocketServer for /osc;
  // its handler checks path filter and ignores non-/osc upgrades).
  // Same pattern here -- we only claim sockets whose URL is /rt.
  httpServer.on("upgrade", (req, clientSocket, head) => {
    if (req.url !== "/rt") return;

    // Origin check -- same policy as the OSC bridge. Same defaults
    // (GH Pages + localhost dev ports). On reject we write a raw
    // 403 + destroy.
    const origin = req.headers.origin || "";
    const originOk =
      allowAnyOrigin ||
      !origin ||  // curl / direct connection
      (allowedOrigins && allowedOrigins.has(origin));
    if (!originOk) {
      logger.warn("[rt-proxy] rejected origin: " + origin);
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

    const peer = clientSocket.remoteAddress + ":" + clientSocket.remotePort;
    logger.log("[rt-proxy] client " + peer + " upgrading; TCP forward → " +
               engineHost + ":" + enginePort);

    // Open raw TCP to the engine.
    const engineSocket = net.connect({ host: engineHost, port: enginePort });

    let engineConnected = false;
    let buffered = [];  // bytes from client we received before engine TCP opens

    // Client → engine. Until engineSocket is connected, buffer the
    // bytes. After connect, write them then pipe directly.
    clientSocket.on("data", (chunk) => {
      if (engineConnected) {
        engineSocket.write(chunk);
      } else {
        buffered.push(chunk);
      }
    });

    engineSocket.on("connect", () => {
      engineConnected = true;
      // Replay the original HTTP upgrade request so the engine's
      // tokio-tungstenite handshake sees the browser's exact headers.
      // This makes the engine's WS handshake be with the BROWSER, not
      // the proxy. Same Sec-WebSocket-Key, same Origin, same
      // extensions offered -- the engine's response goes straight
      // back to the browser through the pipe.
      let upgradeRequest = "GET " + req.url + " HTTP/" + req.httpVersion + "\r\n";
      for (const [key, value] of Object.entries(req.headers)) {
        if (Array.isArray(value)) {
          for (const v of value) upgradeRequest += key + ": " + v + "\r\n";
        } else {
          upgradeRequest += key + ": " + value + "\r\n";
        }
      }
      upgradeRequest += "\r\n";
      engineSocket.write(upgradeRequest);
      // `head` is any bytes received after the headers but before the
      // upgrade event fired (rare, but possible if client pipelines).
      if (head && head.length) engineSocket.write(head);
      // Flush anything the client sent while we were connecting.
      for (const chunk of buffered) engineSocket.write(chunk);
      buffered = [];
      logger.log("[rt-proxy] engine connected for " + peer);
    });

    // Engine → client: straight pipe. The first bytes will be the
    // 101 Switching Protocols response from tokio-tungstenite, which
    // the browser consumes to complete its handshake. After that:
    // WS frames in both directions, opaque to us.
    engineSocket.on("data", (chunk) => {
      try { clientSocket.write(chunk); } catch (_) {}
    });

    // Cleanup: either side disconnects closes the other.
    const closeBoth = () => {
      try { engineSocket.destroy(); } catch (_) {}
      try { clientSocket.destroy(); } catch (_) {}
    };
    clientSocket.on("close", () => {
      logger.log("[rt-proxy] client " + peer + " disconnected");
      closeBoth();
    });
    clientSocket.on("error", (e) => {
      logger.warn("[rt-proxy] client " + peer + " socket error: " + e.message);
      closeBoth();
    });
    engineSocket.on("close", () => {
      logger.log("[rt-proxy] engine disconnected for " + peer);
      closeBoth();
    });
    engineSocket.on("error", (e) => {
      logger.warn("[rt-proxy] engine TCP error for " + peer + ": " + e.message +
        " — is the rt-engine running on " + engineHost + ":" + enginePort + "?");
      // Write a 502-equivalent before the upgrade response, if we
      // haven't already written anything to the client. Best-effort:
      // if the client side hasn't received an HTTP response yet,
      // they'll see a connection failure; otherwise they'll see a
      // mid-stream TCP close which the browser will surface as a
      // WS-level error.
      if (clientSocket.writable && !engineConnected) {
        try {
          clientSocket.write(
            "HTTP/1.1 502 Bad Gateway\r\n" +
            "Content-Type: text/plain\r\n" +
            "Connection: close\r\n" +
            "\r\n" +
            "RT engine unreachable: " + e.message
          );
        } catch (_) {}
      }
      closeBoth();
    });
  });

  // Return value retained for API symmetry with the old WebSocketServer
  // version; nothing in server.js currently uses it, but if it ever
  // does we can plumb something through here. Returning null is the
  // clearest signal that this is no longer a ws.WebSocketServer.
  return null;
}
