// Standalone Node ws client → engine handshake test. Helps isolate
// whether "Handshake not finished" errors come from the proxy layer
// or from Node `ws` library defaults colliding with tokio-tungstenite.
//
// Usage:  node scripts/test-engine-ws.mjs
//
// Expected on success: prints "OPEN" then "MSG: {type:hello,...}".
// On failure: prints whatever Node `ws` says about the connection.

import WebSocket from "ws";

const URL = process.env.ENGINE_URL || "ws://127.0.0.1:9100/";
console.log("[test] connecting to", URL);

// Match the proxy's connection options exactly.
const ws = new WebSocket(URL, { perMessageDeflate: false });

ws.on("open", () => {
  console.log("[test] OPEN");
  console.log("[test] sending hello");
  ws.send(JSON.stringify({ type: "hello" }));
});
ws.on("message", (data, isBinary) => {
  const preview = isBinary ? `<binary ${data.length} bytes>` : data.toString().slice(0, 240);
  console.log("[test] MSG:", preview);
});
ws.on("error", (e) => {
  console.error("[test] ERR:", e.message);
});
ws.on("close", (code, reason) => {
  console.log("[test] CLOSE code=" + code, "reason=" + (reason && reason.toString()));
  process.exit(0);
});

setTimeout(() => {
  console.log("[test] 5s timeout — closing");
  try { ws.close(); } catch (_) {}
}, 5000);
