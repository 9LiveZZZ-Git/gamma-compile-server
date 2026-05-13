//! WebSocket IPC between the Node compile-server (proxy) and the
//! Rust engine. Wire protocol is JSON for control messages + raw
//! binary frames for H.264 NAL units once streaming lands in part 2.
//!
//! Control messages (text frames, JSON):
//!
//!   client → engine:
//!     {type: "hello"}                         handshake
//!     {type: "configure", width, height}      stream init
//!     {type: "scene", patch: {...}}           scene update (full)
//!     {type: "params", patch: {...}}          partial param update
//!     {type: "render-start"}                  begin streaming frames
//!     {type: "render-stop"}                   pause streaming
//!     {type: "shutdown"}                      clean engine exit
//!
//!   engine → client:
//!     {type: "hello", version, capabilities, backend}   handshake reply
//!     {type: "frame-config", codec, width, height}      stream config
//!     <binary>                                          encoded NAL units (H.264)
//!     {type: "error", where, message}                   non-fatal error
//!
//! Sprint 7.5.6.a part 1: only the handshake works. Subsequent
//! messages are parsed but not yet acted upon (the engine just
//! echoes a "not-yet-implemented" reply).

use crate::backend::BackendKind;
use crate::capability::Capabilities;
use anyhow::Context;
use futures_util::{SinkExt, StreamExt};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum ClientMsg {
    Hello,
    Configure { width: u32, height: u32 },
    Scene { patch: serde_json::Value },
    Params { patch: serde_json::Value },
    RenderStart,
    RenderStop,
    Shutdown,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum EngineMsg<'a> {
    Hello {
        version: &'a str,
        backend: &'a str,
        capabilities: &'a Capabilities,
    },
    Error {
        where_: &'a str,
        message: &'a str,
    },
    NotImplemented {
        feature: &'a str,
    },
}

pub async fn serve(
    host: &str,
    port: u16,
    caps: Capabilities,
    backend: BackendKind,
) -> anyhow::Result<()> {
    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr)
        .await
        .with_context(|| format!("could not bind WebSocket on {addr}"))?;
    info!("WebSocket listening on ws://{}/", addr);

    while let Ok((stream, peer)) = listener.accept().await {
        let caps = caps.clone();
        let backend_str = format!("{:?}", backend).to_lowercase();
        tokio::spawn(async move {
            info!("client connected: {}", peer);
            if let Err(e) = handle_connection(stream, caps, backend_str).await {
                warn!("client {} handler error: {}", peer, e);
            }
            info!("client {} disconnected", peer);
        });
    }
    Ok(())
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    caps: Capabilities,
    backend_str: String,
) -> anyhow::Result<()> {
    let ws_stream = accept_async(stream).await.context("ws handshake failed")?;
    let (mut tx, mut rx) = ws_stream.split();

    while let Some(msg) = rx.next().await {
        let msg = msg.context("ws recv failed")?;
        match msg {
            Message::Text(text) => {
                let parsed: Result<ClientMsg, _> = serde_json::from_str(&text);
                match parsed {
                    Ok(ClientMsg::Hello) => {
                        let reply = EngineMsg::Hello {
                            version: env!("CARGO_PKG_VERSION"),
                            backend: &backend_str,
                            capabilities: &caps,
                        };
                        tx.send(Message::Text(serde_json::to_string(&reply)?.into()))
                            .await?;
                    }
                    Ok(ClientMsg::Shutdown) => {
                        info!("client requested shutdown; exiting");
                        return Ok(());
                    }
                    Ok(ClientMsg::Configure { .. })
                    | Ok(ClientMsg::Scene { .. })
                    | Ok(ClientMsg::Params { .. })
                    | Ok(ClientMsg::RenderStart)
                    | Ok(ClientMsg::RenderStop) => {
                        // Part 1: just acknowledge; the actual render
                        // pipeline + streaming lands in part 2.
                        let reply = EngineMsg::NotImplemented {
                            feature: "render pipeline (§5.6.a part 2)",
                        };
                        tx.send(Message::Text(serde_json::to_string(&reply)?.into()))
                            .await?;
                    }
                    Err(e) => {
                        let err_msg = format!("invalid client message: {}", e);
                        let reply = EngineMsg::Error {
                            where_: "parse",
                            message: &err_msg,
                        };
                        tx.send(Message::Text(serde_json::to_string(&reply)?.into()))
                            .await?;
                    }
                }
            }
            Message::Binary(_) => {
                // Client → engine binary is reserved for future use.
                let reply = EngineMsg::Error {
                    where_: "binary",
                    message: "binary client → engine not currently expected",
                };
                tx.send(Message::Text(serde_json::to_string(&reply)?.into()))
                    .await?;
            }
            Message::Close(_) => return Ok(()),
            Message::Ping(p) => tx.send(Message::Pong(p)).await?,
            Message::Pong(_) => {}
            Message::Frame(_) => {} // shouldn't happen with default config
        }
    }
    Ok(())
}
