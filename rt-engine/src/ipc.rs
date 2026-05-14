//! WebSocket IPC between the Node compile-server (proxy) and the
//! Rust engine. Wire protocol is JSON for control messages + raw
//! binary frames for raw RGBA8 pixel data once streaming starts.
//!
//! Control messages (text frames, JSON):
//!
//!   client → engine:
//!     {type: "hello"}                         handshake
//!     {type: "configure", width, height}      stream init (resizes the renderer)
//!     {type: "scene", patch: {...}}           full scene update     (part 2d)
//!     {type: "params", patch: {...}}          partial param update  (part 2d)
//!     {type: "render-start"}                  begin streaming frames
//!     {type: "render-stop"}                   pause streaming
//!     {type: "shutdown"}                      clean engine exit
//!
//!   engine → client:
//!     {type: "hello", version, capabilities, backend}   handshake reply
//!     {type: "frame-config", width, height, format}     stream config
//!     <binary>                                          raw RGBA8 pixel data
//!     {type: "error", where, message}                   non-fatal error
//!
//! Sprint 7.5.6.a part 2c (this commit): render-start / render-stop
//! drive an actual Metal render loop. Each frame goes out as a
//! binary message containing width * height * 4 bytes of RGBA8.

use crate::backend::BackendKind;
use crate::capability::Capabilities;
use crate::render::Renderer;
use crate::scene::{Camera, Light, Scene};
use anyhow::Context;
use futures_util::{SinkExt, StreamExt};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::time::Duration;
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
    FrameConfig {
        width: u32,
        height: u32,
        format: &'a str,
    },
    Error {
        #[serde(rename = "where")]
        where_: &'a str,
        message: &'a str,
    },
}

const DEFAULT_WIDTH: u32 = 800;
const DEFAULT_HEIGHT: u32 = 600;
const TARGET_FRAME_INTERVAL_MS: u64 = 33; // ~30 fps

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
                // {:#} dumps the full anyhow cause chain rather than
                // just the outermost wrapper. Critical for diagnosing
                // "ws handshake failed" -- the wrapper is useless on
                // its own; we need the underlying tungstenite::Error
                // to know which header / state the parser tripped on.
                warn!("client {} handler error: {:#}", peer, e);
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

    // Per-connection rendering state. Renderer is built lazily on
    // first render-start so a client that only does --probe-style
    // hello+shutdown doesn't pay the GPU init cost. Scene + camera
    // updates that arrive before the renderer is built are buffered
    // here and applied on creation.
    let mut renderer: Option<Renderer> = None;
    let mut rendering = false;
    let mut want_dims = (DEFAULT_WIDTH, DEFAULT_HEIGHT);
    let mut frame_counter: u64 = 0;
    let mut pending_scene: Option<Scene> = None;
    let mut pending_camera: Option<Camera> = None;

    let mut tick = tokio::time::interval(Duration::from_millis(TARGET_FRAME_INTERVAL_MS));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            biased;
            msg = rx.next() => {
                let msg = match msg {
                    Some(Ok(m)) => m,
                    Some(Err(e)) => {
                        warn!("ws recv error: {}", e);
                        return Ok(());
                    }
                    None => return Ok(()),
                };
                if !handle_client_msg(
                    msg,
                    &mut tx,
                    &caps,
                    &backend_str,
                    &mut renderer,
                    &mut rendering,
                    &mut want_dims,
                    &mut pending_scene,
                    &mut pending_camera,
                ).await? {
                    // Returned false = clean shutdown requested.
                    return Ok(());
                }
            }
            _ = tick.tick(), if rendering => {
                // Ensure renderer exists at the right dimensions.
                let need_new = renderer
                    .as_ref()
                    .map(|r| r.width != want_dims.0 || r.height != want_dims.1)
                    .unwrap_or(true);
                if need_new {
                    match Renderer::new(want_dims.0, want_dims.1) {
                        Ok(mut r) => {
                            // Apply any scene / camera state that
                            // arrived BEFORE the renderer was built.
                            // Order: scene first (which also seeds
                            // the camera uniform), then any newer
                            // camera-only patch on top.
                            if let Some(scene) = pending_scene.take() {
                                if let Err(e) = r.update_scene(&scene) {
                                    warn!("[stream] pending update_scene failed: {}", e);
                                }
                            }
                            if let Some(cam) = pending_camera.take() {
                                if let Err(e) = r.update_camera(&cam) {
                                    warn!("[stream] pending update_camera failed: {}", e);
                                }
                            }
                            // Notify client of the actual stream
                            // dimensions (defaults if no configure
                            // was sent).
                            let cfg = EngineMsg::FrameConfig {
                                width: r.width,
                                height: r.height,
                                format: "rgba8unorm",
                            };
                            tx.send(Message::Text(serde_json::to_string(&cfg)?.into())).await?;
                            renderer = Some(r);
                            frame_counter = 0;
                            info!("[stream] renderer ready: {}x{}", want_dims.0, want_dims.1);
                        }
                        Err(e) => {
                            warn!("[stream] renderer build failed: {}", e);
                            rendering = false;
                            let err_msg = format!("{}", e);
                            let err = EngineMsg::Error {
                                where_: "render-init",
                                message: &err_msg,
                            };
                            tx.send(Message::Text(serde_json::to_string(&err)?.into())).await?;
                            continue;
                        }
                    }
                }
                // Render + send.
                if let Some(r) = renderer.as_ref() {
                    match r.render_frame() {
                        Ok(pixels) => {
                            // Log every 60th frame so we know
                            // it's still alive without spamming.
                            frame_counter += 1;
                            if frame_counter % 60 == 1 {
                                info!("[stream] frame {} ({} bytes)", frame_counter, pixels.len());
                            }
                            if tx.send(Message::Binary(pixels.into())).await.is_err() {
                                return Ok(());
                            }
                        }
                        Err(e) => {
                            warn!("[stream] render error: {}", e);
                            rendering = false;
                            let err_msg = format!("{}", e);
                            let err = EngineMsg::Error {
                                where_: "render-frame",
                                message: &err_msg,
                            };
                            tx.send(Message::Text(serde_json::to_string(&err)?.into())).await?;
                        }
                    }
                }
            }
        }
    }
}

/// Handle one inbound message. Returns `Ok(false)` on shutdown
/// request (clean exit), `Ok(true)` to keep the loop alive,
/// `Err(_)` on a fatal error.
async fn handle_client_msg(
    msg: Message,
    tx: &mut futures_util::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
        Message,
    >,
    caps: &Capabilities,
    backend_str: &str,
    renderer: &mut Option<Renderer>,
    rendering: &mut bool,
    want_dims: &mut (u32, u32),
    pending_scene: &mut Option<Scene>,
    pending_camera: &mut Option<Camera>,
) -> anyhow::Result<bool> {
    match msg {
        Message::Text(text) => {
            let parsed: Result<ClientMsg, _> = serde_json::from_str(&text);
            match parsed {
                Ok(ClientMsg::Hello) => {
                    let reply = EngineMsg::Hello {
                        version: env!("CARGO_PKG_VERSION"),
                        backend: backend_str,
                        capabilities: caps,
                    };
                    tx.send(Message::Text(serde_json::to_string(&reply)?.into())).await?;
                }
                Ok(ClientMsg::Configure { width, height }) => {
                    *want_dims = (width.max(1).min(4096), height.max(1).min(4096));
                    info!("[stream] configure: {}x{}", want_dims.0, want_dims.1);
                    // Force renderer rebuild on next tick if rendering.
                    *renderer = None;
                }
                Ok(ClientMsg::Scene { patch }) => {
                    // Sprint 7.5.6.a part 2e-1 -- full scene replace.
                    // Editor sends one of these after configure +
                    // before render-start; the engine builds a fresh
                    // AS from the mesh set and uploads camera/colors.
                    match serde_json::from_value::<Scene>(patch) {
                        Ok(scene) => {
                            info!(
                                "[stream] scene: {} mesh(es), camera@{:?}",
                                scene.meshes.len(),
                                scene.camera.pos
                            );
                            if let Some(r) = renderer.as_mut() {
                                if let Err(e) = r.update_scene(&scene) {
                                    warn!("[stream] update_scene failed: {}", e);
                                    let msg = format!("{}", e);
                                    let err = EngineMsg::Error {
                                        where_: "scene",
                                        message: &msg,
                                    };
                                    tx.send(Message::Text(serde_json::to_string(&err)?.into())).await?;
                                }
                            } else {
                                // Renderer not yet built -- buffer
                                // until first render-start tick.
                                *pending_scene = Some(scene);
                            }
                        }
                        Err(e) => {
                            warn!("[stream] scene parse failed: {}", e);
                            let msg = format!("invalid scene patch: {}", e);
                            let err = EngineMsg::Error {
                                where_: "scene-parse",
                                message: &msg,
                            };
                            tx.send(Message::Text(serde_json::to_string(&err)?.into())).await?;
                        }
                    }
                }
                Ok(ClientMsg::Params { patch }) => {
                    // Partial update path. Patch fields handled in
                    // c-1: "camera" (live orbit), "lights" (live
                    // intensity / hue drags). Anything else is
                    // silently ignored -- mesh transforms + per-
                    // material partial updates come in 2e-2 / c-2.
                    if let Some(cam_val) = patch.get("camera") {
                        match serde_json::from_value::<Camera>(cam_val.clone()) {
                            Ok(cam) => {
                                if let Some(r) = renderer.as_mut() {
                                    if let Err(e) = r.update_camera(&cam) {
                                        warn!("[stream] update_camera failed: {}", e);
                                    }
                                } else {
                                    *pending_camera = Some(cam);
                                }
                            }
                            Err(e) => {
                                warn!("[stream] camera-params parse failed: {}", e);
                            }
                        }
                    }
                    if let Some(lights_val) = patch.get("lights") {
                        match serde_json::from_value::<Vec<Light>>(lights_val.clone()) {
                            Ok(lights) => {
                                if let Some(r) = renderer.as_mut() {
                                    if let Err(e) = r.update_lights(&lights) {
                                        warn!("[stream] update_lights failed: {}", e);
                                    }
                                }
                                // If renderer isn't built yet, the
                                // next Scene message will include
                                // lights anyway -- no pending_lights
                                // state needed.
                            }
                            Err(e) => {
                                warn!("[stream] lights-params parse failed: {}", e);
                            }
                        }
                    }
                }
                Ok(ClientMsg::RenderStart) => {
                    *rendering = true;
                    info!("[stream] render-start (dims {}x{})", want_dims.0, want_dims.1);
                }
                Ok(ClientMsg::RenderStop) => {
                    *rendering = false;
                    info!("[stream] render-stop");
                }
                Ok(ClientMsg::Shutdown) => {
                    info!("[stream] shutdown requested");
                    return Ok(false);
                }
                Err(e) => {
                    let err_msg = format!("invalid client message: {}", e);
                    let err = EngineMsg::Error {
                        where_: "parse",
                        message: &err_msg,
                    };
                    tx.send(Message::Text(serde_json::to_string(&err)?.into())).await?;
                }
            }
        }
        Message::Binary(_) => {
            let err = EngineMsg::Error {
                where_: "binary",
                message: "binary client → engine not currently expected",
            };
            tx.send(Message::Text(serde_json::to_string(&err)?.into())).await?;
        }
        Message::Close(_) => return Ok(false),
        Message::Ping(p) => tx.send(Message::Pong(p)).await?,
        Message::Pong(_) => {}
        Message::Frame(_) => {}
    }
    Ok(true)
}
