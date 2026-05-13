//! gamma-rt-engine — native hardware ray-tracing engine for the
//! Gamma Node Editor. Cross-platform via two backends:
//!
//!   - Vulkan-RT (Windows / Linux): VK_KHR_ray_tracing_pipeline +
//!     VK_KHR_acceleration_structure on NVIDIA RTX, AMD RDNA2+,
//!     Intel Arc.
//!   - Metal-RT (macOS): Metal 3+ ray tracing on M3+ Apple Silicon
//!     with hardware acceleration; software traversal via MPS on
//!     M1/M2 (classified "preview only" -- works but slower).
//!
//! Communication: spawned by the Node `gamma-compile-server` as a
//! child process. Binds a local WebSocket on `--port` (default
//! 9100). The Node side proxies between the editor's browser
//! connection + this engine.
//!
//! This is sprint 7.5.6.a part 1 -- the skeleton. Capability
//! detection + IPC handshake only; no actual ray tracing yet.
//! Part 2 of §5.6.a lands the first traced triangle.

use clap::Parser;
use log::{info, warn};

mod backend;
mod capability;
mod ipc;
mod render;
mod scene;

#[derive(Parser, Debug)]
#[command(
    name = "gamma-rt-engine",
    about = "Hardware ray-tracing engine for the Gamma Node Editor",
    long_about = "Binds a WebSocket on the chosen port and accepts scene-rendering requests from the Node compile-server. \
                  Selects a Vulkan-RT or Metal-RT backend based on the host hardware. \
                  Run with --probe to print capabilities + exit."
)]
struct Cli {
    /// WebSocket bind port (Node compile-server defaults to 9100 too).
    #[arg(long, default_value_t = 9100)]
    port: u16,

    /// Bind host (default 127.0.0.1; loopback only).
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Probe the host's RT capabilities, print as JSON, and exit. The
    /// Node side uses this on engine install to know whether to even
    /// offer RT rendering in the editor.
    #[arg(long, default_value_t = false)]
    probe: bool,

    /// Force a specific backend (auto / vulkan / metal / compute-fallback).
    /// Default `auto` picks the best one available on this machine.
    #[arg(long, default_value = "auto")]
    backend: String,

    /// One-shot test render: build a single triangle, intersect per
    /// pixel via Möller-Trumbore in a Metal compute kernel, save the
    /// result to the path specified by --output, exit. Validates the
    /// full Metal pipeline before any streaming / WebSocket work.
    #[arg(long, default_value_t = false)]
    render_test: bool,

    /// Output PNG path for --render-test.
    #[arg(long, default_value = "triangle.png")]
    output: String,

    /// Render-test image width.
    #[arg(long, default_value_t = 800)]
    width: u32,

    /// Render-test image height.
    #[arg(long, default_value_t = 600)]
    height: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let cli = Cli::parse();

    // Probe mode: run the capability detector + emit JSON + exit.
    // This is the cheapest "is RT possible on this box?" check the
    // Node side runs at first-startup time.
    if cli.probe {
        let caps = capability::probe();
        println!("{}", serde_json::to_string_pretty(&caps)?);
        return Ok(());
    }

    // Render-test mode: one-shot test render to a PNG, exit. The
    // smallest end-to-end Metal validation -- no streaming, no
    // scene serialization, just "does the binary produce pixels?".
    if cli.render_test {
        info!(
            "running --render-test: {}x{} → {}",
            cli.width, cli.height, cli.output
        );
        return render::render_test_triangle(cli.width, cli.height, &cli.output);
    }

    info!(
        "gamma-rt-engine starting (version {}, host {}, port {}, backend {})",
        env!("CARGO_PKG_VERSION"),
        cli.host,
        cli.port,
        cli.backend
    );

    let caps = capability::probe();
    info!("Detected capabilities: {:#?}", caps);

    if !caps.has_any_rt() && cli.backend != "compute-fallback" {
        warn!(
            "No hardware ray-tracing detected on this machine. \
             Vulkan-RT extensions missing on PC, or Metal RT unavailable on Mac. \
             Use --backend compute-fallback to run a software path tracer (slow)."
        );
    }

    // Select a backend. The selection is just a stub right now -- in
    // §5.6.a part 2 each branch wires up its real init.
    let backend_choice = backend::select(&cli.backend, &caps);
    info!("Selected backend: {:?}", backend_choice);

    // Bind the IPC WebSocket + handle frames. Currently just echoes
    // hello / capability messages; the render loop lands in §5.6.a
    // part 2 once the backend is real.
    ipc::serve(&cli.host, cli.port, caps, backend_choice).await
}
