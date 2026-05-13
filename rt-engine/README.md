# gamma-rt-engine

Native hardware ray-tracing engine for the Gamma Node Editor.

Lives in the same git repository as the Node `gamma-compile-server`
but is built / distributed separately. The Node side spawns this
binary on demand when the editor first uses a `RayTracedScene` node.

## Status

**Sprint 7.5.6.a part 1 — scaffolding only.** Capability detection +
WebSocket IPC handshake work; no actual ray tracing yet. See
`docs/RAYTRACING.md` at the repo root for the full plan.

## Targets

| Platform | Backend             | RT path                |
|----------|---------------------|------------------------|
| Windows  | Vulkan-RT           | NVIDIA RTX / AMD RDNA2+ / Intel Arc |
| Linux    | Vulkan-RT           | same as Windows |
| macOS    | Metal-RT            | M3+ hardware RT; M1/M2 = preview only via MPS software traversal |
| Anywhere | Compute-fallback    | Slow path-tracing for old hardware (stretch goal) |

## Building

```bash
# From the rt-engine/ directory:
cargo build --release
```

Output: `target/release/gamma-rt-engine` (or `.exe` on Windows).

### Platform prerequisites

**Windows / Linux**:
- Vulkan SDK 1.3+ (https://vulkan.lunarg.com/)
- A GPU + driver with `VK_KHR_ray_tracing_pipeline` + `VK_KHR_acceleration_structure` extensions

**macOS**:
- macOS 14+ (Sonoma) for full Metal 3 ray-tracing API
- Apple Silicon (Intel Macs not supported)

## Running

```bash
# Probe capabilities + exit (used by gamma-compile-server on first
# install to know whether to offer RT in the editor):
gamma-rt-engine --probe

# Run as a long-lived WebSocket server (the normal mode -- spawned
# by the Node compile-server):
gamma-rt-engine --port 9100 --backend auto
```

CLI flags:

- `--port <N>` — WebSocket bind port (default 9100)
- `--host <HOST>` — bind interface (default 127.0.0.1, loopback only)
- `--backend <auto|vulkan|metal|compute-fallback>` — force a backend
- `--probe` — capability scan + JSON output, exit

## Communication protocol

The Node `gamma-compile-server` proxies between the browser editor
and this engine over WebSocket. JSON control plane + binary data
plane (H.264 NAL units for rendered frames).

See `src/ipc.rs` for the message types.

## Directory layout

```
rt-engine/
├── Cargo.toml
├── README.md             (this file)
├── src/
│   ├── main.rs           Entry + CLI
│   ├── capability.rs     Platform capability probe
│   ├── ipc.rs            WebSocket IPC handler
│   ├── scene.rs          Scene state types
│   └── backend/
│       ├── mod.rs        Backend selection
│       ├── vulkan.rs     Vulkan-RT backend (PC)        — part 2
│       ├── metal.rs      Metal-RT backend (Mac)        — part 2
│       └── compute_fallback.rs   Software fallback     — stretch
└── target/                  cargo build output (gitignored)
```

Slang shaders for the path tracer live in `src/shaders/`; part 2
adds a `build.rs` that pre-compiles them to SPIR-V (Vulkan) and MSL
(Metal) via the Slang compiler.

## License

MIT (same as the rest of `gamma-compile-server`).
