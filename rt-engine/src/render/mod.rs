//! Rendering kernels. Right now this contains only the Metal "hello
//! triangle" test path that proves the device → compute pipeline →
//! texture readback → PNG-output chain works end-to-end. The
//! single triangle is intersected per-pixel via Möller-Trumbore
//! directly in the kernel -- no acceleration structure yet.
//!
//! Sprint 7.5.6.a part 2b. The next milestone (part 2c) replaces
//! the in-kernel Möller-Trumbore with a real MTLAccelerationStructure
//! lookup so the renderer scales beyond one primitive.

#[cfg(target_os = "macos")]
pub mod metal_path;

#[cfg(target_os = "macos")]
pub use metal_path::{render_test_triangle, MetalRenderer};

#[cfg(not(target_os = "macos"))]
pub fn render_test_triangle(
    _width: u32,
    _height: u32,
    _output: &str,
) -> anyhow::Result<()> {
    Err(anyhow::anyhow!(
        "--render-test currently requires macOS. PC Vulkan-RT path lands in §5.6.b."
    ))
}

/// Stream-friendly renderer trait that the IPC layer can use without
/// caring whether the backend is Metal or Vulkan. Both backends
/// expose the same shape: build once at fixed dimensions, render a
/// frame to a CPU-readable buffer. The streaming path doesn't
/// allocate per frame -- GPU resources are kept alive between calls.
#[cfg(target_os = "macos")]
pub type Renderer = metal_path::MetalRenderer;

// Non-macOS stub. Fields match the MetalRenderer's public surface so
// ipc.rs can use field access (r.width) identically on both
// platforms. The constructor always returns Err since this is just
// a compilation placeholder until §5.6.b lands the Vulkan backend.
#[cfg(not(target_os = "macos"))]
pub struct Renderer {
    pub width: u32,
    pub height: u32,
}

#[cfg(not(target_os = "macos"))]
impl Renderer {
    pub fn new(_w: u32, _h: u32) -> anyhow::Result<Self> {
        Err(anyhow::anyhow!(
            "RT engine streaming requires macOS in part 2c; PC Vulkan-RT support lands in §5.6.b"
        ))
    }
    pub fn render_frame(&self) -> anyhow::Result<Vec<u8>> {
        Err(anyhow::anyhow!("RT engine streaming requires macOS in part 2c"))
    }
}
