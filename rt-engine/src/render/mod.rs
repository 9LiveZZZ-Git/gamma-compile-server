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
pub use metal_path::render_test_triangle;

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
