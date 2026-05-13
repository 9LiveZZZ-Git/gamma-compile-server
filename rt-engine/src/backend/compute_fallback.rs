//! Software path-tracing fallback via compute shaders on any GPU.
//! Intended for users without hardware RT (older NVIDIA pre-RTX,
//! older AMD pre-RDNA2, M1/M2 Apple Silicon, integrated GPUs).
//!
//! Quality target: 1-2 spp at 720p ~ 15 fps on a GTX 1060-class GPU.
//! Not interactive, but lets the user preview their RT setup before
//! upgrading hardware. Could also run on a CPU as a last resort.
//!
//! Sprint 7.5.6.a part 1 stub; full impl is a stretch goal.

#[allow(dead_code)]
pub fn initialize_stub() {
    log::info!("[compute-fallback] software path tracer stub -- impl in §5.6.g or later");
}
