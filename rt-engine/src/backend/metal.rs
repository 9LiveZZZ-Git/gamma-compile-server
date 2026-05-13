//! Metal-RT backend stub. §5.6.a part 2 will land:
//!
//!   - MTLDevice acquisition (system default)
//!   - Hardware RT capability check (supportsRaytracing,
//!     supportsFamily(MTLGPUFamilyApple9) for M3+ hardware)
//!   - MTLAccelerationStructure building via
//!     MTLAccelerationStructureDescriptor
//!   - MTLComputePipelineState for the path tracer (Metal-RT runs
//!     inside compute shaders, not separate pipelines like Vulkan)
//!   - Per-frame dispatch via MTLComputeCommandEncoder
//!   - Optional: MPSSVGFDenoise integration for the denoiser pass
//!
//! For sprint 7.5.6.a part 1 this is just the module shell.

#[allow(dead_code)]
pub fn initialize_stub() {
    log::info!("[metal] backend stub -- real init in §5.6.a part 2");
}
