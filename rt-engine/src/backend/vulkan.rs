//! Vulkan-RT backend stub. §5.6.a part 2 will land:
//!
//!   - Instance + physical device enumeration
//!   - VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure
//!     extension load
//!   - Logical device with the RT queues
//!   - BLAS / TLAS building via vkCmdBuildAccelerationStructuresKHR
//!   - RT pipeline creation via vkCreateRayTracingPipelinesKHR
//!   - SBT (shader binding table) layout
//!   - Per-frame ray dispatch via vkCmdTraceRaysKHR
//!
//! For sprint 7.5.6.a part 1 this is just the module shell.

#[allow(dead_code)]
pub fn initialize_stub() {
    log::info!("[vulkan] backend stub -- real init in §5.6.a part 2");
}
