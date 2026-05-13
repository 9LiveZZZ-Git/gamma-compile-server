//! Capability detection. Reports what the host machine can actually
//! do for RT. Output is JSON-serializable so the Node compile-server
//! can read it back via the engine's `--probe` mode.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Capabilities {
    /// Whether Vulkan ray-tracing pipeline extensions are present
    /// (PC: NVIDIA RTX, AMD RDNA2+, Intel Arc).
    pub vulkan_rt: bool,
    /// Whether Metal hardware ray tracing is available (M3+ Apple
    /// Silicon). M1/M2 register as Metal-present but
    /// `metal_rt_hardware == false` -- they run via MPSRayIntersector
    /// software traversal as a "preview only" path.
    pub metal_present: bool,
    pub metal_rt_hardware: bool,
    /// Whether a compute-shader software path-tracing fallback is
    /// usable. Always true if any compute-capable GPU is present.
    pub compute_fallback: bool,
    /// Detected GPU vendor + name strings (informational; the Node
    /// side displays these in the RayTracedScene props pane).
    pub gpu_vendor: String,
    pub gpu_name: String,
    /// Operating system family (for the editor's hardware-target
    /// messaging).
    pub os: String,
}

impl Capabilities {
    /// True if ANY hardware RT path is available.
    pub fn has_any_rt(&self) -> bool {
        self.vulkan_rt || self.metal_rt_hardware
    }

    /// The recommended backend string given current capabilities.
    /// Returns "compute-fallback" if no hardware RT is available.
    #[allow(dead_code)]
    pub fn recommended_backend(&self) -> &'static str {
        if self.vulkan_rt {
            "vulkan"
        } else if self.metal_present {
            "metal"
        } else {
            "compute-fallback"
        }
    }
}

/// Probe the host. Stubs out the actual extension queries -- §5.6.a
/// part 2 wires up real Vulkan / Metal device enumeration. For now
/// this returns sane defaults based on the build target so the IPC
/// handshake + Node-side integration can be exercised.
pub fn probe() -> Capabilities {
    let os = std::env::consts::OS.to_string();

    // §5.6.a part 2 will replace these stub values with real checks:
    //   - Windows / Linux: try Entry::load() + check device features
    //     for ray_tracing_pipeline + acceleration_structure
    //   - macOS: MTLDevice + supportsRaytracing + supportsFamily
    //     (Apple9 = M3+)
    let (vulkan_rt, metal_present, metal_rt_hardware) = match os.as_str() {
        "windows" | "linux" => (probe_vulkan_rt(), false, false),
        "macos" => {
            let (mp, mh) = probe_metal();
            (false, mp, mh)
        }
        _ => (false, false, false),
    };

    Capabilities {
        vulkan_rt,
        metal_present,
        metal_rt_hardware,
        compute_fallback: true, // any GPU can do this -- only really false on no-GPU CI runners
        gpu_vendor: "unknown".to_string(),
        gpu_name: "unknown (full detection lands in §5.6.a part 2)".to_string(),
        os,
    }
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn probe_vulkan_rt() -> bool {
    // §5.6.a part 2: instance creation + physical device enumeration
    // + extension property query for VK_KHR_ray_tracing_pipeline +
    // VK_KHR_acceleration_structure. Return true if any physical
    // device supports both.
    //
    // Stub for part 1: assume available on Windows / Linux (the user
    // will see runtime failures if not, which is fine for the
    // scaffolding push).
    log::debug!("probe_vulkan_rt: stub returning true (full detection in §5.6.a part 2)");
    true
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn probe_vulkan_rt() -> bool {
    false
}

#[cfg(target_os = "macos")]
fn probe_metal() -> (bool, bool) {
    // §5.6.a part 2: MTLCreateSystemDefaultDevice + check
    // supportsRaytracing + supportsFamily(MTLGPUFamilyApple9) for
    // hardware RT (M3+) vs software (M1/M2).
    //
    // Stub for part 1: report present + hardware-RT true on macOS so
    // the IPC handshake can be exercised. Dev machine is M4 so this
    // is the realistic case.
    log::debug!("probe_metal: stub returning (true, true) (full detection in §5.6.a part 2)");
    (true, true)
}

#[cfg(not(target_os = "macos"))]
fn probe_metal() -> (bool, bool) {
    (false, false)
}
