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

/// Probe the host. Sprint 7.5.6.a part 2a wired up the Mac path
/// (real MTLDevice + supportsRaytracing + Apple9 family check); the
/// PC path is still stubbed pending the Vulkan backend landing in
/// §5.6.b. Linux + Windows currently report vulkan_rt=true by
/// assumption -- updated when ash is added back to Cargo.toml.
pub fn probe() -> Capabilities {
    let os = std::env::consts::OS.to_string();

    let probe_result = match os.as_str() {
        "windows" | "linux" => ProbeResult {
            vulkan_rt: probe_vulkan_rt(),
            metal_present: false,
            metal_rt_hardware: false,
            gpu_vendor: "unknown".to_string(),
            gpu_name: "unknown (PC detection lands in §5.6.b)".to_string(),
        },
        "macos" => probe_metal(),
        _ => ProbeResult {
            vulkan_rt: false,
            metal_present: false,
            metal_rt_hardware: false,
            gpu_vendor: "unknown".to_string(),
            gpu_name: format!("unsupported os: {}", os),
        },
    };

    Capabilities {
        vulkan_rt: probe_result.vulkan_rt,
        metal_present: probe_result.metal_present,
        metal_rt_hardware: probe_result.metal_rt_hardware,
        compute_fallback: true, // any GPU can do this -- only really false on no-GPU CI runners
        gpu_vendor: probe_result.gpu_vendor,
        gpu_name: probe_result.gpu_name,
        os,
    }
}

struct ProbeResult {
    vulkan_rt: bool,
    metal_present: bool,
    metal_rt_hardware: bool,
    gpu_vendor: String,
    gpu_name: String,
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn probe_vulkan_rt() -> bool {
    // §5.6.b: instance creation + physical device enumeration
    // + extension property query for VK_KHR_ray_tracing_pipeline +
    // VK_KHR_acceleration_structure. Return true if any physical
    // device supports both.
    //
    // Stub until the Vulkan backend lands.
    log::debug!("probe_vulkan_rt: stub returning true (full detection in §5.6.b)");
    true
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn probe_vulkan_rt() -> bool {
    false
}

#[cfg(target_os = "macos")]
fn probe_metal() -> ProbeResult {
    use metal::{Device, MTLGPUFamily};

    let Some(device) = Device::system_default() else {
        log::warn!("probe_metal: no Metal device found (very unusual on macOS)");
        return ProbeResult {
            vulkan_rt: false,
            metal_present: false,
            metal_rt_hardware: false,
            gpu_vendor: "Apple".to_string(),
            gpu_name: "no-Metal-device".to_string(),
        };
    };

    let name = device.name().to_string();
    let supports_rt = device.supports_raytracing();
    // MTLGPUFamilyApple9 = M3 and later; the cutoff between software-
    // emulated RT (MPSRayIntersector, on M1 / M2) and real hardware
    // RT in the GPU cores. M1/M2 still have supports_raytracing ==
    // true via MPS software traversal -- functional but classified
    // "preview only" in the RT plan.
    let apple9_plus = device.supports_family(MTLGPUFamily::Apple9);

    log::info!(
        "probe_metal: device={:?} supports_raytracing={} apple9+={}",
        name, supports_rt, apple9_plus
    );

    ProbeResult {
        vulkan_rt: false,
        // metal_present = "we have a Metal device at all". Always
        // true on macOS unless something is very wrong.
        metal_present: true,
        // metal_rt_hardware = "real hardware ray tracing in the
        // GPU cores" (M3+). M1 / M2 report supports_rt=true but
        // hardware_rt=false (software MPS path).
        metal_rt_hardware: supports_rt && apple9_plus,
        gpu_vendor: "Apple".to_string(),
        gpu_name: name,
    }
}

#[cfg(not(target_os = "macos"))]
fn probe_metal() -> ProbeResult {
    ProbeResult {
        vulkan_rt: false,
        metal_present: false,
        metal_rt_hardware: false,
        gpu_vendor: "n/a".to_string(),
        gpu_name: "n/a (not macOS)".to_string(),
    }
}
