//! Backend selection. The engine has two real backends (Vulkan-RT
//! for PC, Metal-RT for Mac) sharing one Slang shader set. A compute-
//! shader fallback for non-RT hardware is a stretch goal.
//!
//! Sprint 7.5.6.a part 1 only sets up the selection logic + stubs.
//! Part 2 wires the real device init / pipeline creation / BVH
//! building per backend.

use crate::capability::Capabilities;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Vulkan,
    Metal,
    ComputeFallback,
    /// No backend usable on this host -- the engine surfaces this in
    /// the IPC handshake so the editor disables the RayTracedScene
    /// node.
    None,
}

pub fn select(preferred: &str, caps: &Capabilities) -> BackendKind {
    match preferred {
        "vulkan" => {
            if caps.vulkan_rt {
                BackendKind::Vulkan
            } else {
                log::warn!("--backend vulkan requested but Vulkan-RT not detected; falling back");
                fallback(caps)
            }
        }
        "metal" => {
            if caps.metal_present {
                BackendKind::Metal
            } else {
                log::warn!("--backend metal requested but Metal not present; falling back");
                fallback(caps)
            }
        }
        "compute-fallback" => BackendKind::ComputeFallback,
        "auto" | _ => fallback(caps),
    }
}

fn fallback(caps: &Capabilities) -> BackendKind {
    if caps.vulkan_rt {
        BackendKind::Vulkan
    } else if caps.metal_present {
        BackendKind::Metal
    } else if caps.compute_fallback {
        BackendKind::ComputeFallback
    } else {
        BackendKind::None
    }
}

// Backend modules are stubs in part 1 -- they exist so the build
// resolves; real implementations land in §5.6.a part 2.
pub mod vulkan;
pub mod metal;
pub mod compute_fallback;
