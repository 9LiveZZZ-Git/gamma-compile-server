//! Scene state types -- the JSON-serializable shape the editor
//! sends to the engine to describe what to render. Mirrors the
//! editor-side Scene + Camera + Light + material structures so the
//! browser can just stringify the resolved state + ship it.
//!
//! Sprint 7.5.6.a part 1: types only. Actual deserialization +
//! engine consumption land in part 2.

// Suppress unused warnings for the part-1 scaffolding -- everything
// here gets consumed by the renderer in part 2.
#![allow(dead_code)]

use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub camera: Camera,
    pub meshes: Vec<MeshInstance>,
    #[serde(default)]
    pub lights: Vec<Light>,
    #[serde(default)]
    pub environment: Option<Environment>,
    #[serde(default)]
    pub clear_color: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera {
    pub mode: CameraMode,
    pub pos: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov_deg: f32,
    pub near: f32,
    pub far: f32,
    #[serde(default)]
    pub ortho_size: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CameraMode {
    Perspective,
    Orthographic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshInstance {
    pub geometry: GeometryRef,
    pub transform: [f32; 16], // column-major 4x4
    pub material: Material,
}

/// Geometry reference -- either an inline mesh (vertex + index data
/// included in the message) or a cached reference (engine keeps a
/// content-addressable cache keyed by a hash of the vertex data).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum GeometryRef {
    Inline {
        vertices: Vec<f32>,
        indices: Option<Vec<u32>>,
        /// Vertex stride in floats (typically 11: pos.xyz + color.rgb + normal.xyz + uv.xy).
        stride: u32,
    },
    Cached {
        hash: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Material {
    Unlit {
        color: [f32; 3],
        #[serde(default)]
        vertex_mix: f32,
    },
    Phong {
        color: [f32; 3],
        #[serde(default = "default_shininess")]
        shininess: f32,
        #[serde(default = "default_ambient")]
        ambient: f32,
    },
    Pbr {
        color: [f32; 3],
        #[serde(default)]
        metallic: f32,
        #[serde(default = "default_roughness")]
        roughness: f32,
    },
    Glass {
        color: [f32; 3],
        ior: f32,
        #[serde(default)]
        absorption: [f32; 3],
    },
    Mirror {
        tint: [f32; 3],
    },
    /// Shader material -- references a Slang preset by name on the
    /// engine side. Custom user WGSL → Slang transpile is a future
    /// item (cross-compile is tricky); for now we keep a curated
    /// preset list mirroring the editor's ShaderMat options.
    Shader {
        preset: String,
        color: [f32; 3],
        time: f32,
        freq: f32,
        intensity: f32,
        #[serde(default)]
        texture_layer: i32,
    },
}

fn default_shininess() -> f32 { 32.0 }
fn default_ambient() -> f32 { 0.15 }
fn default_roughness() -> f32 { 0.5 }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Light {
    Directional {
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
    },
    Point {
        position: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
    },
    Spot {
        position: [f32; 3],
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
        inner_angle_deg: f32,
        outer_angle_deg: f32,
    },
    /// Rectangular area light -- only meaningful for RT (raster
    /// scene falls back to a regular point or directional approx).
    Area {
        position: [f32; 3],
        normal: [f32; 3],
        width: f32,
        height: f32,
        color: [f32; 3],
        intensity: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Environment {
    /// HDRI / equirectangular env map. The actual texture data is
    /// uploaded via a separate `texture-upload` IPC message and
    /// referenced by id here.
    Hdri { texture_id: String, exposure: f32 },
    /// Procedural sky with atmospheric scattering. Sun-position
    /// driven; cheap to evaluate per-ray.
    ProceduralSky {
        sun_dir: [f32; 3],
        sun_intensity: f32,
        turbidity: f32,
    },
    /// Constant gradient: top + horizon + bottom. Cheapest env.
    Gradient {
        top: [f32; 3],
        horizon: [f32; 3],
        bottom: [f32; 3],
    },
}

/// Helper -- build a column-major Mat4 from glam::Mat4 (engine-side
/// math uses glam internally; this matches the editor's wire format).
#[allow(dead_code)]
pub fn mat_to_array(m: Mat4) -> [f32; 16] {
    m.to_cols_array()
}

#[allow(dead_code)]
pub fn lookat(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    Mat4::look_at_rh(eye, target, up)
}
