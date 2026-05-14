//! Metal renderer with hardware ray-tracing path.
//!
//! Sprint 7.5.6.a part 2e-1. Renderer is now editor-driven: meshes,
//! camera, and clear color all come from a Scene struct supplied by
//! the IPC layer. Per-mesh material support is option (a) -- flat
//! RGB color per mesh, indexed by geometry_id in the kernel.
//!
//! Lifecycle:
//!   - new(): compile kernel, allocate output texture, no AS yet
//!   - update_scene(): rebuild AS from the new mesh set, upload
//!     camera uniform + per-mesh color buffer
//!   - update_camera(): patch the camera uniform in place (cheap;
//!     no AS rebuild). Used for live camera-orbit updates in 2e-2.
//!   - render_frame(): dispatch the kernel; if no scene yet, return
//!     a clear-colored buffer CPU-side (no GPU work).

use anyhow::{anyhow, Context};
use bytemuck::{Pod, Zeroable};
use metal::{
    AccelerationStructure, AccelerationStructureTriangleGeometryDescriptor, Array, Buffer,
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLAttributeFormat,
    MTLIndexType, MTLLanguageVersion, MTLOrigin, MTLPixelFormat, MTLRegion,
    MTLResourceOptions, MTLResourceUsage, MTLSize, MTLStorageMode, MTLTextureUsage,
    PrimitiveAccelerationStructureDescriptor, Texture, TextureDescriptor,
};
use std::ffi::c_void;

use crate::scene::{Camera, GeometryRef, Light, Material, MeshInstance, Scene};
use glam::Vec3;

const KERNEL_SRC: &str = include_str!("../shaders/triangle.metal");

/// Camera uniform layout matching the MSL CameraUniform struct.
/// float4 alignment throughout (16-byte) so Metal and Rust agree on
/// memory layout without packed_float3 / explicit padding tweaks.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    eye: [f32; 4],     // .xyz = eye, .w = unused
    right: [f32; 4],   // .xyz = right basis
    up: [f32; 4],      // .xyz = up basis (orthogonalized)
    forward: [f32; 4], // .xyz = look direction
    misc: [f32; 4],    // .x = tan(fov/2), .y = aspect, .zw = unused
}

/// Per-mesh (= per-geometry-in-the-AS) color buffer entry.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PrimitiveColor {
    color: [f32; 4], // .xyz = RGB, .w = unused
}

/// Per-geometry offset into the global per-primitive normals buffer.
/// The kernel does `flat_normals[geom_offsets[geom_id].triangle_offset
/// + primitive_id]` to fetch the hit triangle's flat normal.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GeomOffset {
    triangle_offset: u32, // index into flat_normals[] for this geometry's first triangle
    _pad: [u32; 3],       // 16-byte alignment for the buffer
}

/// Lights uniform layout matching MSL LightsUniform. Up to 4
/// directional lights; per-c-1 scope, point/spot lights from the
/// editor are parsed in scene.rs but skipped here (added in c-2).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LightsUniform {
    /// .xyz = ambient color, .w = ambient intensity (0..1 typically)
    ambient: [f32; 4],
    /// .x = number of directional lights (0..=4)
    meta: [u32; 4],
    /// `dirs[i].xyz` = direction TO the light (normalized).
    dirs: [[f32; 4]; 4],
    /// `colors[i].xyz` = color, `colors[i].w` = intensity.
    colors: [[f32; 4]; 4],
}

/// Per-mesh build output returned by `build_mesh_buffers`. Holds the
/// GPU buffers the AS needs (positions + optional indices) AND the
/// CPU-side flat-normals vector that we'll concatenate into the
/// global normals buffer used by the kernel for Lambert shading.
struct MeshBuildResult {
    position_buffer: Buffer,
    index_buffer: Option<Buffer>,
    triangle_count: u32,
    flat_normals: Vec<[f32; 4]>, // one entry per triangle, .xyz = normal
}

/// Reusable Metal renderer with hardware RT. Output texture size is
/// fixed at construction (`new(w, h)`); rebuilding for a different
/// resolution means constructing a new MetalRenderer.
pub struct MetalRenderer {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    texture: Texture,
    pub width: u32,
    pub height: u32,

    // Scene state -- None until the IPC layer pushes a Scene message.
    accel: Option<AccelerationStructure>,
    camera_buffer: Option<Buffer>,
    primitive_color_buffer: Option<Buffer>,
    flat_normals_buffer: Option<Buffer>,   // global per-triangle flat normals
    geom_offsets_buffer: Option<Buffer>,   // geom_id → first-triangle index
    lights_buffer: Option<Buffer>,
    // Vertex / index buffers we built for the current AS. Must
    // outlive the AS (Metal's BVH stores GPU pointers into them).
    _scene_buffers: Vec<Buffer>,
    clear_color: [f32; 3],
    has_scene: bool,
}

impl MetalRenderer {
    pub fn new(width: u32, height: u32) -> anyhow::Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| anyhow!("MTLCreateSystemDefaultDevice returned null"))?;

        log::info!(
            "[metal-renderer] device={:?} {}x{} (hardware RT, awaiting scene)",
            device.name(),
            width,
            height
        );

        let queue = device.new_command_queue();

        // Compile the kernel up front; doesn't depend on scene state.
        let options = CompileOptions::new();
        options.set_language_version(MTLLanguageVersion::V2_4);
        let lib = device
            .new_library_with_source(KERNEL_SRC, &options)
            .map_err(|e| anyhow!("MSL compile failed: {}", e))?;
        let func = lib
            .get_function("rt_scene", None)
            .map_err(|e| anyhow!("kernel function 'rt_scene' not found: {}", e))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow!("compute pipeline create failed: {}", e))?;

        let tex_desc = TextureDescriptor::new();
        tex_desc.set_width(width as u64);
        tex_desc.set_height(height as u64);
        tex_desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        tex_desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        tex_desc.set_storage_mode(MTLStorageMode::Shared);
        let texture = device.new_texture(&tex_desc);

        Ok(MetalRenderer {
            device,
            queue,
            pipeline,
            texture,
            width,
            height,
            accel: None,
            camera_buffer: None,
            primitive_color_buffer: None,
            flat_normals_buffer: None,
            geom_offsets_buffer: None,
            lights_buffer: None,
            _scene_buffers: Vec::new(),
            clear_color: [0.05, 0.06, 0.10],
            has_scene: false,
        })
    }

    /// Apply editor-driven scene state. Rebuilds the acceleration
    /// structure from the new mesh set + uploads the camera uniform
    /// + per-mesh color buffer. Existing scene resources are dropped.
    pub fn update_scene(&mut self, scene: &Scene) -> anyhow::Result<()> {
        log::info!(
            "[metal-renderer] update_scene: {} mesh(es), camera@{:?}, clear={:?}",
            scene.meshes.len(),
            scene.camera.pos,
            scene.clear_color
        );
        self.clear_color = scene.clear_color;

        if scene.meshes.is_empty() {
            // Empty scene -- drop the AS, render path falls to the
            // clear-color CPU return below.
            self.accel = None;
            self.camera_buffer = None;
            self.primitive_color_buffer = None;
            self.flat_normals_buffer = None;
            self.geom_offsets_buffer = None;
            self.lights_buffer = None;
            self._scene_buffers.clear();
            self.has_scene = false;
            return Ok(());
        }

        // Build per-mesh GPU resources. Pre-transform vertices on the
        // CPU (option-a simplicity; instance AS with per-instance
        // transforms lands in 2e-2).
        let mut scene_buffers: Vec<Buffer> = Vec::new();
        let mut geom_descs_owned: Vec<AccelerationStructureTriangleGeometryDescriptor> =
            Vec::with_capacity(scene.meshes.len());
        let mut primitive_colors: Vec<PrimitiveColor> = Vec::with_capacity(scene.meshes.len());
        let mut geom_offsets: Vec<GeomOffset> = Vec::with_capacity(scene.meshes.len());
        // Global per-triangle flat-normal buffer concatenated across
        // all meshes; geom_offsets[gid].triangle_offset tells the
        // kernel where this geometry's triangles begin.
        let mut global_flat_normals: Vec<[f32; 4]> = Vec::new();
        let mut total_triangles: u64 = 0;

        for mesh in &scene.meshes {
            let built = self.build_mesh_buffers(mesh)?;
            total_triangles += built.triangle_count as u64;

            // Record where this geometry's triangles start in the
            // global normals buffer BEFORE pushing them.
            geom_offsets.push(GeomOffset {
                triangle_offset: global_flat_normals.len() as u32,
                _pad: [0; 3],
            });
            global_flat_normals.extend_from_slice(&built.flat_normals);

            let gd = AccelerationStructureTriangleGeometryDescriptor::descriptor();
            gd.set_vertex_buffer(Some(&built.position_buffer));
            gd.set_vertex_buffer_offset(0);
            gd.set_vertex_stride(std::mem::size_of::<[f32; 3]>() as u64);
            gd.set_vertex_format(MTLAttributeFormat::Float3);
            gd.set_triangle_count(built.triangle_count as u64);
            if let Some(ref idx_buf) = built.index_buffer {
                gd.set_index_buffer(Some(idx_buf));
                gd.set_index_buffer_offset(0);
                gd.set_index_type(MTLIndexType::UInt32);
            }
            geom_descs_owned.push(gd);

            scene_buffers.push(built.position_buffer);
            if let Some(idx_buf) = built.index_buffer {
                scene_buffers.push(idx_buf);
            }

            // Per-mesh color from the material. Option (a) only uses
            // the .color (or .tint for Mirror); everything else
            // collapses to a flat RGB for now.
            let color = match &mesh.material {
                Material::Unlit { color, .. } => *color,
                Material::Phong { color, .. } => *color,
                Material::Pbr { color, .. } => *color,
                Material::Glass { color, .. } => *color,
                Material::Mirror { tint } => *tint,
                Material::Shader { color, .. } => *color,
            };
            primitive_colors.push(PrimitiveColor {
                color: [color[0], color[1], color[2], 1.0],
            });
        }

        // Upcast each Triangle descriptor to base Geometry for the
        // Array<Geometry> that PrimitiveAccelerationStructureDescriptor
        // wants. metal-rs's From<Triangle> for Geometry bumps the
        // refcount + reinterprets the pointer.
        let geom_descs_base: Vec<metal::AccelerationStructureGeometryDescriptor> =
            geom_descs_owned.iter().map(|g| g.clone().into()).collect();
        let geom_array: &metal::ArrayRef<metal::AccelerationStructureGeometryDescriptor> =
            Array::from_owned_slice(&geom_descs_base);

        let prim_desc = PrimitiveAccelerationStructureDescriptor::descriptor();
        prim_desc.set_geometry_descriptors(geom_array);

        let sizes = self
            .device
            .acceleration_structure_sizes_with_descriptor(&prim_desc);
        log::info!(
            "[metal-renderer] AS sizes: storage={}B scratch={}B refit={}B (for {} mesh(es))",
            sizes.acceleration_structure_size,
            sizes.build_scratch_buffer_size,
            sizes.refit_scratch_buffer_size,
            scene.meshes.len()
        );

        let accel = self
            .device
            .new_acceleration_structure_with_size(sizes.acceleration_structure_size);
        let scratch = self.device.new_buffer(
            sizes.build_scratch_buffer_size,
            MTLResourceOptions::StorageModePrivate,
        );

        let build_cb = self.queue.new_command_buffer();
        let as_enc = build_cb.new_acceleration_structure_command_encoder();
        as_enc.build_acceleration_structure(&accel, &prim_desc, &scratch, 0);
        as_enc.end_encoding();
        build_cb.commit();
        build_cb.wait_until_completed();
        log::info!(
            "[metal-renderer] AS build complete ({} geometry/ies, {} primitive(s) total)",
            geom_descs_owned.len(),
            total_triangles
        );

        // Global flat-normals buffer (one float4 per triangle across
        // all meshes). Indexed by `geom_offsets[geom_id].triangle_offset
        // + primitive_id` in the kernel.
        let flat_norm_buf = self.device.new_buffer_with_data(
            global_flat_normals.as_ptr() as *const c_void,
            (global_flat_normals.len().max(1) * std::mem::size_of::<[f32; 4]>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Per-geometry offset table.
        let geom_off_buf = self.device.new_buffer_with_data(
            geom_offsets.as_ptr() as *const c_void,
            (geom_offsets.len().max(1) * std::mem::size_of::<GeomOffset>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Lights uniform.
        let lights_uniform = build_lights_uniform(&scene.lights);
        let lights_buf = self.device.new_buffer_with_data(
            &lights_uniform as *const _ as *const c_void,
            std::mem::size_of::<LightsUniform>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        log::info!(
            "[metal-renderer] lights: {} directional + ambient {:?}",
            lights_uniform.meta[0],
            &lights_uniform.ambient[..3]
        );

        // Camera uniform.
        let cam_uniform = build_camera_uniform(&scene.camera, self.width, self.height);
        let cam_buf = self.device.new_buffer_with_data(
            &cam_uniform as *const _ as *const c_void,
            std::mem::size_of::<CameraUniform>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Per-mesh color buffer.
        let color_buf = self.device.new_buffer_with_data(
            primitive_colors.as_ptr() as *const c_void,
            (primitive_colors.len() * std::mem::size_of::<PrimitiveColor>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        self.accel = Some(accel);
        self.camera_buffer = Some(cam_buf);
        self.primitive_color_buffer = Some(color_buf);
        self.flat_normals_buffer = Some(flat_norm_buf);
        self.geom_offsets_buffer = Some(geom_off_buf);
        self.lights_buffer = Some(lights_buf);
        self._scene_buffers = scene_buffers;
        self.has_scene = true;
        Ok(())
    }

    /// Update just the camera (cheap; no AS rebuild). Used by Params
    /// IPC messages for live camera orbit.
    pub fn update_camera(&mut self, camera: &Camera) -> anyhow::Result<()> {
        let cam_uniform = build_camera_uniform(camera, self.width, self.height);
        if let Some(buf) = self.camera_buffer.as_ref() {
            // Shared-storage buffer -- contents() gives a CPU pointer
            // we can overwrite directly. Cheaper than reallocating.
            let ptr = buf.contents() as *mut CameraUniform;
            unsafe { ptr.write(cam_uniform); }
        } else {
            let cam_buf = self.device.new_buffer_with_data(
                &cam_uniform as *const _ as *const c_void,
                std::mem::size_of::<CameraUniform>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            self.camera_buffer = Some(cam_buf);
        }
        Ok(())
    }

    /// Update just the lights uniform (cheap; no AS rebuild). Used
    /// by Params IPC messages so live-dragging a light's intensity
    /// or hue slider doesn't churn the BVH.
    pub fn update_lights(&mut self, lights: &[Light]) -> anyhow::Result<()> {
        let uni = build_lights_uniform(lights);
        if let Some(buf) = self.lights_buffer.as_ref() {
            let ptr = buf.contents() as *mut LightsUniform;
            unsafe { ptr.write(uni); }
        } else {
            let lights_buf = self.device.new_buffer_with_data(
                &uni as *const _ as *const c_void,
                std::mem::size_of::<LightsUniform>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            self.lights_buffer = Some(lights_buf);
        }
        Ok(())
    }

    /// Build vertex + index buffers for one mesh. Extracts pos.xyz
    /// from the stride-N interleaved input (editor sends stride=11:
    /// pos.xyz + color.rgb + normal.xyz + uv.xy) and pre-transforms
    /// by the mesh's column-major 4×4 transform. ALSO computes a flat
    /// per-triangle normal from the transformed vertices -- the
    /// kernel uses these for Lambert shading.
    fn build_mesh_buffers(
        &self,
        mesh: &MeshInstance,
    ) -> anyhow::Result<MeshBuildResult> {
        let (vertices, indices_opt, stride) = match &mesh.geometry {
            GeometryRef::Inline {
                vertices,
                indices,
                stride,
            } => (vertices, indices.as_ref(), *stride),
            GeometryRef::Cached { .. } => {
                return Err(anyhow!("cached geometry refs not supported in part 2e-1"));
            }
        };

        let stride_f = stride as usize;
        if stride_f < 3 || vertices.len() % stride_f != 0 {
            return Err(anyhow!(
                "invalid vertex stride {} for {} floats",
                stride,
                vertices.len()
            ));
        }
        let vcount = vertices.len() / stride_f;

        // Pull out pos.xyz, apply transform.
        let tm = mesh.transform; // column-major 4×4
        let mut positions: Vec<f32> = Vec::with_capacity(vcount * 3);
        for vi in 0..vcount {
            let x = vertices[vi * stride_f];
            let y = vertices[vi * stride_f + 1];
            let z = vertices[vi * stride_f + 2];
            let tx = tm[0] * x + tm[4] * y + tm[8] * z + tm[12];
            let ty = tm[1] * x + tm[5] * y + tm[9] * z + tm[13];
            let tz = tm[2] * x + tm[6] * y + tm[10] * z + tm[14];
            positions.push(tx);
            positions.push(ty);
            positions.push(tz);
        }

        let position_buffer = self.device.new_buffer_with_data(
            positions.as_ptr() as *const c_void,
            (positions.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let (index_buffer, triangle_count, indices_for_normals) =
            if let Some(indices) = indices_opt {
                if indices.len() % 3 != 0 {
                    return Err(anyhow!("index count {} not divisible by 3", indices.len()));
                }
                let ib = self.device.new_buffer_with_data(
                    indices.as_ptr() as *const c_void,
                    (indices.len() * std::mem::size_of::<u32>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let tcount = (indices.len() / 3) as u32;
                (Some(ib), tcount, Some(indices.as_slice()))
            } else {
                if vcount % 3 != 0 {
                    return Err(anyhow!(
                        "non-indexed mesh vertex count {} not divisible by 3",
                        vcount
                    ));
                }
                (None, (vcount / 3) as u32, None)
            };

        // Compute per-triangle flat normal from the (transformed)
        // positions. Cross product of two edges, normalized; degenerate
        // triangles get a zero normal (kernel branches on that).
        let mut flat_normals: Vec<[f32; 4]> = Vec::with_capacity(triangle_count as usize);
        let pos_at = |i: usize| -> Vec3 {
            Vec3::new(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2])
        };
        for tri in 0..triangle_count as usize {
            let (i0, i1, i2) = if let Some(idx) = indices_for_normals {
                (
                    idx[tri * 3] as usize,
                    idx[tri * 3 + 1] as usize,
                    idx[tri * 3 + 2] as usize,
                )
            } else {
                (tri * 3, tri * 3 + 1, tri * 3 + 2)
            };
            let v0 = pos_at(i0);
            let v1 = pos_at(i1);
            let v2 = pos_at(i2);
            let n = (v1 - v0).cross(v2 - v0).normalize_or_zero();
            flat_normals.push([n.x, n.y, n.z, 0.0]);
        }

        Ok(MeshBuildResult {
            position_buffer,
            index_buffer,
            triangle_count,
            flat_normals,
        })
    }

    /// Render one frame. If no scene has been pushed yet, returns
    /// clear-color pixels CPU-side (cheap; no GPU dispatch).
    pub fn render_frame(&self) -> anyhow::Result<Vec<u8>> {
        if !self.has_scene {
            let r = (self.clear_color[0].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (self.clear_color[1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (self.clear_color[2].clamp(0.0, 1.0) * 255.0) as u8;
            let mut pixels = Vec::with_capacity((self.width * self.height * 4) as usize);
            for _ in 0..(self.width * self.height) {
                pixels.extend_from_slice(&[r, g, b, 255]);
            }
            return Ok(pixels);
        }

        let accel = self.accel.as_ref().unwrap();
        let cam_buf = self.camera_buffer.as_ref().unwrap();
        let color_buf = self.primitive_color_buffer.as_ref().unwrap();
        let flat_norm_buf = self.flat_normals_buffer.as_ref().unwrap();
        let geom_off_buf = self.geom_offsets_buffer.as_ref().unwrap();
        let lights_buf = self.lights_buffer.as_ref().unwrap();

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_texture(0, Some(&self.texture));
        enc.set_acceleration_structure(0, Some(&**accel));
        enc.use_resource(&**accel, MTLResourceUsage::Read);
        enc.set_buffer(1, Some(cam_buf), 0);
        enc.set_buffer(2, Some(color_buf), 0);
        enc.set_buffer(3, Some(flat_norm_buf), 0);
        enc.set_buffer(4, Some(geom_off_buf), 0);
        enc.set_buffer(5, Some(lights_buf), 0);

        let tg_size = MTLSize { width: 16, height: 16, depth: 1 };
        let tg_count = MTLSize {
            width: ((self.width as u64) + 15) / 16,
            height: ((self.height as u64) + 15) / 16,
            depth: 1,
        };
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let bytes_per_row = (self.width * 4) as u64;
        let mut pixels = vec![0u8; (self.width * self.height * 4) as usize];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: self.width as u64,
                height: self.height as u64,
                depth: 1,
            },
        };
        self.texture.get_bytes(
            pixels.as_mut_ptr() as *mut c_void,
            bytes_per_row,
            region,
            0,
        );
        Ok(pixels)
    }
}

/// Pack the editor-side `lights` array into the kernel's
/// LightsUniform shape. c-1 scope: only Light::Directional entries
/// are honored; Point / Spot / Area are silently skipped (re-added
/// in c-2 when the kernel grows real light evaluation per type).
/// If the scene has no directional lights, we synthesize a warm
/// "sunset-from-up-right" default so shaded scenes aren't pure
/// ambient (which would look completely flat).
fn build_lights_uniform(lights: &[Light]) -> LightsUniform {
    let mut packed: Vec<(Vec3, [f32; 3], f32)> = Vec::new();
    for light in lights {
        if let Light::Directional {
            direction,
            color,
            intensity,
        } = light
        {
            // Editor convention (matches raster path): `direction` is
            // the direction FROM the surface TOWARD the light source
            // -- e.g. (0,1,0) means "light directly overhead, rays
            // shine downward onto surface". For Lambert we use this
            // vector directly: `dot(N, dirs[i].xyz)`. No CPU negation.
            let d = Vec3::from(*direction).normalize_or_zero();
            packed.push((d, *color, *intensity));
            if packed.len() >= 4 {
                break;
            }
        }
        // c-1: ignore non-directional lights. c-2 adds point/spot
        // evaluation per-type in the kernel.
    }
    if packed.is_empty() {
        // Default sunlit angle (matches the DirectionalLight node's
        // registry defaults: upper-right-front, warm white). Looks
        // fine for any scene with shaded materials wired up.
        let d = Vec3::new(0.3, 1.0, 0.4).normalize();
        packed.push((d, [1.0, 0.98, 0.92], 1.0));
    }

    let mut dirs = [[0.0f32; 4]; 4];
    let mut colors = [[0.0f32; 4]; 4];
    for (i, (d, c, intensity)) in packed.iter().enumerate() {
        dirs[i] = [d.x, d.y, d.z, 0.0];
        colors[i] = [c[0], c[1], c[2], *intensity];
    }

    LightsUniform {
        ambient: [0.15, 0.17, 0.20, 1.0],
        meta: [packed.len() as u32, 0, 0, 0],
        dirs,
        colors,
    }
}

/// Convert the editor's pos/target/up/fov camera into a precomputed
/// orthonormal basis + tan(fov/2) + aspect tuple that the kernel can
/// consume without any per-pixel cross() / normalize() work.
fn build_camera_uniform(camera: &Camera, width: u32, height: u32) -> CameraUniform {
    let eye = Vec3::from(camera.pos);
    let target = Vec3::from(camera.target);
    let up_hint = Vec3::from(camera.up);
    let forward = (target - eye).normalize_or_zero();
    let right = forward.cross(up_hint).normalize_or_zero();
    let up = right.cross(forward); // already unit-length
    let fov_rad = camera.fov_deg.to_radians();
    let tan_half_fov = (fov_rad * 0.5).tan();
    let aspect = (width as f32) / (height as f32).max(1.0);
    CameraUniform {
        eye: [eye.x, eye.y, eye.z, 1.0],
        right: [right.x, right.y, right.z, 0.0],
        up: [up.x, up.y, up.z, 0.0],
        forward: [forward.x, forward.y, forward.z, 0.0],
        misc: [tan_half_fov, aspect, 0.0, 0.0],
    }
}

/// One-shot test render: build a renderer, push a hard-coded test
/// scene (since --render-test is invoked without an IPC client),
/// render once, save to PNG, exit. Driven by the --render-test CLI.
pub fn render_test_triangle(width: u32, height: u32, output: &str) -> anyhow::Result<()> {
    let mut renderer = MetalRenderer::new(width, height)?;
    renderer.update_scene(&default_test_scene())?;
    let pixels = renderer.render_frame()?;

    let img = image::RgbaImage::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow!("RgbaImage::from_raw failed (buffer size mismatch)"))?;
    img.save(output)
        .with_context(|| format!("PNG save to {} failed", output))?;

    log::info!("[render-test] wrote {} to {}", img.as_raw().len(), output);
    Ok(())
}

fn default_test_scene() -> Scene {
    use crate::scene::{CameraMode, MeshInstance};
    Scene {
        camera: Camera {
            mode: CameraMode::Perspective,
            pos: [0.0, 0.0, 2.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_deg: 60.0,
            near: 0.1,
            far: 100.0,
            ortho_size: 0.0,
        },
        meshes: vec![MeshInstance {
            geometry: GeometryRef::Inline {
                // pos.xyz + color.rgb + normal.xyz + uv.xy (stride 11)
                vertices: vec![
                    -0.866, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.866, -0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                    0.0, 1.0, 0.0, 0.3, 0.5, 1.0, 0.0, 0.0, 1.0, 0.5, 1.0,
                ],
                indices: None,
                stride: 11,
            },
            transform: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            material: Material::Unlit {
                color: [1.0, 0.4, 0.4],
                vertex_mix: 0.0,
            },
        }],
        lights: vec![],
        environment: None,
        clear_color: [0.05, 0.06, 0.10],
    }
}
