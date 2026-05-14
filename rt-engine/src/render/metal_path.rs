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

/// Per-mesh material uniform. Matches MSL MaterialUniform layout.
/// Single struct discriminated by `flags.x` so the kernel can `switch`
/// on the material type (0=Unlit, 1=Phong, 2=PBR).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MaterialUniform {
    /// .xyz = base color (albedo), .w = vertex_mix [0..1]
    albedo: [f32; 4],
    /// .x = metallic, .y = roughness, .z = shininess, .w = ambient
    params: [f32; 4],
    /// .x = material type tag (0=Unlit, 1=Phong, 2=PBR)
    flags: [u32; 4],
}

/// Per-geometry offset table. For c-2 smooth shading the kernel
/// needs to fetch the three per-vertex normals of the hit triangle
/// from the global concatenated buffers, so we track both the
/// vertex_offset (into vertex_normals) and the index_offset (into
/// vertex_indices, for meshes built with indices). Meshes without
/// indices (raw 3-vertex-per-triangle layout) signal that with
/// `is_indexed == 0` and use direct vertex_offset + primitive*3.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GeomOffset {
    vertex_offset: u32, // into vertex_normals[]
    index_offset: u32,  // into vertex_indices[]
    is_indexed: u32,    // 0 or 1
    _pad: u32,          // 16-byte total
}

/// One light slot in the uniform. Holds enough data for any of the
/// three editor-side light types -- the kernel branches on `flags.x`
/// to pick which fields to read.
///
/// Encoding:
///   flags.x = 0  Directional: pos_or_dir.xyz = direction TO light;
///                range/cones unused.
///   flags.x = 1  Point:       pos_or_dir.xyz = world position;
///                range_cones.x = range (attenuation falls to 0
///                at this distance), .yz unused, spot_dir unused.
///   flags.x = 2  Spot:        pos_or_dir.xyz = world position;
///                range_cones.x = range, .y = cos(inner_angle/2),
///                .z = cos(outer_angle/2); spot_dir.xyz = cone
///                axis (direction the spotlight is shining).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LightSlot {
    pos_or_dir: [f32; 4],   // .xyz = pos (point/spot) or dir-to-light (dir)
    color: [f32; 4],        // .xyz = color, .w = intensity
    spot_dir: [f32; 4],     // .xyz = spot cone axis (where light shines)
    range_cones: [f32; 4],  // .x = range, .y = cos_inner, .z = cos_outer
    flags: [u32; 4],        // .x = light type (0=dir, 1=point, 2=spot)
}

/// Lights uniform: ambient term + up to 4 mixed-type light slots.
/// MSL matching struct is in triangle.metal.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LightsUniform {
    /// .xyz = ambient color, .w = ambient intensity (0..1 typically)
    ambient: [f32; 4],
    /// .x = number of active lights (0..=4)
    meta: [u32; 4],
    /// Per-light slot data. Slot.flags.x discriminates the type.
    slots: [LightSlot; 4],
}

/// Per-mesh build output returned by `build_mesh_buffers`. Holds the
/// GPU buffers the AS needs (positions + optional indices) AND
/// CPU-side per-vertex data (normals + a copy of indices) that gets
/// concatenated into global kernel-side buffers in update_scene.
struct MeshBuildResult {
    position_buffer: Buffer,
    index_buffer: Option<Buffer>,
    triangle_count: u32,
    /// Per-vertex normals (one entry per vertex, .xyz = normal).
    vertex_normals: Vec<[f32; 4]>,
    /// Copy of the indices for kernel-side smooth-normal lookup.
    /// Empty for non-indexed meshes (kernel uses primitive*3 + i).
    indices_for_kernel: Vec<u32>,
    /// Number of vertices in this mesh (= vertex_normals.len()).
    vertex_count: u32,
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
    material_buffer: Option<Buffer>,       // per-mesh MaterialUniform[]
    vertex_normals_buffer: Option<Buffer>, // global per-vertex normals (smooth shading)
    vertex_indices_buffer: Option<Buffer>, // global indices for kernel-side lookup
    geom_offsets_buffer: Option<Buffer>,   // per-mesh GeomOffset[]
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
            material_buffer: None,
            vertex_normals_buffer: None,
            vertex_indices_buffer: None,
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
            self.material_buffer = None;
            self.vertex_normals_buffer = None;
            self.vertex_indices_buffer = None;
            self.geom_offsets_buffer = None;
            self.lights_buffer = None;
            self._scene_buffers.clear();
            self.has_scene = false;
            return Ok(());
        }

        // Build per-mesh GPU resources. Pre-transform vertices on the
        // CPU (option-a simplicity; instance AS with per-instance
        // transforms lands in 2e-2). Accumulate smooth-shading data
        // into global buffers indexed by GeomOffset[geometry_id].
        let mut scene_buffers: Vec<Buffer> = Vec::new();
        let mut geom_descs_owned: Vec<AccelerationStructureTriangleGeometryDescriptor> =
            Vec::with_capacity(scene.meshes.len());
        let mut materials: Vec<MaterialUniform> = Vec::with_capacity(scene.meshes.len());
        let mut geom_offsets: Vec<GeomOffset> = Vec::with_capacity(scene.meshes.len());
        let mut global_vertex_normals: Vec<[f32; 4]> = Vec::new();
        let mut global_vertex_indices: Vec<u32> = Vec::new();
        let mut total_triangles: u64 = 0;

        for mesh in &scene.meshes {
            let built = self.build_mesh_buffers(mesh)?;
            total_triangles += built.triangle_count as u64;

            // Record this geometry's offsets BEFORE appending its data.
            geom_offsets.push(GeomOffset {
                vertex_offset: global_vertex_normals.len() as u32,
                index_offset: global_vertex_indices.len() as u32,
                is_indexed: if built.indices_for_kernel.is_empty() { 0 } else { 1 },
                _pad: 0,
            });
            global_vertex_normals.extend_from_slice(&built.vertex_normals);
            global_vertex_indices.extend_from_slice(&built.indices_for_kernel);

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

            // Per-mesh material → MaterialUniform. The discriminator
            // tag (.flags.x) tells the kernel which BRDF to evaluate.
            materials.push(material_to_uniform(&mesh.material));
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

        // Global per-vertex normals buffer (one float4 per vertex,
        // concatenated across all meshes). Indexed by
        // `geom_offsets[gid].vertex_offset + local_vertex_id`.
        let vnorm_buf = self.device.new_buffer_with_data(
            global_vertex_normals.as_ptr() as *const c_void,
            (global_vertex_normals.len().max(1) * std::mem::size_of::<[f32; 4]>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Global indices buffer for kernel-side smooth-normal lookup.
        // Empty for fully non-indexed scenes (we still allocate a
        // 1-element dummy buffer because Metal's set_buffer doesn't
        // accept zero-length).
        let vidx_buf = if global_vertex_indices.is_empty() {
            self.device.new_buffer(
                4,
                MTLResourceOptions::StorageModeShared,
            )
        } else {
            self.device.new_buffer_with_data(
                global_vertex_indices.as_ptr() as *const c_void,
                (global_vertex_indices.len() * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        // Per-geometry offset table.
        let geom_off_buf = self.device.new_buffer_with_data(
            geom_offsets.as_ptr() as *const c_void,
            (geom_offsets.len() * std::mem::size_of::<GeomOffset>()) as u64,
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

        // Per-mesh material buffer.
        let mat_buf = self.device.new_buffer_with_data(
            materials.as_ptr() as *const c_void,
            (materials.len() * std::mem::size_of::<MaterialUniform>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        self.accel = Some(accel);
        self.camera_buffer = Some(cam_buf);
        self.material_buffer = Some(mat_buf);
        self.vertex_normals_buffer = Some(vnorm_buf);
        self.vertex_indices_buffer = Some(vidx_buf);
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

    /// Update just the per-mesh material array (cheap; no AS rebuild).
    /// Used by Params for live-dragging PhongMat.shininess /
    /// PhysicalMat.metallic / etc. Caller must supply a slice of
    /// the same length the current AS was built from; mismatched
    /// length means the mesh set changed and we should be doing a
    /// full update_scene instead (caller decides via signature).
    pub fn update_materials(&mut self, materials: &[Material]) -> anyhow::Result<()> {
        let Some(buf) = self.material_buffer.as_ref() else {
            return Err(anyhow!("material buffer not yet allocated"));
        };
        let buffer_count = (buf.length() as usize) / std::mem::size_of::<MaterialUniform>();
        if buffer_count != materials.len() {
            return Err(anyhow!(
                "material count mismatch: scene has {} mesh(es) but Params sent {} material(s)",
                buffer_count,
                materials.len()
            ));
        }
        let ptr = buf.contents() as *mut MaterialUniform;
        for (i, m) in materials.iter().enumerate() {
            unsafe { ptr.add(i).write(material_to_uniform(m)); }
        }
        Ok(())
    }

    /// Build per-mesh GPU resources + CPU-side smooth-shading data.
    /// Stride-11 editor vertex layout: pos.xyz + color.rgb + normal.xyz
    /// + uv.xy. We extract pos.xyz (transformed by mesh.transform) for
    /// the AS, and normal.xyz (rotated by the upper-3×3 of the
    /// transform) for kernel-side smooth shading. Indices are copied
    /// for the kernel to use in barycentric vertex lookup.
    ///
    /// Normal transform note: for identity / rotation / uniform-scale
    /// transforms the upper-3×3 of M correctly rotates normals (after
    /// re-normalizing). Non-uniform scale requires the inverse-
    /// transpose; we don't support non-uniform mesh transforms in
    /// 2e-1 / c-2, so the cheap upper-3×3 path is fine.
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
        if stride_f < 9 || vertices.len() % stride_f != 0 {
            return Err(anyhow!(
                "invalid vertex stride {} for {} floats (need >= 9: pos.xyz + color.rgb + normal.xyz)",
                stride,
                vertices.len()
            ));
        }
        let vcount = vertices.len() / stride_f;

        // Pull pos.xyz, apply transform.
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

        // Pull normal.xyz (stride offset 6..8), rotate by upper-3×3
        // of the transform, store as float4 (.xyz = normal, .w pad).
        let mut vertex_normals: Vec<[f32; 4]> = Vec::with_capacity(vcount);
        for vi in 0..vcount {
            let nx = vertices[vi * stride_f + 6];
            let ny = vertices[vi * stride_f + 7];
            let nz = vertices[vi * stride_f + 8];
            // Upper-3×3 rotation (M[0..2], M[4..6], M[8..10] in
            // column-major) applied to the normal.
            let rx = tm[0] * nx + tm[4] * ny + tm[8] * nz;
            let ry = tm[1] * nx + tm[5] * ny + tm[9] * nz;
            let rz = tm[2] * nx + tm[6] * ny + tm[10] * nz;
            let n = Vec3::new(rx, ry, rz).normalize_or_zero();
            vertex_normals.push([n.x, n.y, n.z, 0.0]);
        }

        let (index_buffer, triangle_count, indices_for_kernel) =
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
                (Some(ib), tcount, indices.clone())
            } else {
                if vcount % 3 != 0 {
                    return Err(anyhow!(
                        "non-indexed mesh vertex count {} not divisible by 3",
                        vcount
                    ));
                }
                // Empty indices vec signals "use primitive*3+i directly".
                (None, (vcount / 3) as u32, Vec::new())
            };

        Ok(MeshBuildResult {
            position_buffer,
            index_buffer,
            triangle_count,
            vertex_normals,
            indices_for_kernel,
            vertex_count: vcount as u32,
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
        let mat_buf = self.material_buffer.as_ref().unwrap();
        let vnorm_buf = self.vertex_normals_buffer.as_ref().unwrap();
        let vidx_buf = self.vertex_indices_buffer.as_ref().unwrap();
        let geom_off_buf = self.geom_offsets_buffer.as_ref().unwrap();
        let lights_buf = self.lights_buffer.as_ref().unwrap();

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_texture(0, Some(&self.texture));
        enc.set_acceleration_structure(0, Some(&**accel));
        enc.use_resource(&**accel, MTLResourceUsage::Read);
        enc.set_buffer(1, Some(cam_buf), 0);
        enc.set_buffer(2, Some(mat_buf), 0);
        enc.set_buffer(3, Some(vnorm_buf), 0);
        enc.set_buffer(4, Some(vidx_buf), 0);
        enc.set_buffer(5, Some(geom_off_buf), 0);
        enc.set_buffer(6, Some(lights_buf), 0);

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

/// Convert one editor-side Material variant into the wire-format
/// MaterialUniform the kernel consumes. The flags.x discriminator
/// (0=Unlit, 1=Phong, 2=PBR) decides which BRDF the kernel
/// evaluates; other params are stuffed into albedo / params and
/// only the ones relevant to that BRDF get read.
///
/// Material variants the c-2 kernel handles:
///   Unlit   -> flags.x = 0 (just outputs albedo, ignores lights)
///   Phong   -> flags.x = 1 (Lambert + Blinn-Phong specular)
///   Pbr     -> flags.x = 2 (Cook-Torrance GGX + Schlick)
///
/// Glass / Mirror / Shader -- not in c-2 scope. Glass + Mirror land
/// in 7.5.6.d (refraction); Shader requires Slang transpile work.
/// For now they collapse to Unlit with the base color.
fn material_to_uniform(m: &Material) -> MaterialUniform {
    match m {
        Material::Unlit { color, vertex_mix } => MaterialUniform {
            albedo: [color[0], color[1], color[2], *vertex_mix],
            params: [0.0; 4],
            flags: [0, 0, 0, 0],
        },
        Material::Phong {
            color,
            shininess,
            ambient,
        } => MaterialUniform {
            albedo: [color[0], color[1], color[2], 0.0],
            // .x = metallic (unused), .y = roughness (unused),
            // .z = shininess, .w = ambient.
            params: [0.0, 0.0, *shininess, *ambient],
            flags: [1, 0, 0, 0],
        },
        Material::Pbr {
            color,
            metallic,
            roughness,
        } => MaterialUniform {
            albedo: [color[0], color[1], color[2], 0.0],
            // Clamp roughness to a safe floor; pure 0 produces a
            // mirror with a div-by-zero risk in the GGX denominator.
            params: [*metallic, roughness.max(0.04), 0.0, 0.0],
            flags: [2, 0, 0, 0],
        },
        Material::Mirror { tint } => MaterialUniform {
            albedo: [tint[0], tint[1], tint[2], 0.0],
            params: [0.0; 4],
            // Type tag 3 = Mirror (c-3 added Glass=4 alongside).
            flags: [3, 0, 0, 0],
        },
        Material::Glass {
            color,
            ior,
            absorption: _,  // absorption param could feed Beer-Lambert in
                            // a future pass; c-d v1 uses `color` as
                            // a per-bounce multiplicative tint instead.
        } => MaterialUniform {
            albedo: [color[0], color[1], color[2], 0.0],
            // .x = ior (index of refraction). .y/.z/.w unused for glass.
            params: [*ior, 0.0, 0.0, 0.0],
            flags: [4, 0, 0, 0],
        },
        Material::Shader { color, .. } => MaterialUniform {
            // ShaderMat fallback: render as Unlit with the base color.
            // (Slang transpile is a long-running future item.)
            albedo: [color[0], color[1], color[2], 0.0],
            params: [0.0; 4],
            flags: [0, 0, 0, 0],
        },
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
    let mut slots: Vec<LightSlot> = Vec::new();
    for light in lights {
        if slots.len() >= 4 { break; }
        let slot = match light {
            Light::Directional { direction, color, intensity } => {
                // Direction TO the light (editor convention).
                let d = Vec3::from(*direction).normalize_or_zero();
                LightSlot {
                    pos_or_dir: [d.x, d.y, d.z, 0.0],
                    color: [color[0], color[1], color[2], *intensity],
                    spot_dir: [0.0; 4],
                    range_cones: [0.0; 4],
                    flags: [0, 0, 0, 0],
                }
            }
            Light::Point { position, color, intensity, range } => {
                LightSlot {
                    pos_or_dir: [position[0], position[1], position[2], 0.0],
                    color: [color[0], color[1], color[2], *intensity],
                    spot_dir: [0.0; 4],
                    range_cones: [range.max(0.001), 0.0, 0.0, 0.0],
                    flags: [1, 0, 0, 0],
                }
            }
            Light::Spot {
                position,
                direction,
                color,
                intensity,
                range,
                inner_angle_deg,
                outer_angle_deg,
            } => {
                // Spot direction is "where the light shines" --
                // normalize. cos_inner > cos_outer (smaller angle
                // = larger cosine). The kernel smoothsteps between
                // them for the cone falloff.
                let sd = Vec3::from(*direction).normalize_or_zero();
                let inner_rad = inner_angle_deg.to_radians() * 0.5;
                let outer_rad = outer_angle_deg.to_radians() * 0.5;
                let cos_inner = inner_rad.cos();
                let cos_outer = outer_rad.cos().min(cos_inner - 1e-4);
                LightSlot {
                    pos_or_dir: [position[0], position[1], position[2], 0.0],
                    color: [color[0], color[1], color[2], *intensity],
                    spot_dir: [sd.x, sd.y, sd.z, 0.0],
                    range_cones: [range.max(0.001), cos_inner, cos_outer, 0.0],
                    flags: [2, 0, 0, 0],
                }
            }
            // Area lights deferred to §5.6.d (refraction sprint).
            // For c-3 they degrade to "no light contribution".
            Light::Area { .. } => continue,
        };
        slots.push(slot);
    }

    if slots.is_empty() {
        // Default sunlit angle (matches the DirectionalLight node's
        // registry defaults: upper-right-front, warm white).
        let d = Vec3::new(0.3, 1.0, 0.4).normalize();
        slots.push(LightSlot {
            pos_or_dir: [d.x, d.y, d.z, 0.0],
            color: [1.0, 0.98, 0.92, 1.0],
            spot_dir: [0.0; 4],
            range_cones: [0.0; 4],
            flags: [0, 0, 0, 0],
        });
    }

    let count = slots.len() as u32;
    // Pad up to 4.
    while slots.len() < 4 {
        slots.push(LightSlot::zeroed());
    }

    LightsUniform {
        ambient: [0.15, 0.17, 0.20, 1.0],
        meta: [count, 0, 0, 0],
        slots: [slots[0], slots[1], slots[2], slots[3]],
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
