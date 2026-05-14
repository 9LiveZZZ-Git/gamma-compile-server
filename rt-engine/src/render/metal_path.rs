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
use crate::render::metalfx::{ScalerConfig, TemporalDenoisedScaler};
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

/// Per-frame uniform updated every render_frame call. Holds the
/// frame counter (RNG seed + accumulation count) AND the previous
/// frame's view-projection matrix (for motion-vector computation
/// in the path tracer). Lives in `path_state_buffer`, written via
/// the shared-storage contents() pointer at the start of each
/// render so the kernel sees the latest values without a fresh
/// allocation.
///
/// Layout (96 bytes total):
///   offset  0..3   frame_count: u32
///   offset  4..7   spp: u32              (f.3.d-fix6)
///   offset  8..11  max_bounces: u32      (f.3.d-fix6)
///   offset 12..15  pad: 1× u32
///   offset 16..79  prev_view_proj: column-major float4x4
///   offset 80..95  jitter: float4
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PathStateUniform {
    frame_count: u32,
    /// f.3.d-fix6 -- samples-per-pixel-per-frame. Set via the
    /// `quality` patch on a Params message. Editor resolves the
    /// `quality` preset to {1, 4, 16}; users can override [1, 16].
    /// Kernel clamps to [1, 16] for safety regardless of what we
    /// upload. Higher values target edge / disocclusion noise that
    /// TDS history validation can't clear on its own.
    spp: u32,
    /// f.3.d-fix6 -- max ray-tracing recursion depth. Set via the
    /// `quality` patch on a Params message. Editor resolves the
    /// `quality` preset to {2, 4, 8}; users can override [1, 8].
    /// Kernel clamps to [1, 8] for safety. Replaces the file-scope
    /// `constant int MAX_BOUNCES = 4` the shader used pre-fix6.
    max_bounces: u32,
    _pad: [u32; 1],
    prev_view_proj: [[f32; 4]; 4],
    /// Sprint 7.5.6.f.3.d -- per-frame global sub-pixel jitter.
    /// .xy = Halton(2,3) sequence value in [0, 1]. The kernel uses
    /// this as the same offset for ALL pixels in a given frame
    /// (replacing c-e's per-pixel random offset). MetalFX gets the
    /// same value (centered to [-0.5, 0.5]) via setJitterOffset so
    /// it can correlate this frame's hits with the prior frame's
    /// hits across sub-pixel positions.
    jitter: [f32; 4],
}

/// Halton sequence in base `b`, for index `i`. Returns a low-
/// discrepancy value in [0, 1]. Standard temporal-AA jitter source
/// -- successive frames' jitters cover sub-pixel positions evenly
/// faster than uniform random, giving cleaner reconstruction.
fn halton(base: u32, mut i: u32) -> f32 {
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
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
    // Sprint 7.5.6.f -- separate denoise pipeline. Path-tracing
    // kernel writes accum + normal G-buffer; denoise kernel reads
    // both, applies a 5x5 edge-aware spatial filter, writes display.
    denoise_pipeline: ComputePipelineState,
    texture: Texture,
    pub width: u32,
    pub height: u32,

    // Sprint 7.5.6.e -- path-tracing accumulation. Persistent RGBA32F
    // texture into which each frame's sample contribution gets added.
    // Output = accum / frame_count. Cleared (= "first frame writes
    // initial value") by setting self.frame_count to 0 -- the kernel
    // branches on frame_count == 0 and overwrites instead of reading-
    // adding-writing.
    accum_texture: Texture,
    // Sprint 7.5.6.f -- primary-hit G-buffer. Three textures:
    //   normal_texture  -- .xyz = world-space normal at first opaque
    //                      hit (RGBA32F). f v1: edge-stopping for
    //                      spatial denoise. f.3: input to MetalFX.
    //   depth_texture   -- .x = linear distance from camera to first
    //                      opaque hit (R32F). f.3: MetalFX
    //                      disocclusion detection.
    //   motion_texture  -- .xy = screen-space motion vector in UV
    //                      coordinates (RG16F). f.3: MetalFX temporal
    //                      reprojection. Computed from current hit
    //                      world position + previous-frame view-proj.
    //   albedo_texture  -- .xyz = raw surface color before lighting
    //                      (RGBA8Unorm). f.3: MetalFX demodulates
    //                      so edges between materials don't blur.
    normal_texture: Texture,
    depth_texture: Texture,
    motion_texture: Texture,
    albedo_texture: Texture,
    // Sprint 7.5.6.f.3.c -- noisy single-sample color (RGBA16F).
    // MetalFX's color input: this frame's path-traced sample, NOT
    // the engine-side running accumulation. MetalFX does its own
    // temporal accumulation internally using motion vectors.
    noisy_color_texture: Texture,
    // MetalFX's output texture (RGBA16F). HDR float so the
    // temporal blend doesn't quantize. A separate tonemap kernel
    // then converts to RGBA8 for the display.
    metalfx_output_texture: Texture,
    // f.3.d-fix4 -- additional G-buffers required by MetalFX's
    // denoised scaler (per WWDC25 §211 "Go further with Metal 4
    // games"). Without these the denoiser receives uninitialized
    // auxiliary inputs and never converges on static scenes.
    //   roughness_texture: R16Float, [0,1], 1.0 = fully diffuse
    //   specular_albedo_texture: RGBA16F, F0 (Fresnel @ 0 deg)
    roughness_texture: Texture,
    specular_albedo_texture: Texture,
    // Sprint 7.5.6.f.3 -- MetalFX TemporalDenoisedScaler. Apple's
    // neural-net temporal denoiser for path-traced rendering. None
    // if creation failed (older macOS, unsupported GPU, or build
    // running without MetalFX framework available). f.3.c routes
    // rendering through it when present; falls back to rt_denoise
    // (spatial filter) when None.
    metalfx_scaler: Option<TemporalDenoisedScaler>,
    /// Tonemap pipeline: copies the MetalFX output (RGBA16F) into the
    /// display texture (RGBA8). For c-3 v1 this is a straight clamp +
    /// linear-to-sRGB-ish remap; ACES / Reinhard etc come in polish.
    tonemap_pipeline: ComputePipelineState,
    // Sprint 7.5.6.f.3.b -- camera matrix history for motion vector
    // computation. Two fields:
    //   current_view_proj: the view*projection matrix of the most
    //     recently uploaded camera. Computed in update_camera /
    //     update_scene from the editor-sent pose.
    //   view_proj_history: what was used to render the PREVIOUS
    //     frame. Shifted from current_view_proj at the end of each
    //     render_frame. The kernel reads this (via path_state's
    //     prev_view_proj field) to project current hit points
    //     backward into the previous screen UV.
    // None on both for the very first frame after construction --
    // path_state.prev_view_proj falls back to identity, kernel
    // outputs zero motion (correct for "no history").
    current_view_proj: Option<[f32; 16]>,
    view_proj_history: Option<[f32; 16]>,
    frame_count: u32,
    /// f.3.d-fix6 -- samples-per-pixel-per-frame. Updated via the
    /// "quality" patch on Params messages. Clamped to [1, 16] on set.
    /// Default 1 to match pre-f.3.d-fix6 behavior; the editor's
    /// RayTracedScene node defaults to the "preview" preset (4 spp).
    spp: u32,
    /// f.3.d-fix6 -- max bounce depth for the path tracer. Updated
    /// via the "quality" patch on Params messages. Clamped to [1, 8]
    /// on set. Default 4 to match the pre-fix6 `MAX_BOUNCES` constant
    /// the shader used; editor's RayTracedScene defaults to 4 via the
    /// "preview" preset.
    bounces: u32,
    // Tiny per-frame uniform: just the current frame_count (used to
    // seed RNG + as the denominator for accum averaging). Updated in
    // place each render_frame via the shared-storage buffer's
    // contents() pointer; no reallocation per frame.
    path_state_buffer: Buffer,

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

        // Sprint 7.5.6.f -- second pipeline for the spatial denoiser.
        // Same MSL source; different entry point. Compiled once at
        // construction, dispatched after rt_scene each frame.
        let denoise_func = lib
            .get_function("rt_denoise", None)
            .map_err(|e| anyhow!("kernel function 'rt_denoise' not found: {}", e))?;
        let denoise_pipeline = device
            .new_compute_pipeline_state_with_function(&denoise_func)
            .map_err(|e| anyhow!("denoise pipeline create failed: {}", e))?;

        let tex_desc = TextureDescriptor::new();
        tex_desc.set_width(width as u64);
        tex_desc.set_height(height as u64);
        tex_desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        tex_desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        tex_desc.set_storage_mode(MTLStorageMode::Shared);
        let texture = device.new_texture(&tex_desc);

        // Sprint 7.5.6.e -- accumulation texture for progressive
        // path tracing. RGBA32F so we can sum many noisy samples
        // without quantization. Private storage since only the GPU
        // reads/writes it (no CPU readback).
        let accum_desc = TextureDescriptor::new();
        accum_desc.set_width(width as u64);
        accum_desc.set_height(height as u64);
        accum_desc.set_pixel_format(MTLPixelFormat::RGBA32Float);
        accum_desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        accum_desc.set_storage_mode(MTLStorageMode::Private);
        let accum_texture = device.new_texture(&accum_desc);

        // Sprint 7.5.6.f -- normal G-buffer. f.3.c: format changed
        // from RGBA32Float to RGBA16Float to match the MetalFX
        // scaler's normalTextureFormat. 16-bit float is plenty for
        // normals (15 bits per component = ~1/32k precision, way
        // more than the [-1, 1] range needs). Saves memory + makes
        // MetalFX happy.
        let normal_desc = TextureDescriptor::new();
        normal_desc.set_width(width as u64);
        normal_desc.set_height(height as u64);
        normal_desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
        normal_desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        normal_desc.set_storage_mode(MTLStorageMode::Private);
        let normal_texture = device.new_texture(&normal_desc);

        // Sprint 7.5.6.f.3.b -- depth + motion + albedo G-buffers.
        // Depth = linear world-space distance from camera to first
        // opaque hit. Motion = current_uv - prev_uv per pixel. Albedo
        // = raw surface color before lighting (demodulated).
        let depth_desc = TextureDescriptor::new();
        depth_desc.set_width(width as u64);
        depth_desc.set_height(height as u64);
        depth_desc.set_pixel_format(MTLPixelFormat::R32Float);
        depth_desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        depth_desc.set_storage_mode(MTLStorageMode::Private);
        let depth_texture = device.new_texture(&depth_desc);

        let motion_desc = TextureDescriptor::new();
        motion_desc.set_width(width as u64);
        motion_desc.set_height(height as u64);
        motion_desc.set_pixel_format(MTLPixelFormat::RG16Float);
        motion_desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        motion_desc.set_storage_mode(MTLStorageMode::Private);
        let motion_texture = device.new_texture(&motion_desc);

        let albedo_desc = TextureDescriptor::new();
        albedo_desc.set_width(width as u64);
        albedo_desc.set_height(height as u64);
        albedo_desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        albedo_desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        albedo_desc.set_storage_mode(MTLStorageMode::Private);
        let albedo_texture = device.new_texture(&albedo_desc);

        // Sprint 7.5.6.f.3.c -- noisy single-sample color (path tracer
        // input to MetalFX) + MetalFX output. Both RGBA16Float so
        // we don't quantize HDR samples before the temporal blend.
        let noisy_desc = TextureDescriptor::new();
        noisy_desc.set_width(width as u64);
        noisy_desc.set_height(height as u64);
        noisy_desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
        noisy_desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        noisy_desc.set_storage_mode(MTLStorageMode::Private);
        let noisy_color_texture = device.new_texture(&noisy_desc);

        let fxout_desc = TextureDescriptor::new();
        fxout_desc.set_width(width as u64);
        fxout_desc.set_height(height as u64);
        fxout_desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
        fxout_desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        fxout_desc.set_storage_mode(MTLStorageMode::Private);
        let metalfx_output_texture = device.new_texture(&fxout_desc);

        // f.3.d-fix4 -- roughness + specular albedo G-buffers,
        // required auxiliary inputs for the denoised scaler.
        // Roughness is a single channel in [0,1]; R16Float gives
        // ~11 bits of mantissa over that range, plenty for the
        // denoiser's smooth-vs-rough classification. Specular albedo
        // is per-channel F0, kept at RGBA16F to match the rest of
        // the HDR pipeline.
        let rough_desc = TextureDescriptor::new();
        rough_desc.set_width(width as u64);
        rough_desc.set_height(height as u64);
        rough_desc.set_pixel_format(MTLPixelFormat::R16Float);
        rough_desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        rough_desc.set_storage_mode(MTLStorageMode::Private);
        let roughness_texture = device.new_texture(&rough_desc);

        let spec_desc = TextureDescriptor::new();
        spec_desc.set_width(width as u64);
        spec_desc.set_height(height as u64);
        spec_desc.set_pixel_format(MTLPixelFormat::RGBA16Float);
        spec_desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        spec_desc.set_storage_mode(MTLStorageMode::Private);
        let specular_albedo_texture = device.new_texture(&spec_desc);

        // Tonemap pipeline (RGBA16F MetalFX output -> RGBA8 display).
        let tonemap_func = lib
            .get_function("rt_tonemap", None)
            .map_err(|e| anyhow!("kernel function 'rt_tonemap' not found: {}", e))?;
        let tonemap_pipeline = device
            .new_compute_pipeline_state_with_function(&tonemap_func)
            .map_err(|e| anyhow!("tonemap pipeline create failed: {}", e))?;

        // Per-frame state uniform: frame_count + prev_view_proj.
        // Sized for PathStateUniform (80 bytes). Updated in place
        // each render via the shared buffer's contents() pointer.
        let path_state_buffer = device.new_buffer(
            std::mem::size_of::<PathStateUniform>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Sprint 7.5.6.f.3.a -- attempt to create the MetalFX scaler.
        // Returns Ok(None) if the device / OS doesn't support it; we
        // log + continue with the existing spatial denoiser path in
        // that case. The dimensions match the current render target.
        // f.3.b will add the G-buffer outputs the scaler reads; f.3.c
        // will switch render_frame to use it instead of rt_denoise.
        let metalfx_scaler = match TemporalDenoisedScaler::try_new(
            &device,
            &ScalerConfig {
                input_width: width,
                input_height: height,
                output_width: width,
                output_height: height,
                ..ScalerConfig::default()
            },
        ) {
            Ok(Some(s)) => {
                log::info!(
                    "[metalfx] TemporalDenoisedScaler ready: in {}x{} -> out {}x{}",
                    s.input_width, s.input_height, s.output_width, s.output_height
                );
                Some(s)
            }
            Ok(None) => {
                log::warn!(
                    "[metalfx] TemporalDenoisedScaler not available on this device/OS; \
                     falling back to spatial denoiser for now. Requires macOS 15+ \
                     and an Apple Silicon GPU."
                );
                None
            }
            Err(e) => {
                log::warn!("[metalfx] scaler creation threw: {}; falling back", e);
                None
            }
        };

        Ok(MetalRenderer {
            device,
            queue,
            pipeline,
            denoise_pipeline,
            texture,
            width,
            height,
            accum_texture,
            normal_texture,
            depth_texture,
            motion_texture,
            albedo_texture,
            noisy_color_texture,
            metalfx_output_texture,
            roughness_texture,
            specular_albedo_texture,
            metalfx_scaler,
            tonemap_pipeline,
            current_view_proj: None,
            view_proj_history: None,
            frame_count: 0,
            spp: 1,
            bounces: 4,
            path_state_buffer,
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

    /// Reset path-tracing accumulation. Called whenever the scene
    /// changes (camera move, light tweak, material change, geometry
    /// rebuild) so we don't average a stale image with a fresh one.
    /// Doesn't allocate -- just zeroes the frame counter so the
    /// kernel's "frame_count == 0" branch overwrites accumulation
    /// instead of reading-adding-writing.
    ///
    /// f.3.b: also clears the view-proj history so the next frame's
    /// motion vectors are all zero (no valid history to project
    /// against). MetalFX will then treat the next frame as
    /// "fully new content", same as if the temporal denoiser just
    /// started.
    pub fn reset_accumulation(&mut self) {
        self.frame_count = 0;
        self.view_proj_history = None;
    }

    /// f.3.d-fix6 -- samples per pixel per frame. Mapped from the
    /// editor's `quality` preset (1 / 4 / 16) plus an optional user
    /// override on the `samples` param. Clamped here to [1, 16] so a
    /// stray upload can't pin the GPU on a 1000-spp dispatch.
    ///
    /// Changing SPP also resets accumulation since the per-frame
    /// noise distribution changes character (more samples = lower
    /// variance), and we don't want the running accumulation texture
    /// to blend pre/post-change samples with mismatched magnitudes.
    pub fn set_spp(&mut self, spp: u32) {
        let new_spp = spp.clamp(1, 16);
        if new_spp != self.spp {
            log::info!("[metal-renderer] spp {} -> {}", self.spp, new_spp);
            self.spp = new_spp;
            self.reset_accumulation();
        }
    }

    /// f.3.d-fix6 -- max ray-tracing recursion depth (bounce budget).
    /// Mapped from the editor's `quality` preset (2 / 4 / 8) plus an
    /// optional user override on the `bounces` param. Clamped to
    /// [1, 8] -- the path tracer can't do more without the per-bounce
    /// RNG salt schedule overflowing.
    ///
    /// Changing the bounce budget invalidates the path-tracing
    /// accumulation (different bounce count = different mean radiance
    /// per pixel), so we reset like set_spp does.
    pub fn set_bounces(&mut self, bounces: u32) {
        let new_b = bounces.clamp(1, 8);
        if new_b != self.bounces {
            log::info!("[metal-renderer] bounces {} -> {}", self.bounces, new_b);
            self.bounces = new_b;
            self.reset_accumulation();
        }
    }

    /// Apply editor-driven scene state. Rebuilds the acceleration
    /// structure from the new mesh set + uploads the camera uniform
    /// + per-mesh color buffer. Existing scene resources are dropped.
    pub fn update_scene(&mut self, scene: &Scene) -> anyhow::Result<()> {
        // Any scene change invalidates the path-tracing accumulation
        // (geometry/camera/lights/materials all changed).
        self.reset_accumulation();
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

        // Camera uniform. update_scene also implicitly resets motion
        // history (the path_state's prev_view_proj is shifted from
        // the renderer's view_proj_history, which gets cleared by
        // reset_accumulation -> first frame after a scene change has
        // identity prev_view_proj -> kernel outputs zero motion).
        let (cam_uniform, current_vp) =
            build_camera_uniform(&scene.camera, self.width, self.height);
        self.current_view_proj = Some(current_vp);
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
    /// IPC messages for live camera orbit. Resets accumulation since
    /// the camera move changes which pixels see which world points
    /// (averaging a static scene through a moving camera = motion
    /// blur, not what we want).
    pub fn update_camera(&mut self, camera: &Camera) -> anyhow::Result<()> {
        self.reset_accumulation();
        let (cam_uniform, current_vp) =
            build_camera_uniform(camera, self.width, self.height);
        self.current_view_proj = Some(current_vp);
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
        self.reset_accumulation();
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
        self.reset_accumulation();
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
    ///
    /// `&mut self` because c-e (path tracing) advances `self.frame_count`
    /// after each successful render. Pre-c-e this was `&self`.
    pub fn render_frame(&mut self) -> anyhow::Result<Vec<u8>> {
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

        // Sprint 7.5.6.e + f.3.b + f.3.d -- update per-frame path-
        // tracing state. path_state_buffer is shared storage so we
        // patch the contents() pointer each frame; no reallocation.
        let prev_vp_for_motion = self.view_proj_history.unwrap_or(IDENTITY_VIEW_PROJ);
        let pvp_cols: [[f32; 4]; 4] = [
            [prev_vp_for_motion[0],  prev_vp_for_motion[1],  prev_vp_for_motion[2],  prev_vp_for_motion[3]],
            [prev_vp_for_motion[4],  prev_vp_for_motion[5],  prev_vp_for_motion[6],  prev_vp_for_motion[7]],
            [prev_vp_for_motion[8],  prev_vp_for_motion[9],  prev_vp_for_motion[10], prev_vp_for_motion[11]],
            [prev_vp_for_motion[12], prev_vp_for_motion[13], prev_vp_for_motion[14], prev_vp_for_motion[15]],
        ];
        // f.3.d -- Halton(2,3) per-frame sub-pixel jitter. Used as
        // the SAME offset for every pixel this frame; over many
        // frames the sequence covers the sub-pixel space evenly.
        // Halton index = frame_count + 1 so frame 0 isn't (0, 0)
        // (which puts every ray at pixel corner; trivially correlated).
        let halton_idx = self.frame_count.wrapping_add(1);
        let jitter_x = halton(2, halton_idx);
        let jitter_y = halton(3, halton_idx);
        let path_state = PathStateUniform {
            frame_count: self.frame_count,
            spp: self.spp,
            max_bounces: self.bounces,
            _pad: [0; 1],
            prev_view_proj: pvp_cols,
            jitter: [jitter_x, jitter_y, 0.0, 0.0],
        };
        unsafe {
            let ptr = self.path_state_buffer.contents() as *mut PathStateUniform;
            ptr.write(path_state);
        }

        let cb = self.queue.new_command_buffer();

        // ── Path-tracing dispatch ─────────────────────────────────
        // Writes:  accumTex (RGBA32F) + noisy_color (RGBA16F) +
        //          normal (RGBA32F) + depth (R32F) + motion (RG16F)
        //          + albedo (RGBA8U).
        // Display texture is written by either MetalFX-then-tonemap
        // or the spatial denoiser fallback below.
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_texture(1, Some(&self.accum_texture));
        enc.set_texture(2, Some(&self.normal_texture));
        enc.set_texture(3, Some(&self.depth_texture));
        enc.set_texture(4, Some(&self.motion_texture));
        enc.set_texture(5, Some(&self.albedo_texture));
        enc.set_texture(6, Some(&self.noisy_color_texture));
        // f.3.d-fix4 -- bind roughness (slot 7) + specular albedo
        // (slot 8) G-buffers. The kernel writes them at the primary
        // opaque hit; MetalFX reads them as auxiliary inputs.
        enc.set_texture(7, Some(&self.roughness_texture));
        enc.set_texture(8, Some(&self.specular_albedo_texture));
        enc.set_acceleration_structure(0, Some(&**accel));
        enc.use_resource(&**accel, MTLResourceUsage::Read);
        enc.set_buffer(1, Some(cam_buf), 0);
        enc.set_buffer(2, Some(mat_buf), 0);
        enc.set_buffer(3, Some(vnorm_buf), 0);
        enc.set_buffer(4, Some(vidx_buf), 0);
        enc.set_buffer(5, Some(geom_off_buf), 0);
        enc.set_buffer(6, Some(lights_buf), 0);
        enc.set_buffer(7, Some(&self.path_state_buffer), 0);

        let tg_size = MTLSize { width: 16, height: 16, depth: 1 };
        let tg_count = MTLSize {
            width: ((self.width as u64) + 15) / 16,
            height: ((self.height as u64) + 15) / 16,
            depth: 1,
        };
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();

        // ── Denoise / display dispatch ────────────────────────────
        // Two paths: MetalFX (preferred; temporal denoise + AA) and
        // the c-f spatial filter fallback. They write the same
        // display texture so the readback below doesn't care which.
        if let Some(scaler) = self.metalfx_scaler.as_ref() {
            // MetalFX path.
            //   1. Configure scaler inputs/output for this frame.
            //   2. encodeToCommandBuffer.
            //   3. Tonemap MetalFX's RGBA16F output to RGBA8 display.
            scaler.set_color_texture(&self.noisy_color_texture);
            scaler.set_depth_texture(&self.depth_texture);
            scaler.set_motion_texture(&self.motion_texture);
            scaler.set_normal_texture(&self.normal_texture);
            scaler.set_diffuse_albedo_texture(&self.albedo_texture);
            // f.3.d-fix4 -- newly required auxiliary textures.
            // Without these the denoiser silently uses uninitialized
            // buffers and never converges on static scenes.
            scaler.set_roughness_texture(&self.roughness_texture);
            scaler.set_specular_albedo_texture(&self.specular_albedo_texture);
            scaler.set_output_texture(&self.metalfx_output_texture);
            // f.3.d-fix4 -- depthReversed. Apple defaults to YES
            // (reverse-Z); our depth texture stores LINEAR world-
            // space distance from the camera. If we don't flip this
            // flag, MetalFX interprets near pixels as far and vice
            // versa -> history-validation fails everywhere ->
            // permanent noise (which matches the observed symptom).
            scaler.set_depth_reversed(false);
            // Reset = true on frame 0 after a reset_accumulation so
            // MetalFX drops its temporal history and starts fresh.
            scaler.set_reset(self.frame_count == 0);
            // Motion is stored in UV-space (-1..1). Scale to pixels.
            scaler.set_motion_vector_scale(
                self.width as f32,
                self.height as f32,
            );
            // f.3.d / f.3.d-fix4 -- Halton sub-pixel jitter, centered
            // around 0. Kernel samples at (gid + jitter) in [0, 1]
            // from pixel top-left; MetalFX wants jitter in [-0.5,
            // 0.5] relative to pixel center. Apple's WWDC25 sample
            // explicitly NEGATES the Y component:
            //     scaler.jitterOffsetY = -pixelJitter.y;
            // because MetalFX's Y is "up" while Metal texture Y is
            // "down". Without the negation, MetalFX reprojects this
            // frame's samples to the WRONG sub-pixel offset across
            // frames -> no temporal accumulation -> stays noisy.
            scaler.set_jitter_offset(jitter_x - 0.5, -(jitter_y - 0.5));

            // Debug telemetry: per-frame state visible to whoever's
            // diagnosing the next round of MetalFX issues. Gated on
            // RUST_LOG=debug so production runs stay quiet.
            log::debug!(
                "[metalfx] frame={} reset={} jitter=({:.3},{:.3}) \
                 jitter_mfx=({:.3},{:.3}) motion_scale=({},{}) \
                 prev_vp={} depth_reversed=false",
                self.frame_count,
                self.frame_count == 0,
                jitter_x, jitter_y,
                jitter_x - 0.5, -(jitter_y - 0.5),
                self.width, self.height,
                if self.view_proj_history.is_some() { "valid" } else { "identity" },
            );

            scaler.encode_to_command_buffer(cb);

            // Tonemap: RGBA16F MetalFX output -> RGBA8 display.
            let tm_enc = cb.new_compute_command_encoder();
            tm_enc.set_compute_pipeline_state(&self.tonemap_pipeline);
            tm_enc.set_texture(0, Some(&self.texture));
            tm_enc.set_texture(1, Some(&self.metalfx_output_texture));
            tm_enc.dispatch_thread_groups(tg_count, tg_size);
            tm_enc.end_encoding();
        } else {
            // Fallback: existing c-f spatial denoiser.
            let dn_enc = cb.new_compute_command_encoder();
            dn_enc.set_compute_pipeline_state(&self.denoise_pipeline);
            dn_enc.set_texture(0, Some(&self.texture));
            dn_enc.set_texture(1, Some(&self.accum_texture));
            dn_enc.set_texture(2, Some(&self.normal_texture));
            dn_enc.set_buffer(7, Some(&self.path_state_buffer), 0);
            dn_enc.dispatch_thread_groups(tg_count, tg_size);
            dn_enc.end_encoding();
        }

        cb.commit();
        cb.wait_until_completed();

        // Frame counter advances post-render so the next frame uses
        // the incremented value. Saturate well below 2^24 (the float
        // mantissa limit the MSL kernel uses when it converts back
        // to float for the divide) -- at 30 fps this is years of
        // continuous accumulation; in practice every camera/light
        // tweak resets us long before that.
        self.frame_count = self.frame_count.saturating_add(1).min(16_000_000);

        // f.3.b: shift the view-proj history. What we just rendered
        // with (current_view_proj) becomes "what was used last frame"
        // for the NEXT render's motion-vector computation.
        if let Some(cvp) = self.current_view_proj {
            self.view_proj_history = Some(cvp);
        }

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
/// consume without any per-pixel cross() / normalize() work. Also
/// returns the *current* view-projection matrix so the caller can
/// stash it on the renderer for next frame's motion-vector use
/// (the matrix lives in path_state, not in CameraUniform itself,
/// because path_state is rewritten every frame while CameraUniform
/// is only rewritten on actual camera change).
fn build_camera_uniform(
    camera: &Camera,
    width: u32,
    height: u32,
) -> (CameraUniform, [f32; 16]) {
    use glam::Mat4;
    let eye_v = Vec3::from(camera.pos);
    let target = Vec3::from(camera.target);
    let up_hint = Vec3::from(camera.up);
    let forward = (target - eye_v).normalize_or_zero();
    let right = forward.cross(up_hint).normalize_or_zero();
    let up = right.cross(forward); // already unit-length
    let fov_rad = camera.fov_deg.to_radians();
    let tan_half_fov = (fov_rad * 0.5).tan();
    let aspect = (width as f32) / (height as f32).max(1.0);

    // Current view-projection for motion-vector use NEXT frame.
    let view = Mat4::look_at_rh(eye_v, target, up_hint);
    let proj = Mat4::perspective_rh(fov_rad, aspect, camera.near, camera.far);
    let view_proj = proj * view;
    let current_view_proj = view_proj.to_cols_array();

    let uniform = CameraUniform {
        eye: [eye_v.x, eye_v.y, eye_v.z, 1.0],
        right: [right.x, right.y, right.z, 0.0],
        up: [up.x, up.y, up.z, 0.0],
        forward: [forward.x, forward.y, forward.z, 0.0],
        misc: [tan_half_fov, aspect, 0.0, 0.0],
    };
    (uniform, current_view_proj)
}

/// Identity 4×4 column-major. Used as the initial "previous"
/// view-projection on the very first frame, so motion vectors are
/// effectively zero (no history yet -> MetalFX treats every pixel
/// as new content for the first frame).
const IDENTITY_VIEW_PROJ: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

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
