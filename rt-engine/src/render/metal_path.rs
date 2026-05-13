//! Metal renderer with hardware ray-tracing path.
//!
//! Sprint 7.5.6.a part 2c. Replaces the inline Möller-Trumbore
//! intersection from part 2b with `intersector<>` running against a
//! real `MTLAccelerationStructure`. On Apple9+ GPUs (M3, M4) this
//! lowers to the dedicated hardware RT cores; on older Apple silicon
//! (M1, M2) the same MSL compiles to a software-emulated path. The
//! capability probe at startup reports which path the host supports.
//!
//! The AS is built once at MetalRenderer construction. Each frame:
//!   - bind output texture
//!   - bind AS at buffer(0)
//!   - dispatch a 2D grid; each thread casts a ray, the intersector
//!     traverses the BVH (hardware-side on Apple9+), writes pixel.
//! No GPU work per-frame besides the kernel dispatch + readback.

use anyhow::{anyhow, Context};
use metal::{
    AccelerationStructure, AccelerationStructureTriangleGeometryDescriptor, Array,
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLAttributeFormat,
    MTLLanguageVersion, MTLOrigin, MTLPixelFormat, MTLRegion, MTLResourceOptions,
    MTLResourceUsage, MTLSize, MTLStorageMode, MTLTextureUsage,
    PrimitiveAccelerationStructureDescriptor, Texture, TextureDescriptor,
};
use std::ffi::c_void;

const KERNEL_SRC: &str = include_str!("../shaders/triangle.metal");

// Triangle in the XY plane at z=0. Same vertices as the editor's
// DebugTriangle so visual-diffing against the raster path is
// meaningful. Each row = one vertex (x, y, z).
const TRIANGLE_VERTICES: [f32; 9] = [
    -0.866, -0.5, 0.0,
     0.866, -0.5, 0.0,
     0.0,    1.0, 0.0,
];

/// Reusable Metal renderer with a pre-built AS. Holds the GPU
/// resources (device, queue, pipeline, output texture, AS) so the
/// per-frame work is just kernel dispatch + texture readback.
pub struct MetalRenderer {
    _device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    texture: Texture,
    // Acceleration structure must outlive every command buffer that
    // references it. We build it once in new() and hold it for the
    // life of the renderer.
    accel: AccelerationStructure,
    pub width: u32,
    pub height: u32,
}

impl MetalRenderer {
    pub fn new(width: u32, height: u32) -> anyhow::Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| anyhow!("MTLCreateSystemDefaultDevice returned null"))?;

        log::info!(
            "[metal-renderer] device={:?} {}x{} (hardware RT path)",
            device.name(),
            width,
            height
        );

        let queue = device.new_command_queue();

        // ── Build the acceleration structure (primitive AS / BLAS). ──
        // For one triangle the BVH is trivial, but this is the same
        // code path that'll scale to thousands of primitives in part
        // 2e once we accept editor-driven scene data. Steps:
        //   1. Vertex buffer (host-write, GPU-read)
        //   2. Triangle geometry descriptor pointing at the buffer
        //   3. Primitive AS descriptor wrapping the geometry
        //   4. Query size requirements, allocate AS + scratch buffers
        //   5. Build via the AS command encoder; wait for completion.
        let vertex_buffer = device.new_buffer_with_data(
            TRIANGLE_VERTICES.as_ptr() as *const c_void,
            (TRIANGLE_VERTICES.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let geom_desc = AccelerationStructureTriangleGeometryDescriptor::descriptor();
        geom_desc.set_vertex_buffer(Some(&vertex_buffer));
        geom_desc.set_vertex_buffer_offset(0);
        geom_desc.set_vertex_stride(std::mem::size_of::<[f32; 3]>() as u64);
        geom_desc.set_vertex_format(MTLAttributeFormat::Float3);
        geom_desc.set_triangle_count(1);

        let prim_desc = PrimitiveAccelerationStructureDescriptor::descriptor();
        // metal-rs's set_geometry_descriptors wants
        // &ArrayRef<AccelerationStructureGeometryDescriptor> (base type).
        // Triangle is a subclass; the .into() bumps the refcount and
        // upcasts the ObjC pointer. from_owned_slice returns the
        // ArrayRef directly (no extra `&` needed when passing to the
        // setter).
        let geom_array: &metal::ArrayRef<
            metal::AccelerationStructureGeometryDescriptor,
        > = Array::from_owned_slice(&[geom_desc.clone().into()]);
        prim_desc.set_geometry_descriptors(geom_array);

        let sizes = device.acceleration_structure_sizes_with_descriptor(&prim_desc);
        log::info!(
            "[metal-renderer] AS sizes: storage={}B scratch={}B refit={}B",
            sizes.acceleration_structure_size,
            sizes.build_scratch_buffer_size,
            sizes.refit_scratch_buffer_size
        );

        let accel = device.new_acceleration_structure_with_size(sizes.acceleration_structure_size);
        let scratch = device.new_buffer(
            sizes.build_scratch_buffer_size,
            MTLResourceOptions::StorageModePrivate,
        );

        let build_cb = queue.new_command_buffer();
        let as_enc = build_cb.new_acceleration_structure_command_encoder();
        as_enc.build_acceleration_structure(&accel, &prim_desc, &scratch, 0);
        as_enc.end_encoding();
        build_cb.commit();
        build_cb.wait_until_completed();
        log::info!("[metal-renderer] AS build complete (1 primitive)");

        // ── Compile the metal_raytracing kernel ──
        // intersector<> needs MSL 2.3+; bump the language version on
        // the compile options so #include <metal_raytracing> resolves
        // (default is lower on some host SDKs).
        let options = CompileOptions::new();
        options.set_language_version(MTLLanguageVersion::V2_4);
        let lib = device
            .new_library_with_source(KERNEL_SRC, &options)
            .map_err(|e| anyhow!("MSL compile failed: {}", e))?;
        let func = lib
            .get_function("rt_triangle", None)
            .map_err(|e| anyhow!("kernel function 'rt_triangle' not found: {}", e))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow!("compute pipeline create failed: {}", e))?;

        // Output texture -- RGBA8Unorm so bytes serialize cleanly to
        // PNG (--render-test) AND ship over WebSocket without swizzle.
        let tex_desc = TextureDescriptor::new();
        tex_desc.set_width(width as u64);
        tex_desc.set_height(height as u64);
        tex_desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        tex_desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        tex_desc.set_storage_mode(MTLStorageMode::Shared);
        let texture = device.new_texture(&tex_desc);

        Ok(MetalRenderer {
            _device: device,
            queue,
            pipeline,
            texture,
            accel,
            width,
            height,
        })
    }

    /// Render one frame into the internal texture + read back as
    /// raw RGBA8 bytes (width * height * 4 bytes). Reuses GPU
    /// resources across calls; per-frame cost is one command-buffer
    /// commit + the readback.
    pub fn render_frame(&self) -> anyhow::Result<Vec<u8>> {
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_texture(0, Some(&self.texture));
        // Bind the AS at buffer slot 0 (matches [[buffer(0)]] in MSL).
        // metal-rs signature is (at_buffer_index, Option<&Ref>); the
        // &*self.accel forces Deref from owning AccelerationStructure
        // to AccelerationStructureRef before Some() wraps it.
        // use_resource ensures the AS pages are resident before the
        // dispatch; without it the GPU could fault on the BVH lookup.
        enc.set_acceleration_structure(0, Some(&*self.accel));
        enc.use_resource(&self.accel, MTLResourceUsage::Read);

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

/// One-shot test render: build a renderer, render once, save the
/// result to a PNG, exit. Driven by the --render-test CLI flag.
pub fn render_test_triangle(width: u32, height: u32, output: &str) -> anyhow::Result<()> {
    let renderer = MetalRenderer::new(width, height)?;
    let pixels = renderer.render_frame()?;

    let img = image::RgbaImage::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow!("RgbaImage::from_raw failed (buffer size mismatch)"))?;
    img.save(output)
        .with_context(|| format!("PNG save to {} failed", output))?;

    log::info!("[render-test] wrote {} to {}", img.as_raw().len(), output);
    Ok(())
}
