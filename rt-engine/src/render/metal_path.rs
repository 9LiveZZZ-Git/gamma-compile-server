//! Metal compute-based renderer. Single hard-coded triangle for
//! sprint 7.5.6.a part 2b -- the per-pixel Möller-Trumbore kernel
//! we already validated produces the right image to PNG.
//!
//! Part 2c adds the streaming-ready `MetalRenderer` struct that
//! holds the device + pipeline + reusable output texture so the
//! WebSocket render loop can render at frame rate without re-
//! allocating GPU resources per frame.

use anyhow::{anyhow, Context};
use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLOrigin, MTLPixelFormat,
    MTLRegion, MTLSize, MTLStorageMode, MTLTextureUsage, Texture, TextureDescriptor,
};
use std::ffi::c_void;

const KERNEL_SRC: &str = include_str!("../shaders/triangle.metal");

/// Reusable Metal renderer. Holds the GPU resources (device, queue,
/// pipeline, output texture) so the per-frame work is just kernel
/// dispatch + texture readback. `width` and `height` are fixed at
/// construction; resizing requires building a new renderer.
pub struct MetalRenderer {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    texture: Texture,
    pub width: u32,
    pub height: u32,
}

impl MetalRenderer {
    pub fn new(width: u32, height: u32) -> anyhow::Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| anyhow!("MTLCreateSystemDefaultDevice returned null"))?;

        log::info!(
            "[metal-renderer] device={:?} {}x{}",
            device.name(),
            width,
            height
        );

        let queue = device.new_command_queue();

        // Default compile options for the basic compute kernel.
        // Metal 3 with `metal_raytracing` lands in part 2c when we
        // switch from in-kernel Möller-Trumbore to AS-backed RT.
        let lib = device
            .new_library_with_source(KERNEL_SRC, &CompileOptions::new())
            .map_err(|e| anyhow!("MSL compile failed: {}", e))?;
        let func = lib
            .get_function("rt_triangle", None)
            .map_err(|e| anyhow!("kernel function 'rt_triangle' not found: {}", e))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow!("compute pipeline create failed: {}", e))?;

        // Output texture -- RGBA8Unorm so the bytes serialize cleanly
        // to PNG (--render-test) AND ship over the WebSocket without
        // a swizzle pass. Editor side creates a matching
        // `rgba8unorm` GPUTexture to receive the frame, then samples
        // it via a fullscreen blit into the bgra8unorm framebuffer.
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
