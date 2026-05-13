//! Metal compute-based test renderer. Runs a one-pixel-per-thread
//! compute kernel that does Möller-Trumbore ray-triangle
//! intersection against a hard-coded triangle + writes color into
//! an output texture. The texture is read back to a CPU buffer and
//! saved as PNG.
//!
//! This is intentionally the simplest possible path -- no
//! acceleration structure (the AS API is the next sub-push), no
//! camera uniforms, no per-frame state. The point is to prove the
//! full Metal pipeline (device → library → pipeline → encoder →
//! texture readback) works end-to-end on the dev M4.

use anyhow::{anyhow, Context};
use metal::{
    CompileOptions, Device, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize,
    MTLStorageMode, MTLTextureUsage, TextureDescriptor,
};
use std::ffi::c_void;

const KERNEL_SRC: &str = include_str!("../shaders/triangle.metal");

pub fn render_test_triangle(width: u32, height: u32, output: &str) -> anyhow::Result<()> {
    let device = Device::system_default()
        .ok_or_else(|| anyhow!("MTLCreateSystemDefaultDevice returned null"))?;

    log::info!(
        "[render-test] device={:?} target={}x{} output={}",
        device.name(),
        width,
        height,
        output
    );

    let queue = device.new_command_queue();

    // Default compile options are fine -- our kernel uses basic
    // texture<float, access::write> + cross/dot/normalize. Default
    // language version (Metal 2.x) covers all of this. Metal 3.x is
    // only needed when we move to the `metal_raytracing` include
    // for the AS-backed renderer (part 2c).
    let lib = device
        .new_library_with_source(KERNEL_SRC, &CompileOptions::new())
        .map_err(|e| anyhow!("MSL compile failed: {}", e))?;
    let func = lib
        .get_function("rt_triangle", None)
        .map_err(|e| anyhow!("kernel function 'rt_triangle' not found: {}", e))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow!("compute pipeline create failed: {}", e))?;

    log::debug!("[render-test] pipeline built");

    // Output texture -- RGBA8Unorm so PNG output is trivial.
    let tex_desc = TextureDescriptor::new();
    tex_desc.set_width(width as u64);
    tex_desc.set_height(height as u64);
    tex_desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    tex_desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
    tex_desc.set_storage_mode(MTLStorageMode::Shared);
    let texture = device.new_texture(&tex_desc);

    // Dispatch one thread per pixel.
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_texture(0, Some(&texture));

    // Threadgroup of 16x16; round up the count to cover the image.
    let tg_size = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };
    let tg_count = MTLSize {
        width: ((width as u64) + 15) / 16,
        height: ((height as u64) + 15) / 16,
        depth: 1,
    };
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    log::debug!("[render-test] kernel done");

    // Read back to a CPU buffer.
    let bytes_per_row = (width * 4) as u64;
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    texture.get_bytes(
        pixels.as_mut_ptr() as *mut c_void,
        bytes_per_row,
        region,
        0,
    );

    // Save as PNG.
    let img = image::RgbaImage::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow!("RgbaImage::from_raw failed (buffer size mismatch)"))?;
    img.save(output)
        .with_context(|| format!("PNG save to {} failed", output))?;

    log::info!("[render-test] wrote {} bytes to {}", img.as_raw().len(), output);
    Ok(())
}
