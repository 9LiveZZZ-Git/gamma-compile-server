//! MetalFX TemporalDenoisedScaler FFI.
//!
//! metal-rs (the Rust binding crate we use for Metal proper) doesn't
//! bind the MetalFX framework. For Sprint 7.5.6.f.3 we need
//! `MTLFXTemporalDenoisedScaler` -- Apple's temporal denoiser
//! specifically designed for ray-traced rendering. So we hand-roll
//! the FFI here using the same `objc` 0.2 crate that metal-rs uses
//! internally, going through the ObjC runtime via `msg_send!`.
//!
//! What we bind (just enough for f.3.a / f.3.c):
//!   - `MTLFXTemporalDenoisedScalerDescriptor`
//!     - alloc / init
//!     - input/output dimensions setters
//!     - color / output / depth / motion / normal / albedo format setters
//!     - `newTemporalDenoisedScalerWithDevice:` factory
//!   - `MTLFXTemporalDenoisedScaler` (the resulting scaler)
//!     - `colorTexture` / `depthTexture` / `motionTexture` /
//!       `normalTexture` / `diffuseAlbedoTexture` / `outputTexture` setters
//!     - `reset` flag
//!     - `jitterOffsetX` / `jitterOffsetY` floats
//!     - `motionVectorScaleX` / `motionVectorScaleY` floats
//!     - `encodeToCommandBuffer:` method
//!
//! Class availability: macOS 15+ (Sequoia). Older macOS returns nil
//! from the factory. We handle that gracefully -- the caller decides
//! whether to fall back to the spatial denoiser or surface an error.

#![cfg(target_os = "macos")]
// Silence noisy `cfg(feature = "cargo-clippy")` warnings from objc 0.2's
// sel_impl macro -- they're an artifact of the older objc crate's
// approach to detecting clippy, not anything wrong with our code.
#![allow(unexpected_cfgs)]

use metal::{Device, MTLPixelFormat};
use objc::runtime::{Class, Object, BOOL, NO, YES};
use objc::{msg_send, sel, sel_impl};

/// Owning handle for an `id<MTLFXTemporalDenoisedScaler>`. Drop
/// releases the underlying ObjC object.
pub struct TemporalDenoisedScaler {
    ptr: *mut Object,
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
}

// `*mut Object` (the raw ObjC pointer) is not `Send` by default in
// Rust, which means MetalRenderer (which now contains this scaler)
// wouldn't be `Send` either -- breaks tokio's `spawn` which requires
// a Send future. ObjC objects ARE safe to *move* between threads as
// long as we don't access them concurrently; Rust's `&mut self`
// borrow rules already enforce single-threaded access at the call
// sites. Same pattern metal-rs itself uses for its wrappers.
//
// Sync (= `&` reference shared across threads) we don't actually
// need; we only ever have one owner at a time. Marking it anyway
// for symmetry and to match metal-rs's wrappers.
unsafe impl Send for TemporalDenoisedScaler {}
unsafe impl Sync for TemporalDenoisedScaler {}

impl Drop for TemporalDenoisedScaler {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _: () = msg_send![self.ptr, release];
            }
        }
    }
}

/// Config knobs for scaler creation. We default everything to the
/// formats the rt-engine's path tracer currently outputs / will
/// output once f.3.b lands the G-buffer additions.
pub struct ScalerConfig {
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    /// Color (noisy input). Default RGBA16Float.
    pub color_format: MTLPixelFormat,
    /// Final output. Default RGBA16Float (we'll tonemap to RGBA8 in a
    /// follow-up kernel; MetalFX prefers float output for HDR).
    pub output_format: MTLPixelFormat,
    /// Depth. Single channel float.
    pub depth_format: MTLPixelFormat,
    /// Motion vectors. Two-channel half float.
    pub motion_format: MTLPixelFormat,
    /// Normal. RGBA16Float.
    pub normal_format: MTLPixelFormat,
    /// Diffuse albedo. RGBA8Unorm (sRGB-space color).
    pub diffuse_albedo_format: MTLPixelFormat,
}

impl Default for ScalerConfig {
    fn default() -> Self {
        Self {
            input_width: 800,
            input_height: 600,
            output_width: 800,
            output_height: 600,
            color_format: MTLPixelFormat::RGBA16Float,
            output_format: MTLPixelFormat::RGBA16Float,
            depth_format: MTLPixelFormat::R32Float,
            motion_format: MTLPixelFormat::RG16Float,
            normal_format: MTLPixelFormat::RGBA16Float,
            diffuse_albedo_format: MTLPixelFormat::RGBA8Unorm,
        }
    }
}

impl TemporalDenoisedScaler {
    /// Attempt to create a scaler. Returns Ok(Some(...)) on success,
    /// Ok(None) if MetalFX is not supported on this device (typical
    /// for macOS < 15 or unsupported GPUs). Err for unexpected
    /// failure during creation (e.g. invalid format combination).
    pub fn try_new(device: &Device, cfg: &ScalerConfig) -> anyhow::Result<Option<Self>> {
        unsafe {
            // 1. [MTLFXTemporalDenoisedScalerDescriptor alloc] init
            // Use Class::get instead of the class!() macro so a
            // missing class (= class not present in this macOS
            // version) gracefully returns None instead of panicking.
            let desc_class = match Class::get("MTLFXTemporalDenoisedScalerDescriptor") {
                Some(c) => c,
                None => {
                    return Ok(None);
                }
            };
            let desc: *mut Object = msg_send![desc_class, alloc];
            if desc.is_null() {
                return Ok(None);
            }
            let desc: *mut Object = msg_send![desc, init];
            if desc.is_null() {
                return Ok(None);
            }

            // 2. Configure descriptor. Setters all take NSUInteger
            // (u64 on 64-bit) for dimensions and MTLPixelFormat
            // (NSUInteger underneath) for formats.
            let _: () = msg_send![desc, setInputWidth: cfg.input_width as u64];
            let _: () = msg_send![desc, setInputHeight: cfg.input_height as u64];
            let _: () = msg_send![desc, setOutputWidth: cfg.output_width as u64];
            let _: () = msg_send![desc, setOutputHeight: cfg.output_height as u64];
            let _: () = msg_send![desc, setColorTextureFormat: cfg.color_format as u64];
            let _: () = msg_send![desc, setOutputTextureFormat: cfg.output_format as u64];
            let _: () = msg_send![desc, setDepthTextureFormat: cfg.depth_format as u64];
            let _: () = msg_send![desc, setMotionTextureFormat: cfg.motion_format as u64];
            let _: () = msg_send![desc, setNormalTextureFormat: cfg.normal_format as u64];
            let _: () = msg_send![desc, setDiffuseAlbedoTextureFormat: cfg.diffuse_albedo_format as u64];

            // 3. [descriptor newTemporalDenoisedScalerWithDevice:device]
            //    -- returns nil if the device doesn't support it
            //    (older macOS, older GPU). Treat as "MetalFX not
            //    available" rather than a hard error so the caller
            //    can fall back.
            let device_ptr: *mut Object = device_to_ptr(device);
            let scaler: *mut Object = msg_send![desc, newTemporalDenoisedScalerWithDevice: device_ptr];

            // 4. Release the descriptor; we don't need it past
            //    construction.
            let _: () = msg_send![desc, release];

            if scaler.is_null() {
                return Ok(None);
            }
            Ok(Some(TemporalDenoisedScaler {
                ptr: scaler,
                input_width: cfg.input_width,
                input_height: cfg.input_height,
                output_width: cfg.output_width,
                output_height: cfg.output_height,
            }))
        }
    }

    /// Returns the raw ObjC pointer. Useful for debugging.
    pub fn as_ptr(&self) -> *mut Object {
        self.ptr
    }

    /// Sprint 7.5.6.f.3.c -- texture property setters. Each MetalFX
    /// input/output gets its own setter on the scaler instance.
    /// All are called per-frame before `encode_to_command_buffer`.
    pub fn set_color_texture(&self, tex: &metal::Texture) {
        unsafe {
            let p = texture_to_ptr(tex);
            let _: () = msg_send![self.ptr, setColorTexture: p];
        }
    }
    pub fn set_depth_texture(&self, tex: &metal::Texture) {
        unsafe {
            let p = texture_to_ptr(tex);
            let _: () = msg_send![self.ptr, setDepthTexture: p];
        }
    }
    pub fn set_motion_texture(&self, tex: &metal::Texture) {
        unsafe {
            let p = texture_to_ptr(tex);
            let _: () = msg_send![self.ptr, setMotionTexture: p];
        }
    }
    pub fn set_normal_texture(&self, tex: &metal::Texture) {
        unsafe {
            let p = texture_to_ptr(tex);
            let _: () = msg_send![self.ptr, setNormalTexture: p];
        }
    }
    pub fn set_diffuse_albedo_texture(&self, tex: &metal::Texture) {
        unsafe {
            let p = texture_to_ptr(tex);
            let _: () = msg_send![self.ptr, setDiffuseAlbedoTexture: p];
        }
    }
    pub fn set_output_texture(&self, tex: &metal::Texture) {
        unsafe {
            let p = texture_to_ptr(tex);
            let _: () = msg_send![self.ptr, setOutputTexture: p];
        }
    }

    /// Reset internal history (e.g. on a hard scene change). Tell
    /// MetalFX to drop its temporal accumulation rather than blend
    /// stale samples with the new content.
    pub fn set_reset(&self, reset: bool) {
        let b: BOOL = if reset { YES } else { NO };
        unsafe {
            let _: () = msg_send![self.ptr, setReset: b];
        }
    }

    /// Subpixel jitter offset in the range [-0.5, 0.5]. MetalFX uses
    /// this to do proper sub-pixel reconstruction across frames.
    /// Must match whatever subpixel offset the path tracer used for
    /// the primary ray on this frame.
    pub fn set_jitter_offset(&self, x: f32, y: f32) {
        unsafe {
            let _: () = msg_send![self.ptr, setJitterOffsetX: x];
            let _: () = msg_send![self.ptr, setJitterOffsetY: y];
        }
    }

    /// Motion-vector scale: multiplies the motion texture's values
    /// to get the pixel-space delta. Our motion texture stores
    /// UV-space deltas in [-1, 1], so scale = (width, height) gives
    /// the correct pixel offset MetalFX expects.
    pub fn set_motion_vector_scale(&self, x: f32, y: f32) {
        unsafe {
            let _: () = msg_send![self.ptr, setMotionVectorScaleX: x];
            let _: () = msg_send![self.ptr, setMotionVectorScaleY: y];
        }
    }

    /// Encode the denoise + upscale pass into the given command
    /// buffer. All texture properties must have been set first.
    pub fn encode_to_command_buffer(&self, cb: &metal::CommandBufferRef) {
        unsafe {
            use metal::foreign_types::ForeignTypeRef;
            let cb_ptr = cb.as_ptr() as *mut Object;
            let _: () = msg_send![self.ptr, encodeToCommandBuffer: cb_ptr];
        }
    }
}

fn texture_to_ptr(tex: &metal::Texture) -> *mut Object {
    use metal::foreign_types::ForeignTypeRef;
    use std::ops::Deref;
    let r = tex.deref();
    r.as_ptr() as *mut Object
}

/// Internal: extract the raw ObjC pointer from metal-rs's wrapper.
/// metal-rs's `Device` is a transparent newtype around `*mut Object`
/// so we can `as_ptr()` it via the `ForeignType` trait it implements.
/// We use `into_super` / explicit casts to be safe across crate
/// versions.
fn device_to_ptr(device: &Device) -> *mut Object {
    // metal-rs's `DeviceRef::as_ptr()` returns `*mut MTLDevice` which
    // is `*mut Object` under the hood. We rely on that here. If
    // metal-rs reorganizes the type structure this may break; the
    // fix is straightforward (find the new pointer-extraction API).
    use std::ops::Deref;
    let r = device.deref();
    // ForeignTypeRef::as_ptr returns the raw underlying pointer.
    use metal::foreign_types::ForeignTypeRef;
    r.as_ptr() as *mut Object
}

// Silence "imported but unused" complaints from the BOOL / YES / NO
// imports -- f.3.c will use them for the `reset` setter.
const _: BOOL = YES;
const _: BOOL = NO;
