//! Build script for the rt-engine crate.
//!
//! Currently does just one thing: links the MetalFX framework on
//! macOS so the Sprint 7.5.6.f.3 ObjC FFI to MTLFXTemporalDenoised-
//! Scaler resolves. metal-rs links the Metal framework for us but
//! doesn't know about MetalFX; we need to add it ourselves.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=MetalFX");
        // Re-run the build script if this file changes (it doesn't
        // depend on anything else; this is the safe default).
        println!("cargo:rerun-if-changed=build.rs");
    }
}
