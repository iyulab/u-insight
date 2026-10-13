//! Build script for generating C headers.

fn main() {
    // Generate C header using cbindgen
    let crate_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR must be set by Cargo");
    let config = cbindgen::Config::from_file("cbindgen.toml").unwrap_or_default();

    if let Ok(bindings) = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        // Only into OUT_DIR: a build script must not change the source tree. The
        // committed include/u_insight.h is compared with this file by a test
        // (src/header_contract.rs), which also rewrites it on request.
        let out_path = std::path::Path::new(&out_dir).join("u_insight.h");
        bindings.write_to_file(&out_path);
    }
}
