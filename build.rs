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
        // Always write to OUT_DIR (cargo publish compatible)
        let out_path = std::path::Path::new(&out_dir).join("u_insight.h");
        bindings.write_to_file(&out_path);

        // Also write to include/ for development convenience
        let include_dir = std::path::Path::new(&crate_dir).join("include");
        std::fs::create_dir_all(&include_dir).ok();
        let dev_path = include_dir.join("u_insight.h");
        bindings.write_to_file(dev_path);
    }
}
