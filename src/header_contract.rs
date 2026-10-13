//! The committed C header is the one cbindgen generates from this crate.
//!
//! `build.rs` writes the generated header into `OUT_DIR` only; the C and C#
//! bindings read the committed `include/u_insight.h`. A header that drifted
//! from the `#[repr(C)]` surface would let a C# struct read its neighbour's
//! field without any parse error, so the two are compared here. To accept a
//! change to the FFI surface, regenerate the committed file with
//! `UPDATE_HEADER=1 cargo test header` and commit it.

const GENERATED: &str = include_str!(concat!(env!("OUT_DIR"), "/u_insight.h"));

fn normalized(text: &str) -> String {
    text.replace("\r\n", "\n")
}

#[test]
fn committed_header_matches_the_ffi_surface() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("include/u_insight.h");
    if std::env::var_os("UPDATE_HEADER").is_some() {
        std::fs::write(&path, GENERATED).expect("include/u_insight.h is writable");
        return;
    }
    let committed = std::fs::read_to_string(&path).expect("include/u_insight.h exists");
    assert!(
        normalized(&committed) == normalized(GENERATED),
        "include/u_insight.h is stale: the FFI surface changed. Regenerate it with \
         `UPDATE_HEADER=1 cargo test header` and commit it."
    );
}
