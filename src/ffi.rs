//! C FFI bindings for u-insight.
//!
//! Exposes profiling and analysis functionality via a C-compatible interface.
//!
//! # Design (AD-4)
//!
//! - **Opaque handles**: `*mut ProfileContext` / `*mut AnalysisContext`
//! - **`#[repr(C)]`**: All data transfer structs
//! - **Integer error codes**: 0 = success, negative = error
//! - **Thread-local error message**: `insight_last_error()`
//! - **`catch_unwind`**: All FFI entry points wrapped to prevent panic propagation
//!
//! # Safety
//!
//! All functions use `catch_unwind` to prevent panics from crossing the FFI boundary.
//! Null pointer arguments return error code -1.

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic;
use std::ptr;
use std::slice;

use crate::analysis::mutual_info_classif;
use crate::clustering::{
    dbscan, gap_statistic, hdbscan, hierarchical, kmeans, mini_batch_kmeans, silhouette_samples,
    DbscanConfig, HdbscanConfig, HierarchicalConfig, KMeansConfig, Linkage, MiniBatchKMeansConfig,
};
use crate::csv_parser::CsvParser;
use crate::distribution::{distribution_analysis, DistributionConfig};
use crate::error::{Bound, InsightError};
use crate::feature_importance::{feature_analysis, permutation_importance, FeatureConfig};
use crate::json_parser::JsonParser;
use crate::pca::{pca, PcaConfig};
use crate::profiling::profile_dataframe;
use crate::refusal::{Code, Refusal};

// ── Error handling ────────────────────────────────────────────────────

/// Error codes returned by FFI functions.
pub const INSIGHT_OK: i32 = 0;
pub const INSIGHT_ERR_NULL_PTR: i32 = -1;
pub const INSIGHT_ERR_INVALID_INPUT: i32 = -2;
pub const INSIGHT_ERR_PARSE_FAILED: i32 = -3;
pub const INSIGHT_ERR_ANALYSIS_FAILED: i32 = -4;
pub const INSIGHT_ERR_PANIC: i32 = -99;
// New granular error codes — 1:1 with InsightError variants
pub const INSIGHT_ERR_INSUFFICIENT_DATA: i32 = -5;
pub const INSIGHT_ERR_INVALID_PARAM: i32 = -6;
pub const INSIGHT_ERR_DEGENERATE_DATA: i32 = -7;
pub const INSIGHT_ERR_COMPUTATION_FAILED: i32 = -8;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
    static LAST_ERROR_PARAMETER: RefCell<Option<CString>> = const { RefCell::new(None) };
    /// The last refusal's whole body, and its text form for
    /// [`insight_last_error_json`].
    static LAST_ERROR_BODY: RefCell<Option<(serde_json::Value, CString)>> = const { RefCell::new(None) };
}

/// The `code` an `INSIGHT_ERR_*` category carries when nothing more specific
/// is known about the refusal. Errors from the analyses carry their own
/// (see [`Refusal`]).
fn code_name(rc: i32) -> Code {
    match rc {
        INSIGHT_ERR_NULL_PTR | INSIGHT_ERR_PARSE_FAILED => Code::MalformedInput,
        INSIGHT_ERR_INVALID_INPUT => Code::InvalidInput,
        INSIGHT_ERR_INSUFFICIENT_DATA => Code::InsufficientData,
        INSIGHT_ERR_INVALID_PARAM => Code::InvalidOption,
        INSIGHT_ERR_DEGENERATE_DATA => Code::DegenerateData,
        INSIGHT_ERR_COMPUTATION_FAILED => Code::ComputationFailed,
        _ => Code::Internal,
    }
}

fn error_to_code(e: &InsightError) -> i32 {
    match e {
        InsightError::CsvParse { .. } | InsightError::JsonParse { .. } => INSIGHT_ERR_PARSE_FAILED,
        InsightError::MissingValues { .. }
        | InsightError::ValueNotFinite { .. }
        | InsightError::EmptyInput { .. }
        | InsightError::ColumnNotFound { .. }
        | InsightError::DimensionMismatch { .. } => INSIGHT_ERR_INVALID_INPUT,
        InsightError::InsufficientData { .. } => INSIGHT_ERR_INSUFFICIENT_DATA,
        // A NaN or infinite option is a refused value of that option.
        InsightError::ParameterOutOfRange { .. } | InsightError::InvalidParameter { .. } => {
            INSIGHT_ERR_INVALID_PARAM
        }
        InsightError::DegenerateData { .. } => INSIGHT_ERR_DEGENERATE_DATA,
        InsightError::ComputationFailed { .. } => INSIGHT_ERR_COMPUTATION_FAILED,
        InsightError::Io(_) => INSIGHT_ERR_ANALYSIS_FAILED,
    }
}

/// Records a refusal: its text (`insight_last_error`), the parameter it is
/// about when its fields name one (`insight_last_error_parameter`), and the
/// whole body (`insight_last_error_json`).
fn record(refusal: Refusal) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(refusal.message.as_str()).ok();
    });
    let mut body = serde_json::Map::new();
    body.insert("error".into(), serde_json::Value::String(refusal.message));
    if let serde_json::Value::Object(fields) = refusal.fields {
        body.extend(fields);
    }
    store_body(serde_json::Value::Object(body));
}

/// Keeps `body` as the last refusal's. `insight_last_error_parameter` names
/// the option when the refusal is about one; a refusal of the data (`data`
/// too short, `data[7]` not finite) carries its `parameter` in the body only.
fn store_body(body: serde_json::Value) {
    let about_an_option = matches!(
        body.get("code").and_then(|c| c.as_str()),
        Some("invalid_option" | "parameter_out_of_range" | "unknown_option" | "missing_option")
    );
    let parameter = body
        .get("parameter")
        .and_then(|p| p.as_str())
        .filter(|_| about_an_option)
        .and_then(|p| CString::new(p).ok());
    LAST_ERROR_PARAMETER.with(|cell| {
        *cell.borrow_mut() = parameter;
    });
    let text = CString::new(body.to_string()).ok();
    LAST_ERROR_BODY.with(|cell| {
        *cell.borrow_mut() = text.map(|text| (body, text));
    });
}

/// Records an analysis error with its own code and fields, and returns its
/// `INSIGHT_ERR_*` category.
fn fail(e: &InsightError) -> i32 {
    record(Refusal::from(e));
    error_to_code(e)
}

/// Records an integer selector (`method`, `cost`) that is none of the values
/// the function knows, and returns `INSIGHT_ERR_INVALID_PARAM`.
fn unknown_selector(parameter: &str, got: u32, expected: &[u32], message: String) -> i32 {
    record(Refusal::new(
        Code::UnknownOption,
        message,
        serde_json::json!({ "parameter": parameter, "got": got, "expected": expected }),
    ));
    INSIGHT_ERR_INVALID_PARAM
}

/// Records a refusal known only by its category and text, and returns `rc`.
fn refuse(rc: i32, message: impl Into<String>) -> i32 {
    record(Refusal::new(
        code_name(rc),
        message.into(),
        serde_json::json!({}),
    ));
    rc
}

/// Returns the last error message, or null if no error.
/// The returned string is valid until the next FFI call on this thread.
///
/// # Safety
/// The caller must not free the returned pointer.
#[no_mangle]
pub extern "C" fn insight_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        let borrow = cell.borrow();
        match borrow.as_ref() {
            Some(cstr) => cstr.as_ptr(),
            None => ptr::null(),
        }
    })
}

/// Returns the name of the parameter the last error is about, or null when it
/// is not about one. Set together with an `INSIGHT_ERR_INVALID_PARAM` whose
/// cause is a single named argument or option -- `quantile`, or an option
/// such as LOF's `threshold` -- so a caller can branch on it or map it to its
/// own name without reading the message.
/// The returned string is valid until the next FFI call on this thread.
///
/// # Safety
/// The caller must not free the returned pointer.
#[no_mangle]
pub extern "C" fn insight_last_error_parameter() -> *const c_char {
    LAST_ERROR_PARAMETER.with(|cell| {
        let borrow = cell.borrow();
        match borrow.as_ref() {
            Some(cstr) => cstr.as_ptr(),
            None => ptr::null(),
        }
    })
}

/// Returns the last refusal as a JSON object, or null if no error:
/// `{"error": <message>, "code": <reason>, ...fields}` -- the same `code` and
/// fields the WebAssembly binding puts on its `Error` (`index`, `parameter`,
/// `min`, `got`, `column`, ...), so a caller can say which value was refused
/// and why without parsing the message.
/// The returned string is valid until the next FFI call on this thread.
///
/// # Safety
/// The caller must not free the returned pointer.
#[no_mangle]
pub extern "C" fn insight_last_error_json() -> *const c_char {
    LAST_ERROR_BODY.with(|cell| {
        let borrow = cell.borrow();
        match borrow.as_ref() {
            Some((_, text)) => text.as_ptr(),
            None => ptr::null(),
        }
    })
}

/// Clears the last error message, parameter name and body.
#[no_mangle]
pub extern "C" fn insight_clear_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
    LAST_ERROR_PARAMETER.with(|cell| {
        *cell.borrow_mut() = None;
    });
    LAST_ERROR_BODY.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

// ── Profile Context (opaque handle) ──────────────────────────────────

/// Opaque handle for a profiling context.
/// Holds the parsed DataFrame and computed profiles.
pub struct ProfileContext {
    dataframe: crate::dataframe::DataFrame,
    column_profiles: Vec<crate::profiling::ColumnProfile>,
}

/// C-compatible column profile summary.
#[repr(C)]
pub struct CColumnSummary {
    /// Column index.
    pub index: u32,
    /// Number of valid (non-null) values.
    pub valid_count: u64,
    /// Number of null values.
    pub null_count: u64,
    /// Column data type: 0=Numeric, 1=Boolean, 2=Categorical, 3=Text.
    pub data_type: u32,
    /// For numeric columns: mean. NaN for non-numeric, with no valid values,
    /// or with an infinite one.
    pub mean: f64,
    /// For numeric columns: standard deviation. NaN for non-numeric, below 2
    /// valid values, or with an infinite one.
    pub std_dev: f64,
    /// For numeric columns: minimum. NaN for non-numeric or with no valid values.
    pub min: f64,
    /// For numeric columns: maximum. NaN for non-numeric or with no valid values.
    pub max: f64,
}

/// Creates a profile context from a CSV string.
///
/// # Safety
/// - `csv_data` must be a valid null-terminated UTF-8 string.
/// - The returned handle must be freed with `insight_profile_free`.
#[no_mangle]
pub unsafe extern "C" fn insight_profile_csv(csv_data: *const c_char) -> *mut ProfileContext {
    let result = panic::catch_unwind(|| {
        if csv_data.is_null() {
            refuse(INSIGHT_ERR_NULL_PTR, "null csv_data pointer");
            return ptr::null_mut();
        }

        let c_str = unsafe { CStr::from_ptr(csv_data) };
        let csv = match c_str.to_str() {
            Ok(s) => s,
            Err(e) => {
                refuse(INSIGHT_ERR_PARSE_FAILED, format!("invalid UTF-8: {e}"));
                return ptr::null_mut();
            }
        };

        let df = match CsvParser::new().parse_str(csv) {
            Ok(df) => df,
            Err(e) => {
                fail(&e);
                return ptr::null_mut();
            }
        };

        let profiles = profile_dataframe(&df);

        let ctx = Box::new(ProfileContext {
            dataframe: df,
            column_profiles: profiles,
        });
        Box::into_raw(ctx)
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            refuse(INSIGHT_ERR_PANIC, "panic in insight_profile_csv");
            ptr::null_mut()
        }
    }
}

/// Creates a profile context from a CSV file, read by the engine — the caller
/// does not hold the text, nor marshal it as one string.
///
/// `path` is a UTF-8 file path. Returns null on error (an unreadable file is a
/// parse failure naming the path; see `insight_last_error`).
///
/// # Safety
/// - `path` must be a valid null-terminated UTF-8 string.
/// - The returned handle must be freed with `insight_profile_free`.
#[no_mangle]
pub unsafe extern "C" fn insight_profile_csv_file(path: *const c_char) -> *mut ProfileContext {
    let result = panic::catch_unwind(|| {
        if path.is_null() {
            refuse(INSIGHT_ERR_NULL_PTR, "null path pointer");
            return ptr::null_mut();
        }
        let path = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                refuse(INSIGHT_ERR_PARSE_FAILED, format!("invalid UTF-8 path: {e}"));
                return ptr::null_mut();
            }
        };
        let df = match CsvParser::new().parse_file(path) {
            Ok(df) => df,
            Err(e) => {
                fail(&e);
                return ptr::null_mut();
            }
        };
        let profiles = profile_dataframe(&df);
        Box::into_raw(Box::new(ProfileContext {
            dataframe: df,
            column_profiles: profiles,
        }))
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            refuse(INSIGHT_ERR_PANIC, "panic in insight_profile_csv_file");
            ptr::null_mut()
        }
    }
}

/// Creates a profile context from a column-major JSON string.
///
/// Expected format: `{"col1": [v1, v2, ...], "col2": [...]}`
///
/// Values can be numbers, booleans, strings, or null. Column types are
/// inferred automatically: number → Numeric, bool → Boolean,
/// string → Categorical/Text (based on cardinality), null → missing.
///
/// # Safety
/// - `json_data` must be a valid null-terminated UTF-8 string.
/// - The returned handle must be freed with `insight_profile_free`.
#[no_mangle]
pub unsafe extern "C" fn insight_profile_json(json_data: *const c_char) -> *mut ProfileContext {
    let result = panic::catch_unwind(|| {
        if json_data.is_null() {
            refuse(INSIGHT_ERR_NULL_PTR, "null json_data pointer");
            return ptr::null_mut();
        }

        let c_str = unsafe { CStr::from_ptr(json_data) };
        let json = match c_str.to_str() {
            Ok(s) => s,
            Err(e) => {
                refuse(INSIGHT_ERR_PARSE_FAILED, format!("invalid UTF-8: {e}"));
                return ptr::null_mut();
            }
        };

        let df = match JsonParser::new().parse_str(json) {
            Ok(df) => df,
            Err(e) => {
                fail(&e);
                return ptr::null_mut();
            }
        };

        let profiles = profile_dataframe(&df);

        let ctx = Box::new(ProfileContext {
            dataframe: df,
            column_profiles: profiles,
        });
        Box::into_raw(ctx)
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            refuse(INSIGHT_ERR_PANIC, "panic in insight_profile_json");
            ptr::null_mut()
        }
    }
}

/// Frees a profile context.
///
/// # Safety
/// `ctx` must be a valid pointer from `insight_profile_csv` or `insight_profile_json`, or null.
#[no_mangle]
pub unsafe extern "C" fn insight_profile_free(ctx: *mut ProfileContext) {
    if !ctx.is_null() {
        let _ = unsafe { Box::from_raw(ctx) };
    }
}

/// Returns the number of rows in the profiled dataset.
///
/// # Safety
/// `ctx` must be a valid, non-null profile context.
#[no_mangle]
pub unsafe extern "C" fn insight_profile_row_count(ctx: *const ProfileContext) -> i64 {
    if ctx.is_null() {
        refuse(INSIGHT_ERR_NULL_PTR, "null context");
        return -1;
    }
    let ctx = unsafe { &*ctx };
    ctx.dataframe.row_count() as i64
}

/// Returns the number of columns in the profiled dataset.
///
/// # Safety
/// `ctx` must be a valid, non-null profile context.
#[no_mangle]
pub unsafe extern "C" fn insight_profile_col_count(ctx: *const ProfileContext) -> i64 {
    if ctx.is_null() {
        refuse(INSIGHT_ERR_NULL_PTR, "null context");
        return -1;
    }
    let ctx = unsafe { &*ctx };
    ctx.dataframe.column_count() as i64
}

/// Gets a summary for a specific column.
///
/// # Safety
/// `ctx` must be valid. `out` must point to a valid `CColumnSummary`.
#[no_mangle]
pub unsafe extern "C" fn insight_profile_column(
    ctx: *const ProfileContext,
    col_idx: u32,
    out: *mut CColumnSummary,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if ctx.is_null() || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }
        let ctx = unsafe { &*ctx };
        let idx = col_idx as usize;

        let n_columns = ctx.column_profiles.len();
        if idx >= n_columns {
            return fail(&InsightError::ParameterOutOfRange {
                name: "col_idx".into(),
                low: Bound::Closed(0.0),
                high: Bound::Closed(n_columns as f64 - 1.0),
                got: idx as f64,
            });
        }

        let profile = &ctx.column_profiles[idx];
        let col = ctx.dataframe.column(idx);

        let (data_type, valid_count, null_count) = match col {
            Some(c) => {
                let dt = match c.data_type() {
                    crate::dataframe::DataType::Numeric => 0u32,
                    crate::dataframe::DataType::Boolean => 1,
                    crate::dataframe::DataType::Categorical => 2,
                    crate::dataframe::DataType::Text => 3,
                };
                let vc = c.valid_count() as u64;
                let nc = c.null_count() as u64;
                (dt, vc, nc)
            }
            None => (3, 0, 0),
        };

        let (mean, std_dev, min, max) = match &profile.numeric {
            // A C struct has no absent value: a statistic the data does
            // not define is NaN, as for a non-numeric column.
            Some(np) => (
                np.mean.unwrap_or(f64::NAN),
                np.std_dev.unwrap_or(f64::NAN),
                np.min.unwrap_or(f64::NAN),
                np.max.unwrap_or(f64::NAN),
            ),
            None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        };

        unsafe {
            (*out) = CColumnSummary {
                index: col_idx,
                valid_count,
                null_count,
                data_type,
                mean,
                std_dev,
                min,
                max,
            };
        }

        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_profile_column"),
    }
}

// ── K-Means FFI ──────────────────────────────────────────────────────

/// C-compatible K-Means result.
#[repr(C)]
pub struct CKMeansResult {
    /// Number of clusters.
    pub k: u32,
    /// WCSS value.
    pub wcss: f64,
    /// Number of iterations.
    pub iterations: u32,
    /// Cluster labels (length = n_rows). Caller must free with `insight_free_labels`.
    pub labels: *mut u32,
    /// Number of labels.
    pub n_labels: u32,
}

/// Runs K-Means on row-major data.
///
/// # Safety
/// - `data` must point to `n_rows * n_cols` contiguous f64 values (row-major).
/// - `out` must point to a valid `CKMeansResult`.
/// - The caller must free `out.labels` with `insight_free_labels`.
#[no_mangle]
pub unsafe extern "C" fn insight_kmeans(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    k: u32,
    out: *mut CKMeansResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(data, n_rows) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        let raw = unsafe { view(data, n * d) };

        // Convert to Vec<Vec<f64>>
        let points: Vec<Vec<f64>> = (0..n).map(|i| raw[i * d..(i + 1) * d].to_vec()).collect();

        let config = KMeansConfig::new(k as usize);
        let km_result = match kmeans(&points, &config) {
            Ok(r) => r,
            Err(e) => {
                return fail(&e);
            }
        };

        // Allocate labels array
        let mut labels: Vec<u32> = km_result.labels.iter().map(|&l| l as u32).collect();
        let labels_ptr = labels.as_mut_ptr();
        let labels_len = labels.len() as u32;
        std::mem::forget(labels);

        unsafe {
            (*out) = CKMeansResult {
                k: km_result.k as u32,
                wcss: km_result.wcss,
                iterations: km_result.iterations as u32,
                labels: labels_ptr,
                n_labels: labels_len,
            };
        }

        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_kmeans"),
    }
}

/// Frees a labels array allocated by `insight_kmeans`.
///
/// # Safety
/// `labels` must have been allocated by an insight FFI function, or be null.
#[no_mangle]
pub unsafe extern "C" fn insight_free_labels(labels: *mut u32, count: u32) {
    if !labels.is_null() {
        let _ = unsafe { Vec::from_raw_parts(labels, count as usize, count as usize) };
    }
}

// ── PCA FFI ──────────────────────────────────────────────────────────

/// C-compatible PCA result.
#[repr(C)]
pub struct CPcaResult {
    /// Number of components retained.
    pub n_components: u32,
    /// Number of original features (columns of input data).
    pub n_features: u32,
    /// Number of input rows (samples).
    pub n_samples: u32,
    /// Explained variance ratios (length = n_components).
    /// Caller must free with `insight_free_f64_array`.
    pub explained_variance: *mut f64,
    /// Cumulative explained variance ratios (length = n_components).
    /// Caller must free with `insight_free_f64_array`.
    pub cumulative_variance: *mut f64,
    /// Component loadings, row-major (length = n_components * n_features).
    /// Row k (k-th component) contains the loading weights for original features.
    /// Caller must free with `insight_free_f64_array`.
    pub loadings: *mut f64,
    /// Projected scores, row-major (length = n_samples * n_components).
    /// Row i contains the PC-space coordinates of input row i.
    /// Caller must free with `insight_free_f64_array`.
    pub scores: *mut f64,
}

/// Runs PCA on row-major data.
///
/// # Safety
/// - `data` must point to `n_rows * n_cols` contiguous f64 values (row-major).
/// - `out` must point to a valid `CPcaResult`.
/// - Caller must free each of `out.explained_variance`, `out.cumulative_variance`,
///   `out.loadings`, `out.scores` with `insight_free_f64_array`, passing the
///   appropriate length (see field docs).
#[no_mangle]
pub unsafe extern "C" fn insight_pca(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    n_components: u32,
    auto_scale: i32,
    out: *mut CPcaResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(data, n_rows) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        let raw = unsafe { view(data, n * d) };

        let points: Vec<Vec<f64>> = (0..n).map(|i| raw[i * d..(i + 1) * d].to_vec()).collect();

        let config = PcaConfig::new(n_components as usize).auto_scale(auto_scale != 0);
        let pca_result = match pca(&points, &config) {
            Ok(r) => r,
            Err(e) => {
                return fail(&e);
            }
        };

        let n_components_actual = pca_result.n_components;
        let n_features = pca_result.n_features;
        let n_samples = points.len();

        let evr_ptr = vec_to_raw(pca_result.explained_variance_ratio);
        let cum_ptr = vec_to_raw(pca_result.cumulative_variance_ratio);

        let mut loadings_flat: Vec<f64> =
            Vec::with_capacity(n_components_actual.saturating_mul(n_features));
        for row in &pca_result.loadings {
            loadings_flat.extend_from_slice(row);
        }
        let loadings_ptr = vec_to_raw(loadings_flat);

        let mut scores_flat: Vec<f64> =
            Vec::with_capacity(n_samples.saturating_mul(n_components_actual));
        for row in &pca_result.scores {
            scores_flat.extend_from_slice(row);
        }
        let scores_ptr = vec_to_raw(scores_flat);

        unsafe {
            (*out) = CPcaResult {
                n_components: n_components_actual as u32,
                n_features: n_features as u32,
                n_samples: n_samples as u32,
                explained_variance: evr_ptr,
                cumulative_variance: cum_ptr,
                loadings: loadings_ptr,
                scores: scores_ptr,
            };
        }

        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_pca"),
    }
}

fn vec_to_raw(mut v: Vec<f64>) -> *mut f64 {
    v.shrink_to_fit();
    debug_assert_eq!(v.len(), v.capacity());
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

// ── Silhouette FFI ───────────────────────────────────────────────────

/// C-compatible silhouette analysis result.
#[repr(C)]
pub struct CSilhouetteResult {
    /// Mean silhouette across samples that had a defined silhouette.
    pub avg: f64,
    /// Per-sample silhouette scores (length = n_rows).
    /// Caller must free with `insight_free_f64_array`.
    pub per_sample: *mut f64,
    /// Number of samples (mirrors n_rows the caller passed in).
    pub n_samples: u32,
}

/// Computes silhouette scores for an existing clustering assignment.
///
/// Works with any clustering output (`KMeans`, `MiniBatchKMeans`, `Hierarchical`,
/// `Dbscan`, `Hdbscan`) — the caller passes the already-computed `labels`.
/// `k` is the number of distinct clusters represented in `labels` and is used
/// only as an upper bound on the cluster-id loop.
///
/// # Safety
/// - `data` must point to `n_rows * n_cols` contiguous f64 values (row-major).
/// - `labels` must point to `n_rows` u32 values, each `< k`.
/// - `out` must point to a valid `CSilhouetteResult`.
/// - Caller must free `out.per_sample` with `insight_free_f64_array(ptr, out.n_samples)`.
#[no_mangle]
pub unsafe extern "C" fn insight_silhouette(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    labels: *const u32,
    k: u32,
    out: *mut CSilhouetteResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(data, n_rows) || absent(labels, n_rows) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        if n == 0 || d == 0 {
            return fail(&InsightError::empty("data"));
        }

        let raw = unsafe { view(data, n * d) };
        let raw_labels = unsafe { view(labels, n) };

        let points: Vec<Vec<f64>> = (0..n).map(|i| raw[i * d..(i + 1) * d].to_vec()).collect();
        let labels_usize: Vec<usize> = raw_labels.iter().map(|&l| l as usize).collect();

        // Validate label range
        let k_usize = k as usize;
        if let Some((index, &bad)) = labels_usize.iter().enumerate().find(|(_, &l)| l >= k_usize) {
            record(Refusal::new(
                Code::ParameterOutOfRange,
                format!("label {bad} out of range for k={k_usize}"),
                serde_json::json!({
                    "parameter": "labels",
                    "index": index,
                    "min": 0,
                    "max": k_usize.saturating_sub(1),
                    "got": bad,
                }),
            ));
            return INSIGHT_ERR_INVALID_INPUT;
        }

        let analysis = silhouette_samples(&points, &labels_usize, k_usize);

        let per_sample_ptr = vec_to_raw(analysis.per_sample);

        unsafe {
            (*out) = CSilhouetteResult {
                avg: analysis.avg,
                per_sample: per_sample_ptr,
                n_samples: n as u32,
            };
        }

        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_silhouette"),
    }
}

/// Frees an f64 array allocated by an insight FFI function.
///
/// # Safety
/// `ptr` must have been allocated by an insight FFI function, or be null.
#[no_mangle]
pub unsafe extern "C" fn insight_free_f64_array(ptr: *mut f64, count: u32) {
    if !ptr.is_null() {
        let _ = unsafe { Vec::from_raw_parts(ptr, count as usize, count as usize) };
    }
}

// ── DBSCAN FFI ──────────────────────────────────────────────────────

/// C-compatible DBSCAN result.
#[repr(C)]
pub struct CDbscanResult {
    /// Number of clusters discovered.
    pub n_clusters: u32,
    /// Number of noise points.
    pub noise_count: u32,
    /// Cluster labels (length = n_rows). -1 = noise, >= 0 = cluster id.
    /// Caller must free with `insight_free_i32_array`.
    pub labels: *mut i32,
    /// Number of labels.
    pub n_labels: u32,
}

/// Runs DBSCAN on row-major data.
///
/// # Safety
/// - `data` must point to `n_rows * n_cols` contiguous f64 values (row-major).
/// - `out` must point to a valid `CDbscanResult`.
/// - Caller must free `out.labels` with `insight_free_i32_array`.
#[no_mangle]
pub unsafe extern "C" fn insight_dbscan(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    epsilon: f64,
    min_samples: u32,
    out: *mut CDbscanResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(data, n_rows) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        let raw = unsafe { view(data, n * d) };

        let points: Vec<Vec<f64>> = (0..n).map(|i| raw[i * d..(i + 1) * d].to_vec()).collect();

        let config = DbscanConfig::new(epsilon, min_samples as usize);
        let db_result = match dbscan(&points, &config) {
            Ok(r) => r,
            Err(e) => {
                return fail(&e);
            }
        };

        // Convert labels: None → -1, Some(id) → id as i32
        let mut labels: Vec<i32> = db_result
            .labels
            .iter()
            .map(|l| match l {
                Some(id) => *id as i32,
                None => -1,
            })
            .collect();
        let labels_ptr = labels.as_mut_ptr();
        let labels_len = labels.len() as u32;
        std::mem::forget(labels);

        unsafe {
            (*out) = CDbscanResult {
                n_clusters: db_result.n_clusters as u32,
                noise_count: db_result.noise_count as u32,
                labels: labels_ptr,
                n_labels: labels_len,
            };
        }

        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_dbscan"),
    }
}

/// Frees an i32 array allocated by an insight FFI function.
///
/// # Safety
/// `ptr` must have been allocated by an insight FFI function, or be null.
#[no_mangle]
pub unsafe extern "C" fn insight_free_i32_array(ptr: *mut i32, count: u32) {
    if !ptr.is_null() {
        let _ = unsafe { Vec::from_raw_parts(ptr, count as usize, count as usize) };
    }
}

// ── Distribution Analysis FFI ───────────────────────────────────────

/// C-compatible distribution analysis result (normality assessment).
#[repr(C)]
pub struct CDistributionResult {
    /// Number of observations.
    pub n: u32,
    /// KS test statistic (NaN if unavailable).
    pub ks_statistic: f64,
    /// KS test p-value (NaN if unavailable).
    pub ks_p_value: f64,
    /// Jarque-Bera test statistic (NaN if unavailable).
    pub jb_statistic: f64,
    /// Jarque-Bera test p-value (NaN if unavailable).
    pub jb_p_value: f64,
    /// Shapiro-Wilk W statistic (NaN if unavailable).
    pub sw_statistic: f64,
    /// Shapiro-Wilk p-value (NaN if unavailable).
    pub sw_p_value: f64,
    /// Anderson-Darling A*² statistic (NaN if unavailable).
    pub ad_statistic: f64,
    /// Anderson-Darling p-value (NaN if unavailable).
    pub ad_p_value: f64,
    /// Whether data appears normally distributed. 1 = normal, 0 = not normal.
    pub is_normal: i32,
}

/// Runs distribution analysis (normality testing) on a data vector.
///
/// `significance_level` is the level the normality tests are read at, in
/// (0, 1); outside it the call is refused with `INSIGHT_ERR_INVALID_PARAM`
/// (`parameter_out_of_range`, `parameter` `significance_level`).
///
/// # Safety
/// - `data` must point to `n` contiguous f64 values.
/// - `out` must point to a valid `CDistributionResult`.
#[no_mangle]
pub unsafe extern "C" fn insight_distribution(
    data: *const f64,
    n: u32,
    significance_level: f64,
    out: *mut CDistributionResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(data, n) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let len = n as usize;
        let raw = unsafe { view(data, len) };
        let values: Vec<f64> = raw.to_vec();

        let config = DistributionConfig {
            significance_level,
            compute_ecdf: false,
            compute_histogram: false,
            compute_qq_plot: false,
            ..Default::default()
        };

        let dist_result = match distribution_analysis(&values, &config) {
            Ok(r) => r,
            Err(e) => {
                return fail(&e);
            }
        };

        let (ks_stat, ks_p) = dist_result
            .normality
            .ks_test
            .map_or((f64::NAN, f64::NAN), |t| (t.statistic, t.p_value));
        let (jb_stat, jb_p) = dist_result
            .normality
            .jarque_bera
            .map_or((f64::NAN, f64::NAN), |t| (t.statistic, t.p_value));
        let (sw_stat, sw_p) = dist_result
            .normality
            .shapiro_wilk
            .map_or((f64::NAN, f64::NAN), |t| (t.statistic, t.p_value));
        let (ad_stat, ad_p) = dist_result
            .normality
            .anderson_darling
            .map_or((f64::NAN, f64::NAN), |t| (t.statistic, t.p_value));

        unsafe {
            (*out) = CDistributionResult {
                n: dist_result.n as u32,
                ks_statistic: ks_stat,
                ks_p_value: ks_p,
                jb_statistic: jb_stat,
                jb_p_value: jb_p,
                sw_statistic: sw_stat,
                sw_p_value: sw_p,
                ad_statistic: ad_stat,
                ad_p_value: ad_p,
                is_normal: if dist_result.normality.is_normal {
                    1
                } else {
                    0
                },
            };
        }

        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_distribution"),
    }
}

// ── Feature Importance FFI ──────────────────────────────────────────

/// C-compatible feature importance result.
#[repr(C)]
pub struct CFeatureImportanceResult {
    /// Importance scores per feature (length = n_cols). Higher = more important.
    /// Caller must free with `insight_free_f64_array`.
    pub scores: *mut f64,
    /// Number of scores.
    pub n_scores: u32,
    /// Condition number of the feature correlation matrix.
    pub condition_number: f64,
    /// Number of low-variance features detected.
    pub n_low_variance: u32,
    /// Number of high-correlation pairs detected.
    pub n_high_corr_pairs: u32,
}

/// Runs feature importance analysis on column-major data.
///
/// # Safety
/// - `data` must point to `n_rows * n_cols` contiguous f64 values (row-major).
/// - `out` must point to a valid `CFeatureImportanceResult`.
/// - Caller must free `out.scores` with `insight_free_f64_array`.
#[no_mangle]
pub unsafe extern "C" fn insight_feature_importance(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    out: *mut CFeatureImportanceResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(data, n_rows) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        let raw = unsafe { view(data, n * d) };

        // Convert row-major to column-major vectors
        let columns: Vec<Vec<f64>> = (0..d)
            .map(|col| (0..n).map(|row| raw[row * d + col]).collect())
            .collect();

        let names: Vec<String> = (0..d).map(|i| format!("f{i}")).collect();
        let config = FeatureConfig::default();

        let fi_result = match feature_analysis(&columns, &names, &config) {
            Ok(r) => r,
            Err(e) => {
                return fail(&e);
            }
        };

        // Extract scores in column order
        let mut scores: Vec<f64> = (0..d)
            .map(|i| {
                let name = &names[i];
                fi_result
                    .feature_scores
                    .iter()
                    .find(|s| s.name == *name)
                    .map_or(0.0, |s| s.importance)
            })
            .collect();
        let scores_ptr = scores.as_mut_ptr();
        let scores_len = scores.len() as u32;
        std::mem::forget(scores);

        unsafe {
            (*out) = CFeatureImportanceResult {
                scores: scores_ptr,
                n_scores: scores_len,
                condition_number: fi_result.condition_number,
                n_low_variance: fi_result.low_variance.len() as u32,
                n_high_corr_pairs: fi_result.high_correlations.len() as u32,
            };
        }

        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_feature_importance"),
    }
}

// ── Isolation Forest FFI ─────────────────────────────────────────────

/// C-compatible anomaly detection result (shared by Isolation Forest and LOF).
#[repr(C)]
pub struct CAnomalyResult {
    /// Anomaly score for each point. Higher = more anomalous.
    /// Caller must free with `insight_free_f64_array`.
    pub scores: *mut f64,
    /// Binary anomaly labels (1 = anomaly, 0 = normal).
    /// Caller must free with `insight_free_i32_array`.
    pub anomalies: *mut i32,
    /// Number of data points.
    pub n: u32,
    /// Number of anomalies detected.
    pub anomaly_count: u32,
    /// Threshold used for classification.
    pub threshold: f64,
}

/// Runs Isolation Forest anomaly detection on row-major data.
///
/// # Safety
/// - `data` must point to `n_rows * n_cols` contiguous f64 values (row-major).
/// - `out` must point to a valid `CAnomalyResult`.
/// - Caller must free `out.scores` with `insight_free_f64_array` and
///   `out.anomalies` with `insight_free_i32_array`.
#[no_mangle]
pub unsafe extern "C" fn insight_isolation_forest(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    n_estimators: u32,
    contamination: f64,
    seed: u64,
    out: *mut CAnomalyResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(data, n_rows) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let rows = n_rows as usize;
        let cols = n_cols as usize;
        let raw = unsafe { view(data, rows * cols) };

        let points: Vec<Vec<f64>> = (0..rows)
            .map(|i| raw[i * cols..(i + 1) * cols].to_vec())
            .collect();

        let config = crate::isolation_forest::IsolationForestConfig::default()
            .n_estimators(n_estimators as usize)
            .contamination(contamination)
            .seed(Some(seed));

        let iforest = match crate::isolation_forest::isolation_forest(&points, &config) {
            Ok(r) => r,
            Err(e) => {
                return fail(&e);
            }
        };

        let mut scores = iforest.scores.into_boxed_slice();
        let anomalies_i32: Vec<i32> = iforest.anomalies.iter().map(|&a| a as i32).collect();
        let mut anomalies = anomalies_i32.into_boxed_slice();

        unsafe {
            (*out) = CAnomalyResult {
                scores: scores.as_mut_ptr(),
                anomalies: anomalies.as_mut_ptr(),
                n: n_rows,
                anomaly_count: iforest.anomaly_count as u32,
                threshold: iforest.threshold,
            };
        }

        std::mem::forget(scores);
        std::mem::forget(anomalies);

        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_isolation_forest"),
    }
}

// ── LOF FFI ─────────────────────────────────────────────────────────

/// Runs Local Outlier Factor anomaly detection on row-major data.
///
/// A point is an anomaly when its LOF score exceeds `threshold`, a finite
/// number `> 0` (LOF scores are positive; about 1 for a point as dense as its
/// neighbours). A `threshold` at or below 0, or a `k` of 0, is refused with
/// `INSIGHT_ERR_INVALID_PARAM` (`parameter_out_of_range`).
///
/// # Safety
/// - `data` must point to `n_rows * n_cols` contiguous f64 values (row-major).
/// - `out` must point to a valid `CAnomalyResult`.
/// - Caller must free `out.scores` with `insight_free_f64_array` and
///   `out.anomalies` with `insight_free_i32_array`.
#[no_mangle]
pub unsafe extern "C" fn insight_lof(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    k: u32,
    threshold: f64,
    out: *mut CAnomalyResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(data, n_rows) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let rows = n_rows as usize;
        let cols = n_cols as usize;
        let raw = unsafe { view(data, rows * cols) };

        let points: Vec<Vec<f64>> = (0..rows)
            .map(|i| raw[i * cols..(i + 1) * cols].to_vec())
            .collect();

        let config = crate::lof::LofConfig::default()
            .k(k as usize)
            .threshold(threshold);

        let lof_result = match crate::lof::lof(&points, &config) {
            Ok(r) => r,
            Err(e) => {
                return fail(&e);
            }
        };

        let mut scores = lof_result.scores.into_boxed_slice();
        let anomalies_i32: Vec<i32> = lof_result.anomalies.iter().map(|&a| a as i32).collect();
        let mut anomalies = anomalies_i32.into_boxed_slice();

        unsafe {
            (*out) = CAnomalyResult {
                scores: scores.as_mut_ptr(),
                anomalies: anomalies.as_mut_ptr(),
                n: n_rows,
                anomaly_count: lof_result.anomaly_count as u32,
                threshold: lof_result.threshold,
            };
        }

        std::mem::forget(scores);
        std::mem::forget(anomalies);

        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_lof"),
    }
}

// ── Correlation FFI ─────────────────────────────────────────────────

/// C-compatible correlation result.
#[repr(C)]
pub struct CCorrelationResult {
    /// Number of variables (n).
    pub n_vars: u32,
    /// Flat n×n correlation matrix (row-major). Caller must free with `insight_free_f64_array`.
    pub matrix: *mut f64,
    /// Number of high-correlation pairs found.
    pub n_high_pairs: u32,
}

/// Method codes for [`insight_correlation`].
pub const INSIGHT_CORR_PEARSON: u32 = 0;
/// Spearman rank correlation.
pub const INSIGHT_CORR_SPEARMAN: u32 = 1;
/// Kendall tau-b rank correlation.
pub const INSIGHT_CORR_KENDALL: u32 = 2;

/// Computes a correlation matrix over row-major numeric data.
///
/// `data`: flat array of `n_rows × n_cols` f64 values, row-major.
/// `method`: one of `INSIGHT_CORR_PEARSON` (0) / `_SPEARMAN` (1) / `_KENDALL` (2).
/// `out`: pointer to a `CCorrelationResult`.
///
/// Returns 0 on success, negative on error. Unknown `method` values
/// return `INSIGHT_ERR_INVALID_PARAM` (`unknown_option`). Fewer than 3 rows
/// is `insufficient_data` (`parameter: "data"`, `min: 3`); a constant column
/// is `degenerate_data` naming it by position (`parameter: "data.1"` for the
/// second); no columns is `empty_input`. One column gives the 1 x 1 matrix
/// `[1]`.
///
/// # Safety
/// `data` must point to `n_rows * n_cols` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_correlation(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    method: u32,
    out: *mut CCorrelationResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(data, n_rows) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let nr = n_rows as usize;
        let nc = n_cols as usize;

        let corr_method = match method {
            INSIGHT_CORR_PEARSON => crate::analysis::CorrelationMethod::Pearson,
            INSIGHT_CORR_SPEARMAN => crate::analysis::CorrelationMethod::Spearman,
            INSIGHT_CORR_KENDALL => crate::analysis::CorrelationMethod::Kendall,
            other => {
                return unknown_selector(
                    "method",
                    other,
                    &[
                        INSIGHT_CORR_PEARSON,
                        INSIGHT_CORR_SPEARMAN,
                        INSIGHT_CORR_KENDALL,
                    ],
                    format!("method must be 0 (Pearson), 1 (Spearman) or 2 (Kendall), got {other}"),
                );
            }
        };

        let raw = unsafe { view(data, nr * nc) };

        // Convert row-major flat to column-major Vec<Vec<f64>>
        let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(nr); nc];
        for row in 0..nr {
            for col in 0..nc {
                columns[col].push(raw[row * nc + col]);
            }
        }

        // Columns have no names here: a refusal names one by its position,
        // `data.1` for the second.
        let names: Vec<String> = (0..nc).map(|i| i.to_string()).collect();
        let config = crate::analysis::CorrelationConfig {
            method: corr_method,
            high_threshold: 0.7,
        };

        match crate::analysis::correlation_analysis(&columns, &names, &config) {
            Ok(result) => {
                let out_ref = unsafe { &mut *out };
                out_ref.n_vars = nc as u32;
                out_ref.n_high_pairs = result.high_pairs.len() as u32;

                // Flatten the correlation matrix to row-major
                let mut flat = Vec::with_capacity(nc * nc);
                for r in 0..nc {
                    for c in 0..nc {
                        flat.push(result.matrix.get(r, c));
                    }
                }
                let mut boxed = flat.into_boxed_slice();
                out_ref.matrix = boxed.as_mut_ptr();
                std::mem::forget(boxed);

                INSIGHT_OK
            }
            Err(e) => fail(&e),
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_correlation"),
    }
}

// ── Regression FFI ──────────────────────────────────────────────────

/// C-compatible simple regression result.
#[repr(C)]
pub struct CRegressionResult {
    /// Intercept (β₀).
    pub intercept: f64,
    /// Slope (β₁).
    pub slope: f64,
    /// R² (coefficient of determination).
    pub r_squared: f64,
    /// Adjusted R².
    pub adj_r_squared: f64,
    /// P-value for the F-test.
    pub f_p_value: f64,
}

/// Computes simple linear regression (one predictor).
///
/// `x`: predictor array of length `n`.
/// `y`: target array of length `n`.
/// `out`: pointer to a `CRegressionResult`.
///
/// Returns 0 on success, negative on error.
///
/// # Safety
/// `x` and `y` must point to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_regression(
    x: *const f64,
    y: *const f64,
    n: u32,
    out: *mut CRegressionResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(x, n) || absent(y, n) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let len = n as usize;
        let x_slice = unsafe { view(x, len) };
        let y_slice = unsafe { view(y, len) };

        let x_vecs = vec![x_slice.to_vec()];
        let names = vec!["x".to_string()];

        match crate::analysis::regression_analysis(&x_vecs, &names, y_slice, "y") {
            Ok(reg) => {
                let out_ref = unsafe { &mut *out };
                out_ref.intercept = reg.coefficients[0];
                out_ref.slope = reg.coefficients[1];
                out_ref.r_squared = reg.r_squared;
                out_ref.adj_r_squared = reg.adj_r_squared;
                out_ref.f_p_value = reg.f_p_value;

                INSIGHT_OK
            }
            Err(e) => fail(&e),
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_regression"),
    }
}

// ── Mahalanobis FFI ──────────────────────────────────────────────────

/// C-compatible Mahalanobis distance result.
#[repr(C)]
pub struct CMahalanobisResult {
    /// Mahalanobis distances (length = n_rows).
    /// Caller must free with `insight_free_f64_array`.
    pub distances: *mut f64,
    /// Anomaly flags (1 = outlier, 0 = normal, length = n_rows).
    /// Caller must free with `insight_free_i32_array`.
    pub anomalies: *mut i32,
    /// Number of data points.
    pub n: u32,
    /// Chi-squared threshold used.
    pub threshold: f64,
    /// Number of outliers detected.
    pub outlier_count: u32,
}

/// Runs Mahalanobis distance multivariate outlier detection on row-major data.
///
/// `quantile` is the probability of the threshold's quantile, e.g. 0.975: the
/// squared distance's exact law when the mean and covariance are estimated
/// from the same rows is `(n−1)²/n · Beta(p/2, (n−p−1)/2)`, so under normality
/// a point is flagged with probability `1 − quantile`. A value outside (0, 1)
/// is refused with `INSIGHT_ERR_INVALID_PARAM`. Fewer than `n_cols + 2` rows
/// are refused (`insufficient_data`).
///
/// # Safety
/// - `data` must point to `n_rows * n_cols` contiguous f64 values (row-major).
/// - `out` must point to a valid `CMahalanobisResult`.
/// - Caller must free output arrays.
#[no_mangle]
pub unsafe extern "C" fn insight_mahalanobis(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    quantile: f64,
    out: *mut CMahalanobisResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(data, n_rows) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let nr = n_rows as usize;
        let nc = n_cols as usize;

        let raw = unsafe { view(data, nr * nc) };
        let points: Vec<Vec<f64>> = (0..nr)
            .map(|i| raw[i * nc..(i + 1) * nc].to_vec())
            .collect();

        let config = crate::mahalanobis::MahalanobisConfig { quantile };

        match crate::mahalanobis::mahalanobis(&points, &config) {
            Ok(r) => {
                let out_ref = unsafe { &mut *out };
                out_ref.n = nr as u32;
                out_ref.threshold = r.threshold;
                out_ref.outlier_count = r.outlier_count as u32;

                let mut dists = r.distances.into_boxed_slice();
                out_ref.distances = dists.as_mut_ptr();
                std::mem::forget(dists);

                let anoms: Vec<i32> = r.anomalies.iter().map(|&a| a as i32).collect();
                let mut anoms_box = anoms.into_boxed_slice();
                out_ref.anomalies = anoms_box.as_mut_ptr();
                std::mem::forget(anoms_box);

                INSIGHT_OK
            }
            Err(e) => fail(&e),
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_mahalanobis"),
    }
}

// ── Cramér's V FFI ──────────────────────────────────────────────────

/// C-compatible Cramér's V result.
#[repr(C)]
pub struct CCramersVResult {
    /// Cramér's V (0 to 1).
    pub v: f64,
    /// Chi-squared statistic.
    pub chi_squared: f64,
    /// P-value.
    pub p_value: f64,
}

/// Computes Cramér's V for a contingency table.
///
/// `table`: flat row-major contingency table (observed frequencies).
///
/// Refusals, each with its `code` and `parameter` in `insight_last_error_json`:
/// `n_rows` or `n_cols` below 2, or a negative frequency (`table[i]`) →
/// `INSIGHT_ERR_INVALID_PARAM` (`parameter_out_of_range`); a NaN or infinity →
/// `INSIGHT_ERR_INVALID_INPUT` (`value_not_finite`, `table`, `index`); a table
/// with no observations, or a row or column that sums to 0 →
/// `INSIGHT_ERR_DEGENERATE_DATA` (`degenerate_data`, `table`).
///
/// # Safety
/// `table` must point to `n_rows * n_cols` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_cramers_v(
    table: *const f64,
    n_rows: u32,
    n_cols: u32,
    out: *mut CCramersVResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(table, n_rows) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let nr = n_rows as usize;
        let nc = n_cols as usize;
        let raw = unsafe { view(table, nr * nc) };

        match crate::analysis::cramers_v(raw, nr, nc) {
            Ok(r) => {
                let out_ref = unsafe { &mut *out };
                out_ref.v = r.v;
                out_ref.chi_squared = r.chi_squared;
                out_ref.p_value = r.p_value;
                INSIGHT_OK
            }
            Err(e) => fail(&e),
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_cramers_v"),
    }
}

// ── ANOVA Feature Selection FFI ─────────────────────────────────────

/// C-compatible ANOVA feature result.
#[repr(C)]
pub struct CAnovaFeature {
    /// The feature's column in the input (the list is sorted by p-value).
    pub index: u32,
    /// F-statistic; infinite for a feature that separates the classes
    /// perfectly (constant within every class, not across them).
    pub f_statistic: f64,
    /// P-value; 0 for a perfectly separating feature.
    pub p_value: f64,
}

/// C-compatible ANOVA selection result.
#[repr(C)]
pub struct CAnovaSelectionResult {
    /// Per-feature results sorted by p-value ascending.
    /// Caller must free with `insight_free_anova_features`.
    pub features: *mut CAnovaFeature,
    /// Number of features.
    pub n_features: u32,
    /// Number of significant features.
    pub n_selected: u32,
}

/// Runs ANOVA F-test feature selection.
///
/// `data`: row-major n_rows × n_features.
/// `target`: class labels (u32), length n_rows.
/// `significance_level`: a feature is selected when its p-value is at most
/// this, in (0, 1); outside it the call is refused with
/// `INSIGHT_ERR_INVALID_PARAM` (`parameter_out_of_range`).
///
/// # Safety
/// All pointers must be valid. Caller frees output.
#[no_mangle]
pub unsafe extern "C" fn insight_anova_select(
    data: *const f64,
    n_rows: u32,
    n_features: u32,
    target: *const u32,
    significance_level: f64,
    out: *mut CAnovaSelectionResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(data, n_rows) || absent(target, n_rows) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let nr = n_rows as usize;
        let nf = n_features as usize;
        let raw = unsafe { view(data, nr * nf) };
        let target_raw = unsafe { view(target, nr) };

        // Convert row-major to column-major
        let columns: Vec<Vec<f64>> = (0..nf)
            .map(|col| (0..nr).map(|row| raw[row * nf + col]).collect())
            .collect();

        let names: Vec<String> = (0..nf).map(|i| format!("f{i}")).collect();
        let target_usize: Vec<usize> = target_raw.iter().map(|&t| t as usize).collect();

        match crate::analysis::anova_feature_selection(
            &columns,
            &names,
            &target_usize,
            significance_level,
        ) {
            Ok(r) => {
                let out_ref = unsafe { &mut *out };
                out_ref.n_features = r.features.len() as u32;
                out_ref.n_selected = r.selected_indices.len() as u32;

                let c_features: Vec<CAnovaFeature> = r
                    .features
                    .iter()
                    .map(|f| CAnovaFeature {
                        index: f.index as u32,
                        f_statistic: f.f_statistic,
                        p_value: f.p_value,
                    })
                    .collect();
                let mut boxed = c_features.into_boxed_slice();
                out_ref.features = boxed.as_mut_ptr();
                std::mem::forget(boxed);

                INSIGHT_OK
            }
            Err(e) => fail(&e),
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_anova_select"),
    }
}

/// Frees ANOVA feature result array.
///
/// # Safety
/// `ptr` must have been allocated by `insight_anova_select`, or be null.
#[no_mangle]
pub unsafe extern "C" fn insight_free_anova_features(ptr: *mut CAnovaFeature, count: u32) {
    if !ptr.is_null() {
        let _ = unsafe { Vec::from_raw_parts(ptr, count as usize, count as usize) };
    }
}

// ── Version ──────────────────────────────────────────────────────────

/// Returns the version string of u-insight.
///
/// # Safety
/// The returned string is a static string literal. Do not free it.
#[no_mangle]
pub extern "C" fn insight_version() -> *const c_char {
    // The crate's own version: a literal here reported 0.1.0 for every release.
    static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION.as_ptr().cast()
}

/// Returns every refusal `code` the library answers with, and the fields
/// beside each, as JSON: `{"refusals": [{"code", "fields"}], "violations":
/// []}` -- built from the table the refusals are made from, so a caller that
/// translates codes can check its list against the library's.
///
/// # Safety
/// The returned string lives for the life of the process. Do not free it.
#[no_mangle]
pub extern "C" fn insight_codes() -> *const c_char {
    static CODES: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
    CODES
        .get_or_init(|| {
            std::ffi::CString::new(crate::refusal::code_catalog_json().to_string())
                .expect("JSON has no interior NUL")
        })
        .as_ptr()
}

// ── Hierarchical Clustering FFI ──────────────────────────────────────

/// C-compatible hierarchical clustering result.
#[repr(C)]
pub struct CHierarchicalResult {
    /// Number of flat clusters (0 if no cut).
    pub n_clusters: u32,
    /// Flat cluster labels (length = n_rows). -1 if no labels.
    /// Caller must free with `insight_free_i32_array`.
    pub labels: *mut i32,
    /// Number of labels.
    pub n_labels: u32,
    /// Number of merges in the dendrogram.
    pub n_merges: u32,
    /// Merge distances (length = n_merges).
    /// Caller must free with `insight_free_f64_array`.
    pub merge_distances: *mut f64,
    /// Merge sizes (length = n_merges).
    /// Caller must free with `insight_free_i32_array`.
    pub merge_sizes: *mut i32,
}

/// Linkage codes for [`insight_hierarchical`]: single (nearest neighbour).
pub const INSIGHT_LINKAGE_SINGLE: u32 = 0;
/// Complete linkage (farthest neighbour).
pub const INSIGHT_LINKAGE_COMPLETE: u32 = 1;
/// Average linkage (UPGMA).
pub const INSIGHT_LINKAGE_AVERAGE: u32 = 2;
/// Ward's minimum-variance linkage.
pub const INSIGHT_LINKAGE_WARD: u32 = 3;

/// Runs hierarchical agglomerative clustering on row-major data.
///
/// # Parameters
///
/// - `linkage`: one of `INSIGHT_LINKAGE_SINGLE` (0) / `_COMPLETE` (1) / `_AVERAGE` (2) / `_WARD` (3).
/// - `n_clusters`: Desired number of flat clusters (0 = no cut).
///
/// # Safety
/// - `data` must point to `n_rows * n_cols` contiguous f64 values.
/// - `out` must point to a valid `CHierarchicalResult`.
/// - Caller must free output arrays with appropriate free functions.
#[no_mangle]
pub unsafe extern "C" fn insight_hierarchical(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    linkage: u32,
    n_clusters: u32,
    out: *mut CHierarchicalResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(data, n_rows) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        let raw = unsafe { view(data, n * d) };

        let points: Vec<Vec<f64>> = (0..n).map(|i| raw[i * d..(i + 1) * d].to_vec()).collect();

        // A value past 3 used to be read as Ward, so a caller's out-of-range
        // enum ran a method nobody asked for; it is refused and named.
        let linkage_method = match linkage {
            INSIGHT_LINKAGE_SINGLE => Linkage::Single,
            INSIGHT_LINKAGE_COMPLETE => Linkage::Complete,
            INSIGHT_LINKAGE_AVERAGE => Linkage::Average,
            INSIGHT_LINKAGE_WARD => Linkage::Ward,
            other => {
                record(Refusal::new(
                    Code::UnknownOption,
                    format!(
                        "linkage must be 0 (single), 1 (complete), 2 (average) or 3 (ward), got {other}"
                    ),
                    serde_json::json!({ "parameter": "linkage", "got": other, "expected": [0, 1, 2, 3] }),
                ));
                return INSIGHT_ERR_INVALID_PARAM;
            }
        };

        // The C ABI exposes no max_points override, so preserve prior unlimited
        // behavior (`max_points: 0`) for native callers rather than impose an
        // un-escapable default guard. The Rust/WASM APIs keep the protective
        // default (and can override it).
        let config = if n_clusters > 0 {
            HierarchicalConfig::with_k(n_clusters as usize)
                .linkage(linkage_method)
                .max_points(0)
        } else {
            HierarchicalConfig {
                linkage: linkage_method,
                n_clusters: None,
                distance_threshold: None,
                max_points: 0,
            }
        };

        let hc_result = match hierarchical(&points, &config) {
            Ok(r) => r,
            Err(e) => {
                return fail(&e);
            }
        };

        // Labels
        let (labels_ptr, labels_len, nc) = match &hc_result.labels {
            Some(labels) => {
                let mut l: Vec<i32> = labels.iter().map(|&v| v as i32).collect();
                let ptr = l.as_mut_ptr();
                let len = l.len() as u32;
                std::mem::forget(l);
                (ptr, len, hc_result.n_clusters.unwrap_or(0) as u32)
            }
            None => (ptr::null_mut(), 0, 0),
        };

        // Merge distances and sizes
        let nm = hc_result.merges.len();
        let mut distances: Vec<f64> = hc_result.merges.iter().map(|m| m.distance).collect();
        let mut sizes: Vec<i32> = hc_result.merges.iter().map(|m| m.size as i32).collect();
        let dist_ptr = distances.as_mut_ptr();
        let size_ptr = sizes.as_mut_ptr();
        std::mem::forget(distances);
        std::mem::forget(sizes);

        unsafe {
            (*out) = CHierarchicalResult {
                n_clusters: nc,
                labels: labels_ptr,
                n_labels: labels_len,
                n_merges: nm as u32,
                merge_distances: dist_ptr,
                merge_sizes: size_ptr,
            };
        }

        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_hierarchical"),
    }
}

// ── HDBSCAN FFI ─────────────────────────────────────────────────────

/// C-compatible HDBSCAN result.
#[repr(C)]
pub struct CHdbscanResult {
    /// Number of clusters found (excluding noise).
    pub n_clusters: u32,
    /// Number of noise points.
    pub noise_count: u32,
    /// Cluster labels (length = n_rows). -1 = noise.
    /// Caller must free with `insight_free_i32_array`.
    pub labels: *mut i32,
    /// Membership probabilities (length = n_rows).
    /// Caller must free with `insight_free_f64_array`.
    pub probabilities: *mut f64,
    /// Number of data points.
    pub n_labels: u32,
}

/// Runs HDBSCAN on row-major data.
///
/// # Safety
/// - `data` must point to `n_rows * n_cols` contiguous f64 values.
/// - `out` must point to a valid `CHdbscanResult`.
/// - Caller must free output arrays with appropriate free functions.
#[no_mangle]
pub unsafe extern "C" fn insight_hdbscan(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    min_cluster_size: u32,
    min_samples: u32,
    out: *mut CHdbscanResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if absent(data, n_rows) || out.is_null() {
            return refuse(INSIGHT_ERR_NULL_PTR, "null pointer");
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        let raw = unsafe { view(data, n * d) };

        let points: Vec<Vec<f64>> = (0..n).map(|i| raw[i * d..(i + 1) * d].to_vec()).collect();

        let mut config = HdbscanConfig::new(min_cluster_size as usize);
        if min_samples > 0 {
            config = config.min_samples(min_samples as usize);
        }

        let hdb_result = match hdbscan(&points, &config) {
            Ok(r) => r,
            Err(e) => {
                return fail(&e);
            }
        };

        // Convert labels: None → -1, Some(id) → id as i32
        let mut labels: Vec<i32> = hdb_result
            .labels
            .iter()
            .map(|l| match l {
                Some(id) => *id as i32,
                None => -1,
            })
            .collect();
        let labels_ptr = labels.as_mut_ptr();
        let labels_len = labels.len() as u32;
        std::mem::forget(labels);

        // Probabilities
        let mut probs = hdb_result.probabilities;
        let probs_ptr = probs.as_mut_ptr();
        std::mem::forget(probs);

        unsafe {
            (*out) = CHdbscanResult {
                n_clusters: hdb_result.n_clusters as u32,
                noise_count: hdb_result.noise_count as u32,
                labels: labels_ptr,
                probabilities: probs_ptr,
                n_labels: labels_len,
            };
        }

        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_hdbscan"),
    }
}

// ── Mutual Information FFI ──────────────────────────────────────────

/// Result of mutual information feature selection via FFI.
#[repr(C)]
pub struct CMutualInfoFeature {
    pub index: u32,
    pub mi: f64,
}

/// Aggregate result for mutual information.
#[repr(C)]
pub struct CMutualInfoResult {
    pub features: *mut CMutualInfoFeature,
    pub n_features: u32,
}

/// Computes mutual information between continuous features and a categorical target.
///
/// # Parameters
/// - `data`: row-major feature matrix (n_rows × n_features)
/// - `target`: categorical target array (u32, length n_rows)
/// - `n_bins`: number of bins (0 = auto via Sturges' rule)
///
/// # Safety
/// Caller must free `out.features` via `insight_free_mi_features`.
#[no_mangle]
pub unsafe extern "C" fn insight_mutual_info(
    data: *const f64,
    n_rows: u32,
    n_features: u32,
    target: *const u32,
    n_bins: u32,
    out: *mut CMutualInfoResult,
) -> i32 {
    if absent(data, n_rows) || absent(target, n_rows) || out.is_null() {
        return refuse(INSIGHT_ERR_NULL_PTR, "null pointer argument");
    }

    let result = panic::catch_unwind(|| {
        let nr = n_rows as usize;
        let nf = n_features as usize;
        let raw = view(data, nr * nf);
        let target_raw = view(target, nr);

        let mut features: Vec<Vec<f64>> = Vec::with_capacity(nf);
        for col in 0..nf {
            let mut v = Vec::with_capacity(nr);
            for row in 0..nr {
                v.push(raw[row * nf + col]);
            }
            features.push(v);
        }

        let names: Vec<String> = (0..nf).map(|i| format!("f{}", i)).collect();
        let target_usize: Vec<usize> = target_raw.iter().map(|&t| t as usize).collect();
        let bins = if n_bins == 0 {
            None
        } else {
            Some(n_bins as usize)
        };

        match mutual_info_classif(&features, &names, &target_usize, bins) {
            Ok(mi_result) => {
                let n = mi_result.features.len();
                let mut c_features: Vec<CMutualInfoFeature> = mi_result
                    .features
                    .iter()
                    .map(|f| CMutualInfoFeature {
                        index: f.index as u32,
                        mi: f.mi,
                    })
                    .collect();

                let ptr = c_features.as_mut_ptr();
                std::mem::forget(c_features);

                (*out) = CMutualInfoResult {
                    features: ptr,
                    n_features: n as u32,
                };

                INSIGHT_OK
            }
            Err(e) => fail(&e),
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_mutual_info"),
    }
}

/// Frees a CMutualInfoFeature array allocated by `insight_mutual_info`.
///
/// # Safety
/// `ptr` must have been returned by `insight_mutual_info` with matching `count`.
#[no_mangle]
pub unsafe extern "C" fn insight_free_mi_features(ptr: *mut CMutualInfoFeature, count: u32) {
    if !ptr.is_null() && count > 0 {
        let _ = Vec::from_raw_parts(ptr, count as usize, count as usize);
    }
}

// ── Mini-Batch K-Means FFI ─────────────────────────────────────────

/// Runs mini-batch K-Means clustering.
///
/// # Parameters
/// - `data`: row-major matrix (n_rows × n_cols)
/// - `k`: number of clusters
/// - `batch_size`: mini-batch size (0 = default 100)
/// - `max_iter`: max iterations (0 = default 100)
/// - `seed`: random seed
///
/// # Safety
/// Caller must free `out.labels` via `insight_free_labels` and
/// `out.centroids` via `insight_free_f64_array(out.centroids, k * n_cols)`.
#[no_mangle]
pub unsafe extern "C" fn insight_mini_batch_kmeans(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    k: u32,
    batch_size: u32,
    max_iter: u32,
    seed: u64,
    out: *mut CKMeansResult,
) -> i32 {
    if absent(data, n_rows) || out.is_null() {
        return refuse(INSIGHT_ERR_NULL_PTR, "null pointer argument");
    }

    let result = panic::catch_unwind(|| {
        let nr = n_rows as usize;
        let nc = n_cols as usize;
        let raw = view(data, nr * nc);

        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(nr);
        for r in 0..nr {
            rows.push(raw[r * nc..(r + 1) * nc].to_vec());
        }

        let config = MiniBatchKMeansConfig {
            k: k as usize,
            batch_size: if batch_size == 0 {
                100
            } else {
                batch_size as usize
            },
            max_iter: if max_iter == 0 {
                100
            } else {
                max_iter as usize
            },
            tol: 1e-4,
            seed: Some(seed),
        };

        match mini_batch_kmeans(&rows, &config) {
            Ok(km_result) => {
                let mut labels: Vec<u32> = km_result.labels.iter().map(|&l| l as u32).collect();
                let labels_len = labels.len() as u32;
                let labels_ptr = labels.as_mut_ptr();
                std::mem::forget(labels);

                (*out) = CKMeansResult {
                    k,
                    wcss: km_result.wcss,
                    iterations: km_result.iterations as u32,
                    labels: labels_ptr,
                    n_labels: labels_len,
                };

                INSIGHT_OK
            }
            Err(e) => fail(&e),
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_mini_batch_kmeans"),
    }
}

// ── Gap Statistic FFI ──────────────────────────────────────────────

/// Result of gap statistic analysis via FFI.
#[repr(C)]
pub struct CGapStatResult {
    /// Optimal K selected by gap criterion.
    pub best_k: u32,
    /// Number of K values tested.
    pub n_values: u32,
    /// Array of (k, gap) pairs flattened: [k0, gap0, k1, gap1, ...].
    pub gap_values: *mut f64,
    /// Array of (k, stderr) pairs flattened: [k0, se0, k1, se1, ...].
    pub std_errors: *mut f64,
}

/// Computes gap statistic to find optimal K for clustering.
///
/// # Parameters
/// - `data`: row-major matrix (n_rows × n_cols)
/// - `k_min`, `k_max`: K range to test
/// - `n_refs`: number of reference datasets
/// - `seed`: random seed
///
/// # Safety
/// Caller must free `out.gap_values` and `out.std_errors` via
/// `insight_free_f64_array(ptr, n_values * 2)`.
#[no_mangle]
pub unsafe extern "C" fn insight_gap_statistic(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    k_min: u32,
    k_max: u32,
    n_refs: u32,
    seed: u64,
    out: *mut CGapStatResult,
) -> i32 {
    if absent(data, n_rows) || out.is_null() {
        return refuse(INSIGHT_ERR_NULL_PTR, "null pointer argument");
    }

    let result = panic::catch_unwind(|| {
        let nr = n_rows as usize;
        let nc = n_cols as usize;
        let raw = view(data, nr * nc);

        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(nr);
        for r in 0..nr {
            rows.push(raw[r * nc..(r + 1) * nc].to_vec());
        }

        match gap_statistic(&rows, k_min as usize, k_max as usize, n_refs as usize, seed) {
            Ok(gap_result) => {
                let n = gap_result.gap_values.len();

                // Flatten (k, gap) pairs
                let mut gaps: Vec<f64> = Vec::with_capacity(n * 2);
                for &(k, g) in &gap_result.gap_values {
                    gaps.push(k as f64);
                    gaps.push(g);
                }
                let gaps_ptr = gaps.as_mut_ptr();
                std::mem::forget(gaps);

                // Flatten (k, se) pairs
                let mut ses: Vec<f64> = Vec::with_capacity(n * 2);
                for &(k, s) in &gap_result.std_errors {
                    ses.push(k as f64);
                    ses.push(s);
                }
                let ses_ptr = ses.as_mut_ptr();
                std::mem::forget(ses);

                (*out) = CGapStatResult {
                    best_k: gap_result.best_k as u32,
                    n_values: n as u32,
                    gap_values: gaps_ptr,
                    std_errors: ses_ptr,
                };

                INSIGHT_OK
            }
            Err(e) => fail(&e),
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_gap_statistic"),
    }
}

// ── Permutation Importance FFI ─────────────────────────────────────

/// Per-feature permutation importance result via FFI.
#[repr(C)]
pub struct CPermImportanceFeature {
    pub index: u32,
    pub importance: f64,
    pub std_dev: f64,
}

/// Aggregate permutation importance result via FFI.
#[repr(C)]
pub struct CPermImportanceResult {
    pub baseline_score: f64,
    pub features: *mut CPermImportanceFeature,
    pub n_features: u32,
}

/// Computes permutation importance for regression features.
///
/// # Parameters
/// - `data`: row-major feature matrix (n_rows × n_features)
/// - `target`: continuous target array (length n_rows)
/// - `n_repeats`: number of permutation repetitions
/// - `seed`: random seed
///
/// # Safety
/// Caller must free `out.features` via `insight_free_perm_features`.
#[no_mangle]
pub unsafe extern "C" fn insight_permutation_importance(
    data: *const f64,
    n_rows: u32,
    n_features: u32,
    target: *const f64,
    n_repeats: u32,
    seed: u64,
    out: *mut CPermImportanceResult,
) -> i32 {
    if absent(data, n_rows) || absent(target, n_rows) || out.is_null() {
        return refuse(INSIGHT_ERR_NULL_PTR, "null pointer argument");
    }

    let result = panic::catch_unwind(|| {
        let nr = n_rows as usize;
        let nf = n_features as usize;
        let raw = view(data, nr * nf);
        let target_raw = view(target, nr);

        let mut features: Vec<Vec<f64>> = Vec::with_capacity(nf);
        for col in 0..nf {
            let mut v = Vec::with_capacity(nr);
            for row in 0..nr {
                v.push(raw[row * nf + col]);
            }
            features.push(v);
        }

        let names: Vec<String> = (0..nf).map(|i| format!("f{}", i)).collect();

        match permutation_importance(&features, &names, target_raw, n_repeats as usize, seed) {
            Ok(pi_result) => {
                let n = pi_result.features.len();
                let mut c_features: Vec<CPermImportanceFeature> = pi_result
                    .features
                    .iter()
                    .map(|f| CPermImportanceFeature {
                        index: f.index as u32,
                        importance: f.importance,
                        std_dev: f.std_dev,
                    })
                    .collect();

                let ptr = c_features.as_mut_ptr();
                std::mem::forget(c_features);

                (*out) = CPermImportanceResult {
                    baseline_score: pi_result.baseline_score,
                    features: ptr,
                    n_features: n as u32,
                };

                INSIGHT_OK
            }
            Err(e) => fail(&e),
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => refuse(INSIGHT_ERR_PANIC, "panic in insight_permutation_importance"),
    }
}

/// Frees a CPermImportanceFeature array.
///
/// # Safety
/// `ptr` must have been returned by `insight_permutation_importance` with matching `count`.
#[no_mangle]
pub unsafe extern "C" fn insight_free_perm_features(ptr: *mut CPermImportanceFeature, count: u32) {
    if !ptr.is_null() && count > 0 {
        let _ = Vec::from_raw_parts(ptr, count as usize, count as usize);
    }
}

// ── Input arrays ─────────────────────────────────────────────────────

/// Whether an input array is missing: a null pointer for an array that is
/// not empty. An empty array may come as a null pointer -- `(NULL, 0)` is a
/// valid empty array in C, and .NET pins an empty array to null -- so it is
/// the routine's own empty-input refusal that answers it, not a null-pointer one.
fn absent<T>(ptr: *const T, len: u32) -> bool {
    ptr.is_null() && len > 0
}

/// The `len` values behind `ptr`; an empty slice when `len` is 0, whatever
/// `ptr` is (`slice::from_raw_parts` needs a non-null pointer even then).
///
/// # Safety
/// `ptr` must point to `len` values when `len > 0`.
unsafe fn view<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(ptr, len) }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn ffi_codes_are_the_refusal_table() {
        let s = unsafe { CStr::from_ptr(insight_codes()) }
            .to_str()
            .expect("UTF-8");
        let codes: serde_json::Value = serde_json::from_str(s).expect("a catalog");
        assert_eq!(codes, crate::refusal::code_catalog_json());
        let refusals = codes["refusals"].as_array().expect("refusals");
        assert_eq!(refusals.len(), crate::refusal::Code::ALL.len());
        assert!(refusals.iter().any(|r| r["code"] == "insufficient_data"));
        assert_eq!(codes["violations"], serde_json::json!([]));
        // The same pointer every call: the string lives for the process.
        assert_eq!(insight_codes(), insight_codes());
    }

    #[test]
    fn ffi_version() {
        let v = insight_version();
        let s = unsafe { CStr::from_ptr(v) }.to_str().unwrap();
        assert_eq!(s, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn ffi_error_lifecycle() {
        insight_clear_error();
        assert!(insight_last_error().is_null());

        assert!(insight_last_error_json().is_null());

        refuse(INSIGHT_ERR_INVALID_INPUT, "test error");
        let msg = unsafe { CStr::from_ptr(insight_last_error()) }
            .to_str()
            .unwrap();
        assert_eq!(msg, "test error");
        assert_eq!(
            last_error_body(),
            Some(serde_json::json!({ "error": "test error", "code": "invalid_input" }))
        );

        insight_clear_error();
        assert!(insight_last_error().is_null());
        assert!(insight_last_error_json().is_null());
    }

    fn last_error_body() -> Option<serde_json::Value> {
        let ptr = insight_last_error_json();
        (!ptr.is_null()).then(|| {
            let text = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
            serde_json::from_str(&text).expect("the body is JSON")
        })
    }

    #[test]
    fn ffi_profile_csv_roundtrip() {
        let csv = CString::new("name,value\nAlice,1.5\nBob,2.3\n").unwrap();
        let ctx = unsafe { insight_profile_csv(csv.as_ptr()) };
        assert!(!ctx.is_null());

        let rows = unsafe { insight_profile_row_count(ctx) };
        assert_eq!(rows, 2);

        let cols = unsafe { insight_profile_col_count(ctx) };
        assert_eq!(cols, 2);

        // Get numeric column profile (column 1 = "value")
        let mut summary = CColumnSummary {
            index: 0,
            valid_count: 0,
            null_count: 0,
            data_type: 0,
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
        };
        let rc = unsafe { insight_profile_column(ctx, 1, &mut summary) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(summary.data_type, 0); // Numeric
        assert!((summary.mean - 1.9).abs() < 0.01);

        unsafe { insight_profile_free(ctx) };
    }

    #[test]
    fn ffi_profile_csv_file_reads_the_file_and_names_what_it_refuses() {
        let path = std::env::temp_dir().join(format!("u-insight-{}.csv", std::process::id()));
        std::fs::write(&path, "name,value\nAlice,1.5\nBob,2.3\n").unwrap();
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let ctx = unsafe { insight_profile_csv_file(c_path.as_ptr()) };
        assert!(!ctx.is_null());
        assert_eq!(unsafe { insight_profile_row_count(ctx) }, 2);
        unsafe { insight_profile_free(ctx) };

        // A row of the wrong width is refused with its line, as from a string.
        std::fs::write(&path, "a,b\n1,2\n3\n").unwrap();
        assert!(unsafe { insight_profile_csv_file(c_path.as_ptr()) }.is_null());
        let body = unsafe { CStr::from_ptr(insight_last_error_json()) }
            .to_str()
            .unwrap()
            .to_string();
        assert!(body.contains("\"line\":3"), "{body}");
        std::fs::remove_file(&path).unwrap();

        // A missing file is refused, not a crash.
        assert!(unsafe { insight_profile_csv_file(c_path.as_ptr()) }.is_null());
        assert!(!insight_last_error().is_null());
        assert!(unsafe { insight_profile_csv_file(ptr::null()) }.is_null());
    }

    #[test]
    fn ffi_profile_null_ptr() {
        let ctx = unsafe { insight_profile_csv(ptr::null()) };
        assert!(ctx.is_null());
    }

    #[test]
    fn ffi_kmeans_basic() {
        // 4 points, 2D, 2 clusters
        let data: Vec<f64> = vec![0.0, 0.0, 0.5, 0.5, 10.0, 10.0, 10.5, 10.5];

        let mut result = CKMeansResult {
            k: 0,
            wcss: 0.0,
            iterations: 0,
            labels: ptr::null_mut(),
            n_labels: 0,
        };

        let rc = unsafe { insight_kmeans(data.as_ptr(), 4, 2, 2, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.k, 2);
        assert_eq!(result.n_labels, 4);

        // Read labels
        let labels = unsafe { slice::from_raw_parts(result.labels, result.n_labels as usize) };
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);

        // Clean up
        unsafe { insight_free_labels(result.labels, result.n_labels) };
    }

    fn empty_pca_result() -> CPcaResult {
        CPcaResult {
            n_components: 0,
            n_features: 0,
            n_samples: 0,
            explained_variance: ptr::null_mut(),
            cumulative_variance: ptr::null_mut(),
            loadings: ptr::null_mut(),
            scores: ptr::null_mut(),
        }
    }

    unsafe fn free_pca_result(result: &CPcaResult) {
        insight_free_f64_array(result.explained_variance, result.n_components);
        insight_free_f64_array(result.cumulative_variance, result.n_components);
        insight_free_f64_array(
            result.loadings,
            result.n_components.saturating_mul(result.n_features),
        );
        insight_free_f64_array(
            result.scores,
            result.n_samples.saturating_mul(result.n_components),
        );
    }

    #[test]
    fn ffi_pca_basic() {
        // 4 points, 2D
        let data: Vec<f64> = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];

        let mut result = empty_pca_result();
        let rc = unsafe { insight_pca(data.as_ptr(), 4, 2, 1, 0, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_components, 1);
        assert_eq!(result.n_features, 2);
        assert_eq!(result.n_samples, 4);

        let evr = unsafe {
            slice::from_raw_parts(result.explained_variance, result.n_components as usize)
        };
        assert!(
            (evr[0] - 1.0).abs() < 1e-10,
            "PC1 should explain all variance"
        );

        unsafe { free_pca_result(&result) };
    }

    #[test]
    fn ffi_pca_loadings_scores_shapes() {
        // 5 samples, 3 features — synthetic correlated data
        let data: Vec<f64> = vec![
            1.0, 1.0, 0.0, //
            2.0, 2.0, 0.0, //
            3.0, 3.0, 0.0, //
            4.0, 4.0, 0.0, //
            5.0, 5.0, 0.0, //
        ];

        let mut result = empty_pca_result();
        let rc = unsafe { insight_pca(data.as_ptr(), 5, 3, 2, 0, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_components, 2);
        assert_eq!(result.n_features, 3);
        assert_eq!(result.n_samples, 5);

        // Cumulative variance is monotonic non-decreasing and sums consistently
        let evr = unsafe {
            slice::from_raw_parts(result.explained_variance, result.n_components as usize)
        };
        let cum = unsafe {
            slice::from_raw_parts(result.cumulative_variance, result.n_components as usize)
        };
        assert!((cum[0] - evr[0]).abs() < 1e-10);
        assert!(cum[1] >= cum[0] - 1e-10);
        assert!((cum[1] - (evr[0] + evr[1])).abs() < 1e-10);

        // Loadings: shape n_components × n_features, each row is unit-norm
        let loadings = unsafe {
            slice::from_raw_parts(
                result.loadings,
                (result.n_components * result.n_features) as usize,
            )
        };
        for k in 0..result.n_components as usize {
            let row =
                &loadings[k * result.n_features as usize..(k + 1) * result.n_features as usize];
            let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "loading row {k} should be unit-norm, got {norm}"
            );
        }

        // Scores: shape n_samples × n_components
        let scores = unsafe {
            slice::from_raw_parts(
                result.scores,
                (result.n_samples * result.n_components) as usize,
            )
        };
        assert_eq!(scores.len(), 5 * 2);
        // Scores along PC1 should be monotonic for the correlated input above
        let pc1: Vec<f64> = (0..5).map(|i| scores[i * 2]).collect();
        assert!(
            pc1.windows(2).all(|w| w[1] >= w[0] - 1e-10)
                || pc1.windows(2).all(|w| w[1] <= w[0] + 1e-10)
        );

        unsafe { free_pca_result(&result) };
    }

    #[test]
    fn ffi_kmeans_null_ptr() {
        let mut result = CKMeansResult {
            k: 0,
            wcss: 0.0,
            iterations: 0,
            labels: ptr::null_mut(),
            n_labels: 0,
        };
        let rc = unsafe { insight_kmeans(ptr::null(), 4, 2, 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── Silhouette FFI tests ─────────────────────────────────────

    #[test]
    fn ffi_silhouette_well_separated() {
        // Two well-separated 2D clusters
        let data: Vec<f64> = vec![
            0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, // cluster A
            10.0, 10.0, 10.5, 10.5, 10.0, 10.5, 10.5, 10.0, // cluster B
        ];
        let labels: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let mut result = CSilhouetteResult {
            avg: 0.0,
            per_sample: ptr::null_mut(),
            n_samples: 0,
        };
        let rc =
            unsafe { insight_silhouette(data.as_ptr(), 8, 2, labels.as_ptr(), 2, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_samples, 8);
        assert!(
            result.avg > 0.9,
            "well-separated clusters should have high silhouette: {}",
            result.avg
        );

        let per_sample =
            unsafe { slice::from_raw_parts(result.per_sample, result.n_samples as usize) };
        for &s in per_sample {
            assert!(
                (-1.0..=1.0).contains(&s),
                "per-sample silhouette out of range: {s}"
            );
        }

        unsafe { insight_free_f64_array(result.per_sample, result.n_samples) };
    }

    #[test]
    fn ffi_silhouette_label_out_of_range() {
        let data: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0];
        let labels: Vec<u32> = vec![0, 5]; // 5 >= k=2

        let mut result = CSilhouetteResult {
            avg: 0.0,
            per_sample: ptr::null_mut(),
            n_samples: 0,
        };
        let rc =
            unsafe { insight_silhouette(data.as_ptr(), 2, 2, labels.as_ptr(), 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_INPUT);
        let body = last_error_body().expect("a body");
        assert_eq!(body["code"], "parameter_out_of_range");
        assert_eq!(body["parameter"], "labels");
        assert_eq!(
            (&body["index"], &body["max"], &body["got"]),
            (
                &serde_json::json!(1),
                &serde_json::json!(1),
                &serde_json::json!(5)
            )
        );
    }

    #[test]
    fn ffi_silhouette_null_ptr() {
        let mut result = CSilhouetteResult {
            avg: 0.0,
            per_sample: ptr::null_mut(),
            n_samples: 0,
        };
        let rc = unsafe { insight_silhouette(ptr::null(), 4, 2, ptr::null(), 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── DBSCAN FFI tests ─────────────────────────────────────────

    #[test]
    fn ffi_dbscan_basic() {
        // 2 clusters + 1 noise point
        let data: Vec<f64> = vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, // cluster A
            10.0, 10.0, 10.5, 10.0, 10.0, 10.5, // cluster B
            50.0, 50.0, // noise
        ];

        let mut result = CDbscanResult {
            n_clusters: 0,
            noise_count: 0,
            labels: ptr::null_mut(),
            n_labels: 0,
        };

        let rc = unsafe { insight_dbscan(data.as_ptr(), 7, 2, 1.5, 2, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.noise_count, 1);
        assert_eq!(result.n_labels, 7);

        let labels = unsafe { slice::from_raw_parts(result.labels, result.n_labels as usize) };
        // Noise point (last) should be -1
        assert_eq!(labels[6], -1);
        // First 3 in same cluster
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        // Next 3 in same cluster
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        // Different clusters
        assert_ne!(labels[0], labels[3]);

        unsafe { insight_free_i32_array(result.labels, result.n_labels) };
    }

    #[test]
    fn ffi_dbscan_null_ptr() {
        let mut result = CDbscanResult {
            n_clusters: 0,
            noise_count: 0,
            labels: ptr::null_mut(),
            n_labels: 0,
        };
        let rc = unsafe { insight_dbscan(ptr::null(), 4, 2, 1.0, 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── Distribution FFI tests ───────────────────────────────────

    #[test]
    fn ffi_distribution_basic() {
        // Roughly normal data
        let data: Vec<f64> = vec![
            -2.5, -2.0, -1.8, -1.5, -1.2, -1.0, -0.8, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.8, 1.0,
            1.2, 1.5, 1.8, 2.0, 2.5,
        ];

        let mut result = CDistributionResult {
            n: 0,
            ks_statistic: 0.0,
            ks_p_value: 0.0,
            jb_statistic: 0.0,
            jb_p_value: 0.0,
            sw_statistic: 0.0,
            sw_p_value: 0.0,
            ad_statistic: 0.0,
            ad_p_value: 0.0,
            is_normal: 0,
        };

        let rc = unsafe { insight_distribution(data.as_ptr(), 20, 0.05, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n, 20);
        assert!(result.ks_statistic > 0.0);
        assert!(result.ks_p_value > 0.0);
        assert!(!result.jb_statistic.is_nan());
        assert!(!result.sw_statistic.is_nan());
        assert!(result.sw_p_value > 0.0);
        assert!(!result.ad_statistic.is_nan());
        assert!(result.ad_p_value > 0.0);
        assert_eq!(result.is_normal, 1); // should be normal
    }

    #[test]
    fn ffi_distribution_null_ptr() {
        let mut result = CDistributionResult {
            n: 0,
            ks_statistic: 0.0,
            ks_p_value: 0.0,
            jb_statistic: 0.0,
            jb_p_value: 0.0,
            sw_statistic: 0.0,
            sw_p_value: 0.0,
            ad_statistic: 0.0,
            ad_p_value: 0.0,
            is_normal: 0,
        };
        let rc = unsafe { insight_distribution(ptr::null(), 10, 0.05, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── Feature Importance FFI tests ─────────────────────────────

    #[test]
    fn ffi_feature_importance_basic() {
        // 3 features, 10 rows (row-major)
        let data: Vec<f64> = vec![
            1.0, 2.0, 5.0, 2.0, 4.0, 4.0, 3.0, 6.0, 3.0, 4.0, 8.0, 2.0, 5.0, 10.0, 1.0, 6.0, 12.0,
            6.0, 7.0, 14.0, 5.0, 8.0, 16.0, 4.0, 9.0, 18.0, 3.0, 10.0, 20.0, 2.0,
        ];

        let mut result = CFeatureImportanceResult {
            scores: ptr::null_mut(),
            n_scores: 0,
            condition_number: 0.0,
            n_low_variance: 0,
            n_high_corr_pairs: 0,
        };

        let rc = unsafe { insight_feature_importance(data.as_ptr(), 10, 3, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_scores, 3);

        let scores = unsafe { slice::from_raw_parts(result.scores, result.n_scores as usize) };
        // All scores should be in [0, 1]
        for &s in scores {
            assert!((0.0..=1.0).contains(&s), "score {s} out of range");
        }

        // f0 and f1 are perfectly correlated → should have high_corr_pairs
        assert!(result.n_high_corr_pairs > 0);

        unsafe { insight_free_f64_array(result.scores, result.n_scores) };
    }

    #[test]
    fn ffi_feature_importance_null_ptr() {
        let mut result = CFeatureImportanceResult {
            scores: ptr::null_mut(),
            n_scores: 0,
            condition_number: 0.0,
            n_low_variance: 0,
            n_high_corr_pairs: 0,
        };
        let rc = unsafe { insight_feature_importance(ptr::null(), 10, 3, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── Isolation Forest FFI tests ──

    #[test]
    fn ffi_isolation_forest_roundtrip() {
        // Dense cluster + outlier (row-major: 2 cols)
        let mut data: Vec<f64> = Vec::new();
        for i in 0..20 {
            data.push(i as f64 * 0.1);
            data.push(0.0);
        }
        data.push(100.0);
        data.push(100.0);

        let mut result = CAnomalyResult {
            scores: ptr::null_mut(),
            anomalies: ptr::null_mut(),
            n: 0,
            anomaly_count: 0,
            threshold: 0.0,
        };

        let rc =
            unsafe { insight_isolation_forest(data.as_ptr(), 21, 2, 50, 0.1, 42, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n, 21);

        let scores = unsafe { slice::from_raw_parts(result.scores, 21) };
        // Outlier (last point) should have highest score
        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 20);

        unsafe {
            insight_free_f64_array(result.scores, result.n);
            insight_free_i32_array(result.anomalies, result.n);
        }
    }

    #[test]
    fn ffi_isolation_forest_null() {
        let mut result = CAnomalyResult {
            scores: ptr::null_mut(),
            anomalies: ptr::null_mut(),
            n: 0,
            anomaly_count: 0,
            threshold: 0.0,
        };
        let rc = unsafe { insight_isolation_forest(ptr::null(), 10, 2, 50, 0.1, 42, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── LOF FFI tests ──

    #[test]
    fn ffi_lof_roundtrip() {
        // Dense cluster + outlier (row-major: 2 cols)
        let mut data: Vec<f64> = Vec::new();
        for i in 0..20 {
            data.push(i as f64 * 0.1);
            data.push(0.0);
        }
        data.push(100.0);
        data.push(100.0);

        let mut result = CAnomalyResult {
            scores: ptr::null_mut(),
            anomalies: ptr::null_mut(),
            n: 0,
            anomaly_count: 0,
            threshold: 0.0,
        };

        let rc = unsafe { insight_lof(data.as_ptr(), 21, 2, 5, 1.5, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n, 21);

        let scores = unsafe { slice::from_raw_parts(result.scores, 21) };
        // Outlier should have highest LOF
        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 20);
        assert!(scores[20] > 1.5, "outlier LOF = {}", scores[20]);

        unsafe {
            insight_free_f64_array(result.scores, result.n);
            insight_free_i32_array(result.anomalies, result.n);
        }
    }

    #[test]
    fn ffi_lof_null() {
        let mut result = CAnomalyResult {
            scores: ptr::null_mut(),
            anomalies: ptr::null_mut(),
            n: 0,
            anomaly_count: 0,
            threshold: 0.0,
        };
        let rc = unsafe { insight_lof(ptr::null(), 10, 2, 5, 1.5, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── Hierarchical FFI tests ──────────────────────────────────────

    #[test]
    fn ffi_hierarchical_basic() {
        // 6 points in 2D, 2 obvious clusters
        let data: Vec<f64> = vec![
            0.0, 0.0, 0.1, 0.1, 0.05, 0.05, // cluster 1
            10.0, 10.0, 10.1, 10.1, 10.05, 10.05, // cluster 2
        ];
        let mut result = CHierarchicalResult {
            n_clusters: 0,
            labels: ptr::null_mut(),
            n_labels: 0,
            n_merges: 0,
            merge_distances: ptr::null_mut(),
            merge_sizes: ptr::null_mut(),
        };

        let rc = unsafe { insight_hierarchical(data.as_ptr(), 6, 2, 3, 2, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.n_labels, 6);
        assert_eq!(result.n_merges, 5); // n-1 merges

        // Verify labels
        let labels = unsafe { slice::from_raw_parts(result.labels, result.n_labels as usize) };
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);

        // Clean up
        unsafe {
            insight_free_i32_array(result.labels, result.n_labels);
            insight_free_f64_array(result.merge_distances, result.n_merges);
            insight_free_i32_array(result.merge_sizes, result.n_merges);
        }
    }

    #[test]
    fn ffi_hierarchical_null_ptr() {
        let mut result = CHierarchicalResult {
            n_clusters: 0,
            labels: ptr::null_mut(),
            n_labels: 0,
            n_merges: 0,
            merge_distances: ptr::null_mut(),
            merge_sizes: ptr::null_mut(),
        };
        let rc = unsafe { insight_hierarchical(ptr::null(), 6, 2, 3, 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    /// A linkage past 3 used to run Ward; it is refused and named.
    #[test]
    fn ffi_hierarchical_refuses_an_unknown_linkage() {
        let data: Vec<f64> = vec![0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1];
        let mut result = CHierarchicalResult {
            n_clusters: 0,
            labels: ptr::null_mut(),
            n_labels: 0,
            n_merges: 0,
            merge_distances: ptr::null_mut(),
            merge_sizes: ptr::null_mut(),
        };
        let rc = unsafe { insight_hierarchical(data.as_ptr(), 4, 2, 4, 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
        assert_eq!(last_error_parameter().as_deref(), Some("linkage"));
        assert!(result.labels.is_null(), "nothing is allocated on refusal");
    }

    // ── HDBSCAN FFI tests ───────────────────────────────────────────

    #[test]
    fn ffi_hdbscan_basic() {
        let data: Vec<f64> = vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            10.1, 10.1, 10.05, 10.05, 50.0, 50.0, // noise
        ];
        let mut result = CHdbscanResult {
            n_clusters: 0,
            noise_count: 0,
            labels: ptr::null_mut(),
            probabilities: ptr::null_mut(),
            n_labels: 0,
        };

        let rc = unsafe { insight_hdbscan(data.as_ptr(), 11, 2, 3, 0, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert!(result.n_clusters >= 1);
        assert_eq!(result.n_labels, 11);

        // Verify labels and probs are allocated
        assert!(!result.labels.is_null());
        assert!(!result.probabilities.is_null());

        let probs =
            unsafe { slice::from_raw_parts(result.probabilities, result.n_labels as usize) };
        for &p in probs {
            assert!((0.0..=1.0).contains(&p) || p == 0.0);
        }

        // Clean up
        unsafe {
            insight_free_i32_array(result.labels, result.n_labels);
            insight_free_f64_array(result.probabilities, result.n_labels);
        }
    }

    #[test]
    fn ffi_hdbscan_null_ptr() {
        let mut result = CHdbscanResult {
            n_clusters: 0,
            noise_count: 0,
            labels: ptr::null_mut(),
            probabilities: ptr::null_mut(),
            n_labels: 0,
        };
        let rc = unsafe { insight_hdbscan(ptr::null(), 10, 2, 3, 0, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── Correlation FFI tests ─────────────────────────────────────

    #[test]
    fn ffi_correlation_basic() {
        // 3 columns, 5 rows (row-major): c0 and c1 correlated, c2 independent
        let data: Vec<f64> = vec![
            1.0, 2.0, 5.0, 2.0, 4.0, 3.0, 3.0, 6.0, 7.0, 4.0, 8.0, 1.0, 5.0, 10.0, 4.0,
        ];

        let mut result = CCorrelationResult {
            n_vars: 0,
            matrix: ptr::null_mut(),
            n_high_pairs: 0,
        };

        let rc =
            unsafe { insight_correlation(data.as_ptr(), 5, 3, INSIGHT_CORR_PEARSON, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_vars, 3);

        let mat = unsafe { slice::from_raw_parts(result.matrix, 9) };
        // Diagonal should be 1.0
        assert!((mat[0] - 1.0).abs() < 1e-10, "r(0,0) = {}", mat[0]);
        assert!((mat[4] - 1.0).abs() < 1e-10, "r(1,1) = {}", mat[4]);
        assert!((mat[8] - 1.0).abs() < 1e-10, "r(2,2) = {}", mat[8]);
        // c0 and c1 are perfectly correlated → r ≈ 1.0
        assert!((mat[1] - 1.0).abs() < 1e-10, "r(0,1) = {}", mat[1]);
        // At least 1 high pair (c0,c1)
        assert!(result.n_high_pairs >= 1);

        unsafe { insight_free_f64_array(result.matrix, 9) };
    }

    #[test]
    fn ffi_correlation_null_ptr() {
        let mut result = CCorrelationResult {
            n_vars: 0,
            matrix: ptr::null_mut(),
            n_high_pairs: 0,
        };
        let rc =
            unsafe { insight_correlation(ptr::null(), 5, 3, INSIGHT_CORR_PEARSON, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    /// Two rows give no p-value: refused as too few rows, with the counts,
    /// before anything is computed -- for every method.
    #[test]
    fn ffi_correlation_too_few() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 5.0]; // 2 rows, 2 cols
        for method in [
            INSIGHT_CORR_PEARSON,
            INSIGHT_CORR_SPEARMAN,
            INSIGHT_CORR_KENDALL,
        ] {
            let mut result = CCorrelationResult {
                n_vars: 0,
                matrix: ptr::null_mut(),
                n_high_pairs: 0,
            };
            let rc = unsafe { insight_correlation(data.as_ptr(), 2, 2, method, &mut result) };
            assert_eq!(rc, INSIGHT_ERR_INSUFFICIENT_DATA);
            let body = last_error_body().expect("a body");
            assert_eq!(
                body,
                serde_json::json!({
                    "error": body["error"],
                    "code": "insufficient_data",
                    "parameter": "data",
                    "min": 3,
                    "got": 2,
                })
            );
        }
    }

    /// A constant column is named by its position.
    #[test]
    fn ffi_correlation_names_a_constant_column() {
        // 4 rows x 2 cols; the second column is constant.
        let data: Vec<f64> = vec![1.0, 7.0, 2.0, 7.0, 3.0, 7.0, 4.0, 7.0];
        let mut result = CCorrelationResult {
            n_vars: 0,
            matrix: ptr::null_mut(),
            n_high_pairs: 0,
        };
        let rc =
            unsafe { insight_correlation(data.as_ptr(), 4, 2, INSIGHT_CORR_PEARSON, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_DEGENERATE_DATA);
        let body = last_error_body().expect("a body");
        assert_eq!(body["code"], "degenerate_data");
        assert_eq!(body["parameter"], "data.1");
    }

    #[test]
    fn ffi_correlation_invalid_method() {
        let data: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let mut result = CCorrelationResult {
            n_vars: 0,
            matrix: ptr::null_mut(),
            n_high_pairs: 0,
        };
        let rc = unsafe { insight_correlation(data.as_ptr(), 5, 3, 99, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
        let body = last_error_body().expect("a body");
        assert_eq!(body["code"], "unknown_option");
        assert_eq!(body["parameter"], "method");
        assert_eq!(body["expected"], serde_json::json!([0, 1, 2]));
    }

    #[test]
    fn ffi_correlation_kendall() {
        // 3 columns × 5 rows: c0 perfectly monotonic with c1, anti-monotonic with c2.
        let data: Vec<f64> = vec![
            1.0, 2.0, 5.0, // row 0
            2.0, 4.0, 4.0, // row 1
            3.0, 6.0, 3.0, // row 2
            4.0, 8.0, 2.0, // row 3
            5.0, 10.0, 1.0, // row 4
        ];
        let mut result = CCorrelationResult {
            n_vars: 0,
            matrix: ptr::null_mut(),
            n_high_pairs: 0,
        };
        let rc =
            unsafe { insight_correlation(data.as_ptr(), 5, 3, INSIGHT_CORR_KENDALL, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_vars, 3);

        let mat = unsafe { slice::from_raw_parts(result.matrix, 9) };
        // c0 and c1 perfectly concordant → tau ≈ 1.0
        assert!((mat[1] - 1.0).abs() < 1e-10, "r(0,1) = {}", mat[1]);
        // c0 and c2 perfectly discordant → tau ≈ -1.0
        assert!((mat[2] + 1.0).abs() < 1e-10, "r(0,2) = {}", mat[2]);

        unsafe { insight_free_f64_array(result.matrix, 9) };
    }

    // ── Regression FFI tests ──────────────────────────────────────

    #[test]
    fn ffi_regression_basic() {
        // y = 2x + 1, perfect linear
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let mut result = CRegressionResult {
            intercept: 0.0,
            slope: 0.0,
            r_squared: 0.0,
            adj_r_squared: 0.0,
            f_p_value: 0.0,
        };

        let rc = unsafe { insight_regression(x.as_ptr(), y.as_ptr(), 5, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert!(
            (result.intercept - 1.0).abs() < 1e-6,
            "intercept = {}",
            result.intercept
        );
        assert!(
            (result.slope - 2.0).abs() < 1e-6,
            "slope = {}",
            result.slope
        );
        assert!(
            (result.r_squared - 1.0).abs() < 1e-6,
            "R² = {}",
            result.r_squared
        );
    }

    #[test]
    fn ffi_regression_null_ptr() {
        let mut result = CRegressionResult {
            intercept: 0.0,
            slope: 0.0,
            r_squared: 0.0,
            adj_r_squared: 0.0,
            f_p_value: 0.0,
        };
        let rc = unsafe { insight_regression(ptr::null(), ptr::null(), 5, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_regression_too_few() {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![3.0, 5.0];
        let mut result = CRegressionResult {
            intercept: 0.0,
            slope: 0.0,
            r_squared: 0.0,
            adj_r_squared: 0.0,
            f_p_value: 0.0,
        };
        let rc = unsafe { insight_regression(x.as_ptr(), y.as_ptr(), 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INSUFFICIENT_DATA);
        let body = last_error_body().expect("a body");
        assert_eq!(
            (&body["min"], &body["got"]),
            (&serde_json::json!(3), &serde_json::json!(2))
        );
    }

    // ── Mahalanobis FFI tests ─────────────────────────────────────

    #[test]
    fn ffi_mahalanobis_basic() {
        // 8 inliers + 1 outlier, 2D
        let mut data: Vec<f64> = Vec::new();
        for i in 0..8 {
            let x = (i as f64) * 0.5 + (i as f64 * 1.3).sin() * 0.3;
            let y = (i as f64) * 0.4 + (i as f64 * 0.7).cos() * 0.2;
            data.push(x);
            data.push(y);
        }
        data.push(100.0);
        data.push(100.0);

        let mut result = CMahalanobisResult {
            distances: ptr::null_mut(),
            anomalies: ptr::null_mut(),
            n: 0,
            threshold: 0.0,
            outlier_count: 0,
        };

        let rc = unsafe { insight_mahalanobis(data.as_ptr(), 9, 2, 0.975, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n, 9);
        assert!(result.threshold > 0.0);

        // Outlier should have largest distance
        let dists = unsafe { slice::from_raw_parts(result.distances, 9) };
        let max_idx = dists
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 8);

        unsafe {
            insight_free_f64_array(result.distances, result.n);
            insight_free_i32_array(result.anomalies, result.n);
        }
    }

    #[test]
    fn ffi_mahalanobis_null() {
        let mut result = CMahalanobisResult {
            distances: ptr::null_mut(),
            anomalies: ptr::null_mut(),
            n: 0,
            threshold: 0.0,
            outlier_count: 0,
        };
        let rc = unsafe { insight_mahalanobis(ptr::null(), 5, 2, 0.975, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── Cramér's V FFI tests ──────────────────────────────────────

    #[test]
    fn ffi_cramers_v_basic() {
        let table: Vec<f64> = vec![50.0, 0.0, 0.0, 50.0]; // 2x2 perfect
        let mut result = CCramersVResult {
            v: 0.0,
            chi_squared: 0.0,
            p_value: 0.0,
        };

        let rc = unsafe { insight_cramers_v(table.as_ptr(), 2, 2, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert!(result.v > 0.9, "V = {}", result.v);
    }

    /// Each reason a table cannot be read reaches C with its code and the
    /// argument it is about -- not one fieldless "computation failed".
    #[test]
    fn ffi_cramers_v_refusals_name_their_reason() {
        let mut result = CCramersVResult {
            v: 0.0,
            chi_squared: 0.0,
            p_value: 0.0,
        };
        let table = [10.0_f64, 20.0];
        let rc = unsafe { insight_cramers_v(table.as_ptr(), 1, 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
        let body = last_error_body().expect("a body");
        assert_eq!(
            (
                &body["code"],
                &body["parameter"],
                &body["min"],
                &body["got"]
            ),
            (
                &serde_json::json!("parameter_out_of_range"),
                &serde_json::json!("n_rows"),
                &serde_json::json!(2.0),
                &serde_json::json!(1.0)
            )
        );

        let table = [1.0_f64, 2.0, 0.0, 0.0];
        let rc = unsafe { insight_cramers_v(table.as_ptr(), 2, 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_DEGENERATE_DATA);
        let body = last_error_body().expect("a body");
        assert_eq!(
            (&body["code"], &body["parameter"]),
            (
                &serde_json::json!("degenerate_data"),
                &serde_json::json!("table")
            )
        );

        let table = [1.0_f64, f64::INFINITY, 1.0, 1.0];
        let rc = unsafe { insight_cramers_v(table.as_ptr(), 2, 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_INPUT);
        let body = last_error_body().expect("a body");
        assert_eq!(
            (&body["code"], &body["parameter"], &body["index"]),
            (
                &serde_json::json!("value_not_finite"),
                &serde_json::json!("table"),
                &serde_json::json!(1)
            )
        );

        let table = [1.0_f64, -1.0, 1.0, 1.0];
        let rc = unsafe { insight_cramers_v(table.as_ptr(), 2, 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
        let body = last_error_body().expect("a body");
        assert_eq!(
            (&body["code"], &body["parameter"], &body["min"]),
            (
                &serde_json::json!("parameter_out_of_range"),
                &serde_json::json!("table[1]"),
                &serde_json::json!(0.0)
            )
        );
    }

    /// The options the C ABI passes to the analysis are refused there with
    /// their interval: a LOF threshold `> 0`, a significance level in (0, 1).
    #[test]
    fn ffi_options_out_of_range_carry_their_interval() {
        let points: Vec<f64> = (0..20).flat_map(|i| [i as f64, (i % 7) as f64]).collect();
        let mut anomalies = CAnomalyResult {
            scores: ptr::null_mut(),
            anomalies: ptr::null_mut(),
            n: 0,
            anomaly_count: 0,
            threshold: 0.0,
        };
        let rc = unsafe { insight_lof(points.as_ptr(), 20, 2, 5, 0.0, &mut anomalies) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
        assert_eq!(
            last_error_body().expect("a body"),
            serde_json::json!({
                "error": "invalid parameter 'threshold': must be > 0, got 0",
                "code": "parameter_out_of_range",
                "parameter": "threshold",
                "exclusive_min": 0.0,
                "got": 0.0,
            })
        );

        let series: Vec<f64> = (0..30).map(|i| ((i * 37) % 11) as f64).collect();
        for bad in [0.0, 1.0] {
            let mut dist = CDistributionResult {
                n: 0,
                ks_statistic: 0.0,
                ks_p_value: 0.0,
                jb_statistic: 0.0,
                jb_p_value: 0.0,
                sw_statistic: 0.0,
                sw_p_value: 0.0,
                ad_statistic: 0.0,
                ad_p_value: 0.0,
                is_normal: 0,
            };
            let rc = unsafe { insight_distribution(series.as_ptr(), 30, bad, &mut dist) };
            assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
            let body = last_error_body().expect("a body");
            assert_eq!(
                (
                    &body["code"],
                    &body["parameter"],
                    &body["exclusive_min"],
                    &body["exclusive_max"]
                ),
                (
                    &serde_json::json!("parameter_out_of_range"),
                    &serde_json::json!("significance_level"),
                    &serde_json::json!(0.0),
                    &serde_json::json!(1.0)
                )
            );
        }

        let target: Vec<u32> = (0..20).map(|i| (i % 2) as u32).collect();
        let mut anova = CAnovaSelectionResult {
            features: ptr::null_mut(),
            n_features: 0,
            n_selected: 0,
        };
        let rc = unsafe {
            insight_anova_select(points.as_ptr(), 20, 2, target.as_ptr(), 1.5, &mut anova)
        };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
        let body = last_error_body().expect("a body");
        assert_eq!(
            (&body["parameter"], &body["got"]),
            (
                &serde_json::json!("significance_level"),
                &serde_json::json!(1.5)
            )
        );
    }

    /// A column index past the last column is a range refusal with the
    /// columns there are.
    #[test]
    fn ffi_profile_column_index_out_of_range() {
        let csv = CString::new("a,b\n1,2\n3,4\n").unwrap();
        let ctx = unsafe { insight_profile_csv(csv.as_ptr()) };
        assert!(!ctx.is_null());
        let mut summary = CColumnSummary {
            index: 0,
            valid_count: 0,
            null_count: 0,
            data_type: 0,
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
        };
        let rc = unsafe { insight_profile_column(ctx, 2, &mut summary) };
        unsafe { insight_profile_free(ctx) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
        let body = last_error_body().expect("a body");
        assert_eq!(
            (
                &body["code"],
                &body["parameter"],
                &body["min"],
                &body["max"],
                &body["got"]
            ),
            (
                &serde_json::json!("parameter_out_of_range"),
                &serde_json::json!("col_idx"),
                &serde_json::json!(0.0),
                &serde_json::json!(1.0),
                &serde_json::json!(2.0)
            )
        );
    }

    #[test]
    fn ffi_cramers_v_null() {
        let mut result = CCramersVResult {
            v: 0.0,
            chi_squared: 0.0,
            p_value: 0.0,
        };
        let rc = unsafe { insight_cramers_v(ptr::null(), 2, 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── ANOVA Selection FFI tests ─────────────────────────────────

    #[test]
    fn ffi_anova_select_basic() {
        // 6 points, 2 features (row-major), 2 classes
        let data: Vec<f64> = vec![1.0, 3.0, 1.1, 3.1, 1.2, 2.9, 5.0, 3.0, 5.1, 3.1, 5.2, 2.9];
        let target: Vec<u32> = vec![0, 0, 0, 1, 1, 1];

        let mut result = CAnovaSelectionResult {
            features: ptr::null_mut(),
            n_features: 0,
            n_selected: 0,
        };

        let rc = unsafe {
            insight_anova_select(data.as_ptr(), 6, 2, target.as_ptr(), 0.05, &mut result)
        };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_features, 2);
        assert!(result.n_selected >= 1); // at least f0 should be significant

        unsafe { insight_free_anova_features(result.features, result.n_features) };
    }

    #[test]
    fn ffi_anova_select_null() {
        let mut result = CAnovaSelectionResult {
            features: ptr::null_mut(),
            n_features: 0,
            n_selected: 0,
        };
        let rc = unsafe { insight_anova_select(ptr::null(), 6, 2, ptr::null(), 0.05, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── Mutual Information FFI tests ────────────────────────────────

    #[test]
    fn ffi_mutual_info_basic() {
        // 6 points, 2 features, 2 classes; feature 0 separates classes, feature 1 does not
        let data: Vec<f64> = vec![1.0, 5.0, 1.1, 5.1, 1.2, 4.9, 5.0, 5.0, 5.1, 5.1, 5.2, 4.9];
        let target: Vec<u32> = vec![0, 0, 0, 1, 1, 1];

        let mut result = CMutualInfoResult {
            features: ptr::null_mut(),
            n_features: 0,
        };

        let rc =
            unsafe { insight_mutual_info(data.as_ptr(), 6, 2, target.as_ptr(), 0, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_features, 2);

        unsafe { insight_free_mi_features(result.features, result.n_features) };
    }

    #[test]
    fn ffi_mutual_info_null() {
        let mut result = CMutualInfoResult {
            features: ptr::null_mut(),
            n_features: 0,
        };
        let rc = unsafe { insight_mutual_info(ptr::null(), 6, 2, ptr::null(), 0, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── Mini-Batch K-Means FFI tests ────────────────────────────────

    #[test]
    fn ffi_mini_batch_kmeans_basic() {
        // Two obvious clusters
        let data: Vec<f64> = vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 5.0, 5.0, 5.1, 5.1, 5.2, 5.0];
        let mut result = CKMeansResult {
            k: 0,
            wcss: 0.0,
            iterations: 0,
            labels: ptr::null_mut(),
            n_labels: 0,
        };

        let rc =
            unsafe { insight_mini_batch_kmeans(data.as_ptr(), 6, 2, 2, 3, 50, 42, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.k, 2);
        assert_eq!(result.n_labels, 6);

        unsafe { insight_free_labels(result.labels, result.n_labels) };
    }

    #[test]
    fn ffi_mini_batch_kmeans_null() {
        let mut result = CKMeansResult {
            k: 0,
            wcss: 0.0,
            iterations: 0,
            labels: ptr::null_mut(),
            n_labels: 0,
        };
        let rc = unsafe { insight_mini_batch_kmeans(ptr::null(), 6, 2, 2, 3, 50, 42, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── Gap Statistic FFI tests ─────────────────────────────────────

    #[test]
    fn ffi_gap_statistic_basic() {
        // Two clusters
        let data: Vec<f64> = vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2,
        ];
        let mut result = CGapStatResult {
            best_k: 0,
            n_values: 0,
            gap_values: ptr::null_mut(),
            std_errors: ptr::null_mut(),
        };

        let rc = unsafe { insight_gap_statistic(data.as_ptr(), 8, 2, 1, 4, 3, 42, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert!(result.best_k >= 1 && result.best_k <= 4);
        assert!(result.n_values > 0);

        unsafe {
            insight_free_f64_array(result.gap_values, result.n_values * 2);
            insight_free_f64_array(result.std_errors, result.n_values * 2);
        }
    }

    #[test]
    fn ffi_gap_statistic_null() {
        let mut result = CGapStatResult {
            best_k: 0,
            n_values: 0,
            gap_values: ptr::null_mut(),
            std_errors: ptr::null_mut(),
        };
        let rc = unsafe { insight_gap_statistic(ptr::null(), 8, 2, 1, 4, 3, 42, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    // ── Permutation Importance FFI tests ────────────────────────────

    #[test]
    fn ffi_permutation_importance_basic() {
        // y = 2*x0 + noise; x1 = noise
        let data: Vec<f64> = vec![1.0, 0.5, 2.0, 0.3, 3.0, 0.8, 4.0, 0.1, 5.0, 0.9];
        let target: Vec<f64> = vec![2.1, 4.0, 6.2, 7.9, 10.1];

        let mut result = CPermImportanceResult {
            baseline_score: 0.0,
            features: ptr::null_mut(),
            n_features: 0,
        };

        let rc = unsafe {
            insight_permutation_importance(data.as_ptr(), 5, 2, target.as_ptr(), 5, 42, &mut result)
        };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_features, 2);
        assert!(result.baseline_score > 0.5); // should have decent R²

        unsafe { insight_free_perm_features(result.features, result.n_features) };
    }

    #[test]
    fn ffi_permutation_importance_null() {
        let mut result = CPermImportanceResult {
            baseline_score: 0.0,
            features: ptr::null_mut(),
            n_features: 0,
        };
        let rc = unsafe {
            insight_permutation_importance(ptr::null(), 5, 2, ptr::null(), 5, 42, &mut result)
        };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    fn last_error_parameter() -> Option<String> {
        let ptr = insight_last_error_parameter();
        (!ptr.is_null()).then(|| {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        })
    }

    /// An option the core refuses names its parameter across the C ABI, and
    /// the body carries the interval; the next error that is not about one
    /// clears it.
    #[test]
    fn ffi_last_error_parameter_follows_invalid_parameter() {
        let data: Vec<f64> = (0..40).map(|i| ((i * 37) % 11) as f64).collect();
        let mut out = CMahalanobisResult {
            distances: ptr::null_mut(),
            anomalies: ptr::null_mut(),
            n: 0,
            threshold: 0.0,
            outlier_count: 0,
        };
        let rc = unsafe { insight_mahalanobis(data.as_ptr(), 20, 2, 1.5, &mut out) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
        assert_eq!(last_error_parameter().as_deref(), Some("quantile"));
        let body = last_error_body().expect("a body");
        assert_eq!(
            body,
            serde_json::json!({
                "error": body["error"],
                "code": "parameter_out_of_range",
                "parameter": "quantile",
                "exclusive_min": 0.0,
                "exclusive_max": 1.0,
                "got": 1.5,
            })
        );

        let rc = unsafe { insight_mahalanobis(ptr::null(), 20, 2, 0.975, &mut out) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
        assert_eq!(last_error_parameter(), None);
        assert_eq!(
            last_error_body().expect("a body")["code"],
            "malformed_input"
        );

        let rc = unsafe { insight_mahalanobis(data.as_ptr(), 20, 2, 1.5, &mut out) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
        insight_clear_error();
        assert_eq!(last_error_parameter(), None);
    }

    /// The named codes are the ABI: a binding that mirrors them as an enum
    /// (C# `Linkage`) depends on these exact values.
    #[test]
    fn ffi_enum_codes_are_stable() {
        assert_eq!(
            [
                INSIGHT_LINKAGE_SINGLE,
                INSIGHT_LINKAGE_COMPLETE,
                INSIGHT_LINKAGE_AVERAGE,
                INSIGHT_LINKAGE_WARD
            ],
            [0, 1, 2, 3]
        );
    }

    #[test]
    fn ffi_an_empty_array_given_as_null_is_refused_as_empty_not_as_a_null_pointer() {
        // (NULL, 0) is how C passes an empty array and how .NET pins one.
        let mut dist: CDistributionResult = unsafe { std::mem::zeroed() };
        let rc = unsafe { insight_distribution(ptr::null(), 0, 0.05, &mut dist) };
        assert_ne!(rc, INSIGHT_ERR_NULL_PTR);
        assert_eq!(
            last_error_body().expect("a refusal body")["code"],
            "empty_input"
        );

        let mut reg: CRegressionResult = unsafe { std::mem::zeroed() };
        let rc = unsafe { insight_regression(ptr::null(), ptr::null(), 0, &mut reg) };
        assert_ne!(rc, INSIGHT_ERR_NULL_PTR);

        let mut km: CKMeansResult = unsafe { std::mem::zeroed() };
        let rc = unsafe { insight_kmeans(ptr::null(), 0, 3, 2, &mut km) };
        assert_ne!(rc, INSIGHT_ERR_NULL_PTR);

        // A null pointer to an array that is not empty is still a null pointer.
        let rc = unsafe { insight_distribution(ptr::null(), 4, 0.05, &mut dist) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_array_inputs_are_read_through_the_empty_aware_helpers() {
        // Every routine's input array goes through `absent` / `view`: a bare
        // `data.is_null()` refusal (or `slice::from_raw_parts(data, ..)`)
        // would refuse an empty array as a null pointer again.
        let source = include_str!("ffi.rs");
        let library = &source[..source.find("#[cfg(test)]").expect("a test module")];
        // A whole identifier: `csv_data.is_null()` and `ctx.is_null()` are not inputs of this kind,
        // and `!labels.is_null()` / `r.x.is_null()` are about outputs, not inputs.
        let whole = |needle: &str| {
            library
                .match_indices(needle)
                .filter(|&(at, _)| {
                    !library[..at]
                        .chars()
                        .next_back()
                        .is_some_and(|c| c.is_alphanumeric() || matches!(c, '_' | '!' | '.'))
                })
                .count()
        };
        for name in ["data", "labels", "target", "table", "x", "y"] {
            for bare in [
                format!("{name}.is_null()"),
                format!("slice::from_raw_parts({name},"),
            ] {
                assert_eq!(whole(&bare), 0, "{bare} outside the helpers");
            }
        }
    }
}
