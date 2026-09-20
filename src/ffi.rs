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
use crate::feature_importance::{feature_analysis, permutation_importance, FeatureConfig};
use crate::json_parser::JsonParser;
use crate::pca::{pca, PcaConfig};
use crate::profiling::profile_dataframe;

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
}

fn error_to_code(e: &crate::error::InsightError) -> i32 {
    use crate::error::InsightError;
    match e {
        InsightError::CsvParse { .. } | InsightError::JsonParse { .. } => INSIGHT_ERR_PARSE_FAILED,
        InsightError::MissingValues { .. }
        | InsightError::ColumnNotFound { .. }
        | InsightError::DimensionMismatch { .. } => INSIGHT_ERR_INVALID_INPUT,
        InsightError::InsufficientData { .. } => INSIGHT_ERR_INSUFFICIENT_DATA,
        InsightError::InvalidParameter { .. } => INSIGHT_ERR_INVALID_PARAM,
        InsightError::DegenerateData { .. } => INSIGHT_ERR_DEGENERATE_DATA,
        InsightError::ComputationFailed { .. } => INSIGHT_ERR_COMPUTATION_FAILED,
        InsightError::Io(_) => INSIGHT_ERR_ANALYSIS_FAILED,
    }
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg).ok();
    });
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

/// Clears the last error message.
#[no_mangle]
pub extern "C" fn insight_clear_error() {
    LAST_ERROR.with(|cell| {
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
    /// For numeric columns: mean. NaN for non-numeric.
    pub mean: f64,
    /// For numeric columns: standard deviation. NaN for non-numeric.
    pub std_dev: f64,
    /// For numeric columns: minimum. NaN for non-numeric.
    pub min: f64,
    /// For numeric columns: maximum. NaN for non-numeric.
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
            set_last_error("null csv_data pointer");
            return ptr::null_mut();
        }

        let c_str = unsafe { CStr::from_ptr(csv_data) };
        let csv = match c_str.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&format!("invalid UTF-8: {e}"));
                return ptr::null_mut();
            }
        };

        let df = match CsvParser::new().parse_str(csv) {
            Ok(df) => df,
            Err(e) => {
                set_last_error(&format!("CSV parse error: {e}"));
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
            set_last_error("panic in insight_profile_csv");
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
            set_last_error("null json_data pointer");
            return ptr::null_mut();
        }

        let c_str = unsafe { CStr::from_ptr(json_data) };
        let json = match c_str.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&format!("invalid UTF-8: {e}"));
                return ptr::null_mut();
            }
        };

        let df = match JsonParser::new().parse_str(json) {
            Ok(df) => df,
            Err(e) => {
                set_last_error(&format!("JSON parse error: {e}"));
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
            set_last_error("panic in insight_profile_json");
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
        set_last_error("null context");
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
        set_last_error("null context");
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
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }
        let ctx = unsafe { &*ctx };
        let idx = col_idx as usize;

        if idx >= ctx.column_profiles.len() {
            set_last_error("column index out of range");
            return INSIGHT_ERR_INVALID_INPUT;
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
            Some(np) => (np.mean, np.std_dev, np.min, np.max),
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
        Err(_) => {
            set_last_error("panic in insight_profile_column");
            INSIGHT_ERR_PANIC
        }
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
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        let raw = unsafe { slice::from_raw_parts(data, n * d) };

        // Convert to Vec<Vec<f64>>
        let points: Vec<Vec<f64>> = (0..n).map(|i| raw[i * d..(i + 1) * d].to_vec()).collect();

        let config = KMeansConfig::new(k as usize);
        let km_result = match kmeans(&points, &config) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(&e.to_string());
                return error_to_code(&e);
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
        Err(_) => {
            set_last_error("panic in insight_kmeans");
            INSIGHT_ERR_PANIC
        }
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
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        let raw = unsafe { slice::from_raw_parts(data, n * d) };

        let points: Vec<Vec<f64>> = (0..n).map(|i| raw[i * d..(i + 1) * d].to_vec()).collect();

        let config = PcaConfig::new(n_components as usize).auto_scale(auto_scale != 0);
        let pca_result = match pca(&points, &config) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(&e.to_string());
                return error_to_code(&e);
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
        Err(_) => {
            set_last_error("panic in insight_pca");
            INSIGHT_ERR_PANIC
        }
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
        if data.is_null() || labels.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        if n == 0 || d == 0 {
            set_last_error("empty input");
            return INSIGHT_ERR_INVALID_INPUT;
        }

        let raw = unsafe { slice::from_raw_parts(data, n * d) };
        let raw_labels = unsafe { slice::from_raw_parts(labels, n) };

        let points: Vec<Vec<f64>> = (0..n).map(|i| raw[i * d..(i + 1) * d].to_vec()).collect();
        let labels_usize: Vec<usize> = raw_labels.iter().map(|&l| l as usize).collect();

        // Validate label range
        let k_usize = k as usize;
        if let Some(&bad) = labels_usize.iter().find(|&&l| l >= k_usize) {
            set_last_error(&format!("label {bad} out of range for k={k_usize}"));
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
        Err(_) => {
            set_last_error("panic in insight_silhouette");
            INSIGHT_ERR_PANIC
        }
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
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        let raw = unsafe { slice::from_raw_parts(data, n * d) };

        let points: Vec<Vec<f64>> = (0..n).map(|i| raw[i * d..(i + 1) * d].to_vec()).collect();

        let config = DbscanConfig::new(epsilon, min_samples as usize);
        let db_result = match dbscan(&points, &config) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(&e.to_string());
                return error_to_code(&e);
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
        Err(_) => {
            set_last_error("panic in insight_dbscan");
            INSIGHT_ERR_PANIC
        }
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
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        let raw = unsafe { slice::from_raw_parts(data, len) };
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
                set_last_error(&e.to_string());
                return error_to_code(&e);
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
        Err(_) => {
            set_last_error("panic in insight_distribution");
            INSIGHT_ERR_PANIC
        }
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
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        let raw = unsafe { slice::from_raw_parts(data, n * d) };

        // Convert row-major to column-major vectors
        let columns: Vec<Vec<f64>> = (0..d)
            .map(|col| (0..n).map(|row| raw[row * d + col]).collect())
            .collect();

        let names: Vec<String> = (0..d).map(|i| format!("f{i}")).collect();
        let config = FeatureConfig::default();

        let fi_result = match feature_analysis(&columns, &names, &config) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(&e.to_string());
                return error_to_code(&e);
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
        Err(_) => {
            set_last_error("panic in insight_feature_importance");
            INSIGHT_ERR_PANIC
        }
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
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let rows = n_rows as usize;
        let cols = n_cols as usize;
        let raw = unsafe { slice::from_raw_parts(data, rows * cols) };

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
                set_last_error(&e.to_string());
                return error_to_code(&e);
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
        Err(_) => {
            set_last_error("panic in insight_isolation_forest");
            INSIGHT_ERR_PANIC
        }
    }
}

// ── LOF FFI ─────────────────────────────────────────────────────────

/// Runs Local Outlier Factor anomaly detection on row-major data.
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
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let rows = n_rows as usize;
        let cols = n_cols as usize;
        let raw = unsafe { slice::from_raw_parts(data, rows * cols) };

        let points: Vec<Vec<f64>> = (0..rows)
            .map(|i| raw[i * cols..(i + 1) * cols].to_vec())
            .collect();

        let config = crate::lof::LofConfig::default()
            .k(k as usize)
            .threshold(threshold);

        let lof_result = match crate::lof::lof(&points, &config) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(&e.to_string());
                return error_to_code(&e);
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
        Err(_) => {
            set_last_error("panic in insight_lof");
            INSIGHT_ERR_PANIC
        }
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
/// return `INSIGHT_ERR_INVALID_PARAM`.
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
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let nr = n_rows as usize;
        let nc = n_cols as usize;
        if nr < 2 || nc < 2 {
            set_last_error("need at least 2 rows and 2 columns");
            return INSIGHT_ERR_INVALID_INPUT;
        }

        let corr_method = match method {
            INSIGHT_CORR_PEARSON => crate::analysis::CorrelationMethod::Pearson,
            INSIGHT_CORR_SPEARMAN => crate::analysis::CorrelationMethod::Spearman,
            INSIGHT_CORR_KENDALL => crate::analysis::CorrelationMethod::Kendall,
            _ => {
                set_last_error("invalid correlation method (use 0=Pearson, 1=Spearman, 2=Kendall)");
                return INSIGHT_ERR_INVALID_PARAM;
            }
        };

        let raw = unsafe { slice::from_raw_parts(data, nr * nc) };

        // Convert row-major flat to column-major Vec<Vec<f64>>
        let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(nr); nc];
        for row in 0..nr {
            for col in 0..nc {
                columns[col].push(raw[row * nc + col]);
            }
        }

        let names: Vec<String> = (0..nc).map(|i| format!("c{i}")).collect();
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
            Err(e) => {
                set_last_error(&e.to_string());
                error_to_code(&e)
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_correlation");
            INSIGHT_ERR_PANIC
        }
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
        if x.is_null() || y.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        if len < 3 {
            set_last_error("need at least 3 data points");
            return INSIGHT_ERR_INVALID_INPUT;
        }

        let x_slice = unsafe { slice::from_raw_parts(x, len) };
        let y_slice = unsafe { slice::from_raw_parts(y, len) };

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
            Err(e) => {
                set_last_error(&e.to_string());
                error_to_code(&e)
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_regression");
            INSIGHT_ERR_PANIC
        }
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
/// # Safety
/// - `data` must point to `n_rows * n_cols` contiguous f64 values (row-major).
/// - `out` must point to a valid `CMahalanobisResult`.
/// - Caller must free output arrays.
#[no_mangle]
pub unsafe extern "C" fn insight_mahalanobis(
    data: *const f64,
    n_rows: u32,
    n_cols: u32,
    chi2_quantile: f64,
    out: *mut CMahalanobisResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let nr = n_rows as usize;
        let nc = n_cols as usize;
        if nr < nc + 1 {
            set_last_error("need n > p for Mahalanobis distance");
            return INSIGHT_ERR_INVALID_INPUT;
        }

        let raw = unsafe { slice::from_raw_parts(data, nr * nc) };
        let points: Vec<Vec<f64>> = (0..nr)
            .map(|i| raw[i * nc..(i + 1) * nc].to_vec())
            .collect();

        let config = crate::mahalanobis::MahalanobisConfig {
            chi2_quantile: if chi2_quantile > 0.0 && chi2_quantile < 1.0 {
                chi2_quantile
            } else {
                0.975
            },
        };

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
            Err(e) => {
                set_last_error(&e.to_string());
                error_to_code(&e)
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_mahalanobis");
            INSIGHT_ERR_PANIC
        }
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
        if table.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let nr = n_rows as usize;
        let nc = n_cols as usize;
        let raw = unsafe { slice::from_raw_parts(table, nr * nc) };

        match crate::analysis::cramers_v(raw, nr, nc) {
            Some(r) => {
                let out_ref = unsafe { &mut *out };
                out_ref.v = r.v;
                out_ref.chi_squared = r.chi_squared;
                out_ref.p_value = r.p_value;
                INSIGHT_OK
            }
            None => {
                set_last_error("Cramér's V computation failed");
                INSIGHT_ERR_ANALYSIS_FAILED
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_cramers_v");
            INSIGHT_ERR_PANIC
        }
    }
}

// ── ANOVA Feature Selection FFI ─────────────────────────────────────

/// C-compatible ANOVA feature result.
#[repr(C)]
pub struct CAnovaFeature {
    /// Feature index.
    pub index: u32,
    /// F-statistic.
    pub f_statistic: f64,
    /// P-value.
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
        if data.is_null() || target.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let nr = n_rows as usize;
        let nf = n_features as usize;
        let raw = unsafe { slice::from_raw_parts(data, nr * nf) };
        let target_raw = unsafe { slice::from_raw_parts(target, nr) };

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
                    .enumerate()
                    .map(|(i, f)| CAnovaFeature {
                        index: i as u32,
                        f_statistic: f.f_statistic,
                        p_value: f.p_value,
                    })
                    .collect();
                let mut boxed = c_features.into_boxed_slice();
                out_ref.features = boxed.as_mut_ptr();
                std::mem::forget(boxed);

                INSIGHT_OK
            }
            Err(e) => {
                set_last_error(&e.to_string());
                error_to_code(&e)
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_anova_select");
            INSIGHT_ERR_PANIC
        }
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
    // "0.1.0\0"
    c"0.1.0".as_ptr()
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

/// Runs hierarchical agglomerative clustering on row-major data.
///
/// # Parameters
///
/// - `linkage`: 0 = Single, 1 = Complete, 2 = Average, 3 = Ward.
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
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        let raw = unsafe { slice::from_raw_parts(data, n * d) };

        let points: Vec<Vec<f64>> = (0..n).map(|i| raw[i * d..(i + 1) * d].to_vec()).collect();

        let linkage_method = match linkage {
            0 => Linkage::Single,
            1 => Linkage::Complete,
            2 => Linkage::Average,
            _ => Linkage::Ward,
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
                set_last_error(&e.to_string());
                return error_to_code(&e);
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
        Err(_) => {
            set_last_error("panic in insight_hierarchical");
            INSIGHT_ERR_PANIC
        }
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
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let n = n_rows as usize;
        let d = n_cols as usize;
        let raw = unsafe { slice::from_raw_parts(data, n * d) };

        let points: Vec<Vec<f64>> = (0..n).map(|i| raw[i * d..(i + 1) * d].to_vec()).collect();

        let mut config = HdbscanConfig::new(min_cluster_size as usize);
        if min_samples > 0 {
            config = config.min_samples(min_samples as usize);
        }

        let hdb_result = match hdbscan(&points, &config) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(&e.to_string());
                return error_to_code(&e);
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
        Err(_) => {
            set_last_error("panic in insight_hdbscan");
            INSIGHT_ERR_PANIC
        }
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
    if data.is_null() || target.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return INSIGHT_ERR_NULL_PTR;
    }

    let result = panic::catch_unwind(|| {
        let nr = n_rows as usize;
        let nf = n_features as usize;
        let raw = slice::from_raw_parts(data, nr * nf);
        let target_raw = slice::from_raw_parts(target, nr);

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
            Err(e) => {
                set_last_error(&e.to_string());
                error_to_code(&e)
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_mutual_info");
            INSIGHT_ERR_PANIC
        }
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
    if data.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return INSIGHT_ERR_NULL_PTR;
    }

    let result = panic::catch_unwind(|| {
        let nr = n_rows as usize;
        let nc = n_cols as usize;
        let raw = slice::from_raw_parts(data, nr * nc);

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
            Err(e) => {
                set_last_error(&e.to_string());
                error_to_code(&e)
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_mini_batch_kmeans");
            INSIGHT_ERR_PANIC
        }
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
    if data.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return INSIGHT_ERR_NULL_PTR;
    }

    let result = panic::catch_unwind(|| {
        let nr = n_rows as usize;
        let nc = n_cols as usize;
        let raw = slice::from_raw_parts(data, nr * nc);

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
            Err(e) => {
                set_last_error(&e.to_string());
                error_to_code(&e)
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_gap_statistic");
            INSIGHT_ERR_PANIC
        }
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
    if data.is_null() || target.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return INSIGHT_ERR_NULL_PTR;
    }

    let result = panic::catch_unwind(|| {
        let nr = n_rows as usize;
        let nf = n_features as usize;
        let raw = slice::from_raw_parts(data, nr * nf);
        let target_raw = slice::from_raw_parts(target, nr);

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
            Err(e) => {
                set_last_error(&e.to_string());
                error_to_code(&e)
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_permutation_importance");
            INSIGHT_ERR_PANIC
        }
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

// ── PELT changepoint detection ───────────────────────────────────────

/// Result of PELT changepoint detection.
#[repr(C)]
pub struct CPeltResult {
    /// Detected changepoint indices (0-based). Caller must free with
    /// `insight_free_pelt_result`.
    pub changepoints: *mut u32,
    /// Number of changepoints detected.
    pub n_changepoints: u32,
}

/// Runs PELT changepoint detection on a univariate time series.
///
/// # Parameters
///
/// - `data`: pointer to `n` contiguous f64 values
/// - `n`: number of data points
/// - `cost`: cost function (0 = L2 mean change, 1 = Normal mean+variance)
/// - `penalty`: penalty value. Pass 0.0 to use BIC (automatic).
/// - `min_segment_len`: minimum segment length (must be >= 2)
/// - `out`: pointer to `CPeltResult` (filled on success)
///
/// # Safety
///
/// - `data` must point to `n` contiguous f64 values.
/// - `out` must point to a valid `CPeltResult`.
/// - Caller must free `out` with `insight_free_pelt_result`.
#[no_mangle]
pub unsafe extern "C" fn insight_pelt(
    data: *const f64,
    n: u32,
    cost: u32,
    penalty: f64,
    min_segment_len: u32,
    out: *mut CPeltResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        let raw = unsafe { slice::from_raw_parts(data, len) };

        let cost_fn = match cost {
            0 => u_analytics::detection::CostFunction::L2,
            1 => u_analytics::detection::CostFunction::Normal,
            _ => {
                set_last_error("cost must be 0 (L2) or 1 (Normal)");
                return INSIGHT_ERR_INVALID_PARAM;
            }
        };

        let pen = if penalty == 0.0 {
            u_analytics::detection::Penalty::Bic
        } else if penalty > 0.0 && penalty.is_finite() {
            u_analytics::detection::Penalty::Custom(penalty)
        } else {
            set_last_error("penalty must be 0.0 (BIC) or a positive finite number");
            return INSIGHT_ERR_INVALID_PARAM;
        };

        let min_seg = min_segment_len as usize;
        let pelt = match u_analytics::detection::Pelt::with_min_segment_len(cost_fn, pen, min_seg) {
            Some(p) => p,
            None => {
                set_last_error("invalid parameters (min_segment_len must be >= 2)");
                return INSIGHT_ERR_INVALID_PARAM;
            }
        };

        let pelt_result = pelt.detect(raw);

        let mut changepoints: Vec<u32> = pelt_result
            .changepoints
            .iter()
            .map(|&cp| cp as u32)
            .collect();
        let n_cp = changepoints.len() as u32;
        let cp_ptr = if changepoints.is_empty() {
            ptr::null_mut()
        } else {
            let p = changepoints.as_mut_ptr();
            std::mem::forget(changepoints);
            p
        };

        unsafe {
            (*out) = CPeltResult {
                changepoints: cp_ptr,
                n_changepoints: n_cp,
            };
        }

        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_pelt");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Runs PELT on multi-signal (multivariate) data.
///
/// # Parameters
///
/// - `data`: row-major array of shape `[n_samples, n_channels]`.
///   Each row is one observation across all channels — same convention
///   as every other multi-dimensional FFI in this crate
///   (PCA, KMeans, DBSCAN, IsolationForest, etc.).
/// - `n_samples`: number of time-series observations (rows)
/// - `n_channels`: number of signal channels (columns)
/// - Other params same as `insight_pelt`.
///
/// # Safety
///
/// - `data` must point to `n_samples * n_channels` contiguous f64 values.
/// - `out` must point to a valid `CPeltResult`.
/// - Caller must free `out` with `insight_free_pelt_result`.
#[no_mangle]
pub unsafe extern "C" fn insight_pelt_multi(
    data: *const f64,
    n_samples: u32,
    n_channels: u32,
    cost: u32,
    penalty: f64,
    min_segment_len: u32,
    out: *mut CPeltResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let ns = n_samples as usize;
        let ch = n_channels as usize;
        let raw = unsafe { slice::from_raw_parts(data, ns * ch) };

        // Transpose row-major [n_samples, n_channels] into per-channel slices
        // expected by `Pelt::detect_multi` (one slice per channel).
        let signals: Vec<Vec<f64>> = (0..ch)
            .map(|c| (0..ns).map(|s| raw[s * ch + c]).collect())
            .collect();
        let refs: Vec<&[f64]> = signals.iter().map(|s| s.as_slice()).collect();

        let cost_fn = match cost {
            0 => u_analytics::detection::CostFunction::L2,
            1 => u_analytics::detection::CostFunction::Normal,
            _ => {
                set_last_error("cost must be 0 (L2) or 1 (Normal)");
                return INSIGHT_ERR_INVALID_PARAM;
            }
        };

        let pen = if penalty == 0.0 {
            u_analytics::detection::Penalty::Bic
        } else if penalty > 0.0 && penalty.is_finite() {
            u_analytics::detection::Penalty::Custom(penalty)
        } else {
            set_last_error("penalty must be 0.0 (BIC) or positive finite");
            return INSIGHT_ERR_INVALID_PARAM;
        };

        let min_seg = min_segment_len as usize;
        let pelt = match u_analytics::detection::Pelt::with_min_segment_len(cost_fn, pen, min_seg) {
            Some(p) => p,
            None => {
                set_last_error("invalid parameters");
                return INSIGHT_ERR_INVALID_PARAM;
            }
        };

        let pelt_result = match pelt.detect_multi(&refs) {
            Some(r) => r,
            None => {
                set_last_error("all signals must have the same length");
                return INSIGHT_ERR_INVALID_INPUT;
            }
        };

        let mut changepoints: Vec<u32> = pelt_result
            .changepoints
            .iter()
            .map(|&cp| cp as u32)
            .collect();
        let n_cp = changepoints.len() as u32;
        let cp_ptr = if changepoints.is_empty() {
            ptr::null_mut()
        } else {
            let p = changepoints.as_mut_ptr();
            std::mem::forget(changepoints);
            p
        };

        unsafe {
            (*out) = CPeltResult {
                changepoints: cp_ptr,
                n_changepoints: n_cp,
            };
        }

        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_pelt_multi");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Frees a `CPeltResult` allocated by `insight_pelt` or `insight_pelt_multi`.
///
/// # Safety
///
/// The result must have been allocated by `insight_pelt` or `insight_pelt_multi`
/// and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn insight_free_pelt_result(result: *mut CPeltResult) {
    if !result.is_null() {
        let r = unsafe { &*result };
        if !r.changepoints.is_null() && r.n_changepoints > 0 {
            let _ = unsafe {
                Vec::from_raw_parts(
                    r.changepoints,
                    r.n_changepoints as usize,
                    r.n_changepoints as usize,
                )
            };
        }
    }
}

// ── Mann-Kendall trend test FFI ─────────────────────────────────────────

/// C-compatible Mann-Kendall trend test result.
#[repr(C)]
pub struct CMannKendallResult {
    /// Mann-Kendall S statistic: Σ sign(xⱼ - xᵢ) for all i < j.
    pub s_statistic: i64,
    /// Variance of S (with tie correction).
    pub variance: f64,
    /// Z statistic (with continuity correction).
    pub z_statistic: f64,
    /// Two-tailed p-value.
    pub p_value: f64,
    /// Kendall's tau: S / [n(n-1)/2]. Range [-1, 1].
    pub kendall_tau: f64,
    /// Sen's slope estimator (median of pairwise slopes).
    pub sen_slope: f64,
}

/// Mann-Kendall non-parametric trend test with Sen's slope estimator.
///
/// `data`: time-ordered observations, length `n`.
/// `out`: pointer to a `CMannKendallResult`.
///
/// Returns 0 on success, negative on error (fewer than 4 points, non-finite
/// values, or zero variance — e.g. all values identical).
///
/// # Safety
/// `data` must point to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_mann_kendall(
    data: *const f64,
    n: u32,
    out: *mut CMannKendallResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        if len < 4 {
            set_last_error("need at least 4 data points");
            return INSIGHT_ERR_INSUFFICIENT_DATA;
        }

        let raw = unsafe { slice::from_raw_parts(data, len) };

        match u_analytics::testing::mann_kendall_test(raw) {
            Some(r) => {
                unsafe {
                    (*out) = CMannKendallResult {
                        s_statistic: r.s_statistic,
                        variance: r.variance,
                        z_statistic: r.z_statistic,
                        p_value: r.p_value,
                        kendall_tau: r.kendall_tau,
                        sen_slope: r.sen_slope,
                    };
                }
                INSIGHT_OK
            }
            None => {
                set_last_error("invalid input (non-finite values or zero variance)");
                INSIGHT_ERR_INVALID_INPUT
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_mann_kendall");
            INSIGHT_ERR_PANIC
        }
    }
}

// ── Kernel Density Estimation FFI ───────────────────────────────────────

/// Silverman's rule of thumb bandwidth.
pub const INSIGHT_KDE_SILVERMAN: u32 = 0;
/// Scott's rule bandwidth.
pub const INSIGHT_KDE_SCOTT: u32 = 1;
/// Manually specified bandwidth (see `bandwidth` parameter of [`insight_kde`]).
pub const INSIGHT_KDE_MANUAL: u32 = 2;

/// C-compatible kernel density estimation result.
#[repr(C)]
pub struct CKdeResult {
    /// Evaluation points (x-axis), length `n_points`. Caller must free with `insight_free_kde_result`.
    pub x: *mut f64,
    /// Density estimates at each evaluation point (y-axis), length `n_points`.
    pub density: *mut f64,
    /// Number of evaluation grid points (length of `x` and `density`).
    pub n_points: u32,
    /// Bandwidth actually used (echoes the manual value, or the computed automatic one).
    pub bandwidth: f64,
}

/// Gaussian kernel density estimation.
///
/// `data`: sample observations, length `n`.
/// `method`: one of `INSIGHT_KDE_SILVERMAN` (0) / `_SCOTT` (1) / `_MANUAL` (2).
/// `bandwidth`: used only when `method == INSIGHT_KDE_MANUAL`; ignored otherwise.
/// `n_points`: number of evaluation grid points (typical: 256–1024).
/// `out`: pointer to a `CKdeResult`.
///
/// Returns 0 on success, negative on error. Caller must free `out` with
/// `insight_free_kde_result`.
///
/// # Safety
/// `data` must point to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_kde(
    data: *const f64,
    n: u32,
    method: u32,
    bandwidth: f64,
    n_points: u32,
    out: *mut CKdeResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let bw_method = match method {
            INSIGHT_KDE_SILVERMAN => u_analytics::distribution::BandwidthMethod::Silverman,
            INSIGHT_KDE_SCOTT => u_analytics::distribution::BandwidthMethod::Scott,
            INSIGHT_KDE_MANUAL => u_analytics::distribution::BandwidthMethod::Manual(bandwidth),
            _ => {
                set_last_error("invalid method (use 0=Silverman, 1=Scott, 2=Manual)");
                return INSIGHT_ERR_INVALID_PARAM;
            }
        };

        let len = n as usize;
        let raw = unsafe { slice::from_raw_parts(data, len) };

        match u_analytics::distribution::kde(raw, bw_method, n_points as usize) {
            Some(r) => {
                let mut x = r.x.into_boxed_slice();
                let mut density = r.density.into_boxed_slice();
                let x_ptr = x.as_mut_ptr();
                let density_ptr = density.as_mut_ptr();
                std::mem::forget(x);
                std::mem::forget(density);

                unsafe {
                    (*out) = CKdeResult {
                        x: x_ptr,
                        density: density_ptr,
                        n_points,
                        bandwidth: r.bandwidth,
                    };
                }
                INSIGHT_OK
            }
            None => {
                set_last_error(
                    "invalid input (need >= 2 data points and >= 2 grid points, finite values, \
                     nonzero variance for automatic bandwidth)",
                );
                INSIGHT_ERR_INVALID_INPUT
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_kde");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Frees a `CKdeResult` allocated by `insight_kde`.
///
/// # Safety
/// The result must have been allocated by `insight_kde` and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn insight_free_kde_result(result: *mut CKdeResult) {
    if !result.is_null() {
        let r = unsafe { &*result };
        if !r.x.is_null() && r.n_points > 0 {
            let _ = unsafe { Vec::from_raw_parts(r.x, r.n_points as usize, r.n_points as usize) };
        }
        if !r.density.is_null() && r.n_points > 0 {
            let _ =
                unsafe { Vec::from_raw_parts(r.density, r.n_points as usize, r.n_points as usize) };
        }
    }
}

// ── SPC Variables Control Charts FFI (X-bar-R, X-bar-S, Individual-MR) ──

/// A single point on a control chart, with Nelson-rule violations as a bitmask.
///
/// Bit `i` of `violation_mask` is set when Nelson rule `i + 1` fires at this
/// point (bit 0 = beyond limits ... bit 7 = eight beyond 1-sigma). Zero means
/// no violations were detected at this point.
#[repr(C)]
pub struct CSpcChartPoint {
    /// The computed statistic value (subgroup mean, range, standard deviation,
    /// individual observation, or moving range — depending on which series
    /// this point belongs to).
    pub value: f64,
    /// Bitmask of Nelson-rule violations detected at this point.
    pub violation_mask: u32,
}

/// C-compatible result for a two-series variables control chart
/// (X-bar-R, X-bar-S, or Individual-MR).
///
/// The primary series is the mean/individual chart; the secondary series is
/// the variation chart (R, S, or MR). `n_primary_points` and
/// `n_secondary_points` may differ — the MR chart has one fewer point than
/// the I chart (the first moving range is undefined).
#[repr(C)]
pub struct CVariablesChartResult {
    /// Primary (X-bar or Individual) chart upper control limit.
    pub primary_ucl: f64,
    /// Primary chart center line.
    pub primary_cl: f64,
    /// Primary chart lower control limit.
    pub primary_lcl: f64,
    /// Primary chart points. Caller must free with `insight_free_variables_chart_result`.
    pub primary_points: *mut CSpcChartPoint,
    /// Number of primary chart points.
    pub n_primary_points: u32,
    /// Secondary (R, S, or MR) chart upper control limit.
    pub secondary_ucl: f64,
    /// Secondary chart center line.
    pub secondary_cl: f64,
    /// Secondary chart lower control limit.
    pub secondary_lcl: f64,
    /// Secondary chart points.
    pub secondary_points: *mut CSpcChartPoint,
    /// Number of secondary chart points.
    pub n_secondary_points: u32,
    /// Estimate of the within-subgroup (short-term) sigma from the variation
    /// chart: `R-bar / d2` (X-bar-R), `S-bar / c4` (X-bar-S) or `MR-bar / d2`
    /// (Individual-MR). Pass it to `insight_process_capability` as
    /// `sigma_within` to obtain the short-term indices. NaN when the chart
    /// could not estimate it.
    pub sigma_hat: f64,
    /// 1 if no Nelson-rule violations were detected on either series, 0 otherwise.
    pub in_control: u8,
}

fn spc_violation_mask(violations: &[u_analytics::spc::ViolationType]) -> u32 {
    use u_analytics::spc::ViolationType::*;
    let mut mask = 0u32;
    for v in violations {
        let bit = match v {
            BeyondLimits => 0,
            NineOneSide => 1,
            SixTrend => 2,
            FourteenAlternating => 3,
            TwoOfThreeBeyond2Sigma => 4,
            FourOfFiveBeyond1Sigma => 5,
            FifteenWithin1Sigma => 6,
            EightBeyond1Sigma => 7,
        };
        mask |= 1 << bit;
    }
    mask
}

fn spc_points_to_c_array(points: &[u_analytics::spc::ChartPoint]) -> (*mut CSpcChartPoint, u32) {
    if points.is_empty() {
        return (ptr::null_mut(), 0);
    }
    let arr: Vec<CSpcChartPoint> = points
        .iter()
        .map(|p| CSpcChartPoint {
            value: p.value,
            violation_mask: spc_violation_mask(&p.violations),
        })
        .collect();
    let n = arr.len() as u32;
    let mut boxed = arr.into_boxed_slice();
    let out_ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    (out_ptr, n)
}

fn build_variables_chart_result(
    primary_limits: u_analytics::spc::ControlLimits,
    primary_points: &[u_analytics::spc::ChartPoint],
    secondary_limits: u_analytics::spc::ControlLimits,
    secondary_points: &[u_analytics::spc::ChartPoint],
    sigma_hat: Option<f64>,
    in_control: bool,
) -> CVariablesChartResult {
    let (primary_ptr, n_primary) = spc_points_to_c_array(primary_points);
    let (secondary_ptr, n_secondary) = spc_points_to_c_array(secondary_points);
    CVariablesChartResult {
        primary_ucl: primary_limits.ucl,
        primary_cl: primary_limits.cl,
        primary_lcl: primary_limits.lcl,
        primary_points: primary_ptr,
        n_primary_points: n_primary,
        secondary_ucl: secondary_limits.ucl,
        secondary_cl: secondary_limits.cl,
        secondary_lcl: secondary_limits.lcl,
        secondary_points: secondary_ptr,
        n_secondary_points: n_secondary,
        sigma_hat: sigma_hat.unwrap_or(f64::NAN),
        in_control: u8::from(in_control),
    }
}

/// The first value a variables chart would drop without saying so.
fn first_non_finite(values: &[f64]) -> Option<usize> {
    values.iter().position(|v| !v.is_finite())
}

/// Computes an X-bar-R control chart (subgroup mean + range).
///
/// `data`: row-major array of shape `[n_subgroups, subgroup_size]`.
/// `subgroup_size`: fixed subgroup size. The supported range is u-analytics'
/// (2 to 25); a size outside it is rejected with that range in the message.
/// A non-finite value is rejected with its index rather than skipped, so every
/// point stays in line with the subgroup it came from.
/// `out`: pointer to a `CVariablesChartResult` — primary series is X-bar,
/// secondary series is R. Its `sigma_hat` (`R-bar / d2`) is the within
/// sigma `insight_process_capability` needs for the short-term indices.
///
/// Returns 0 on success, negative on error. Caller must free `out` with
/// `insight_free_variables_chart_result`.
///
/// # Safety
/// `data` must point to `n_subgroups * subgroup_size` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_xbar_r_chart(
    data: *const f64,
    n_subgroups: u32,
    subgroup_size: u32,
    out: *mut CVariablesChartResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        use u_analytics::spc::ControlChart;

        let ns = n_subgroups as usize;
        let sg = subgroup_size as usize;
        let raw = unsafe { slice::from_raw_parts(data, ns * sg) };

        let mut chart = match u_analytics::spc::XBarRChart::new(sg) {
            Ok(chart) => chart,
            Err(e) => {
                set_last_error(&e.to_string());
                return INSIGHT_ERR_INVALID_PARAM;
            }
        };
        if let Some(i) = first_non_finite(raw) {
            set_last_error(&format!(
                "data[{i}] (subgroup {}) is not a finite number",
                i / sg
            ));
            return INSIGHT_ERR_INVALID_PARAM;
        }
        if let Err(code) = add_rows(ns, |g| chart.add_sample(&raw[g * sg..(g + 1) * sg])) {
            return code;
        }

        match (chart.control_limits(), chart.r_limits()) {
            (Some(xbar_limits), Some(r_limits)) => {
                unsafe {
                    (*out) = build_variables_chart_result(
                        xbar_limits,
                        chart.points(),
                        r_limits,
                        chart.r_points(),
                        chart.sigma_hat(),
                        chart.is_in_control(),
                    );
                }
                INSIGHT_OK
            }
            _ => {
                set_last_error("need at least 1 subgroup");

                INSIGHT_ERR_INSUFFICIENT_DATA
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_xbar_r_chart");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Feeds subgroups `0..count` to a chart, naming the subgroup a rejection
/// came from -- the chart knows why a sample is unusable but not where it sat
/// in the caller's arrays. The inputs are validated before this runs, so a
/// rejection here means the two checks have drifted apart.
fn add_rows(
    count: usize,
    mut add: impl FnMut(usize) -> Result<(), u_analytics::spc::ControlChartError>,
) -> Result<(), i32> {
    for i in 0..count {
        if let Err(e) = add(i) {
            set_last_error(&format!("subgroup {i}: {e}"));
            return Err(INSIGHT_ERR_INVALID_PARAM);
        }
    }
    Ok(())
}

/// Computes an X-bar-S control chart (subgroup mean + standard deviation).
///
/// `data`: row-major array of shape `[n_subgroups, subgroup_size]`.
/// `subgroup_size`: fixed subgroup size. The supported range is u-analytics'
/// (2 to 25); a size outside it is rejected with that range in the message.
/// A non-finite value is rejected with its index rather than skipped, so every
/// point stays in line with the subgroup it came from.
/// `out`: pointer to a `CVariablesChartResult` — primary series is X-bar,
/// secondary series is S. Its `sigma_hat` (`S-bar / c4`) is the within
/// sigma `insight_process_capability` needs for the short-term indices.
///
/// Returns 0 on success, negative on error. Caller must free `out` with
/// `insight_free_variables_chart_result`.
///
/// # Safety
/// `data` must point to `n_subgroups * subgroup_size` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_xbar_s_chart(
    data: *const f64,
    n_subgroups: u32,
    subgroup_size: u32,
    out: *mut CVariablesChartResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        use u_analytics::spc::ControlChart;

        let ns = n_subgroups as usize;
        let sg = subgroup_size as usize;
        let raw = unsafe { slice::from_raw_parts(data, ns * sg) };

        let mut chart = match u_analytics::spc::XBarSChart::new(sg) {
            Ok(chart) => chart,
            Err(e) => {
                set_last_error(&e.to_string());
                return INSIGHT_ERR_INVALID_PARAM;
            }
        };
        if let Some(i) = first_non_finite(raw) {
            set_last_error(&format!(
                "data[{i}] (subgroup {}) is not a finite number",
                i / sg
            ));
            return INSIGHT_ERR_INVALID_PARAM;
        }
        if let Err(code) = add_rows(ns, |g| chart.add_sample(&raw[g * sg..(g + 1) * sg])) {
            return code;
        }

        match (chart.control_limits(), chart.s_limits()) {
            (Some(xbar_limits), Some(s_limits)) => {
                unsafe {
                    (*out) = build_variables_chart_result(
                        xbar_limits,
                        chart.points(),
                        s_limits,
                        chart.s_points(),
                        chart.sigma_hat(),
                        chart.is_in_control(),
                    );
                }
                INSIGHT_OK
            }
            _ => {
                set_last_error("need at least 1 subgroup");

                INSIGHT_ERR_INSUFFICIENT_DATA
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_xbar_s_chart");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Computes an Individual-MR control chart (single observations + moving range).
///
/// `data`: individual observations, length `n`.
/// `out`: pointer to a `CVariablesChartResult` — primary series is
/// Individual (I), secondary series is Moving Range (MR, one fewer point
/// than I — the first moving range is undefined).
///
/// Returns 0 on success, negative on error. Caller must free `out` with
/// `insight_free_variables_chart_result`.
///
/// # Safety
/// `data` must point to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_individual_mr_chart(
    data: *const f64,
    n: u32,
    out: *mut CVariablesChartResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        use u_analytics::spc::ControlChart;

        let len = n as usize;
        let raw = unsafe { slice::from_raw_parts(data, len) };

        if let Some(i) = first_non_finite(raw) {
            set_last_error(&format!("data[{i}] is not a finite number"));
            return INSIGHT_ERR_INVALID_PARAM;
        }
        let mut chart = u_analytics::spc::IndividualMRChart::new();
        if let Err(code) = add_rows(len, |i| chart.add_sample(&raw[i..=i])) {
            return code;
        }

        match (chart.control_limits(), chart.mr_limits()) {
            (Some(i_limits), Some(mr_limits)) => {
                unsafe {
                    (*out) = build_variables_chart_result(
                        i_limits,
                        chart.points(),
                        mr_limits,
                        chart.mr_points(),
                        chart.sigma_hat(),
                        chart.is_in_control(),
                    );
                }
                INSIGHT_OK
            }
            _ => {
                set_last_error("need at least 2 observations");
                INSIGHT_ERR_INSUFFICIENT_DATA
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_individual_mr_chart");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Frees a `CVariablesChartResult` allocated by `insight_xbar_r_chart`,
/// `insight_xbar_s_chart`, or `insight_individual_mr_chart`.
///
/// # Safety
/// The result must have been allocated by one of those functions and not
/// yet freed.
#[no_mangle]
pub unsafe extern "C" fn insight_free_variables_chart_result(result: *mut CVariablesChartResult) {
    if result.is_null() {
        return;
    }
    let r = unsafe { &*result };
    if !r.primary_points.is_null() && r.n_primary_points > 0 {
        let _ = unsafe {
            Vec::from_raw_parts(
                r.primary_points,
                r.n_primary_points as usize,
                r.n_primary_points as usize,
            )
        };
    }
    if !r.secondary_points.is_null() && r.n_secondary_points > 0 {
        let _ = unsafe {
            Vec::from_raw_parts(
                r.secondary_points,
                r.n_secondary_points as usize,
                r.n_secondary_points as usize,
            )
        };
    }
}

// ── SPC Attributes Control Charts FFI (P, NP, C, U) ─────────────────────

/// A single point on an attributes control chart.
///
/// Unlike variables charts, attributes charts may have control limits that
/// vary per point (P and U charts, when sample sizes/inspection areas
/// differ) — so each point carries its own `ucl`/`cl`/`lcl`.
#[repr(C)]
pub struct CAttributeChartPoint {
    /// The computed statistic (proportion, count, or rate).
    pub value: f64,
    /// Upper control limit at this point.
    pub ucl: f64,
    /// Center line at this point.
    pub cl: f64,
    /// Lower control limit at this point.
    pub lcl: f64,
    /// 1 if this point is beyond its control limits, 0 otherwise.
    pub out_of_control: u8,
}

/// C-compatible result for a single-series attributes control chart
/// (P, NP, C, or U).
#[repr(C)]
pub struct CAttributeChartResult {
    /// Chart points. Caller must free with `insight_free_attribute_chart_result`.
    pub points: *mut CAttributeChartPoint,
    /// Number of chart points.
    pub n_points: u32,
}

fn attribute_points_to_c_result(
    points: &[u_analytics::spc::AttributeChartPoint],
) -> CAttributeChartResult {
    if points.is_empty() {
        return CAttributeChartResult {
            points: ptr::null_mut(),
            n_points: 0,
        };
    }
    let arr: Vec<CAttributeChartPoint> = points
        .iter()
        .map(|p| CAttributeChartPoint {
            value: p.value,
            ucl: p.ucl,
            cl: p.cl,
            lcl: p.lcl,
            out_of_control: u8::from(p.out_of_control),
        })
        .collect();
    let n = arr.len() as u32;
    let mut boxed = arr.into_boxed_slice();
    let out_ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    CAttributeChartResult {
        points: out_ptr,
        n_points: n,
    }
}

/// The first subgroup a proportion chart cannot use: nothing inspected, or
/// more defectives than inspected.
fn first_invalid_proportion(defectives: &[u64], sample_sizes: &[u64]) -> Option<usize> {
    defectives
        .iter()
        .zip(sample_sizes)
        .position(|(&d, &n)| n == 0 || d > n)
}

/// The first subgroup whose inspected units are not a positive number.
fn first_invalid_units(units: &[f64]) -> Option<usize> {
    units.iter().position(|&u| !(u.is_finite() && u > 0.0))
}

/// Computes a P chart (proportion nonconforming, variable sample size).
///
/// `defectives` / `sample_sizes`: parallel arrays of length `n` — number of
/// defective items and total sample size for each subgroup. A subgroup with
/// `sample_size == 0`, or more defectives than `sample_size`, is rejected with
/// `INSIGHT_ERR_INVALID_PARAM` and its index rather than skipped: the points
/// carry no index, so a skipped subgroup would leave every later point out of
/// line with its input.
/// `out`: pointer to a `CAttributeChartResult`.
///
/// Returns 0 on success, negative on error. Caller must free `out` with
/// `insight_free_attribute_chart_result`.
///
/// # Safety
/// `defectives` and `sample_sizes` must each point to `n` u64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_p_chart(
    defectives: *const u64,
    sample_sizes: *const u64,
    n: u32,
    out: *mut CAttributeChartResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if defectives.is_null() || sample_sizes.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        let defs = unsafe { slice::from_raw_parts(defectives, len) };
        let sizes = unsafe { slice::from_raw_parts(sample_sizes, len) };

        if let Some(i) = first_invalid_proportion(defs, sizes) {
            set_last_error(&format!(
                "subgroup {i}: {} defectives out of {} (need sample_size > 0 and \
                 defectives <= sample_size)",
                defs[i], sizes[i]
            ));
            return INSIGHT_ERR_INVALID_PARAM;
        }
        let mut chart = u_analytics::spc::PChart::new();
        if let Err(code) = add_rows(len, |i| chart.add_sample(defs[i], sizes[i])) {
            return code;
        }

        if chart.p_bar().is_none() {
            set_last_error("need at least 1 subgroup");

            return INSIGHT_ERR_INSUFFICIENT_DATA;
        }

        unsafe {
            (*out) = attribute_points_to_c_result(chart.points());
        }
        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_p_chart");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Computes an NP chart (count nonconforming, constant sample size).
///
/// `defective_counts`: defective count per subgroup, length `n`.
/// `sample_size`: constant sample size (> 0). A subgroup with more defectives
/// than `sample_size` is rejected with `INSIGHT_ERR_INVALID_PARAM` and its
/// index rather than skipped.
/// `out`: pointer to a `CAttributeChartResult`.
///
/// Returns 0 on success, negative on error. Caller must free `out` with
/// `insight_free_attribute_chart_result`.
///
/// # Safety
/// `defective_counts` must point to `n` u64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_np_chart(
    defective_counts: *const u64,
    n: u32,
    sample_size: u64,
    out: *mut CAttributeChartResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if defective_counts.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        let counts = unsafe { slice::from_raw_parts(defective_counts, len) };

        let mut chart = match u_analytics::spc::NPChart::new(sample_size) {
            Ok(chart) => chart,
            Err(e) => {
                set_last_error(&e.to_string());
                return INSIGHT_ERR_INVALID_PARAM;
            }
        };
        if let Some(i) = counts.iter().position(|&c| c > sample_size) {
            set_last_error(&format!(
                "subgroup {i}: {} defectives out of {sample_size}",
                counts[i]
            ));
            return INSIGHT_ERR_INVALID_PARAM;
        }
        if let Err(code) = add_rows(len, |i| chart.add_sample(counts[i])) {
            return code;
        }

        if chart.control_limits().is_none() {
            set_last_error("need at least 1 subgroup");
            return INSIGHT_ERR_INSUFFICIENT_DATA;
        }

        unsafe {
            (*out) = attribute_points_to_c_result(chart.points());
        }
        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_np_chart");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Computes a C chart (defect count, constant area of opportunity).
///
/// `defect_counts`: defect count per inspection unit, length `n`.
/// `out`: pointer to a `CAttributeChartResult`.
///
/// Returns 0 on success, negative on error. Caller must free `out` with
/// `insight_free_attribute_chart_result`.
///
/// # Safety
/// `defect_counts` must point to `n` u64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_c_chart(
    defect_counts: *const u64,
    n: u32,
    out: *mut CAttributeChartResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if defect_counts.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        let counts = unsafe { slice::from_raw_parts(defect_counts, len) };

        let mut chart = u_analytics::spc::CChart::new();
        for &c in counts {
            chart.add_sample(c);
        }

        if chart.control_limits().is_none() {
            set_last_error("no samples provided");
            return INSIGHT_ERR_INSUFFICIENT_DATA;
        }

        unsafe {
            (*out) = attribute_points_to_c_result(chart.points());
        }
        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_c_chart");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Computes a U chart (defects per unit, variable area of opportunity).
///
/// `defects` / `units_inspected`: parallel arrays of length `n` — defect
/// count and units inspected for each subgroup. A subgroup whose
/// `units_inspected` is not a positive number is rejected with
/// `INSIGHT_ERR_INVALID_PARAM` and its index rather than skipped.
/// `out`: pointer to a `CAttributeChartResult`.
///
/// Returns 0 on success, negative on error. Caller must free `out` with
/// `insight_free_attribute_chart_result`.
///
/// # Safety
/// `defects` must point to `n` u64s, `units_inspected` to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_u_chart(
    defects: *const u64,
    units_inspected: *const f64,
    n: u32,
    out: *mut CAttributeChartResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if defects.is_null() || units_inspected.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        let defs = unsafe { slice::from_raw_parts(defects, len) };
        let units = unsafe { slice::from_raw_parts(units_inspected, len) };

        if let Some(i) = first_invalid_units(units) {
            set_last_error(&format!(
                "subgroup {i}: units_inspected must be a positive number, got {}",
                units[i]
            ));
            return INSIGHT_ERR_INVALID_PARAM;
        }
        let mut chart = u_analytics::spc::UChart::new();
        if let Err(code) = add_rows(len, |i| chart.add_sample(defs[i], units[i])) {
            return code;
        }

        if chart.u_bar().is_none() {
            set_last_error("need at least 1 subgroup");
            return INSIGHT_ERR_INSUFFICIENT_DATA;
        }

        unsafe {
            (*out) = attribute_points_to_c_result(chart.points());
        }
        INSIGHT_OK
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_u_chart");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Frees a `CAttributeChartResult` allocated by `insight_p_chart`,
/// `insight_np_chart`, `insight_c_chart`, or `insight_u_chart`.
///
/// # Safety
/// The result must have been allocated by one of those functions and not
/// yet freed.
#[no_mangle]
pub unsafe extern "C" fn insight_free_attribute_chart_result(result: *mut CAttributeChartResult) {
    if result.is_null() {
        return;
    }
    let r = unsafe { &*result };
    if !r.points.is_null() && r.n_points > 0 {
        let _ = unsafe { Vec::from_raw_parts(r.points, r.n_points as usize, r.n_points as usize) };
    }
}

// ── SPC Laney P'/U' + G/T Charts FFI ─────────────────────────────────────

/// C-compatible result for a Laney P' or U' chart (overdispersion-adjusted
/// attributes chart).
#[repr(C)]
pub struct CLaneyChartResult {
    /// Overall proportion defective (P') or defect rate (U').
    pub bar: f64,
    /// Overdispersion/underdispersion correction factor. 1.0 means no
    /// correction was needed (equivalent to an ordinary P or U chart).
    pub phi: f64,
    /// Per-subgroup chart points. Caller must free with `insight_free_laney_chart_result`.
    pub points: *mut CAttributeChartPoint,
    /// Number of chart points.
    pub n_points: u32,
}

/// C-compatible result for a G or T chart (rare-event monitoring).
#[repr(C)]
pub struct CRareEventChartResult {
    /// Mean inter-event conforming count (G chart) or inter-event time (T chart).
    pub bar: f64,
    /// Per-observation chart points. Caller must free with `insight_free_rare_event_chart_result`.
    pub points: *mut CAttributeChartPoint,
    /// Number of chart points.
    pub n_points: u32,
}

fn attribute_points_from<'a, T: 'a>(
    points: &'a [T],
    field: impl Fn(&'a T) -> (f64, f64, f64, f64, bool),
) -> (*mut CAttributeChartPoint, u32) {
    if points.is_empty() {
        return (ptr::null_mut(), 0);
    }
    let arr: Vec<CAttributeChartPoint> = points
        .iter()
        .map(|p| {
            let (value, ucl, cl, lcl, out_of_control) = field(p);
            CAttributeChartPoint {
                value,
                ucl,
                cl,
                lcl,
                out_of_control: u8::from(out_of_control),
            }
        })
        .collect();
    let n = arr.len() as u32;
    let mut boxed = arr.into_boxed_slice();
    let out_ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    (out_ptr, n)
}

/// Computes a Laney P' chart (overdispersion-adjusted proportion nonconforming).
///
/// `defectives` / `sample_sizes`: parallel arrays of length `n` (needs at
/// least 3 subgroups). A subgroup with `sample_size == 0`, or more defectives
/// than `sample_size`, is rejected with `INSIGHT_ERR_INVALID_PARAM` and its
/// index.
/// `out`: pointer to a `CLaneyChartResult`.
///
/// Returns 0 on success, negative on error. Caller must free `out` with
/// `insight_free_laney_chart_result`.
///
/// # Safety
/// `defectives` and `sample_sizes` must each point to `n` u64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_laney_p_chart(
    defectives: *const u64,
    sample_sizes: *const u64,
    n: u32,
    out: *mut CLaneyChartResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if defectives.is_null() || sample_sizes.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        let defs = unsafe { slice::from_raw_parts(defectives, len) };
        let sizes = unsafe { slice::from_raw_parts(sample_sizes, len) };
        let samples: Vec<(u64, u64)> = defs.iter().zip(sizes).map(|(&d, &s)| (d, s)).collect();

        match u_analytics::spc::laney_p_chart(&samples, None) {
            Ok(chart) => {
                let (points_ptr, n_points) = attribute_points_from(&chart.points, |p| {
                    (p.value, p.ucl, p.cl, p.lcl, p.out_of_control)
                });
                unsafe {
                    (*out) = CLaneyChartResult {
                        bar: chart.p_bar,
                        phi: chart.phi,
                        points: points_ptr,
                        n_points,
                    };
                }
                INSIGHT_OK
            }
            Err(e) => laney_input_error(&e),
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_laney_p_chart");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Reports a Laney chart's refusal: a subgroup it cannot chart (with its
/// position) as `INSIGHT_ERR_INVALID_PARAM`, too few subgroups as
/// `INSIGHT_ERR_INSUFFICIENT_DATA`.
fn laney_input_error(e: &u_analytics::spc::ChartInputError) -> i32 {
    use u_analytics::spc::ChartInputError as E;
    match e {
        E::Sample { index, error } => {
            set_last_error(&format!("subgroup {index}: {error}"));
            INSIGHT_ERR_INVALID_PARAM
        }
        E::TooFewSamples { min, .. } => {
            set_last_error(&format!("need at least {min} subgroups"));
            INSIGHT_ERR_INSUFFICIENT_DATA
        }
        other => {
            set_last_error(&other.to_string());
            INSIGHT_ERR_INVALID_PARAM
        }
    }
}

/// Computes a Laney U' chart (overdispersion-adjusted defect rate).
///
/// `defects` / `units_inspected`: parallel arrays of length `n` (needs at
/// least 3 subgroups, all `units_inspected` positive and finite).
/// `out`: pointer to a `CLaneyChartResult`.
///
/// Returns 0 on success, negative on error. Caller must free `out` with
/// `insight_free_laney_chart_result`.
///
/// # Safety
/// `defects` must point to `n` u64s, `units_inspected` to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_laney_u_chart(
    defects: *const u64,
    units_inspected: *const f64,
    n: u32,
    out: *mut CLaneyChartResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if defects.is_null() || units_inspected.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        let defs = unsafe { slice::from_raw_parts(defects, len) };
        let units = unsafe { slice::from_raw_parts(units_inspected, len) };
        let samples: Vec<(u64, f64)> = defs.iter().zip(units).map(|(&d, &u)| (d, u)).collect();

        match u_analytics::spc::laney_u_chart(&samples, None) {
            Ok(chart) => {
                let (points_ptr, n_points) = attribute_points_from(&chart.points, |p| {
                    (p.value, p.ucl, p.cl, p.lcl, p.out_of_control)
                });
                unsafe {
                    (*out) = CLaneyChartResult {
                        bar: chart.u_bar,
                        phi: chart.phi,
                        points: points_ptr,
                        n_points,
                    };
                }
                INSIGHT_OK
            }
            Err(e) => laney_input_error(&e),
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_laney_u_chart");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Frees a `CLaneyChartResult` allocated by `insight_laney_p_chart` or
/// `insight_laney_u_chart`.
///
/// # Safety
/// The result must have been allocated by one of those functions and not
/// yet freed.
#[no_mangle]
pub unsafe extern "C" fn insight_free_laney_chart_result(result: *mut CLaneyChartResult) {
    if result.is_null() {
        return;
    }
    let r = unsafe { &*result };
    if !r.points.is_null() && r.n_points > 0 {
        let _ = unsafe { Vec::from_raw_parts(r.points, r.n_points as usize, r.n_points as usize) };
    }
}

/// Computes a G chart (geometric distribution — inter-defect conforming
/// count) for rare-event monitoring.
///
/// `inter_event_counts`: conforming-unit counts between successive defects,
/// length `n`.
/// `out`: pointer to a `CRareEventChartResult`.
///
/// Returns 0 on success, negative on error. Caller must free `out` with
/// `insight_free_rare_event_chart_result`.
///
/// # Safety
/// `inter_event_counts` must point to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_g_chart(
    inter_event_counts: *const f64,
    n: u32,
    out: *mut CRareEventChartResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if inter_event_counts.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        let raw = unsafe { slice::from_raw_parts(inter_event_counts, len) };

        match u_analytics::spc::g_chart(raw) {
            Some(chart) => {
                let (points_ptr, n_points) = attribute_points_from(&chart.points, |p| {
                    (p.value, p.ucl, p.cl, p.lcl, p.out_of_control)
                });
                unsafe {
                    (*out) = CRareEventChartResult {
                        bar: chart.g_bar,
                        points: points_ptr,
                        n_points,
                    };
                }
                INSIGHT_OK
            }
            None => {
                set_last_error("invalid input (need at least 3 points, all finite and >= 0.0)");
                INSIGHT_ERR_INSUFFICIENT_DATA
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_g_chart");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Computes a T chart (exponential distribution — inter-defect time) for
/// rare-event monitoring.
///
/// `inter_event_times`: time between successive defects, length `n`.
/// `out`: pointer to a `CRareEventChartResult`.
///
/// Returns 0 on success, negative on error. Caller must free `out` with
/// `insight_free_rare_event_chart_result`.
///
/// # Safety
/// `inter_event_times` must point to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_t_chart(
    inter_event_times: *const f64,
    n: u32,
    out: *mut CRareEventChartResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if inter_event_times.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        let raw = unsafe { slice::from_raw_parts(inter_event_times, len) };

        match u_analytics::spc::t_chart(raw) {
            Some(chart) => {
                let (points_ptr, n_points) = attribute_points_from(&chart.points, |p| {
                    (p.value, p.ucl, p.cl, p.lcl, p.out_of_control)
                });
                unsafe {
                    (*out) = CRareEventChartResult {
                        bar: chart.t_bar,
                        points: points_ptr,
                        n_points,
                    };
                }
                INSIGHT_OK
            }
            None => {
                set_last_error("invalid input (need at least 3 points, all finite and > 0.0)");
                INSIGHT_ERR_INSUFFICIENT_DATA
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_t_chart");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Frees a `CRareEventChartResult` allocated by `insight_g_chart` or
/// `insight_t_chart`.
///
/// # Safety
/// The result must have been allocated by one of those functions and not
/// yet freed.
#[no_mangle]
pub unsafe extern "C" fn insight_free_rare_event_chart_result(result: *mut CRareEventChartResult) {
    if result.is_null() {
        return;
    }
    let r = unsafe { &*result };
    if !r.points.is_null() && r.n_points > 0 {
        let _ = unsafe { Vec::from_raw_parts(r.points, r.n_points as usize, r.n_points as usize) };
    }
}

// ── Process Capability FFI ───────────────────────────────────────────────

/// C-compatible process capability indices.
///
/// Fields are `f64::NAN` when the corresponding index could not be computed
/// (e.g. Cp requires both USL and LSL) — the same "NaN means absent"
/// convention already used elsewhere in this module for optional statistics.
#[repr(C)]
pub struct CCapabilityIndices {
    /// Cp = (USL - LSL) / (6 * sigma_within). NaN unless both limits are set.
    pub cp: f64,
    /// Cpk = min(Cpu, Cpl). NaN unless at least one limit is set.
    pub cpk: f64,
    /// Cpu = (USL - mean) / (3 * sigma_within). NaN unless USL is set.
    pub cpu: f64,
    /// Cpl = (mean - LSL) / (3 * sigma_within). NaN unless LSL is set.
    pub cpl: f64,
    /// Pp = (USL - LSL) / (6 * sigma_overall). NaN unless both limits are set.
    pub pp: f64,
    /// Ppk = min(Ppu, Ppl). NaN unless at least one limit is set.
    pub ppk: f64,
    /// Ppu = (USL - mean) / (3 * sigma_overall). NaN unless USL is set.
    pub ppu: f64,
    /// Ppl = (mean - LSL) / (3 * sigma_overall). NaN unless LSL is set.
    pub ppl: f64,
    /// Cpm (Taguchi index), from the spread of the data about the target. NaN
    /// unless both limits and a target within them are given.
    pub cpm: f64,
    /// Sample mean of the data.
    pub mean: f64,
    /// Short-term (within-group) standard deviation -- the `sigma_within` the
    /// caller supplied. NaN when none was, together with `cp`, `cpk`, `cpu`
    /// and `cpl`.
    pub std_dev_within: f64,
    /// Long-term (overall) standard deviation.
    pub std_dev_overall: f64,
}

impl From<u_analytics::capability::CapabilityIndices> for CCapabilityIndices {
    fn from(idx: u_analytics::capability::CapabilityIndices) -> Self {
        CCapabilityIndices {
            cp: idx.cp.unwrap_or(f64::NAN),
            cpk: idx.cpk.unwrap_or(f64::NAN),
            cpu: idx.cpu.unwrap_or(f64::NAN),
            cpl: idx.cpl.unwrap_or(f64::NAN),
            pp: idx.pp.unwrap_or(f64::NAN),
            ppk: idx.ppk.unwrap_or(f64::NAN),
            ppu: idx.ppu.unwrap_or(f64::NAN),
            ppl: idx.ppl.unwrap_or(f64::NAN),
            cpm: idx.cpm.unwrap_or(f64::NAN),
            mean: idx.mean,
            std_dev_within: idx.std_dev_within.unwrap_or(f64::NAN),
            std_dev_overall: idx.std_dev_overall,
        }
    }
}

/// Computes standard process capability indices (Cp, Cpk, Pp, Ppk, Cpm).
///
/// `data`: process observations, length `n`.
/// `usl` / `lsl`: specification limits — pass `NaN` for "not set" (at least
/// one of the two must be a real number).
/// `target`: target value for Cpm — pass `NaN` when there is none, and `cpm`
/// comes back NaN. Pass `(usl + lsl) / 2` if the midpoint is the target.
/// `sigma_within`: short-term standard deviation (e.g. from a control
/// chart's R-bar/d2, S-bar/c4 or MR-bar/d2). Pass `NaN` when there is none
/// — a flat vector with no subgroup structure — and only the long-term
/// indices are reported: `cp`, `cpk`, `cpu`, `cpl` and `std_dev_within` come
/// back NaN. They are not filled from the overall sigma, which would make
/// `cp` equal `pp` for every input.
/// `out`: pointer to a `CCapabilityIndices`.
///
/// Returns 0 on success, negative on error.
///
/// # Safety
/// `data` must point to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_process_capability(
    data: *const f64,
    n: u32,
    usl: f64,
    lsl: f64,
    target: f64,
    sigma_within: f64,
    out: *mut CCapabilityIndices,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let usl_opt = if usl.is_nan() { None } else { Some(usl) };
        let lsl_opt = if lsl.is_nan() { None } else { Some(lsl) };

        let mut spec = match u_analytics::capability::ProcessCapability::new(usl_opt, lsl_opt) {
            Ok(s) => s,
            Err(msg) => {
                set_last_error(msg);
                return INSIGHT_ERR_INVALID_PARAM;
            }
        };
        if !target.is_nan() {
            spec = spec.with_target(target);
        }

        let len = n as usize;
        let raw = unsafe { slice::from_raw_parts(data, len) };

        let indices = if sigma_within.is_nan() {
            spec.compute_overall(raw)
        } else {
            spec.compute(raw, sigma_within)
        };

        match indices {
            Some(idx) => {
                unsafe {
                    (*out) = idx.into();
                }
                INSIGHT_OK
            }
            None => {
                set_last_error(
                    "invalid input (need >= 2 data points, all finite; sigma_within must be \
                     positive and finite when provided)",
                );
                INSIGHT_ERR_INSUFFICIENT_DATA
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_process_capability");
            INSIGHT_ERR_PANIC
        }
    }
}

/// C-compatible result of Box-Cox-based non-normal process capability analysis.
#[repr(C)]
pub struct CBoxCoxCapabilityResult {
    /// Estimated optimal Box-Cox transformation parameter lambda.
    pub lambda: f64,
    /// Capability indices computed on the Box-Cox-transformed scale.
    ///
    /// Only the long-term indices (`pp`, `ppk`, `ppu`, `ppl`) carry a value.
    /// `cp`, `cpk`, `cpu` and `cpl` are always `NaN` here: they are defined
    /// against a within-subgroup sigma, and a flat observation vector carries
    /// no subgroup structure to estimate one from. Reporting them from the
    /// overall sigma instead would make `cp` equal `pp` for every input.
    ///
    /// Every field is `NaN` when neither specification limit was given.
    pub indices: CCapabilityIndices,
    /// 1 when the likelihood maximum lies on an end of the lambda search range
    /// (the likelihood was still rising there, so `lambda` is that limit rather
    /// than an interior estimate), 0 otherwise.
    pub lambda_at_bound: u8,
}

impl CCapabilityIndices {
    /// Every index and statistic absent (`NaN`).
    fn absent() -> Self {
        CCapabilityIndices {
            cp: f64::NAN,
            cpk: f64::NAN,
            cpu: f64::NAN,
            cpl: f64::NAN,
            pp: f64::NAN,
            ppk: f64::NAN,
            ppu: f64::NAN,
            ppl: f64::NAN,
            cpm: f64::NAN,
            mean: f64::NAN,
            std_dev_within: f64::NAN,
            std_dev_overall: f64::NAN,
        }
    }
}

/// Computes process capability for non-normal data via Box-Cox transformation.
///
/// `data`: process observations, length `n` — must all be strictly positive.
/// `usl` / `lsl`: specification limits — pass `NaN` for "not set". Each set
/// limit must be positive. With neither, only `lambda` and `lambda_at_bound`
/// are estimated and every index is `NaN`.
/// `lambda_min` / `lambda_max`: the lambda search range — pass `NaN` for both
/// to use the default `[-5, 5]`; otherwise both must be finite with
/// `lambda_min < lambda_max`.
/// `out`: pointer to a `CBoxCoxCapabilityResult`.
///
/// Returns 0 on success, negative on error.
///
/// # Safety
/// `data` must point to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_boxcox_capability(
    data: *const f64,
    n: u32,
    usl: f64,
    lsl: f64,
    lambda_min: f64,
    lambda_max: f64,
    out: *mut CBoxCoxCapabilityResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let usl_opt = if usl.is_nan() { None } else { Some(usl) };
        let lsl_opt = if lsl.is_nan() { None } else { Some(lsl) };
        let range = if lambda_min.is_nan() && lambda_max.is_nan() {
            u_analytics::capability::DEFAULT_LAMBDA_RANGE
        } else {
            (lambda_min, lambda_max)
        };

        let len = n as usize;
        let raw = unsafe { slice::from_raw_parts(data, len) };

        match u_analytics::capability::boxcox_capability(raw, usl_opt, lsl_opt, range) {
            Ok(r) => {
                unsafe {
                    (*out) = CBoxCoxCapabilityResult {
                        lambda: r.lambda,
                        indices: r
                            .indices
                            .map_or_else(CCapabilityIndices::absent, Into::into),
                        lambda_at_bound: u8::from(r.lambda_at_bound),
                    };
                }
                INSIGHT_OK
            }
            Err(e) => {
                use u_analytics::capability::NonNormalCapabilityError as E;
                let code = match e {
                    E::InsufficientData => INSIGHT_ERR_INSUFFICIENT_DATA,
                    E::SpecTransformError | E::InvalidLambdaRange => INSIGHT_ERR_INVALID_PARAM,
                    E::NonPositiveData | E::NonFiniteData => INSIGHT_ERR_INVALID_INPUT,
                    E::CapabilityError => INSIGHT_ERR_DEGENERATE_DATA,
                };
                set_last_error(&e.to_string());
                code
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_boxcox_capability");
            INSIGHT_ERR_PANIC
        }
    }
}

/// C-compatible result of percentile-based (ISO 22514-2) process capability analysis.
#[repr(C)]
pub struct CPercentileCapabilityResult {
    /// Cp* = (USL - LSL) / (X_99.865 - X_0.135). NaN unless both limits are set.
    pub cp_star: f64,
    /// Cpk* = min(Cpu*, Cpl*). NaN unless at least one limit is set.
    pub cpk_star: f64,
    /// Cpu* = (USL - median) / (X_99.865 - median). NaN unless USL is set.
    pub cpu_star: f64,
    /// Cpl* = (median - LSL) / (median - X_0.135). NaN unless LSL is set.
    pub cpl_star: f64,
    /// Sample median.
    pub median: f64,
    /// 0.135th percentile value (lower natural process limit).
    pub percentile_lower: f64,
    /// 99.865th percentile value (upper natural process limit).
    pub percentile_upper: f64,
}

/// Computes percentile-based process capability indices (ISO 22514-2).
///
/// `data`: process observations, length `n` (needs at least 20 points).
/// `lsl` / `usl`: specification limits — pass `NaN` for "not set" (at least
/// one must be a real number).
/// `out`: pointer to a `CPercentileCapabilityResult`.
///
/// Returns 0 on success, negative on error.
///
/// # Safety
/// `data` must point to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_percentile_capability(
    data: *const f64,
    n: u32,
    lsl: f64,
    usl: f64,
    out: *mut CPercentileCapabilityResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let lsl_opt = if lsl.is_nan() { None } else { Some(lsl) };
        let usl_opt = if usl.is_nan() { None } else { Some(usl) };

        let len = n as usize;
        let raw = unsafe { slice::from_raw_parts(data, len) };

        match u_analytics::capability::percentile_capability(raw, lsl_opt, usl_opt) {
            Ok(r) => {
                unsafe {
                    (*out) = CPercentileCapabilityResult {
                        cp_star: r.cp_star.unwrap_or(f64::NAN),
                        cpk_star: r.cpk_star.unwrap_or(f64::NAN),
                        cpu_star: r.cpu_star.unwrap_or(f64::NAN),
                        cpl_star: r.cpl_star.unwrap_or(f64::NAN),
                        median: r.median,
                        percentile_lower: r.percentile_lower,
                        percentile_upper: r.percentile_upper,
                    };
                }
                INSIGHT_OK
            }
            Err(msg) => {
                set_last_error(msg);
                INSIGHT_ERR_INVALID_INPUT
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_percentile_capability");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Converts a sigma quality level to a parts-per-million (PPM) defect rate,
/// using the standard Motorola 1.5-sigma shift convention.
#[no_mangle]
pub extern "C" fn insight_sigma_to_ppm(sigma: f64) -> f64 {
    u_analytics::capability::sigma_to_ppm(sigma)
}

/// Converts a parts-per-million (PPM) defect rate to a sigma quality level
/// (inverse of `insight_sigma_to_ppm`). Returns `NaN` if `ppm` is outside
/// the valid range `(0, 1_000_000)` exclusive, or is itself `NaN`.
#[no_mangle]
pub extern "C" fn insight_ppm_to_sigma(ppm: f64) -> f64 {
    u_analytics::capability::ppm_to_sigma(ppm).unwrap_or(f64::NAN)
}

// ── Weibull Reliability FFI ──────────────────────────────────────────────

/// C-compatible result of Weibull Maximum Likelihood Estimation.
#[repr(C)]
pub struct CWeibullMleResult {
    /// Shape parameter (beta).
    pub shape: f64,
    /// Scale parameter (eta).
    pub scale: f64,
    /// Log-likelihood at the fitted parameters.
    pub log_likelihood: f64,
    /// Number of Newton-Raphson iterations used.
    pub iterations: u32,
}

/// Fits Weibull distribution parameters via Maximum Likelihood Estimation.
///
/// `failure_times`: positive failure times, length `n` (needs at least 2 values).
/// `out`: pointer to a `CWeibullMleResult`.
///
/// Returns 0 on success, negative on error (insufficient data, non-positive
/// values, or non-convergence).
///
/// # Safety
/// `failure_times` must point to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_weibull_mle(
    failure_times: *const f64,
    n: u32,
    out: *mut CWeibullMleResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if failure_times.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        let raw = unsafe { slice::from_raw_parts(failure_times, len) };

        match u_analytics::weibull::weibull_mle(raw) {
            Some(r) => {
                unsafe {
                    (*out) = CWeibullMleResult {
                        shape: r.shape,
                        scale: r.scale,
                        log_likelihood: r.log_likelihood,
                        iterations: r.iterations as u32,
                    };
                }
                INSIGHT_OK
            }
            None => {
                set_last_error(
                    "invalid input (need >= 2 positive finite values, and Newton-Raphson must converge)",
                );
                INSIGHT_ERR_INSUFFICIENT_DATA
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_weibull_mle");
            INSIGHT_ERR_PANIC
        }
    }
}

/// C-compatible result of Weibull Median Rank Regression fitting.
#[repr(C)]
pub struct CWeibullMrrResult {
    /// Shape parameter (beta).
    pub shape: f64,
    /// Scale parameter (eta).
    pub scale: f64,
    /// Coefficient of determination (R-squared) measuring goodness of fit.
    pub r_squared: f64,
}

/// Fits Weibull distribution parameters via Median Rank Regression.
///
/// `failure_times`: positive failure times, length `n` (needs at least 2 values).
/// `out`: pointer to a `CWeibullMrrResult`.
///
/// Returns 0 on success, negative on error.
///
/// # Safety
/// `failure_times` must point to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_weibull_mrr(
    failure_times: *const f64,
    n: u32,
    out: *mut CWeibullMrrResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if failure_times.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }

        let len = n as usize;
        let raw = unsafe { slice::from_raw_parts(failure_times, len) };

        match u_analytics::weibull::weibull_mrr(raw) {
            Some(r) => {
                unsafe {
                    (*out) = CWeibullMrrResult {
                        shape: r.shape,
                        scale: r.scale,
                        r_squared: r.r_squared,
                    };
                }
                INSIGHT_OK
            }
            None => {
                set_last_error("invalid input (need >= 2 positive finite values)");
                INSIGHT_ERR_INSUFFICIENT_DATA
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_weibull_mrr");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Weibull reliability (survival) function R(t) = exp(-(t/eta)^beta).
///
/// Returns `NaN` if `shape` or `scale` is non-positive or non-finite.
/// For `t <= 0`, returns 1.0 (no failure before time zero).
#[no_mangle]
pub extern "C" fn insight_weibull_reliability(shape: f64, scale: f64, t: f64) -> f64 {
    match u_analytics::weibull::ReliabilityAnalysis::new(shape, scale) {
        Some(ra) => ra.reliability(t),
        None => f64::NAN,
    }
}

/// Weibull hazard (instantaneous failure) rate at time t.
///
/// Returns `NaN` if `shape` or `scale` is non-positive or non-finite.
/// For `t <= 0`, returns 0.0.
#[no_mangle]
pub extern "C" fn insight_weibull_hazard_rate(shape: f64, scale: f64, t: f64) -> f64 {
    match u_analytics::weibull::ReliabilityAnalysis::new(shape, scale) {
        Some(ra) => ra.hazard_rate(t),
        None => f64::NAN,
    }
}

/// Mean Time Between Failures (MTBF) = eta * Gamma(1 + 1/beta).
///
/// Returns `NaN` if `shape` or `scale` is non-positive or non-finite.
#[no_mangle]
pub extern "C" fn insight_weibull_mtbf(shape: f64, scale: f64) -> f64 {
    match u_analytics::weibull::ReliabilityAnalysis::new(shape, scale) {
        Some(ra) => ra.mtbf(),
        None => f64::NAN,
    }
}

/// Time at which reliability drops to level `p` (solves R(t) = p for t).
///
/// Returns `NaN` if `shape`/`scale` are invalid, or `p` is outside `(0, 1)`.
#[no_mangle]
pub extern "C" fn insight_weibull_time_to_reliability(shape: f64, scale: f64, p: f64) -> f64 {
    match u_analytics::weibull::ReliabilityAnalysis::new(shape, scale) {
        Some(ra) => ra.time_to_reliability(p).unwrap_or(f64::NAN),
        None => f64::NAN,
    }
}

/// B-life: time at which `fraction_failed` of the population has failed
/// (e.g. `fraction_failed = 0.10` gives the B10 life).
///
/// Returns `NaN` if `shape`/`scale` are invalid, or `fraction_failed` is
/// outside `(0, 1)`.
#[no_mangle]
pub extern "C" fn insight_weibull_b_life(shape: f64, scale: f64, fraction_failed: f64) -> f64 {
    match u_analytics::weibull::ReliabilityAnalysis::new(shape, scale) {
        Some(ra) => ra.b_life(fraction_failed).unwrap_or(f64::NAN),
        None => f64::NAN,
    }
}

// ── Time series: period estimation and spectral residual ───────────────

/// One validated period candidate.
#[repr(C)]
pub struct CPeriodCandidate {
    /// Integer period, in observations.
    pub period: u32,
    /// Autocorrelation at that lag (the strength of the periodicity).
    pub acf: f64,
    /// Periodogram bin (1-based, of the padded transform) that produced it.
    pub bin: u32,
    /// Periodogram power of that bin.
    pub power: f64,
    /// That bin's share of the total periodogram power.
    pub power_share: f64,
}

/// C-compatible result of `insight_estimate_period`.
#[repr(C)]
pub struct CPeriodEstimate {
    /// The dominant period, or 0 when no periodicity passed both stages
    /// (a constant, a pure trend, white noise) — explicit, not an error.
    pub period: u32,
    /// Number of observations.
    pub n: u32,
    /// The 95% white-noise bound on the ACF, `1.96 / sqrt(n)`.
    pub acf_threshold: f64,
    /// Periodogram power a bin had to exceed to become a candidate.
    pub power_threshold: f64,
    /// Every validated candidate, strongest first. Caller must free with
    /// `insight_free_period_estimate`.
    pub candidates: *mut CPeriodCandidate,
    /// Number of candidates.
    pub n_candidates: u32,
}

/// Estimates the dominant period of a univariate series (AutoPeriod:
/// Vlachos, Yu & Castelli 2005 — permutation-thresholded periodogram peaks
/// refined on the autocorrelation function). Deterministic for a series.
///
/// `data`: `n` observations (at least 8, all finite). `out`: pointer to a
/// `CPeriodEstimate`. Returns 0 on success, negative on error. Caller must
/// free `out` with `insight_free_period_estimate`.
///
/// # Safety
/// `data` must point to `n` f64s. `out` must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_estimate_period(
    data: *const f64,
    n: u32,
    out: *mut CPeriodEstimate,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }
        let raw = unsafe { slice::from_raw_parts(data, n as usize) };
        match u_analytics::seasonality::estimate_period(raw) {
            Some(r) => {
                let mut candidates: Vec<CPeriodCandidate> = r
                    .candidates
                    .iter()
                    .map(|c| CPeriodCandidate {
                        period: c.period as u32,
                        acf: c.acf,
                        bin: c.bin as u32,
                        power: c.power,
                        power_share: c.power_share,
                    })
                    .collect();
                let n_candidates = candidates.len() as u32;
                let candidates_ptr = if candidates.is_empty() {
                    ptr::null_mut()
                } else {
                    let p = candidates.as_mut_ptr();
                    std::mem::forget(candidates);
                    p
                };
                unsafe {
                    (*out) = CPeriodEstimate {
                        period: r.period.unwrap_or(0) as u32,
                        n: r.n as u32,
                        acf_threshold: r.acf_threshold,
                        power_threshold: r.power_threshold,
                        candidates: candidates_ptr,
                        n_candidates,
                    };
                }
                INSIGHT_OK
            }
            None => {
                set_last_error("invalid input (need at least 8 finite observations)");
                INSIGHT_ERR_INSUFFICIENT_DATA
            }
        }
    });
    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_estimate_period");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Frees the candidates of a `CPeriodEstimate` allocated by
/// `insight_estimate_period`.
///
/// # Safety
/// The result must have been allocated by that function and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn insight_free_period_estimate(result: *mut CPeriodEstimate) {
    if result.is_null() {
        return;
    }
    let r = unsafe { &*result };
    if !r.candidates.is_null() && r.n_candidates > 0 {
        let _ = unsafe {
            Vec::from_raw_parts(
                r.candidates,
                r.n_candidates as usize,
                r.n_candidates as usize,
            )
        };
    }
}

/// Options for `insight_spectral_residual`. Pass a null pointer for the
/// defaults of Ren et al. (2019): q = 3, z = 40, threshold 3, z-score gate
/// 1.5, 70% band, no batching.
#[repr(C)]
pub struct CSpectralResidualOptions {
    /// Moving-average width on the log amplitude spectrum (>= 1).
    pub averaging_window: u32,
    /// Preceding saliencies a point is scored against (>= 1).
    pub judgement_window: u32,
    /// Score above which a point is an anomaly (> 0).
    pub threshold: f64,
    /// Minimum z-score of the point against the window before it (>= 0; 0 disables).
    pub min_zscore: f64,
    /// Coverage in percent of the band around the expected value (0 < s < 100).
    pub sensitivity: f64,
    /// Score in consecutive batches of this size (>= 12); 0 = one batch.
    pub batch_size: u32,
}

/// One scored point of `insight_spectral_residual`.
#[repr(C)]
pub struct CSrPoint {
    /// Position in the input series.
    pub index: u32,
    /// The observed value.
    pub value: f64,
    /// Spectral residual saliency (>= 0).
    pub saliency: f64,
    /// Saliency relative to the preceding judgement window (>= 0).
    pub score: f64,
    /// Low-frequency reconstruction of the series with anomalies removed.
    pub expected: f64,
    /// `expected - margin`.
    pub lower: f64,
    /// `expected + margin`.
    pub upper: f64,
    /// 1 when the point is an anomaly.
    pub is_anomaly: bool,
}

/// C-compatible result of `insight_spectral_residual`.
#[repr(C)]
pub struct CSpectralResidualResult {
    /// One point per observation, in order. Caller must free with
    /// `insight_free_spectral_residual_result`.
    pub points: *mut CSrPoint,
    /// Number of points (= n).
    pub n_points: u32,
    /// Number of points flagged as anomalies.
    pub n_anomalies: u32,
}

/// Scores every point of a series for anomalies by spectral residual
/// saliency (Ren et al. 2019): spikes, steps and dropouts, without a trained
/// model and without assuming a period.
///
/// `data`: `n` observations (at least 12, all finite). `options`: null for
/// the defaults. `out`: pointer to a `CSpectralResidualResult`. Returns 0 on
/// success, negative on error. Caller must free `out` with
/// `insight_free_spectral_residual_result`.
///
/// # Safety
/// `data` must point to `n` f64s. `options` must be null or valid. `out`
/// must be valid.
#[no_mangle]
pub unsafe extern "C" fn insight_spectral_residual(
    data: *const f64,
    n: u32,
    options: *const CSpectralResidualOptions,
    out: *mut CSpectralResidualResult,
) -> i32 {
    let result = panic::catch_unwind(|| {
        if data.is_null() || out.is_null() {
            set_last_error("null pointer");
            return INSIGHT_ERR_NULL_PTR;
        }
        let raw = unsafe { slice::from_raw_parts(data, n as usize) };
        let mut sr = u_analytics::detection::SpectralResidual::new();
        if !options.is_null() {
            let o = unsafe { &*options };
            sr = sr
                .with_averaging_window(o.averaging_window as usize)
                .with_judgement_window(o.judgement_window as usize)
                .with_threshold(o.threshold)
                .with_min_zscore(o.min_zscore)
                .with_sensitivity(o.sensitivity)
                .with_batch_size((o.batch_size > 0).then_some(o.batch_size as usize));
        }
        match sr.analyze(raw) {
            Ok(points) => {
                let n_anomalies = points.iter().filter(|p| p.is_anomaly).count() as u32;
                let mut c_points: Vec<CSrPoint> = points
                    .iter()
                    .map(|p| CSrPoint {
                        index: p.index as u32,
                        value: p.value,
                        saliency: p.saliency,
                        score: p.score,
                        expected: p.expected,
                        lower: p.lower,
                        upper: p.upper,
                        is_anomaly: p.is_anomaly,
                    })
                    .collect();
                let n_points = c_points.len() as u32;
                let points_ptr = if c_points.is_empty() {
                    ptr::null_mut()
                } else {
                    let p = c_points.as_mut_ptr();
                    std::mem::forget(c_points);
                    p
                };
                unsafe {
                    (*out) = CSpectralResidualResult {
                        points: points_ptr,
                        n_points,
                        n_anomalies,
                    };
                }
                INSIGHT_OK
            }
            // One condition, named -- not the whole rulebook for the caller
            // to match its own settings against.
            Err(e) => {
                set_last_error(&e.to_string());
                INSIGHT_ERR_INVALID_PARAM
            }
        }
    });
    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in insight_spectral_residual");
            INSIGHT_ERR_PANIC
        }
    }
}

/// Frees the points of a `CSpectralResidualResult` allocated by
/// `insight_spectral_residual`.
///
/// # Safety
/// The result must have been allocated by that function and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn insight_free_spectral_residual_result(
    result: *mut CSpectralResidualResult,
) {
    if result.is_null() {
        return;
    }
    let r = unsafe { &*result };
    if !r.points.is_null() && r.n_points > 0 {
        let _ = unsafe { Vec::from_raw_parts(r.points, r.n_points as usize, r.n_points as usize) };
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn ffi_version() {
        let v = insight_version();
        let s = unsafe { CStr::from_ptr(v) }.to_str().unwrap();
        assert_eq!(s, "0.1.0");
    }

    #[test]
    fn ffi_error_lifecycle() {
        insight_clear_error();
        assert!(insight_last_error().is_null());

        set_last_error("test error");
        let msg = unsafe { CStr::from_ptr(insight_last_error()) }
            .to_str()
            .unwrap();
        assert_eq!(msg, "test error");

        insight_clear_error();
        assert!(insight_last_error().is_null());
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

    #[test]
    fn ffi_correlation_too_few() {
        let data: Vec<f64> = vec![1.0, 2.0]; // 1 row, 2 cols
        let mut result = CCorrelationResult {
            n_vars: 0,
            matrix: ptr::null_mut(),
            n_high_pairs: 0,
        };
        let rc =
            unsafe { insight_correlation(data.as_ptr(), 1, 2, INSIGHT_CORR_PEARSON, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_INPUT);
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
        assert_eq!(rc, INSIGHT_ERR_INVALID_INPUT);
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

    // ── PELT tests ───────────────────────────────────────────────────

    #[test]
    fn ffi_estimate_period_sawtooth_and_line() {
        let saw: Vec<f64> = (0..40).map(|i| (i % 7) as f64).collect();
        let mut out = CPeriodEstimate {
            period: 0,
            n: 0,
            acf_threshold: 0.0,
            power_threshold: 0.0,
            candidates: ptr::null_mut(),
            n_candidates: 0,
        };
        let rc = unsafe { insight_estimate_period(saw.as_ptr(), 40, &mut out) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(out.period, 7);
        assert_eq!(out.n, 40);
        assert!(out.n_candidates >= 1 && !out.candidates.is_null());
        let first = unsafe { &*out.candidates };
        assert_eq!(first.period, 7);
        assert!(first.power > out.power_threshold);
        unsafe { insight_free_period_estimate(&mut out) };

        let line: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let rc = unsafe { insight_estimate_period(line.as_ptr(), 40, &mut out) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(out.period, 0, "a line has no period");
        assert_eq!(out.n_candidates, 0);
        unsafe { insight_free_period_estimate(&mut out) };

        let rc = unsafe { insight_estimate_period(line.as_ptr(), 5, &mut out) };
        assert_eq!(rc, INSIGHT_ERR_INSUFFICIENT_DATA);
        let rc = unsafe { insight_estimate_period(ptr::null(), 40, &mut out) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_spectral_residual_flags_the_spike() {
        let mut data = vec![1.0_f64; 40];
        data[25] = 9.0;
        let mut out = CSpectralResidualResult {
            points: ptr::null_mut(),
            n_points: 0,
            n_anomalies: 0,
        };
        let rc = unsafe { insight_spectral_residual(data.as_ptr(), 40, ptr::null(), &mut out) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(out.n_points, 40);
        assert_eq!(out.n_anomalies, 1);
        let points = unsafe { slice::from_raw_parts(out.points, 40) };
        assert!(points[25].is_anomaly);
        assert_eq!(points[25].index, 25);
        assert!(points[25].value > points[25].upper);
        assert!(points
            .iter()
            .all(|p| p.lower <= p.expected && p.expected <= p.upper));
        unsafe { insight_free_spectral_residual_result(&mut out) };

        let options = CSpectralResidualOptions {
            averaging_window: 3,
            judgement_window: 20,
            threshold: 3.0,
            min_zscore: 1.5,
            sensitivity: 95.0,
            batch_size: 0,
        };
        let rc = unsafe { insight_spectral_residual(data.as_ptr(), 40, &options, &mut out) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(out.n_anomalies, 1);
        unsafe { insight_free_spectral_residual_result(&mut out) };

        let bad = CSpectralResidualOptions {
            threshold: 0.0,
            ..options
        };
        let rc = unsafe { insight_spectral_residual(data.as_ptr(), 40, &bad, &mut out) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
        // The refusal names the option that was wrong. It used to restate
        // every rule, which left the C# and TS consumers re-validating the
        // options to be able to say which one their user had to change.
        let message = unsafe { CStr::from_ptr(insight_last_error()) }
            .to_string_lossy()
            .into_owned();
        assert_eq!(message, "threshold must be a finite number > 0");
        for other in [
            "sensitivity",
            "batch_size",
            "averaging_window",
            "observations",
        ] {
            assert!(!message.contains(other), "{message}");
        }

        let bad = CSpectralResidualOptions {
            sensitivity: 100.0,
            ..options
        };
        let rc = unsafe { insight_spectral_residual(data.as_ptr(), 40, &bad, &mut out) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
        let message = unsafe { CStr::from_ptr(insight_last_error()) }
            .to_string_lossy()
            .into_owned();
        assert!(message.starts_with("sensitivity"), "{message}");

        let rc = unsafe { insight_spectral_residual(data.as_ptr(), 5, ptr::null(), &mut out) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
        let message = unsafe { CStr::from_ptr(insight_last_error()) }
            .to_string_lossy()
            .into_owned();
        assert_eq!(message, "needs at least 12 observations, got 5");
    }

    #[test]
    fn ffi_pelt_single_changepoint() {
        let mut data = vec![0.0_f64; 50];
        data.extend(vec![5.0; 50]);

        let mut result = CPeltResult {
            changepoints: ptr::null_mut(),
            n_changepoints: 0,
        };

        let rc = unsafe { insight_pelt(data.as_ptr(), 100, 0, 0.0, 2, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_changepoints, 1);
        assert!(!result.changepoints.is_null());

        let cp = unsafe { *result.changepoints };
        assert!(
            (cp as i64 - 50).unsigned_abs() <= 2,
            "changepoint near 50, got {}",
            cp
        );

        unsafe { insight_free_pelt_result(&mut result) };
    }

    #[test]
    fn ffi_pelt_no_changepoint() {
        let data = vec![5.0_f64; 100];

        let mut result = CPeltResult {
            changepoints: ptr::null_mut(),
            n_changepoints: 0,
        };

        let rc = unsafe { insight_pelt(data.as_ptr(), 100, 0, 0.0, 2, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_changepoints, 0);
        assert!(result.changepoints.is_null());
    }

    #[test]
    fn ffi_pelt_null_pointer() {
        let mut result = CPeltResult {
            changepoints: ptr::null_mut(),
            n_changepoints: 0,
        };
        let rc = unsafe { insight_pelt(ptr::null(), 10, 0, 0.0, 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_pelt_invalid_cost() {
        let data = [1.0_f64; 10];
        let mut result = CPeltResult {
            changepoints: ptr::null_mut(),
            n_changepoints: 0,
        };
        let rc = unsafe { insight_pelt(data.as_ptr(), 10, 99, 0.0, 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
    }

    #[test]
    fn ffi_pelt_multi_two_channels() {
        // Row-major [n_samples=100, n_channels=2]:
        // sample s, channel c → data[s * 2 + c]
        let mut data = Vec::with_capacity(200);
        for s in 0..100 {
            // Channel 0: [0..50]=0, [50..100]=5
            data.push(if s < 50 { 0.0_f64 } else { 5.0 });
            // Channel 1: [0..50]=0, [50..100]=3
            data.push(if s < 50 { 0.0_f64 } else { 3.0 });
        }

        let mut result = CPeltResult {
            changepoints: ptr::null_mut(),
            n_changepoints: 0,
        };

        // n_samples=100, n_channels=2
        let rc = unsafe { insight_pelt_multi(data.as_ptr(), 100, 2, 0, 0.0, 2, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_changepoints, 1);

        let cp = unsafe { *result.changepoints };
        assert!(
            (cp as i64 - 50).unsigned_abs() <= 2,
            "changepoint near 50, got {}",
            cp
        );

        unsafe { insight_free_pelt_result(&mut result) };
    }

    #[test]
    fn ffi_pelt_multi_asymmetric_layout() {
        // Asymmetric shape (n_samples=80 ≠ n_channels=3) so any swapped
        // dimensions or off-by-one transpose surfaces immediately.
        // Channel 0: step at sample 40, level jump 0 → 4
        // Channel 1: step at sample 40, level jump 0 → 6
        // Channel 2: step at sample 40, level jump 0 → 2
        let n_samples = 80usize;
        let n_channels = 3usize;
        let mut data = Vec::with_capacity(n_samples * n_channels);
        for s in 0..n_samples {
            let pre = s < 40;
            data.push(if pre { 0.0_f64 } else { 4.0 });
            data.push(if pre { 0.0_f64 } else { 6.0 });
            data.push(if pre { 0.0_f64 } else { 2.0 });
        }

        let mut result = CPeltResult {
            changepoints: ptr::null_mut(),
            n_changepoints: 0,
        };

        let rc = unsafe {
            insight_pelt_multi(
                data.as_ptr(),
                n_samples as u32,
                n_channels as u32,
                0,
                0.0,
                2,
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_changepoints, 1);

        let cp = unsafe { *result.changepoints };
        assert!(
            (cp as i64 - 40).unsigned_abs() <= 2,
            "changepoint near 40, got {}",
            cp
        );

        unsafe { insight_free_pelt_result(&mut result) };
    }

    #[test]
    fn ffi_pelt_multi_null() {
        let mut result = CPeltResult {
            changepoints: ptr::null_mut(),
            n_changepoints: 0,
        };
        // n_samples=50, n_channels=2
        let rc = unsafe { insight_pelt_multi(ptr::null(), 50, 2, 0, 0.0, 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_mann_kendall_upward_trend() {
        let data: Vec<f64> = vec![1.0, 2.3, 3.1, 4.5, 5.2, 6.8, 7.1, 8.9, 9.5, 10.2];
        let mut result = CMannKendallResult {
            s_statistic: 0,
            variance: 0.0,
            z_statistic: 0.0,
            p_value: 0.0,
            kendall_tau: 0.0,
            sen_slope: 0.0,
        };

        let rc = unsafe { insight_mann_kendall(data.as_ptr(), data.len() as u32, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert!(result.p_value < 0.01);
        assert!(result.kendall_tau > 0.8);
        assert!(result.sen_slope > 0.0);
    }

    #[test]
    fn ffi_mann_kendall_insufficient_data() {
        let data = [1.0_f64, 2.0, 3.0];
        let mut result = CMannKendallResult {
            s_statistic: 0,
            variance: 0.0,
            z_statistic: 0.0,
            p_value: 0.0,
            kendall_tau: 0.0,
            sen_slope: 0.0,
        };

        let rc = unsafe { insight_mann_kendall(data.as_ptr(), 3, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INSUFFICIENT_DATA);
    }

    #[test]
    fn ffi_mann_kendall_null_pointer() {
        let mut result = CMannKendallResult {
            s_statistic: 0,
            variance: 0.0,
            z_statistic: 0.0,
            p_value: 0.0,
            kendall_tau: 0.0,
            sen_slope: 0.0,
        };
        let rc = unsafe { insight_mann_kendall(ptr::null(), 10, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_kde_silverman_basic() {
        let data = [1.0_f64, 1.1, 1.2, 2.0, 2.1, 2.2, 5.0];
        let mut result = CKdeResult {
            x: ptr::null_mut(),
            density: ptr::null_mut(),
            n_points: 0,
            bandwidth: 0.0,
        };

        let rc = unsafe {
            insight_kde(
                data.as_ptr(),
                data.len() as u32,
                INSIGHT_KDE_SILVERMAN,
                0.0,
                512,
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_points, 512);
        assert!(result.bandwidth > 0.0);
        assert!(!result.x.is_null());
        assert!(!result.density.is_null());

        unsafe { insight_free_kde_result(&mut result) };
    }

    #[test]
    fn ffi_kde_manual_bandwidth() {
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut result = CKdeResult {
            x: ptr::null_mut(),
            density: ptr::null_mut(),
            n_points: 0,
            bandwidth: 0.0,
        };

        let rc = unsafe {
            insight_kde(
                data.as_ptr(),
                data.len() as u32,
                INSIGHT_KDE_MANUAL,
                0.5,
                128,
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.bandwidth, 0.5);

        unsafe { insight_free_kde_result(&mut result) };
    }

    #[test]
    fn ffi_kde_invalid_method() {
        let data = [1.0_f64, 2.0, 3.0];
        let mut result = CKdeResult {
            x: ptr::null_mut(),
            density: ptr::null_mut(),
            n_points: 0,
            bandwidth: 0.0,
        };
        let rc = unsafe { insight_kde(data.as_ptr(), 3, 99, 0.0, 256, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
    }

    #[test]
    fn ffi_kde_null_pointer() {
        let mut result = CKdeResult {
            x: ptr::null_mut(),
            density: ptr::null_mut(),
            n_points: 0,
            bandwidth: 0.0,
        };
        let rc = unsafe {
            insight_kde(
                ptr::null(),
                10,
                INSIGHT_KDE_SILVERMAN,
                0.0,
                256,
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    fn empty_variables_result() -> CVariablesChartResult {
        CVariablesChartResult {
            primary_ucl: 0.0,
            primary_cl: 0.0,
            primary_lcl: 0.0,
            primary_points: ptr::null_mut(),
            n_primary_points: 0,
            secondary_ucl: 0.0,
            secondary_cl: 0.0,
            secondary_lcl: 0.0,
            secondary_points: ptr::null_mut(),
            n_secondary_points: 0,
            sigma_hat: 0.0,
            in_control: 0,
        }
    }

    #[test]
    fn ffi_xbar_r_chart_in_control() {
        // 10 subgroups of size 5, small stable variation.
        let data: Vec<f64> = vec![
            25.0, 26.0, 24.5, 25.5, 25.0, 25.2, 24.8, 25.1, 24.9, 25.3, 25.1, 25.0, 24.7, 25.3,
            24.9, 24.9, 25.2, 25.0, 24.8, 25.1, 25.0, 24.9, 25.1, 25.0, 24.9, 25.2, 24.8, 25.0,
            25.1, 24.9, 24.9, 25.0, 25.1, 24.8, 25.2, 25.0, 25.0, 24.9, 25.1, 25.0, 25.1, 24.9,
            25.0, 25.0, 25.0, 24.9, 25.1, 25.0, 24.9, 25.1,
        ];
        let mut result = empty_variables_result();

        let rc = unsafe { insight_xbar_r_chart(data.as_ptr(), 10, 5, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_primary_points, 10);
        assert_eq!(result.n_secondary_points, 10);
        assert!(result.primary_ucl > result.primary_cl);
        assert!(result.primary_cl > result.primary_lcl);
        assert!(result.secondary_ucl > result.secondary_lcl);

        unsafe { insight_free_variables_chart_result(&mut result) };
    }

    #[test]
    fn ffi_xbar_r_chart_detects_violation() {
        let mut data: Vec<f64> = vec![
            25.0, 26.0, 24.5, 25.5, 25.0, 25.2, 24.8, 25.1, 24.9, 25.3, 25.1, 25.0, 24.7, 25.3,
            24.9, 24.9, 25.2, 25.0, 24.8, 25.1,
        ];
        // One wildly out-of-range subgroup to trigger a BeyondLimits violation.
        data.extend_from_slice(&[100.0, 101.0, 99.0, 100.5, 99.5]);

        let mut result = empty_variables_result();
        let rc = unsafe { insight_xbar_r_chart(data.as_ptr(), 5, 5, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.in_control, 0);

        let points = unsafe {
            slice::from_raw_parts(result.primary_points, result.n_primary_points as usize)
        };
        assert!(points.iter().any(|p| p.violation_mask & 1 != 0));

        unsafe { insight_free_variables_chart_result(&mut result) };
    }

    #[test]
    fn ffi_xbar_r_chart_invalid_subgroup_size() {
        let data = [1.0_f64; 20];
        let mut result = empty_variables_result();
        let rc = unsafe { insight_xbar_r_chart(data.as_ptr(), 2, 1, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
    }

    #[test]
    fn ffi_xbar_r_chart_null_pointer() {
        let mut result = empty_variables_result();
        let rc = unsafe { insight_xbar_r_chart(ptr::null(), 10, 5, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_xbar_s_chart_in_control() {
        let data: Vec<f64> = vec![
            25.0, 26.0, 24.5, 25.5, 25.0, 25.2, 24.8, 25.1, 24.9, 25.3, 25.1, 25.0, 24.7, 25.3,
            24.9, 24.9, 25.2, 25.0, 24.8, 25.1,
        ];
        let mut result = empty_variables_result();

        let rc = unsafe { insight_xbar_s_chart(data.as_ptr(), 4, 5, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_primary_points, 4);
        assert_eq!(result.n_secondary_points, 4);
        assert!(result.primary_ucl > result.primary_lcl);
        assert!(result.secondary_ucl >= result.secondary_lcl);

        unsafe { insight_free_variables_chart_result(&mut result) };
    }

    #[test]
    fn ffi_xbar_s_chart_null_pointer() {
        let mut result = empty_variables_result();
        let rc = unsafe { insight_xbar_s_chart(ptr::null(), 10, 5, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_individual_mr_chart_basic() {
        let data: Vec<f64> = vec![10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7, 10.1, 9.9];
        let mut result = empty_variables_result();

        let rc = unsafe { insight_individual_mr_chart(data.as_ptr(), 10, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_primary_points, 10);
        // Moving-range chart has one fewer point than the individual chart.
        assert_eq!(result.n_secondary_points, 9);
        assert!(result.primary_ucl > result.primary_cl);
        assert!(result.primary_cl > result.primary_lcl);

        unsafe { insight_free_variables_chart_result(&mut result) };
    }

    #[test]
    fn ffi_individual_mr_chart_insufficient_data() {
        let data = [10.0_f64];
        let mut result = empty_variables_result();
        let rc = unsafe { insight_individual_mr_chart(data.as_ptr(), 1, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INSUFFICIENT_DATA);
    }

    #[test]
    fn ffi_individual_mr_chart_null_pointer() {
        let mut result = empty_variables_result();
        let rc = unsafe { insight_individual_mr_chart(ptr::null(), 10, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    fn empty_attribute_result() -> CAttributeChartResult {
        CAttributeChartResult {
            points: ptr::null_mut(),
            n_points: 0,
        }
    }

    #[test]
    fn ffi_p_chart_basic() {
        let defectives: [u64; 4] = [3, 5, 2, 4];
        let sample_sizes: [u64; 4] = [100, 100, 100, 100];
        let mut result = empty_attribute_result();

        let rc =
            unsafe { insight_p_chart(defectives.as_ptr(), sample_sizes.as_ptr(), 4, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_points, 4);

        let points = unsafe { slice::from_raw_parts(result.points, result.n_points as usize) };
        assert!((points[0].value - 0.03).abs() < 1e-9);
        assert!(points[0].ucl > points[0].cl);
        assert!(points[0].cl > points[0].lcl || points[0].lcl == 0.0);

        unsafe { insight_free_attribute_chart_result(&mut result) };
    }

    #[test]
    fn ffi_p_chart_insufficient_data() {
        // No subgroups at all is too little data. A subgroup that cannot be
        // charted is a different error: see the test below.
        let defectives: [u64; 1] = [0];
        let sample_sizes: [u64; 1] = [0];
        let mut result = empty_attribute_result();
        let rc =
            unsafe { insight_p_chart(defectives.as_ptr(), sample_sizes.as_ptr(), 0, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INSUFFICIENT_DATA);
    }

    /// A subgroup the chart cannot use used to be skipped. The points carry no
    /// index, so every later point then sat against the wrong input row.
    #[test]
    fn ffi_attribute_charts_reject_a_row_instead_of_skipping_it() {
        let defectives: [u64; 3] = [3, 5, 2];
        let sizes: [u64; 3] = [100, 0, 100];
        let mut result = empty_attribute_result();
        let rc = unsafe { insight_p_chart(defectives.as_ptr(), sizes.as_ptr(), 3, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);

        let counts: [u64; 3] = [3, 101, 2];
        let mut result = empty_attribute_result();
        let rc = unsafe { insight_np_chart(counts.as_ptr(), 3, 100, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);

        let defects: [u64; 3] = [3, 5, 2];
        let units: [f64; 3] = [1.0, 0.0, 1.0];
        let mut result = empty_attribute_result();
        let rc = unsafe { insight_u_chart(defects.as_ptr(), units.as_ptr(), 3, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);

        let defectives: [u64; 4] = [3, 5, 2, 4];
        let sizes: [u64; 4] = [100, 0, 100, 100];
        let mut laney = CLaneyChartResult {
            bar: 0.0,
            phi: 0.0,
            points: ptr::null_mut(),
            n_points: 0,
        };
        let rc =
            unsafe { insight_laney_p_chart(defectives.as_ptr(), sizes.as_ptr(), 4, &mut laney) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
    }

    #[test]
    fn ffi_variables_charts_take_the_crate_range_and_reject_non_finite_values() {
        // 12 is beyond the bound this crate used to restate for itself.
        let data: Vec<f64> = (0..36).map(|i| 10.0 + 0.1 * (i % 7) as f64).collect();
        let mut result = empty_variables_result();
        let rc = unsafe { insight_xbar_r_chart(data.as_ptr(), 3, 12, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        unsafe { insight_free_variables_chart_result(&mut result) };

        // 26 is beyond the crate's own tables, and the crate says so.
        let mut result = empty_variables_result();
        let rc = unsafe { insight_xbar_s_chart(data.as_ptr(), 1, 26, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);

        let mut bad = data.clone();
        bad[14] = f64::NAN;
        let mut result = empty_variables_result();
        let rc = unsafe { insight_xbar_r_chart(bad.as_ptr(), 3, 12, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
        let mut result = empty_variables_result();
        let rc = unsafe { insight_individual_mr_chart(bad.as_ptr(), 20, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
    }

    #[test]
    fn ffi_p_chart_null_pointer() {
        let sample_sizes: [u64; 1] = [100];
        let mut result = empty_attribute_result();
        let rc = unsafe { insight_p_chart(ptr::null(), sample_sizes.as_ptr(), 1, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_np_chart_basic() {
        let counts: [u64; 4] = [3, 5, 2, 4];
        let mut result = empty_attribute_result();

        let rc = unsafe { insight_np_chart(counts.as_ptr(), 4, 100, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_points, 4);

        unsafe { insight_free_attribute_chart_result(&mut result) };
    }

    #[test]
    fn ffi_np_chart_invalid_sample_size() {
        let counts: [u64; 1] = [3];
        let mut result = empty_attribute_result();
        let rc = unsafe { insight_np_chart(counts.as_ptr(), 1, 0, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
    }

    #[test]
    fn ffi_c_chart_basic() {
        let counts: [u64; 5] = [2, 3, 1, 4, 2];
        let mut result = empty_attribute_result();

        let rc = unsafe { insight_c_chart(counts.as_ptr(), 5, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_points, 5);

        let points = unsafe { slice::from_raw_parts(result.points, result.n_points as usize) };
        assert!((points[0].cl - 2.4).abs() < 1e-9);

        unsafe { insight_free_attribute_chart_result(&mut result) };
    }

    #[test]
    fn ffi_c_chart_null_pointer() {
        let mut result = empty_attribute_result();
        let rc = unsafe { insight_c_chart(ptr::null(), 5, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_u_chart_basic() {
        let defects: [u64; 4] = [2, 3, 1, 4];
        let units: [f64; 4] = [10.0, 12.0, 8.0, 15.0];
        let mut result = empty_attribute_result();

        let rc = unsafe { insight_u_chart(defects.as_ptr(), units.as_ptr(), 4, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_points, 4);

        unsafe { insight_free_attribute_chart_result(&mut result) };
    }

    #[test]
    fn ffi_u_chart_null_pointer() {
        let units: [f64; 1] = [10.0];
        let mut result = empty_attribute_result();
        let rc = unsafe { insight_u_chart(ptr::null(), units.as_ptr(), 1, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    fn empty_laney_result() -> CLaneyChartResult {
        CLaneyChartResult {
            bar: 0.0,
            phi: 0.0,
            points: ptr::null_mut(),
            n_points: 0,
        }
    }

    fn empty_rare_event_result() -> CRareEventChartResult {
        CRareEventChartResult {
            bar: 0.0,
            points: ptr::null_mut(),
            n_points: 0,
        }
    }

    #[test]
    fn ffi_laney_p_chart_basic() {
        let defectives: [u64; 5] = [3, 5, 2, 4, 6];
        let sample_sizes: [u64; 5] = [100, 150, 80, 120, 200];
        let mut result = empty_laney_result();

        let rc = unsafe {
            insight_laney_p_chart(defectives.as_ptr(), sample_sizes.as_ptr(), 5, &mut result)
        };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_points, 5);
        assert!(result.bar > 0.0);

        unsafe { insight_free_laney_chart_result(&mut result) };
    }

    #[test]
    fn ffi_laney_p_chart_insufficient_data() {
        let defectives: [u64; 2] = [3, 5];
        let sample_sizes: [u64; 2] = [100, 100];
        let mut result = empty_laney_result();
        let rc = unsafe {
            insight_laney_p_chart(defectives.as_ptr(), sample_sizes.as_ptr(), 2, &mut result)
        };
        assert_eq!(rc, INSIGHT_ERR_INSUFFICIENT_DATA);
    }

    #[test]
    fn ffi_laney_p_chart_null_pointer() {
        let sample_sizes: [u64; 3] = [100, 100, 100];
        let mut result = empty_laney_result();
        let rc =
            unsafe { insight_laney_p_chart(ptr::null(), sample_sizes.as_ptr(), 3, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_laney_u_chart_basic() {
        let defects: [u64; 5] = [2, 3, 1, 4, 2];
        let units: [f64; 5] = [10.0, 12.0, 8.0, 15.0, 11.0];
        let mut result = empty_laney_result();

        let rc = unsafe { insight_laney_u_chart(defects.as_ptr(), units.as_ptr(), 5, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_points, 5);

        unsafe { insight_free_laney_chart_result(&mut result) };
    }

    #[test]
    fn ffi_laney_u_chart_null_pointer() {
        let units: [f64; 3] = [10.0, 12.0, 8.0];
        let mut result = empty_laney_result();
        let rc = unsafe { insight_laney_u_chart(ptr::null(), units.as_ptr(), 3, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_g_chart_basic() {
        let gaps: [f64; 5] = [100.0, 120.0, 95.0, 110.0, 105.0];
        let mut result = empty_rare_event_result();

        let rc = unsafe { insight_g_chart(gaps.as_ptr(), 5, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_points, 5);
        assert!(result.bar > 0.0);

        let points = unsafe { slice::from_raw_parts(result.points, result.n_points as usize) };
        assert!(points[0].ucl > points[0].cl);

        unsafe { insight_free_rare_event_chart_result(&mut result) };
    }

    #[test]
    fn ffi_g_chart_insufficient_data() {
        let gaps: [f64; 2] = [100.0, 120.0];
        let mut result = empty_rare_event_result();
        let rc = unsafe { insight_g_chart(gaps.as_ptr(), 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INSUFFICIENT_DATA);
    }

    #[test]
    fn ffi_g_chart_null_pointer() {
        let mut result = empty_rare_event_result();
        let rc = unsafe { insight_g_chart(ptr::null(), 5, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_t_chart_basic() {
        let times: [f64; 5] = [24.0, 30.0, 18.0, 26.0, 22.0];
        let mut result = empty_rare_event_result();

        let rc = unsafe { insight_t_chart(times.as_ptr(), 5, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.n_points, 5);
        assert!(result.bar > 0.0);

        unsafe { insight_free_rare_event_chart_result(&mut result) };
    }

    #[test]
    fn ffi_t_chart_insufficient_data() {
        let times: [f64; 2] = [24.0, 30.0];
        let mut result = empty_rare_event_result();
        let rc = unsafe { insight_t_chart(times.as_ptr(), 2, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INSUFFICIENT_DATA);
    }

    #[test]
    fn ffi_t_chart_null_pointer() {
        let mut result = empty_rare_event_result();
        let rc = unsafe { insight_t_chart(ptr::null(), 5, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    fn empty_capability_result() -> CCapabilityIndices {
        CCapabilityIndices {
            cp: 0.0,
            cpk: 0.0,
            cpu: 0.0,
            cpl: 0.0,
            pp: 0.0,
            ppk: 0.0,
            ppu: 0.0,
            ppl: 0.0,
            cpm: 0.0,
            mean: 0.0,
            std_dev_within: 0.0,
            std_dev_overall: 0.0,
        }
    }

    #[test]
    fn ffi_process_capability_two_sided() {
        let data: Vec<f64> = vec![
            208.0, 209.0, 210.0, 211.0, 212.0, 208.5, 209.5, 210.5, 211.5, 210.0, 209.0, 211.0,
            210.0, 209.5, 210.5, 210.0, 210.0, 210.0, 209.0, 211.0,
        ];
        let mut result = empty_capability_result();

        let rc = unsafe {
            insight_process_capability(
                data.as_ptr(),
                data.len() as u32,
                220.0,
                200.0,
                210.0,
                2.0,
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_OK);
        assert!((result.cp - 1.6667).abs() < 0.001);
        assert!(result.cpk > 0.0);
        assert!(!result.cpm.is_nan());

        // Without a target there is no Cpm -- not one against the midpoint.
        let rc = unsafe {
            insight_process_capability(
                data.as_ptr(),
                data.len() as u32,
                220.0,
                200.0,
                f64::NAN,
                2.0,
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_OK);
        assert!(result.cpm.is_nan());
    }

    #[test]
    fn ffi_process_capability_one_sided_no_lsl() {
        let data: Vec<f64> = vec![7.0, 8.0, 9.0, 7.5, 8.5, 8.0, 7.0, 9.0, 8.0, 8.5];
        let mut result = empty_capability_result();

        let rc = unsafe {
            insight_process_capability(
                data.as_ptr(),
                data.len() as u32,
                10.0,
                f64::NAN,
                f64::NAN,
                0.5,
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_OK);
        assert!(!result.cpu.is_nan());
        assert!(result.cpl.is_nan());
        assert!(result.cp.is_nan(), "Cp requires both limits");
    }

    #[test]
    fn imr_sigma_hat_feeds_process_capability_the_short_term_indices() {
        // The route to Cp/Cpk for individual data: the I-MR chart estimates
        // the within sigma (MR-bar / d2), and the capability entry point
        // takes it explicitly. Without this field the caller would have to
        // rebuild the estimate from the MR center line by hand.
        let data: Vec<f64> = vec![
            208.0, 209.0, 210.0, 211.0, 212.0, 208.5, 209.5, 210.5, 211.5, 210.0,
        ];
        let mut chart = empty_variables_result();
        let rc =
            unsafe { insight_individual_mr_chart(data.as_ptr(), data.len() as u32, &mut chart) };
        assert_eq!(rc, INSIGHT_OK);
        assert!(chart.sigma_hat.is_finite() && chart.sigma_hat > 0.0);
        let expected = {
            use u_analytics::spc::ControlChart;
            let mut c = u_analytics::spc::IndividualMRChart::new();
            for &v in &data {
                c.add_sample(&[v]).unwrap();
            }
            c.sigma_hat().unwrap()
        };
        assert!((chart.sigma_hat - expected).abs() < 1e-12);
        unsafe { insight_free_variables_chart_result(&mut chart) };

        let mut result = empty_capability_result();
        let rc = unsafe {
            insight_process_capability(
                data.as_ptr(),
                data.len() as u32,
                220.0,
                200.0,
                f64::NAN,
                chart.sigma_hat,
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_OK);
        assert!((result.std_dev_within - expected).abs() < 1e-12);
        assert!((result.cp - 20.0 / (6.0 * expected)).abs() < 1e-9);
        assert!(result.cp != result.pp);
    }

    #[test]
    fn ffi_process_capability_without_sigma_within_reports_long_term_only() {
        // This test used to assert `cp == pp` here -- pinning the overall
        // sigma being reported under the short-term names. Without a within
        // sigma there is no short-term index to report.
        let data: Vec<f64> = vec![
            208.0, 209.0, 210.0, 211.0, 212.0, 208.5, 209.5, 210.5, 211.5, 210.0,
        ];
        let mut result = empty_capability_result();

        let rc = unsafe {
            insight_process_capability(
                data.as_ptr(),
                data.len() as u32,
                220.0,
                200.0,
                f64::NAN,
                f64::NAN, // no within sigma
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_OK);
        assert!(result.cp.is_nan(), "no within sigma, no Cp");
        assert!(result.cpk.is_nan());
        assert!(result.cpu.is_nan());
        assert!(result.cpl.is_nan());
        assert!(result.std_dev_within.is_nan());
        assert!(result.pp.is_finite() && result.pp > 0.0);
        assert!(result.ppk.is_finite() && result.ppk > 0.0);
        assert!(result.std_dev_overall.is_finite() && result.std_dev_overall > 0.0);
    }

    #[test]
    fn ffi_process_capability_no_limits() {
        let data = [1.0_f64, 2.0, 3.0];
        let mut result = empty_capability_result();
        let rc = unsafe {
            insight_process_capability(
                data.as_ptr(),
                3,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                1.0,
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
    }

    #[test]
    fn ffi_process_capability_null_pointer() {
        let mut result = empty_capability_result();
        let rc = unsafe {
            insight_process_capability(ptr::null(), 10, 10.0, 0.0, f64::NAN, 1.0, &mut result)
        };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    fn empty_boxcox_result() -> CBoxCoxCapabilityResult {
        CBoxCoxCapabilityResult {
            lambda: 0.0,
            indices: empty_capability_result(),
            lambda_at_bound: 9,
        }
    }

    #[test]
    fn ffi_boxcox_capability_basic() {
        let data: Vec<f64> = (1..=20).map(|i| (i as f64 * 0.3_f64).exp()).collect();
        let mut result = empty_boxcox_result();

        let rc = unsafe {
            insight_boxcox_capability(
                data.as_ptr(),
                data.len() as u32,
                100.0,
                1.0,
                f64::NAN,
                f64::NAN,
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_OK);
        assert!(result.lambda.is_finite());
        assert_eq!(result.lambda_at_bound, 0);
        assert!(!result.indices.ppk.is_nan());
        // Short-term indices are absent by construction on this path, and
        // absence is NaN across this FFI surface.
        assert!(result.indices.cp.is_nan());
        assert!(result.indices.cpk.is_nan());
    }

    #[test]
    fn ffi_boxcox_capability_narrow_range_and_no_limits() {
        // Normal quantiles through the inverse Box-Cox at lambda = 4.
        let n = 100;
        let data: Vec<f64> = (1..=n)
            .map(|i| {
                let p = (i as f64 - 0.5) / n as f64;
                let z = 10.0 + 2.0 * u_numflow::special::inverse_normal_cdf(p);
                (4.0 * z + 1.0).powf(0.25)
            })
            .collect();
        let mut result = empty_boxcox_result();
        let rc = unsafe {
            insight_boxcox_capability(
                data.as_ptr(),
                n as u32,
                f64::NAN,
                f64::NAN,
                -2.0,
                2.0,
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_OK);
        assert_eq!(result.lambda, 2.0);
        assert_eq!(result.lambda_at_bound, 1);
        assert!(result.indices.ppk.is_nan() && result.indices.mean.is_nan());

        // A half-given range is not the default — it is refused.
        let rc = unsafe {
            insight_boxcox_capability(
                data.as_ptr(),
                n as u32,
                40.0,
                f64::NAN,
                -2.0,
                f64::NAN,
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_ERR_INVALID_PARAM);
    }

    #[test]
    fn ffi_boxcox_capability_insufficient_data() {
        let data = [1.0_f64, 2.0];
        let mut result = empty_boxcox_result();
        let rc = unsafe {
            insight_boxcox_capability(
                data.as_ptr(),
                2,
                100.0,
                1.0,
                f64::NAN,
                f64::NAN,
                &mut result,
            )
        };
        assert_eq!(rc, INSIGHT_ERR_INSUFFICIENT_DATA);
    }

    #[test]
    fn ffi_boxcox_capability_null_pointer() {
        let mut result = empty_boxcox_result();
        let rc = unsafe {
            insight_boxcox_capability(ptr::null(), 10, 100.0, 1.0, f64::NAN, f64::NAN, &mut result)
        };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_percentile_capability_basic() {
        let data: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64) * 0.1).collect();
        let mut result = CPercentileCapabilityResult {
            cp_star: 0.0,
            cpk_star: 0.0,
            cpu_star: 0.0,
            cpl_star: 0.0,
            median: 0.0,
            percentile_lower: 0.0,
            percentile_upper: 0.0,
        };

        let rc = unsafe {
            insight_percentile_capability(data.as_ptr(), data.len() as u32, 5.0, 15.0, &mut result)
        };
        assert_eq!(rc, INSIGHT_OK);
        assert!(!result.cp_star.is_nan());
        assert!(result.median > 0.0);
    }

    #[test]
    fn ffi_percentile_capability_insufficient_data() {
        let data = [1.0_f64, 2.0, 3.0];
        let mut result = CPercentileCapabilityResult {
            cp_star: 0.0,
            cpk_star: 0.0,
            cpu_star: 0.0,
            cpl_star: 0.0,
            median: 0.0,
            percentile_lower: 0.0,
            percentile_upper: 0.0,
        };
        let rc = unsafe { insight_percentile_capability(data.as_ptr(), 3, 5.0, 15.0, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INVALID_INPUT);
    }

    #[test]
    fn ffi_percentile_capability_null_pointer() {
        let mut result = CPercentileCapabilityResult {
            cp_star: 0.0,
            cpk_star: 0.0,
            cpu_star: 0.0,
            cpl_star: 0.0,
            median: 0.0,
            percentile_lower: 0.0,
            percentile_upper: 0.0,
        };
        let rc = unsafe { insight_percentile_capability(ptr::null(), 10, 5.0, 15.0, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_sigma_to_ppm_six_sigma() {
        let ppm = insight_sigma_to_ppm(6.0);
        assert!((ppm - 3.4).abs() < 1.0);
    }

    #[test]
    fn ffi_ppm_to_sigma_six_sigma() {
        let sigma = insight_ppm_to_sigma(3.4);
        assert!((sigma - 6.0).abs() < 0.1);
    }

    #[test]
    fn ffi_ppm_to_sigma_out_of_range() {
        let sigma = insight_ppm_to_sigma(-1.0);
        assert!(sigma.is_nan());
        let sigma = insight_ppm_to_sigma(2_000_000.0);
        assert!(sigma.is_nan());
    }

    #[test]
    fn ffi_weibull_mle_basic() {
        let data: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let mut result = CWeibullMleResult {
            shape: 0.0,
            scale: 0.0,
            log_likelihood: 0.0,
            iterations: 0,
        };

        let rc = unsafe { insight_weibull_mle(data.as_ptr(), data.len() as u32, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert!(result.shape > 0.0);
        assert!(result.scale > 0.0);
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn ffi_weibull_mle_insufficient_data() {
        let data = [10.0_f64];
        let mut result = CWeibullMleResult {
            shape: 0.0,
            scale: 0.0,
            log_likelihood: 0.0,
            iterations: 0,
        };
        let rc = unsafe { insight_weibull_mle(data.as_ptr(), 1, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_INSUFFICIENT_DATA);
    }

    #[test]
    fn ffi_weibull_mle_null_pointer() {
        let mut result = CWeibullMleResult {
            shape: 0.0,
            scale: 0.0,
            log_likelihood: 0.0,
            iterations: 0,
        };
        let rc = unsafe { insight_weibull_mle(ptr::null(), 10, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_weibull_mrr_basic() {
        let data: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let mut result = CWeibullMrrResult {
            shape: 0.0,
            scale: 0.0,
            r_squared: 0.0,
        };

        let rc = unsafe { insight_weibull_mrr(data.as_ptr(), data.len() as u32, &mut result) };
        assert_eq!(rc, INSIGHT_OK);
        assert!(result.shape > 0.0);
        assert!(result.scale > 0.0);
        assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
    }

    #[test]
    fn ffi_weibull_mrr_null_pointer() {
        let mut result = CWeibullMrrResult {
            shape: 0.0,
            scale: 0.0,
            r_squared: 0.0,
        };
        let rc = unsafe { insight_weibull_mrr(ptr::null(), 10, &mut result) };
        assert_eq!(rc, INSIGHT_ERR_NULL_PTR);
    }

    #[test]
    fn ffi_weibull_reliability_basic() {
        let r0 = insight_weibull_reliability(2.0, 100.0, 0.0);
        assert!((r0 - 1.0).abs() < 1e-10);
        let r_eta = insight_weibull_reliability(2.0, 100.0, 100.0);
        assert!((r_eta - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn ffi_weibull_reliability_invalid_params() {
        let r = insight_weibull_reliability(-1.0, 100.0, 50.0);
        assert!(r.is_nan());
    }

    #[test]
    fn ffi_weibull_hazard_rate_basic() {
        let h1 = insight_weibull_hazard_rate(2.0, 100.0, 50.0);
        let h2 = insight_weibull_hazard_rate(2.0, 100.0, 80.0);
        assert!(h1 > 0.0);
        assert!(h2 > h1, "hazard rate should increase for shape > 1");
    }

    #[test]
    fn ffi_weibull_mtbf_basic() {
        let mtbf = insight_weibull_mtbf(1.0, 50.0);
        assert!(
            (mtbf - 50.0).abs() < 1e-8,
            "MTBF should equal scale when shape=1"
        );
    }

    #[test]
    fn ffi_weibull_mtbf_invalid_params() {
        let mtbf = insight_weibull_mtbf(0.0, 50.0);
        assert!(mtbf.is_nan());
    }

    #[test]
    fn ffi_weibull_time_to_reliability_basic() {
        let t = insight_weibull_time_to_reliability(2.0, 100.0, 0.9);
        assert!(t > 0.0 && t < 100.0);
    }

    #[test]
    fn ffi_weibull_time_to_reliability_out_of_range() {
        let t = insight_weibull_time_to_reliability(2.0, 100.0, 1.5);
        assert!(t.is_nan());
    }

    #[test]
    fn ffi_weibull_b_life_ordering() {
        let b5 = insight_weibull_b_life(2.0, 100.0, 0.05);
        let b10 = insight_weibull_b_life(2.0, 100.0, 0.10);
        let b50 = insight_weibull_b_life(2.0, 100.0, 0.50);
        assert!(b5 < b10 && b10 < b50);
    }

    #[test]
    fn ffi_weibull_b_life_out_of_range() {
        let b = insight_weibull_b_life(2.0, 100.0, 0.0);
        assert!(b.is_nan());
    }
}
