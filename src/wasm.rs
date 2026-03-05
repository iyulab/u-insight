//! WASM bindings for u-insight.
//!
//! Exposes statistical analysis and profiling functionality for JavaScript/TypeScript.
//!
//! # API
//!
//! - `describe(data_json)` — Descriptive statistics per column
//! - `correlation_matrix(data_json)` — Pearson correlation matrix
//! - `kmeans(data_json, k)` — K-Means++ clustering
//! - `pca(data_json, n_components)` — Principal Component Analysis
//!
//! # Input Formats
//!
//! `describe` / `correlation_matrix` accept column-major JSON:
//! ```json
//! { "col1": [1.0, 2.0, 3.0], "col2": [4.0, 5.0, 6.0] }
//! ```
//!
//! `kmeans` / `pca` accept row-major JSON:
//! ```json
//! [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
//! ```

use serde::Serialize;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

// ── Serializable DTOs ─────────────────────────────────────────────────

/// Descriptive statistics for a single numeric column.
#[derive(Serialize)]
struct NumericStats {
    count: usize,
    null_count: usize,
    missing_pct: f64,
    min: f64,
    max: f64,
    mean: f64,
    median: f64,
    std_dev: f64,
    variance: f64,
    skewness: f64,
    kurtosis: f64,
    q1: f64,
    q3: f64,
    iqr: f64,
    p5: f64,
    p95: f64,
}

/// Summary statistics for a boolean column.
#[derive(Serialize)]
struct BoolStats {
    count: usize,
    null_count: usize,
    missing_pct: f64,
    true_count: usize,
    false_count: usize,
    true_ratio: f64,
}

/// Summary statistics for a categorical column.
#[derive(Serialize)]
struct CatStats {
    count: usize,
    null_count: usize,
    missing_pct: f64,
    distinct_count: usize,
    top_values: Vec<(String, usize)>,
    mode_ratio: f64,
    is_constant: bool,
}

/// Summary statistics for a text column.
#[derive(Serialize)]
struct TextStats {
    count: usize,
    null_count: usize,
    missing_pct: f64,
    distinct_count: usize,
    min_length: usize,
    max_length: usize,
    mean_length: f64,
    empty_count: usize,
}

/// Column profile result returned by `describe`.
#[derive(Serialize)]
struct ColumnResult {
    name: String,
    data_type: String,
    numeric: Option<NumericStats>,
    boolean: Option<BoolStats>,
    categorical: Option<CatStats>,
    text: Option<TextStats>,
}

/// Result of correlation analysis.
#[derive(Serialize)]
struct CorrelationResult {
    /// Column names (in order).
    names: Vec<String>,
    /// Flattened n×n matrix (row-major).
    matrix: Vec<f64>,
    /// n — dimension of the square matrix.
    n: usize,
    /// Pairs with |r| > 0.7, sorted by |r| descending.
    high_pairs: Vec<CorrelationPairDto>,
}

#[derive(Serialize)]
struct CorrelationPairDto {
    col_a: String,
    col_b: String,
    r: f64,
    p_value: f64,
}

/// Result of K-Means clustering.
#[derive(Serialize)]
struct KMeansDto {
    k: usize,
    labels: Vec<usize>,
    centroids: Vec<Vec<f64>>,
    wcss: f64,
    iterations: usize,
    cluster_sizes: Vec<usize>,
}

/// Result of PCA.
#[derive(Serialize)]
struct PcaDto {
    n_components: usize,
    n_features: usize,
    eigenvalues: Vec<f64>,
    explained_variance_ratio: Vec<f64>,
    cumulative_variance_ratio: Vec<f64>,
    loadings: Vec<Vec<f64>>,
    scores: Vec<Vec<f64>>,
    means: Vec<f64>,
    stds: Vec<f64>,
}

// ── WASM entry points ─────────────────────────────────────────────────

/// Returns descriptive statistics for each column in a column-major dataset.
///
/// # Input
/// ```json
/// { "col1": [1.0, 2.0, 3.0], "col2": [4.0, 5.0, 6.0] }
/// ```
///
/// # Output
/// Array of column profile objects, one per column.
#[wasm_bindgen]
pub fn describe(data_json: JsValue) -> Result<JsValue, JsValue> {
    let raw: HashMap<String, Vec<f64>> =
        serde_wasm_bindgen::from_value(data_json).map_err(js_err)?;

    if raw.is_empty() {
        return Err(js_err("data_json must contain at least one column"));
    }

    // Build a DataFrame from numeric columns using CsvParser isn't practical here;
    // instead use profiling directly via the Column API.
    use crate::dataframe::{Column, DataFrame, ValidityBitmap};
    use crate::profiling::profile_dataframe;

    let mut df = DataFrame::new();
    // Sort keys for deterministic column order
    let mut keys: Vec<String> = raw.keys().cloned().collect();
    keys.sort();

    for key in &keys {
        let values = raw[key].clone();
        let n = values.len();
        let validity = ValidityBitmap::all_valid(n);
        let col = Column::numeric(values, validity);
        df.add_column(key.clone(), col).map_err(js_err)?;
    }

    let profiles = profile_dataframe(&df);

    let results: Vec<ColumnResult> = profiles
        .into_iter()
        .map(|p| {
            let data_type = format!("{:?}", p.data_type);
            let numeric = p.numeric.map(|n| NumericStats {
                count: p.row_count,
                null_count: p.null_count,
                missing_pct: p.missing_pct,
                min: n.min,
                max: n.max,
                mean: n.mean,
                median: n.median,
                std_dev: n.std_dev,
                variance: n.variance,
                skewness: n.skewness,
                kurtosis: n.kurtosis,
                q1: n.q1,
                q3: n.q3,
                iqr: n.iqr,
                p5: n.p5,
                p95: n.p95,
            });
            let boolean = p.boolean.map(|b| BoolStats {
                count: p.row_count,
                null_count: p.null_count,
                missing_pct: p.missing_pct,
                true_count: b.true_count,
                false_count: b.false_count,
                true_ratio: b.true_ratio,
            });
            let categorical = p.categorical.map(|c| CatStats {
                count: p.row_count,
                null_count: p.null_count,
                missing_pct: p.missing_pct,
                distinct_count: c.distinct_count,
                top_values: c.top_values,
                mode_ratio: c.mode_ratio,
                is_constant: c.is_constant,
            });
            let text = p.text.map(|t| TextStats {
                count: p.row_count,
                null_count: p.null_count,
                missing_pct: p.missing_pct,
                distinct_count: t.distinct_count,
                min_length: t.min_length,
                max_length: t.max_length,
                mean_length: t.mean_length,
                empty_count: t.empty_count,
            });
            ColumnResult {
                name: p.name,
                data_type,
                numeric,
                boolean,
                categorical,
                text,
            }
        })
        .collect();

    serde_wasm_bindgen::to_value(&results).map_err(js_err)
}

/// Computes a Pearson correlation matrix for a column-major dataset.
///
/// # Input
/// ```json
/// { "col1": [1.0, 2.0, 3.0], "col2": [4.0, 5.0, 6.0] }
/// ```
///
/// # Output
/// `{ names, matrix (flattened n×n), n, high_pairs }`
#[wasm_bindgen]
pub fn correlation_matrix(data_json: JsValue) -> Result<JsValue, JsValue> {
    let raw: HashMap<String, Vec<f64>> =
        serde_wasm_bindgen::from_value(data_json).map_err(js_err)?;

    if raw.is_empty() {
        return Err(js_err("data_json must contain at least one column"));
    }

    // Sort keys for deterministic order
    let mut names: Vec<String> = raw.keys().cloned().collect();
    names.sort();

    let columns: Vec<Vec<f64>> = names.iter().map(|n| raw[n].clone()).collect();

    use crate::analysis::{correlation_analysis, CorrelationConfig};
    let config = CorrelationConfig::default();
    let result = correlation_analysis(&columns, &names, &config).map_err(js_err)?;

    let n = names.len();
    // Flatten the n×n matrix to a Vec<f64>
    let mut flat_matrix = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            flat_matrix.push(result.matrix.get(i, j));
        }
    }

    let high_pairs: Vec<CorrelationPairDto> = result
        .high_pairs
        .into_iter()
        .map(|p| CorrelationPairDto {
            col_a: p.col_a,
            col_b: p.col_b,
            r: p.r,
            p_value: p.p_value,
        })
        .collect();

    let dto = CorrelationResult {
        names,
        matrix: flat_matrix,
        n,
        high_pairs,
    };

    serde_wasm_bindgen::to_value(&dto).map_err(js_err)
}

/// Runs K-Means++ clustering on row-major data.
///
/// # Input
/// ```json
/// [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
/// ```
///
/// # Output
/// `{ k, labels, centroids, wcss, iterations, cluster_sizes }`
#[wasm_bindgen]
pub fn kmeans(data_json: JsValue, k: usize) -> Result<JsValue, JsValue> {
    let data: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(data_json).map_err(js_err)?;

    use crate::clustering::{kmeans as kmeans_fn, KMeansConfig};
    let config = KMeansConfig::new(k);
    let result = kmeans_fn(&data, &config).map_err(js_err)?;

    let dto = KMeansDto {
        k: result.k,
        labels: result.labels,
        centroids: result.centroids,
        wcss: result.wcss,
        iterations: result.iterations,
        cluster_sizes: result.cluster_sizes,
    };

    serde_wasm_bindgen::to_value(&dto).map_err(js_err)
}

/// Runs Principal Component Analysis on row-major data.
///
/// # Input
/// ```json
/// [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]]
/// ```
///
/// # Output
/// `{ n_components, n_features, eigenvalues, explained_variance_ratio, ... }`
#[wasm_bindgen]
pub fn pca(data_json: JsValue, n_components: usize) -> Result<JsValue, JsValue> {
    let data: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(data_json).map_err(js_err)?;

    use crate::pca::{pca as pca_fn, PcaConfig};
    let config = PcaConfig::new(n_components);
    let result = pca_fn(&data, &config).map_err(js_err)?;

    let dto = PcaDto {
        n_components: result.n_components,
        n_features: result.n_features,
        eigenvalues: result.eigenvalues,
        explained_variance_ratio: result.explained_variance_ratio,
        cumulative_variance_ratio: result.cumulative_variance_ratio,
        loadings: result.loadings,
        scores: result.scores,
        means: result.means,
        stds: result.stds,
    };

    serde_wasm_bindgen::to_value(&dto).map_err(js_err)
}
