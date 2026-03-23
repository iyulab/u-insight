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
//! - `dbscan(data_json, config_json)` — DBSCAN density-based clustering
//! - `hierarchical(data_json, config_json)` — Hierarchical agglomerative clustering
//! - `isolation_forest(data_json, config_json)` — Isolation Forest anomaly detection
//! - `lof(data_json, config_json)` — Local Outlier Factor anomaly detection
//! - `distribution_analysis(data_json, config_json)` — Distribution analysis + normality tests
//! - `regression(data_json)` — OLS regression (simple or multiple)
//! - `feature_importance(data_json)` — Feature importance (permutation / ANOVA / mutual info)
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

use serde::{Deserialize, Serialize};
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
///
/// Accepts mixed-type columns (numbers, booleans, strings, null):
/// ```json
/// { "age": [30, 25, null], "name": ["Alice", "Bob", null], "active": [true, false, true] }
/// ```
///
/// Also accepts numeric-only columns (backward-compatible):
/// ```json
/// { "col1": [1.0, 2.0, 3.0], "col2": [4.0, 5.0, 6.0] }
/// ```
///
/// # Output
/// Array of column profile objects, one per column.
#[wasm_bindgen]
pub fn describe(data_json: JsValue) -> Result<JsValue, JsValue> {
    use crate::json_parser::JsonParser;
    use crate::profiling::profile_dataframe;

    let raw: serde_json::Value = serde_wasm_bindgen::from_value(data_json).map_err(js_err)?;

    let df = JsonParser::new().parse_value(&raw).map_err(js_err)?;

    if df.is_empty() {
        return Err(js_err("data_json must contain at least one column"));
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

// ── DBSCAN ──────────────────────────────────────────────────────────

/// DBSCAN configuration input.
#[derive(Deserialize)]
struct DbscanConfigDto {
    epsilon: f64,
    min_samples: usize,
}

/// DBSCAN clustering result.
#[derive(Serialize)]
struct DbscanDto {
    /// Cluster label per point: null = noise, number = cluster id.
    labels: Vec<Option<usize>>,
    n_clusters: usize,
    noise_count: usize,
    cluster_sizes: Vec<usize>,
    core_points: Vec<bool>,
}

/// Runs DBSCAN density-based clustering on row-major data.
///
/// # Input
///
/// `data_json`: row-major points `[[x,y,...], ...]`
///
/// `config_json`: `{ "epsilon": 1.5, "min_samples": 3 }`
///
/// # Output
///
/// `{ labels, n_clusters, noise_count, cluster_sizes, core_points }`
#[wasm_bindgen]
pub fn dbscan(data_json: JsValue, config_json: JsValue) -> Result<JsValue, JsValue> {
    let data: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(data_json).map_err(js_err)?;
    let cfg: DbscanConfigDto = serde_wasm_bindgen::from_value(config_json).map_err(js_err)?;

    use crate::clustering::{dbscan as dbscan_fn, DbscanConfig};
    let config = DbscanConfig::new(cfg.epsilon, cfg.min_samples);
    let result = dbscan_fn(&data, &config).map_err(js_err)?;

    let dto = DbscanDto {
        labels: result.labels,
        n_clusters: result.n_clusters,
        noise_count: result.noise_count,
        cluster_sizes: result.cluster_sizes,
        core_points: result.core_points,
    };

    serde_wasm_bindgen::to_value(&dto).map_err(js_err)
}

// ── Hierarchical Clustering ─────────────────────────────────────────

/// Hierarchical clustering configuration input.
#[derive(Deserialize)]
struct HierarchicalConfigDto {
    /// Linkage: "single", "complete", "average", "ward". Default: "ward".
    #[serde(default = "default_linkage")]
    linkage: String,
    /// Number of flat clusters (mutually exclusive with distance_threshold).
    n_clusters: Option<usize>,
    /// Distance threshold for dendrogram cut (mutually exclusive with n_clusters).
    distance_threshold: Option<f64>,
}

fn default_linkage() -> String {
    "ward".into()
}

/// A single merge step in the dendrogram.
#[derive(Serialize)]
struct MergeDto {
    cluster_a: usize,
    cluster_b: usize,
    distance: f64,
    size: usize,
}

/// Hierarchical clustering result.
#[derive(Serialize)]
struct HierarchicalDto {
    merges: Vec<MergeDto>,
    labels: Option<Vec<usize>>,
    n_clusters: Option<usize>,
}

/// Runs hierarchical agglomerative clustering on row-major data.
///
/// # Input
///
/// `data_json`: row-major points `[[x,y,...], ...]`
///
/// `config_json`: `{ "linkage": "ward", "n_clusters": 3 }` or
/// `{ "linkage": "single", "distance_threshold": 5.0 }`
///
/// # Output
///
/// `{ merges, labels, n_clusters }`
#[wasm_bindgen]
pub fn hierarchical(data_json: JsValue, config_json: JsValue) -> Result<JsValue, JsValue> {
    let data: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(data_json).map_err(js_err)?;
    let cfg: HierarchicalConfigDto = serde_wasm_bindgen::from_value(config_json).map_err(js_err)?;

    use crate::clustering::{hierarchical as hier_fn, HierarchicalConfig, Linkage};

    let linkage = match cfg.linkage.to_lowercase().as_str() {
        "single" => Linkage::Single,
        "complete" => Linkage::Complete,
        "average" => Linkage::Average,
        "ward" => Linkage::Ward,
        _ => Linkage::Ward,
    };

    let config = match (cfg.n_clusters, cfg.distance_threshold) {
        (Some(k), _) => HierarchicalConfig::with_k(k).linkage(linkage),
        (None, Some(t)) => HierarchicalConfig::with_threshold(t).linkage(linkage),
        (None, None) => {
            return Err(js_err(
                "config must specify either n_clusters or distance_threshold",
            ))
        }
    };

    let result = hier_fn(&data, &config).map_err(js_err)?;

    let dto = HierarchicalDto {
        merges: result
            .merges
            .into_iter()
            .map(|m| MergeDto {
                cluster_a: m.cluster_a,
                cluster_b: m.cluster_b,
                distance: m.distance,
                size: m.size,
            })
            .collect(),
        labels: result.labels,
        n_clusters: result.n_clusters,
    };

    serde_wasm_bindgen::to_value(&dto).map_err(js_err)
}

// ── Isolation Forest ────────────────────────────────────────────────

/// Isolation Forest configuration input.
#[derive(Deserialize)]
struct IsolationForestConfigDto {
    /// Number of trees. Default: 100.
    #[serde(default = "default_n_estimators")]
    n_estimators: usize,
    /// Subsample size per tree. 0 = auto (min(256, n)). Default: 0.
    #[serde(default)]
    max_samples: usize,
    /// Expected contamination rate (0.0–1.0). Default: 0.1.
    #[serde(default = "default_contamination")]
    contamination: f64,
    /// Random seed. Default: 42.
    #[serde(default = "default_seed")]
    seed: Option<u64>,
}

fn default_n_estimators() -> usize {
    100
}
fn default_contamination() -> f64 {
    0.1
}
fn default_seed() -> Option<u64> {
    Some(42)
}

/// Isolation Forest anomaly detection result.
#[derive(Serialize)]
struct IsolationForestDto {
    scores: Vec<f64>,
    anomalies: Vec<bool>,
    threshold: f64,
    anomaly_count: usize,
    anomaly_fraction: f64,
}

/// Runs Isolation Forest anomaly detection on row-major data.
///
/// # Input
///
/// `data_json`: row-major points `[[x,y,...], ...]`
///
/// `config_json`: `{ "n_estimators": 100, "contamination": 0.1, "seed": 42 }`
///
/// # Output
///
/// `{ scores, anomalies, threshold, anomaly_count, anomaly_fraction }`
#[wasm_bindgen]
pub fn isolation_forest(data_json: JsValue, config_json: JsValue) -> Result<JsValue, JsValue> {
    let data: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(data_json).map_err(js_err)?;
    let cfg: IsolationForestConfigDto =
        serde_wasm_bindgen::from_value(config_json).map_err(js_err)?;

    use crate::isolation_forest::{isolation_forest as iforest_fn, IsolationForestConfig};

    let config = IsolationForestConfig {
        n_estimators: cfg.n_estimators,
        max_samples: cfg.max_samples,
        contamination: cfg.contamination,
        seed: cfg.seed,
    };

    let result = iforest_fn(&data, &config).map_err(js_err)?;

    let dto = IsolationForestDto {
        scores: result.scores,
        anomalies: result.anomalies,
        threshold: result.threshold,
        anomaly_count: result.anomaly_count,
        anomaly_fraction: result.anomaly_fraction,
    };

    serde_wasm_bindgen::to_value(&dto).map_err(js_err)
}

// ── LOF (Local Outlier Factor) ──────────────────────────────────────

/// LOF configuration input.
#[derive(Deserialize)]
struct LofConfigDto {
    /// Number of nearest neighbors. Default: 20.
    #[serde(default = "default_lof_k")]
    k: usize,
    /// LOF threshold for outlier classification. Default: 1.5.
    #[serde(default = "default_lof_threshold")]
    threshold: f64,
}

fn default_lof_k() -> usize {
    20
}
fn default_lof_threshold() -> f64 {
    1.5
}

/// LOF anomaly detection result.
#[derive(Serialize)]
struct LofDto {
    scores: Vec<f64>,
    anomalies: Vec<bool>,
    threshold: f64,
    anomaly_count: usize,
    anomaly_fraction: f64,
}

/// Runs Local Outlier Factor anomaly detection on row-major data.
///
/// # Input
///
/// `data_json`: row-major points `[[x,y,...], ...]`
///
/// `config_json`: `{ "k": 20, "threshold": 1.5 }`
///
/// # Output
///
/// `{ scores, anomalies, threshold, anomaly_count, anomaly_fraction }`
#[wasm_bindgen]
pub fn lof(data_json: JsValue, config_json: JsValue) -> Result<JsValue, JsValue> {
    let data: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(data_json).map_err(js_err)?;
    let cfg: LofConfigDto = serde_wasm_bindgen::from_value(config_json).map_err(js_err)?;

    use crate::lof::{lof as lof_fn, LofConfig};

    let config = LofConfig::default().k(cfg.k).threshold(cfg.threshold);
    let result = lof_fn(&data, &config).map_err(js_err)?;

    let dto = LofDto {
        scores: result.scores,
        anomalies: result.anomalies,
        threshold: result.threshold,
        anomaly_count: result.anomaly_count,
        anomaly_fraction: result.anomaly_fraction,
    };

    serde_wasm_bindgen::to_value(&dto).map_err(js_err)
}

// ── Distribution Analysis ───────────────────────────────────────────

/// Distribution analysis configuration input.
#[derive(Deserialize)]
struct DistributionConfigDto {
    /// Bin method: "sturges", "scott", "freedman_diaconis". Default: "freedman_diaconis".
    #[serde(default = "default_bin_method")]
    bin_method: String,
    /// Significance level for normality tests. Default: 0.05.
    #[serde(default = "default_significance")]
    significance_level: f64,
    /// Whether to compute ECDF. Default: true.
    #[serde(default = "default_true")]
    compute_ecdf: bool,
    /// Whether to compute histogram. Default: true.
    #[serde(default = "default_true")]
    compute_histogram: bool,
    /// Whether to compute QQ-plot. Default: true.
    #[serde(default = "default_true")]
    compute_qq_plot: bool,
    /// Whether to fit distributions. Default: false.
    #[serde(default)]
    fit_distributions: bool,
}

fn default_bin_method() -> String {
    "freedman_diaconis".into()
}
fn default_significance() -> f64 {
    0.05
}
fn default_true() -> bool {
    true
}

#[derive(Serialize)]
struct EcdfDto {
    values: Vec<f64>,
    probabilities: Vec<f64>,
}

#[derive(Serialize)]
struct HistogramDto {
    n_bins: usize,
    bin_width: f64,
    edges: Vec<f64>,
    counts: Vec<usize>,
    method: String,
}

#[derive(Serialize)]
struct QQPlotDto {
    theoretical: Vec<f64>,
    sample: Vec<f64>,
}

#[derive(Serialize)]
struct NormalityTestDto {
    statistic: f64,
    p_value: f64,
    rejected: bool,
}

#[derive(Serialize)]
struct NormalityDto {
    ks_test: Option<NormalityTestDto>,
    jarque_bera: Option<NormalityTestDto>,
    shapiro_wilk: Option<NormalityTestDto>,
    anderson_darling: Option<NormalityTestDto>,
    is_normal: bool,
    significance_level: f64,
}

#[derive(Serialize)]
struct FitResultDto {
    distribution: String,
    parameters: Vec<(String, f64)>,
    log_likelihood: f64,
    aic: f64,
    bic: f64,
    n_params: usize,
}

#[derive(Serialize)]
struct DistributionAnalysisDto {
    n: usize,
    ecdf: Option<EcdfDto>,
    histogram: Option<HistogramDto>,
    qq_plot: Option<QQPlotDto>,
    normality: NormalityDto,
    fits: Vec<FitResultDto>,
}

/// Runs distribution analysis on a 1-D numeric array.
///
/// # Input
///
/// `data_json`: flat array `[1.0, 2.0, 3.0, ...]`
///
/// `config_json`: `{ "bin_method": "freedman_diaconis", "significance_level": 0.05,
///   "compute_ecdf": true, "compute_histogram": true, "compute_qq_plot": true,
///   "fit_distributions": false }`
///
/// # Output
///
/// `{ n, ecdf, histogram, qq_plot, normality, fits }`
#[wasm_bindgen]
pub fn distribution_analysis(data_json: JsValue, config_json: JsValue) -> Result<JsValue, JsValue> {
    let data: Vec<f64> = serde_wasm_bindgen::from_value(data_json).map_err(js_err)?;
    let cfg: DistributionConfigDto = serde_wasm_bindgen::from_value(config_json).map_err(js_err)?;

    use crate::distribution::{distribution_analysis as dist_fn, BinMethod, DistributionConfig};

    let bin_method = match cfg.bin_method.to_lowercase().as_str() {
        "sturges" => BinMethod::Sturges,
        "scott" => BinMethod::Scott,
        "freedman_diaconis" => BinMethod::FreedmanDiaconis,
        _ => BinMethod::FreedmanDiaconis,
    };

    let config = DistributionConfig {
        bin_method,
        significance_level: cfg.significance_level,
        compute_ecdf: cfg.compute_ecdf,
        compute_histogram: cfg.compute_histogram,
        compute_qq_plot: cfg.compute_qq_plot,
        fit_distributions: cfg.fit_distributions,
    };

    let result = dist_fn(&data, &config).map_err(js_err)?;

    fn map_test(t: Option<crate::distribution::NormalityTestResult>) -> Option<NormalityTestDto> {
        t.map(|r| NormalityTestDto {
            statistic: r.statistic,
            p_value: r.p_value,
            rejected: r.rejected,
        })
    }

    let dto = DistributionAnalysisDto {
        n: result.n,
        ecdf: result.ecdf.map(|e| EcdfDto {
            values: e.values,
            probabilities: e.probabilities,
        }),
        histogram: result.histogram.map(|h| HistogramDto {
            n_bins: h.n_bins,
            bin_width: h.bin_width,
            edges: h.edges,
            counts: h.counts,
            method: format!("{:?}", h.method),
        }),
        qq_plot: result.qq_plot.map(|q| QQPlotDto {
            theoretical: q.theoretical,
            sample: q.sample,
        }),
        normality: NormalityDto {
            ks_test: map_test(result.normality.ks_test),
            jarque_bera: map_test(result.normality.jarque_bera),
            shapiro_wilk: map_test(result.normality.shapiro_wilk),
            anderson_darling: map_test(result.normality.anderson_darling),
            is_normal: result.normality.is_normal,
            significance_level: result.normality.significance_level,
        },
        fits: result
            .fits
            .into_iter()
            .map(|f| FitResultDto {
                distribution: f.distribution,
                parameters: f.parameters,
                log_likelihood: f.log_likelihood,
                aic: f.aic,
                bic: f.bic,
                n_params: f.n_params,
            })
            .collect(),
    };

    serde_wasm_bindgen::to_value(&dto).map_err(js_err)
}

// ── Regression Analysis ─────────────────────────────────────────────

/// Regression input: column-major predictors + target.
#[derive(Deserialize)]
struct RegressionInputDto {
    /// Predictor columns: `{ "x1": [1,2,3], "x2": [4,5,6] }`.
    predictors: HashMap<String, Vec<f64>>,
    /// Target column values.
    target: Vec<f64>,
    /// Target column name.
    target_name: String,
}

/// Regression analysis result.
#[derive(Serialize)]
struct RegressionDto {
    target_name: String,
    predictor_names: Vec<String>,
    r_squared: f64,
    adj_r_squared: f64,
    coefficients: Vec<f64>,
    p_values: Vec<f64>,
    vif: Vec<f64>,
    f_p_value: f64,
}

/// Runs OLS regression analysis.
///
/// # Input
///
/// `data_json`:
/// ```json
/// {
///   "predictors": { "x1": [1,2,3,4,5], "x2": [2,4,6,8,10] },
///   "target": [2.1, 3.9, 6.1, 7.9, 10.1],
///   "target_name": "y"
/// }
/// ```
///
/// # Output
///
/// `{ target_name, predictor_names, r_squared, adj_r_squared, coefficients, p_values, vif, f_p_value }`
#[wasm_bindgen]
pub fn regression(data_json: JsValue) -> Result<JsValue, JsValue> {
    let input: RegressionInputDto = serde_wasm_bindgen::from_value(data_json).map_err(js_err)?;

    use crate::analysis::regression_analysis;

    if input.predictors.is_empty() {
        return Err(js_err("predictors must contain at least one column"));
    }

    // Sort predictor names for deterministic order
    let mut pred_names: Vec<String> = input.predictors.keys().cloned().collect();
    pred_names.sort();

    let pred_columns: Vec<Vec<f64>> = pred_names
        .iter()
        .map(|n| input.predictors[n].clone())
        .collect();

    let result = regression_analysis(
        &pred_columns,
        &pred_names,
        &input.target,
        &input.target_name,
    )
    .map_err(js_err)?;

    let dto = RegressionDto {
        target_name: result.target_name,
        predictor_names: result.predictor_names,
        r_squared: result.r_squared,
        adj_r_squared: result.adj_r_squared,
        coefficients: result.coefficients,
        p_values: result.p_values,
        vif: result.vif,
        f_p_value: result.f_p_value,
    };

    serde_wasm_bindgen::to_value(&dto).map_err(js_err)
}

// ── Feature Importance ──────────────────────────────────────────────

/// Feature importance input.
#[derive(Deserialize)]
struct FeatureImportanceInputDto {
    /// Feature columns: `{ "f1": [1,2,3], "f2": [4,5,6] }`.
    features: HashMap<String, Vec<f64>>,
    /// Target values (continuous for permutation, categorical class ids for ANOVA/MI).
    target: Vec<f64>,
    /// Method: "permutation", "anova", or "mutual_info". Default: "permutation".
    #[serde(default = "default_fi_method")]
    method: String,
    /// Significance level for ANOVA. Default: 0.05.
    #[serde(default = "default_significance")]
    significance_level: f64,
    /// Number of permutation repeats. Default: 5.
    #[serde(default = "default_n_repeats")]
    n_repeats: usize,
    /// Random seed for permutation. Default: 42.
    #[serde(default = "default_fi_seed")]
    seed: u64,
    /// Number of bins for mutual information. None = auto (Sturges' rule).
    n_bins: Option<usize>,
}

fn default_fi_method() -> String {
    "permutation".into()
}
fn default_n_repeats() -> usize {
    5
}
fn default_fi_seed() -> u64 {
    42
}

/// A single feature's importance result.
#[derive(Serialize)]
struct FeatureImportanceItemDto {
    name: String,
    index: usize,
    score: f64,
    /// Only present for permutation importance.
    #[serde(skip_serializing_if = "Option::is_none")]
    std_dev: Option<f64>,
    /// Only present for ANOVA.
    #[serde(skip_serializing_if = "Option::is_none")]
    p_value: Option<f64>,
}

/// Feature importance result.
#[derive(Serialize)]
struct FeatureImportanceDto {
    method: String,
    features: Vec<FeatureImportanceItemDto>,
    /// Only present for permutation importance.
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline_score: Option<f64>,
    /// Only present for ANOVA.
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_indices: Option<Vec<usize>>,
}

/// Computes feature importance using one of three methods.
///
/// # Input
///
/// `data_json`:
/// ```json
/// {
///   "features": { "f1": [1,2,3,4,5], "f2": [5,4,3,2,1] },
///   "target": [0, 0, 1, 1, 1],
///   "method": "permutation",
///   "n_repeats": 5,
///   "seed": 42
/// }
/// ```
///
/// Methods: `"permutation"` (regression target), `"anova"` (class target),
/// `"mutual_info"` (class target).
///
/// # Output
///
/// `{ method, features: [{ name, index, score, std_dev?, p_value? }], baseline_score?, selected_indices? }`
#[wasm_bindgen]
pub fn feature_importance(data_json: JsValue) -> Result<JsValue, JsValue> {
    let input: FeatureImportanceInputDto =
        serde_wasm_bindgen::from_value(data_json).map_err(js_err)?;

    if input.features.is_empty() {
        return Err(js_err("features must contain at least one column"));
    }

    // Sort feature names for deterministic order
    let mut feat_names: Vec<String> = input.features.keys().cloned().collect();
    feat_names.sort();

    let feat_columns: Vec<Vec<f64>> = feat_names
        .iter()
        .map(|n| input.features[n].clone())
        .collect();

    match input.method.to_lowercase().as_str() {
        "anova" => {
            use crate::analysis::anova_feature_selection;

            // Convert f64 target to usize class labels
            let class_target: Vec<usize> = input.target.iter().map(|&v| v as usize).collect();

            let result = anova_feature_selection(
                &feat_columns,
                &feat_names,
                &class_target,
                input.significance_level,
            )
            .map_err(js_err)?;

            let dto = FeatureImportanceDto {
                method: "anova".into(),
                features: result
                    .features
                    .iter()
                    .enumerate()
                    .map(|(i, f)| FeatureImportanceItemDto {
                        name: f.name.clone(),
                        index: i,
                        score: f.f_statistic,
                        std_dev: None,
                        p_value: Some(f.p_value),
                    })
                    .collect(),
                baseline_score: None,
                selected_indices: Some(result.selected_indices),
            };

            serde_wasm_bindgen::to_value(&dto).map_err(js_err)
        }
        "mutual_info" => {
            use crate::analysis::mutual_info_classif;

            let class_target: Vec<usize> = input.target.iter().map(|&v| v as usize).collect();

            let result =
                mutual_info_classif(&feat_columns, &feat_names, &class_target, input.n_bins)
                    .map_err(js_err)?;

            let dto = FeatureImportanceDto {
                method: "mutual_info".into(),
                features: result
                    .features
                    .iter()
                    .map(|f| FeatureImportanceItemDto {
                        name: f.name.clone(),
                        index: f.index,
                        score: f.mi,
                        std_dev: None,
                        p_value: None,
                    })
                    .collect(),
                baseline_score: None,
                selected_indices: None,
            };

            serde_wasm_bindgen::to_value(&dto).map_err(js_err)
        }
        _ => {
            // Default: permutation importance
            use crate::feature_importance::permutation_importance;

            let result = permutation_importance(
                &feat_columns,
                &feat_names,
                &input.target,
                input.n_repeats,
                input.seed,
            )
            .map_err(js_err)?;

            let dto = FeatureImportanceDto {
                method: "permutation".into(),
                features: result
                    .features
                    .iter()
                    .map(|f| FeatureImportanceItemDto {
                        name: f.name.clone(),
                        index: f.index,
                        score: f.importance,
                        std_dev: Some(f.std_dev),
                        p_value: None,
                    })
                    .collect(),
                baseline_score: Some(result.baseline_score),
                selected_indices: None,
            };

            serde_wasm_bindgen::to_value(&dto).map_err(js_err)
        }
    }
}
