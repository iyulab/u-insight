//! WASM bindings for u-insight.
//!
//! Exposes statistical analysis and profiling functionality for JavaScript/TypeScript.
//!
//! # API
//!
//! - `describe(data)` — Descriptive statistics per column
//! - `correlation_matrix(data)` — Pearson correlation matrix
//! - `kmeans(data, k)` — K-Means++ clustering
//! - `pca(data, config)` — Principal Component Analysis (standardised by default)
//! - `dbscan(data, config)` — DBSCAN density-based clustering
//! - `hierarchical(data, config)` — Hierarchical agglomerative clustering
//! - `isolation_forest(data, config)` — Isolation Forest anomaly detection
//! - `lof(data, config)` — Local Outlier Factor anomaly detection
//! - `distribution_analysis(data, config)` — Distribution analysis + normality tests
//! - `regression(data)` — OLS regression (simple or multiple)
//! - `feature_importance(data)` — Feature importance (permutation / ANOVA / mutual info)
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
//!
//! # Errors
//!
//! Every refusal throws an `Error` whose `message` is readable text and which
//! carries `code` -- a stable reason -- and the values behind it
//! (`parameter`, `index`, `got`, `expected`, ...). See the README's *Errors*.

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// ── Refusals ────────────────────────────────────────────────────────

use crate::refusal::Refusal as WireError;

impl From<u_analytics::detection::SpectralResidualError> for WireError {
    fn from(e: u_analytics::detection::SpectralResidualError) -> Self {
        WireError::from(&e)
    }
}

/// A result that could not be turned into a JavaScript value.
impl From<serde_wasm_bindgen::Error> for WireError {
    fn from(e: serde_wasm_bindgen::Error) -> Self {
        WireError::malformed_input("result", e.to_string())
    }
}

/// Every refusal crosses into JavaScript as an `Error` whose `message` is the
/// readable text and which carries `code` -- a stable reason -- and the values
/// behind it as further properties. A program branches on `err.code` and
/// reads the fields; `err.message` reads as it always did.
fn js_err(error: impl Into<WireError>) -> JsValue {
    let error = error.into();
    let js = js_sys::Error::new(&error.message);
    // `json_compatible` turns the map into a plain object; the default would
    // produce a JavaScript `Map`, which `Object.assign` does not read.
    if let Ok(fields) = error
        .fields
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
    {
        js_sys::Object::assign(&js, &fields.into());
    }
    js.into()
}

/// A NaN or ±Infinity found in a JS argument, and where it sits.
///
/// JSON has no non-finite numbers, so on the way to the wire schema
/// `serde_json` turns one into `null` and the caller would be told a value has
/// the wrong type. [`find_non_finite`] looks before that happens, so the
/// refusal names the real reason and the place.
struct NonFinite {
    /// The argument's name, then `.key` and `[i]` steps down to the array or
    /// field that holds the number.
    parameter: String,
    /// The number's position, when it is an array element.
    index: Option<usize>,
    value: f64,
}

impl NonFinite {
    fn message(&self) -> String {
        let at = match self.index {
            Some(i) => format!("{}[{i}]", self.parameter),
            None => self.parameter.clone(),
        };
        let got = if self.value.is_nan() {
            "NaN"
        } else if self.value > 0.0 {
            "Infinity"
        } else {
            "-Infinity"
        };
        format!("{at}: expected a finite number, got {got}")
    }

    /// `parameter` and `index` (`null` when the number is not an array element).
    fn fields(&self) -> serde_json::Value {
        serde_json::json!({ "parameter": self.parameter, "index": self.index })
    }
}

/// The first NaN or ±Infinity in `value`, searching arrays, iterables and
/// plain objects. `allow_nan` lets NaN through for an input that reads it as a
/// missing value; it then arrives as `null`.
fn find_non_finite(value: &JsValue, parameter: &str, allow_nan: bool) -> Option<NonFinite> {
    let refused = |n: f64| !n.is_finite() && !(allow_nan && n.is_nan());
    let found = |index: Option<usize>, value: f64| NonFinite {
        parameter: parameter.to_string(),
        index,
        value,
    };
    if let Some(n) = value.as_f64() {
        return refused(n).then(|| found(None, n));
    }
    if !value.is_object() {
        return None;
    }
    if let Ok(Some(items)) = js_sys::try_iter(value) {
        for (i, item) in items.enumerate() {
            // An iterator that throws is left for serde to report.
            let item = item.ok()?;
            match item.as_f64() {
                Some(n) if refused(n) => return Some(found(Some(i), n)),
                Some(_) => {}
                None => {
                    let inner = find_non_finite(&item, &format!("{parameter}[{i}]"), allow_nan);
                    if inner.is_some() {
                        return inner;
                    }
                }
            }
        }
        return None;
    }
    let object: &js_sys::Object = wasm_bindgen::JsCast::unchecked_ref(value);
    for entry in js_sys::Object::entries(object).iter() {
        let pair: js_sys::Array = wasm_bindgen::JsCast::unchecked_into(entry);
        let key = pair.get(0).as_string().unwrap_or_default();
        let inner = find_non_finite(&pair.get(1), &format!("{parameter}.{key}"), allow_nan);
        if inner.is_some() {
            return inner;
        }
    }
    None
}

/// Deserialize a native JS value, rejecting JSON strings with an actionable
/// message and prefixing the offending parameter name to any serde error.
fn from_js<T: serde::de::DeserializeOwned>(value: JsValue, param: &str) -> Result<T, JsValue> {
    from_js_with(value, param, false)
}

/// [`from_js`], letting NaN through as a missing value when `allow_nan`
/// (it arrives as `null`, which [`describe`] counts as missing). ±Infinity is
/// refused either way.
fn from_js_with<T: serde::de::DeserializeOwned>(
    value: JsValue,
    param: &str,
    allow_nan: bool,
) -> Result<T, JsValue> {
    let refuse = |message: String| js_err(WireError::malformed_input(param, message));
    if value.as_string().is_some() {
        return Err(refuse(format!(
            "{param}: expected a native JS object/array, got a string — \
             pass the value directly, not JSON.stringify(...)"
        )));
    }
    if let Some(found) = find_non_finite(&value, param, allow_nan) {
        return Err(js_err(WireError::new(
            "value_not_finite",
            found.message(),
            found.fields(),
        )));
    }
    // serde-wasm-bindgen reads only a struct's declared fields from a JS
    // object, so `deny_unknown_fields` never sees extra keys. Round-trip
    // through serde_json::Value so the strict wire schema is enforced.
    let json: serde_json::Value =
        serde_wasm_bindgen::from_value(value).map_err(|e| refuse(format!("{param}: {e}")))?;
    serde_json::from_value(json).map_err(|e| refuse(format!("{param}: {e}")))
}

/// Strict column-extractor for column-major JSON inputs.
///
/// WASM is a system boundary — we reject malformed input loudly instead of
/// silently coercing it to an empty / partial vector. Returns an `InvalidInput`-style
/// JS error if the value is not a JSON array, or if any element is not a JSON number.
fn extract_numeric_array(value: &serde_json::Value, name: &str) -> Result<Vec<f64>, JsValue> {
    let refuse = |message: String| {
        js_err(WireError::new(
            "malformed_input",
            message,
            json!({ "parameter": "data", "column": name }),
        ))
    };
    let arr = value
        .as_array()
        .ok_or_else(|| refuse(format!("column '{name}' must be a numeric array")))?;
    let parsed: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
    if parsed.len() != arr.len() {
        return Err(refuse(format!(
            "column '{name}' contains {} non-numeric value(s)",
            arr.len() - parsed.len()
        )));
    }
    Ok(parsed)
}

// ── TypeScript declarations for column-major inputs ─────────────────

/// Column-major inputs keyed by column name. Declared by hand: their keys are
/// the caller's column names, with a reserved `_`-prefixed key for an option,
/// so no struct describes them.
#[wasm_bindgen(typescript_custom_section)]
const COLUMN_INPUTS_TS: &'static str = r#"
/** Columns by name; a column may mix numbers, strings, booleans and nulls. */
export type DescribeInput = Record<string, (number | string | boolean | null)[]>;
/** A correlation method. */
export type CorrelationMethod = "pearson" | "spearman" | "kendall";
/** Numeric columns by name, and optionally `_method` (default `"pearson"`). */
export interface CorrelationInput {
    _method?: CorrelationMethod;
    [column: string]: number[] | CorrelationMethod | undefined;
}
/** Numeric columns by name, and optionally `_threshold` (default 10). */
export interface VifInput {
    _threshold?: number;
    [column: string]: number[] | number | undefined;
}
"#;

// ── Serializable DTOs ─────────────────────────────────────────────────

/// Descriptive statistics for a single numeric column.
#[derive(Serialize, tsify::Tsify)]
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
#[derive(Serialize, tsify::Tsify)]
struct BoolStats {
    count: usize,
    null_count: usize,
    missing_pct: f64,
    true_count: usize,
    false_count: usize,
    true_ratio: f64,
}

/// Summary statistics for a categorical column.
#[derive(Serialize, tsify::Tsify)]
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
#[derive(Serialize, tsify::Tsify)]
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
#[derive(Serialize, tsify::Tsify)]
struct ColumnResult {
    name: String,
    data_type: String,
    numeric: Option<NumericStats>,
    boolean: Option<BoolStats>,
    categorical: Option<CatStats>,
    text: Option<TextStats>,
}

/// Result of correlation analysis.
#[derive(Serialize, tsify::Tsify)]
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

#[derive(Serialize, tsify::Tsify)]
struct CorrelationPairDto {
    col_a: String,
    col_b: String,
    r: f64,
    p_value: f64,
}

/// Result of K-Means clustering.
#[derive(Serialize, tsify::Tsify)]
struct KMeansDto {
    k: usize,
    labels: Vec<usize>,
    centroids: Vec<Vec<f64>>,
    wcss: f64,
    iterations: usize,
    cluster_sizes: Vec<usize>,
}

/// Result of silhouette analysis.
#[derive(Serialize, tsify::Tsify)]
struct SilhouetteDto {
    avg: f64,
    per_sample: Vec<f64>,
}

/// Result of PCA.
#[derive(Serialize, tsify::Tsify)]
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
#[wasm_bindgen(unchecked_return_type = "ColumnResult[]")]
pub fn describe(
    #[wasm_bindgen(unchecked_param_type = "DescribeInput")] data: JsValue,
) -> Result<JsValue, JsValue> {
    use crate::json_parser::JsonParser;
    use crate::profiling::profile_dataframe;

    // NaN is a missing value here, like null; ±Infinity is still refused.
    let raw: serde_json::Value = from_js_with(data, "data", true)?;

    let df = JsonParser::new().parse_value(&raw).map_err(js_err)?;

    if df.is_empty() {
        return Err(js_err(WireError::empty_input(
            "data",
            "data must contain at least one column",
        )));
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

/// A classification target as class labels: whole numbers `>= 0`. A label
/// such as 1.7 or -2 used to become 1 or 0 by a cast, merging classes the
/// caller kept apart.
fn class_labels(target: &[f64]) -> Result<Vec<usize>, WireError> {
    target
        .iter()
        .enumerate()
        .map(|(index, &v)| {
            if v >= 0.0 && v.fract() == 0.0 && v <= u32::MAX as f64 {
                Ok(v as usize)
            } else {
                Err(WireError::new(
                    "not_a_class_label",
                    format!("target[{index}] is {v}; a class label is a whole number >= 0"),
                    json!({ "parameter": "target", "index": index, "got": v }),
                ))
            }
        })
        .collect()
}

/// Computes a correlation matrix for a column-major dataset.
///
/// # Input
/// ```json
/// { "col1": [1.0, 2.0, 3.0], "col2": [4.0, 5.0, 6.0], "_method": "pearson" }
/// ```
///
/// `_method` ∈ `{"pearson", "spearman", "kendall"}` — optional, defaults
/// to `"pearson"`. Reserved key (prefix `_`) so it never collides with a
/// column name.
///
/// # Output
/// `{ names, matrix (flattened n×n), n, high_pairs }`
#[wasm_bindgen(unchecked_return_type = "CorrelationResult")]
pub fn correlation_matrix(
    #[wasm_bindgen(unchecked_param_type = "CorrelationInput")] data: JsValue,
) -> Result<JsValue, JsValue> {
    let mut raw: HashMap<String, serde_json::Value> = from_js(data, "data")?;

    let method_str = raw
        .remove("_method")
        .map(|v| match v.as_str() {
            Some(s) => Ok(s.to_string()),
            None => Err(js_err(crate::error::InsightError::InvalidParameter {
                name: "_method".into(),
                message: format!("expected a string, got {v}"),
            })),
        })
        .transpose()?
        .unwrap_or_else(|| "pearson".to_string());

    if raw.is_empty() {
        return Err(js_err(WireError::empty_input(
            "data",
            "data must contain at least one column",
        )));
    }

    // Sort keys for deterministic order
    let mut names: Vec<String> = raw.keys().cloned().collect();
    names.sort();

    let columns: Vec<Vec<f64>> = names
        .iter()
        .map(|n| extract_numeric_array(&raw[n], n))
        .collect::<Result<_, _>>()?;

    use crate::analysis::{correlation_analysis, CorrelationConfig, CorrelationMethod};
    let method = match method_str.as_str() {
        "pearson" => CorrelationMethod::Pearson,
        "spearman" => CorrelationMethod::Spearman,
        "kendall" => CorrelationMethod::Kendall,
        other => {
            return Err(js_err(WireError::unknown_option(
                "_method",
                other,
                &["pearson", "spearman", "kendall"],
            )))
        }
    };
    let config = CorrelationConfig {
        method,
        high_threshold: 0.7,
    };
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
#[wasm_bindgen(unchecked_return_type = "KMeansDto")]
pub fn kmeans(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] data: JsValue,
    k: usize,
) -> Result<JsValue, JsValue> {
    let data: Vec<Vec<f64>> = from_js(data, "data")?;

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

/// Computes silhouette scores for an existing clustering assignment.
///
/// # Input
/// `data`: row-major points `[[x,y,...], ...]`
/// `labels`: cluster id per sample `[0, 0, 1, 1, ...]` (each value `< k`)
/// `k`: number of distinct clusters
///
/// # Output
/// `{ avg, per_sample }` — `avg` is the mean silhouette across samples that
/// had a defined silhouette; `per_sample[i]` is the silhouette of sample `i`
/// (0.0 for singleton-cluster points).
///
/// O(n²) — use sparingly on very large inputs.
#[wasm_bindgen(unchecked_return_type = "SilhouetteDto")]
pub fn silhouette(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number[]")] labels: JsValue,
    k: usize,
) -> Result<JsValue, JsValue> {
    let data: Vec<Vec<f64>> = from_js(data, "data")?;
    let labels: Vec<usize> = from_js(labels, "labels")?;

    if labels.len() != data.len() {
        return Err(js_err(WireError::new(
            "dimension_mismatch",
            format!(
                "labels length ({}) must match data row count ({})",
                labels.len(),
                data.len()
            ),
            json!({ "parameter": "labels", "expected": data.len(), "got": labels.len() }),
        )));
    }
    if let Some((index, &bad)) = labels.iter().enumerate().find(|(_, &l)| l >= k) {
        return Err(js_err(WireError::new(
            "parameter_out_of_range",
            format!("label {bad} out of range for k={k}"),
            json!({
                "parameter": "labels",
                "index": index,
                "min": 0,
                "max": k.saturating_sub(1),
                "got": bad,
            }),
        )));
    }

    use crate::clustering::silhouette_samples;
    let analysis = silhouette_samples(&data, &labels, k);
    let dto = SilhouetteDto {
        avg: analysis.avg,
        per_sample: analysis.per_sample,
    };

    serde_wasm_bindgen::to_value(&dto).map_err(js_err)
}

/// PCA configuration input.
#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct PcaConfigDto {
    /// Number of principal components to keep.
    n_components: usize,
    /// Standardise each column to unit variance before the decomposition
    /// (correlation-matrix PCA). Default: `true` — without it a column in
    /// large units takes the leading components. `false` gives
    /// covariance-matrix PCA.
    #[serde(default = "default_true")]
    #[tsify(optional)]
    auto_scale: bool,
}

fn pca_config(dto: &PcaConfigDto) -> crate::pca::PcaConfig {
    crate::pca::PcaConfig::new(dto.n_components).auto_scale(dto.auto_scale)
}

/// Runs Principal Component Analysis on row-major data.
///
/// # Input
/// ```json
/// [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]]
/// ```
/// Config: `{ n_components: 2, auto_scale?: true }`.
///
/// # Output
/// `{ n_components, n_features, eigenvalues, explained_variance_ratio, ... }`;
/// `stds` are the column standard deviations used for scaling (all `1` when
/// `auto_scale` is `false`).
#[wasm_bindgen(unchecked_return_type = "PcaDto")]
pub fn pca(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "PcaConfigDto")] config: JsValue,
) -> Result<JsValue, JsValue> {
    let data: Vec<Vec<f64>> = from_js(data, "data")?;
    let config: PcaConfigDto = from_js(config, "config")?;

    use crate::pca::pca as pca_fn;
    let config = pca_config(&config);
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
#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct DbscanConfigDto {
    epsilon: f64,
    min_samples: usize,
}

/// DBSCAN clustering result.
#[derive(Serialize, tsify::Tsify)]
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
/// `data`: row-major points `[[x,y,...], ...]`
///
/// `config`: `{ "epsilon": 1.5, "min_samples": 3 }`
///
/// # Output
///
/// `{ labels, n_clusters, noise_count, cluster_sizes, core_points }`
#[wasm_bindgen(unchecked_return_type = "DbscanDto")]
pub fn dbscan(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "DbscanConfigDto")] config: JsValue,
) -> Result<JsValue, JsValue> {
    let data: Vec<Vec<f64>> = from_js(data, "data")?;
    let cfg: DbscanConfigDto = from_js(config, "config")?;

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
#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct HierarchicalConfigDto {
    /// Linkage: "single", "complete", "average", "ward". Default: "ward".
    #[serde(default = "default_linkage")]
    #[tsify(optional)]
    #[tsify(type = "\"single\" | \"complete\" | \"average\" | \"ward\"")]
    linkage: String,
    /// Number of flat clusters (mutually exclusive with distance_threshold).
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    n_clusters: Option<usize>,
    /// Distance threshold for dendrogram cut (mutually exclusive with n_clusters).
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    distance_threshold: Option<f64>,
    /// Max input points (memory guard). Omit for the default; `0` disables it.
    /// Rejects oversized inputs before allocating the O(n²) distance matrix.
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    max_points: Option<usize>,
}

fn default_linkage() -> String {
    "ward".into()
}

/// Reads a linkage name, case-insensitively. A name it does not know is
/// refused: it used to be read as `"ward"`, so a misspelling ran a different
/// method than the one asked for and nothing said so.
fn parse_linkage(name: &str) -> Result<crate::clustering::Linkage, WireError> {
    use crate::clustering::Linkage;
    match name.to_lowercase().as_str() {
        "single" => Ok(Linkage::Single),
        "complete" => Ok(Linkage::Complete),
        "average" => Ok(Linkage::Average),
        "ward" => Ok(Linkage::Ward),
        _ => Err(WireError::unknown_option(
            "linkage",
            name,
            &["single", "complete", "average", "ward"],
        )),
    }
}

/// A single merge step in the dendrogram.
#[derive(Serialize, tsify::Tsify)]
struct MergeDto {
    cluster_a: usize,
    cluster_b: usize,
    distance: f64,
    size: usize,
}

/// Hierarchical clustering result.
#[derive(Serialize, tsify::Tsify)]
struct HierarchicalDto {
    merges: Vec<MergeDto>,
    labels: Option<Vec<usize>>,
    n_clusters: Option<usize>,
}

/// Runs hierarchical agglomerative clustering on row-major data.
///
/// # Input
///
/// `data`: row-major points `[[x,y,...], ...]`
///
/// `config`: `{ "linkage": "ward", "n_clusters": 3 }` or
/// `{ "linkage": "single", "distance_threshold": 5.0 }`
///
/// # Output
///
/// `{ merges, labels, n_clusters }`
#[wasm_bindgen(unchecked_return_type = "HierarchicalDto")]
pub fn hierarchical(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "HierarchicalConfigDto")] config: JsValue,
) -> Result<JsValue, JsValue> {
    let data: Vec<Vec<f64>> = from_js(data, "data")?;
    let cfg: HierarchicalConfigDto = from_js(config, "config")?;

    use crate::clustering::{hierarchical as hier_fn, HierarchicalConfig};

    let linkage = parse_linkage(&cfg.linkage).map_err(js_err)?;

    let mut config = match (cfg.n_clusters, cfg.distance_threshold) {
        (Some(_), Some(_)) => {
            return Err(js_err(WireError::new(
                "invalid_option",
                "config gives both n_clusters and distance_threshold; they are mutually \
                 exclusive -- give one"
                    .to_string(),
                json!({ "parameter": "config" }),
            )))
        }
        (Some(k), None) => HierarchicalConfig::with_k(k).linkage(linkage),
        (None, Some(t)) => HierarchicalConfig::with_threshold(t).linkage(linkage),
        (None, None) => {
            return Err(js_err(WireError::new(
                "missing_option",
                "config must specify either n_clusters or distance_threshold".to_string(),
                json!({ "parameter": "config", "expected": ["n_clusters", "distance_threshold"] }),
            )))
        }
    };
    if let Some(mp) = cfg.max_points {
        config.max_points = mp;
    }

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
#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct IsolationForestConfigDto {
    /// Number of trees. Default: 100.
    #[serde(default = "default_n_estimators")]
    #[tsify(optional)]
    n_estimators: usize,
    /// Subsample size per tree. 0 = auto (min(256, n)). Default: 0.
    #[serde(default)]
    #[tsify(optional)]
    max_samples: usize,
    /// Expected contamination rate (0.0–1.0). Default: 0.1.
    #[serde(default = "default_contamination")]
    #[tsify(optional)]
    contamination: f64,
    /// Random seed. Default: 42.
    #[serde(default = "default_seed")]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
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
#[derive(Serialize, tsify::Tsify)]
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
/// `data`: row-major points `[[x,y,...], ...]`
///
/// `config`: `{ "n_estimators": 100, "contamination": 0.1, "seed": 42 }`
///
/// # Output
///
/// `{ scores, anomalies, threshold, anomaly_count, anomaly_fraction }`
#[wasm_bindgen(unchecked_return_type = "IsolationForestDto")]
pub fn isolation_forest(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "IsolationForestConfigDto")] config: JsValue,
) -> Result<JsValue, JsValue> {
    let data: Vec<Vec<f64>> = from_js(data, "data")?;
    let cfg: IsolationForestConfigDto = from_js(config, "config")?;

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
#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct LofConfigDto {
    /// Number of nearest neighbors. Default: 20.
    #[serde(default = "default_lof_k")]
    #[tsify(optional)]
    k: usize,
    /// LOF threshold for outlier classification. Default: 1.5.
    #[serde(default = "default_lof_threshold")]
    #[tsify(optional)]
    threshold: f64,
}

fn default_lof_k() -> usize {
    20
}
fn default_lof_threshold() -> f64 {
    1.5
}

/// LOF anomaly detection result.
#[derive(Serialize, tsify::Tsify)]
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
/// `data`: row-major points `[[x,y,...], ...]`
///
/// `config`: `{ "k": 20, "threshold": 1.5 }`
///
/// # Output
///
/// `{ scores, anomalies, threshold, anomaly_count, anomaly_fraction }`
#[wasm_bindgen(unchecked_return_type = "LofDto")]
pub fn lof(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "LofConfigDto")] config: JsValue,
) -> Result<JsValue, JsValue> {
    let data: Vec<Vec<f64>> = from_js(data, "data")?;
    let cfg: LofConfigDto = from_js(config, "config")?;

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
#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct DistributionConfigDto {
    /// Bin method: "sturges", "scott", "freedman_diaconis". Default: "freedman_diaconis".
    #[serde(default = "default_bin_method")]
    #[tsify(optional)]
    #[tsify(type = "\"sturges\" | \"scott\" | \"freedman_diaconis\"")]
    bin_method: String,
    /// Explicit bin count (>= 1). When set, takes precedence over `bin_method`.
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    bins: Option<usize>,
    /// Significance level for normality tests. Default: 0.05.
    #[serde(default = "default_significance")]
    #[tsify(optional)]
    significance_level: f64,
    /// Whether to compute ECDF. Default: true.
    #[serde(default = "default_true")]
    #[tsify(optional)]
    compute_ecdf: bool,
    /// Whether to compute histogram. Default: true.
    #[serde(default = "default_true")]
    #[tsify(optional)]
    compute_histogram: bool,
    /// Whether to compute QQ-plot. Default: true.
    #[serde(default = "default_true")]
    #[tsify(optional)]
    compute_qq_plot: bool,
    /// Whether to fit distributions. Default: false.
    #[serde(default)]
    #[tsify(optional)]
    fit_distributions: bool,
}

fn default_bin_method() -> String {
    "freedman_diaconis".into()
}

/// Resolves the effective bin method: an explicit `bins` count wins over
/// the `bin_method` name. The name is checked either way -- a misspelt one is
/// refused rather than read as the default.
fn resolve_bin_method(
    bin_method: &str,
    bins: Option<usize>,
) -> Result<crate::distribution::BinMethod, WireError> {
    use crate::distribution::BinMethod;
    let named = match bin_method.to_lowercase().as_str() {
        "sturges" => BinMethod::Sturges,
        "scott" => BinMethod::Scott,
        "freedman_diaconis" => BinMethod::FreedmanDiaconis,
        _ => {
            return Err(WireError::unknown_option(
                "bin_method",
                bin_method,
                &["sturges", "scott", "freedman_diaconis"],
            ))
        }
    };
    Ok(match bins {
        Some(n) => BinMethod::Fixed(n),
        None => named,
    })
}
fn default_significance() -> f64 {
    0.05
}
fn default_true() -> bool {
    true
}

#[derive(Serialize, tsify::Tsify)]
struct EcdfDto {
    values: Vec<f64>,
    probabilities: Vec<f64>,
}

#[derive(Serialize, tsify::Tsify)]
struct HistogramDto {
    n_bins: usize,
    bin_width: f64,
    edges: Vec<f64>,
    counts: Vec<usize>,
    method: String,
}

#[derive(Serialize, tsify::Tsify)]
struct QQPlotDto {
    theoretical: Vec<f64>,
    sample: Vec<f64>,
}

#[derive(Serialize, tsify::Tsify)]
struct NormalityTestDto {
    statistic: f64,
    p_value: f64,
    rejected: bool,
}

#[derive(Serialize, tsify::Tsify)]
struct NormalityDto {
    ks_test: Option<NormalityTestDto>,
    jarque_bera: Option<NormalityTestDto>,
    shapiro_wilk: Option<NormalityTestDto>,
    anderson_darling: Option<NormalityTestDto>,
    is_normal: bool,
    significance_level: f64,
}

#[derive(Serialize, tsify::Tsify)]
struct FitResultDto {
    distribution: String,
    parameters: Vec<(String, f64)>,
    log_likelihood: f64,
    aic: f64,
    bic: f64,
    n_params: usize,
}

#[derive(Serialize, tsify::Tsify)]
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
/// `data`: flat array `[1.0, 2.0, 3.0, ...]`
///
/// `config`: `{ "bin_method": "freedman_diaconis", "bins": null,
///   "significance_level": 0.05, "compute_ecdf": true, "compute_histogram": true,
///   "compute_qq_plot": true, "fit_distributions": false }`
///
/// `bins` (optional, >= 1): explicit histogram bin count; when set it takes
/// precedence over `bin_method`. The histogram `method` field echoes
/// `"Fixed(n)"` in that case.
///
/// # Output
///
/// `{ n, ecdf, histogram, qq_plot, normality, fits }`
#[wasm_bindgen(unchecked_return_type = "DistributionAnalysisDto")]
pub fn distribution_analysis(
    #[wasm_bindgen(unchecked_param_type = "number[]")] data: JsValue,
    #[wasm_bindgen(unchecked_param_type = "DistributionConfigDto")] config: JsValue,
) -> Result<JsValue, JsValue> {
    let data: Vec<f64> = from_js(data, "data")?;
    let cfg: DistributionConfigDto = from_js(config, "config")?;

    use crate::distribution::{distribution_analysis as dist_fn, DistributionConfig};

    let bin_method = resolve_bin_method(&cfg.bin_method, cfg.bins).map_err(js_err)?;

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
#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct RegressionInputDto {
    /// Predictor columns: `{ "x1": [1,2,3], "x2": [4,5,6] }`.
    #[tsify(type = "Record<string, number[]>")]
    predictors: HashMap<String, Vec<f64>>,
    /// Target column values.
    target: Vec<f64>,
    /// Target column name.
    target_name: String,
}

/// Regression analysis result.
#[derive(Serialize, tsify::Tsify)]
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
/// `data`:
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
#[wasm_bindgen(unchecked_return_type = "RegressionDto")]
pub fn regression(
    #[wasm_bindgen(unchecked_param_type = "RegressionInputDto")] data: JsValue,
) -> Result<JsValue, JsValue> {
    let input: RegressionInputDto = from_js(data, "data")?;

    use crate::analysis::regression_analysis;

    if input.predictors.is_empty() {
        return Err(js_err(WireError::empty_input(
            "predictors",
            "predictors must contain at least one column",
        )));
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
#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct FeatureImportanceInputDto {
    /// Feature columns: `{ "f1": [1,2,3], "f2": [4,5,6] }`.
    #[tsify(type = "Record<string, number[]>")]
    features: HashMap<String, Vec<f64>>,
    /// Target values (continuous for permutation, categorical class ids for ANOVA/MI).
    target: Vec<f64>,
    /// Method: "permutation", "anova", or "mutual_info". Default: "permutation".
    #[serde(default = "default_fi_method")]
    #[tsify(optional)]
    #[tsify(type = "\"permutation\" | \"anova\" | \"mutual_info\"")]
    method: String,
    /// Significance level for ANOVA. Default: 0.05.
    #[serde(default = "default_significance")]
    #[tsify(optional)]
    significance_level: f64,
    /// Number of permutation repeats. Default: 5.
    #[serde(default = "default_n_repeats")]
    #[tsify(optional)]
    n_repeats: usize,
    /// Random seed for permutation. Default: 42.
    #[serde(default = "default_fi_seed")]
    #[tsify(optional)]
    seed: u64,
    /// Number of bins for mutual information. None = auto (Sturges' rule).
    #[tsify(optional)]
    #[tsify(type = "number | null")]
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
#[derive(Serialize, tsify::Tsify)]
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
#[derive(Serialize, tsify::Tsify)]
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
/// `data`:
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
#[wasm_bindgen(unchecked_return_type = "FeatureImportanceDto")]
pub fn feature_importance(
    #[wasm_bindgen(unchecked_param_type = "FeatureImportanceInputDto")] data: JsValue,
) -> Result<JsValue, JsValue> {
    let input: FeatureImportanceInputDto = from_js(data, "data")?;

    if input.features.is_empty() {
        return Err(js_err(WireError::empty_input(
            "features",
            "features must contain at least one column",
        )));
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

            let class_target = class_labels(&input.target).map_err(js_err)?;

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

            let class_target = class_labels(&input.target).map_err(js_err)?;

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
        "permutation" => {
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
        other => Err(js_err(WireError::unknown_option(
            "method",
            other,
            &["permutation", "anova", "mutual_info"],
        ))),
    }
}

// ── Multicollinearity Diagnostics (VIF / Condition Number) ───────────

#[derive(Serialize, tsify::Tsify)]
struct VifDto {
    vif_per_column: Vec<f64>,
    high_vif_columns: Vec<u32>,
    threshold: f64,
    names: Vec<String>,
}

/// Variance Inflation Factor diagnostics for column-major numeric data.
///
/// # Input
/// ```json
/// { "col1": [...], "col2": [...], "_threshold": 10.0 }
/// ```
/// `_threshold` is optional (default 10.0).
///
/// # Output
/// `{ vif_per_column, high_vif_columns, threshold, names }`
#[wasm_bindgen(unchecked_return_type = "VifDto")]
pub fn vif_diagnostic(
    #[wasm_bindgen(unchecked_param_type = "VifInput")] data: JsValue,
) -> Result<JsValue, JsValue> {
    let mut raw: HashMap<String, serde_json::Value> = from_js(data, "data")?;

    let threshold = raw
        .remove("_threshold")
        .map(|v| {
            v.as_f64().ok_or_else(|| {
                js_err(crate::error::InsightError::InvalidParameter {
                    name: "_threshold".into(),
                    message: format!("expected a number, got {v}"),
                })
            })
        })
        .transpose()?
        .unwrap_or(10.0);

    if raw.is_empty() {
        return Err(js_err(WireError::empty_input(
            "data",
            "input must contain at least one numeric column",
        )));
    }

    let mut names: Vec<String> = raw.keys().cloned().collect();
    names.sort();
    let columns: Vec<Vec<f64>> = names
        .iter()
        .map(|n| extract_numeric_array(&raw[n], n))
        .collect::<Result<_, _>>()?;

    let r = crate::analysis::vif_analysis(&columns, &names, threshold).map_err(js_err)?;

    serde_wasm_bindgen::to_value(&VifDto {
        vif_per_column: r.vif_per_column,
        high_vif_columns: r.high_vif_columns,
        threshold: r.threshold,
        names: r.names,
    })
    .map_err(js_err)
}

#[derive(Serialize, tsify::Tsify)]
struct ConditionNumberDto {
    condition_number: f64,
    names: Vec<String>,
}

/// 2-norm condition number of the sample covariance matrix.
///
/// # Input
/// ```json
/// { "col1": [...], "col2": [...] }
/// ```
///
/// # Output
/// `{ condition_number, names }`
///
/// Standard threshold: `cond > 30` indicates multicollinearity (Belsley 1991).
/// Returns `Infinity` for numerically singular input.
#[wasm_bindgen(unchecked_return_type = "ConditionNumberDto")]
pub fn condition_number_diagnostic(
    #[wasm_bindgen(unchecked_param_type = "Record<string, number[]>")] data: JsValue,
) -> Result<JsValue, JsValue> {
    let raw: HashMap<String, Vec<f64>> = from_js(data, "data")?;

    if raw.is_empty() {
        return Err(js_err(WireError::empty_input(
            "data",
            "input must contain at least one numeric column",
        )));
    }

    let mut names: Vec<String> = raw.keys().cloned().collect();
    names.sort();
    let columns: Vec<Vec<f64>> = names.iter().map(|n| raw[n].clone()).collect();

    let cond = crate::analysis::condition_number(&columns, &names).map_err(js_err)?;

    serde_wasm_bindgen::to_value(&ConditionNumberDto {
        condition_number: cond,
        names,
    })
    .map_err(js_err)
}

// ── Univariate Outlier Detection ──────────────────────────────────────

/// Univariate outlier detection on a flat numeric vector.
///
/// # Input
/// ```json
/// { "data": [1.0, 2.0, 3.0, 100.0], "method": "iqr" }
/// ```
/// `method` ∈ `{"iqr"|"tukey", "zscore"|"three_sigma", "modified_zscore"|"hampel"}`.
/// Optional, defaults to `"iqr"`.
///
/// # Output
/// `{ method, indices, scores, count, pct, lower_fence, upper_fence, center, spread }`
///
/// `method` aliases: `tukey` → IQR Tukey fences (k=1.5), `three_sigma` →
/// mean ± 3·σ, `hampel` → robust median ± 3.5·(MAD/0.6745).
#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct OutlierInputDto {
    data: Vec<f64>,
    #[serde(default = "default_outlier_method")]
    #[tsify(optional)]
    #[tsify(
        type = "\"iqr\" | \"tukey\" | \"zscore\" | \"three_sigma\" | \"modified_zscore\" | \"hampel\""
    )]
    method: String,
}
fn default_outlier_method() -> String {
    "iqr".to_string()
}

#[derive(Serialize, tsify::Tsify)]
struct UnivariateOutlierDto {
    method: String,
    indices: Vec<usize>,
    scores: Vec<f64>,
    count: usize,
    pct: f64,
    lower_fence: f64,
    upper_fence: f64,
    center: f64,
    spread: f64,
}

#[wasm_bindgen(unchecked_return_type = "UnivariateOutlierDto")]
pub fn detect_univariate_outliers(
    #[wasm_bindgen(unchecked_param_type = "OutlierInputDto")] data: JsValue,
) -> Result<JsValue, JsValue> {
    let req: OutlierInputDto = from_js(data, "data")?;

    use crate::profiling::{detect_outliers_slice, OutlierMethod};
    let method = match req.method.as_str() {
        "iqr" | "tukey" => OutlierMethod::Iqr,
        "zscore" | "three_sigma" => OutlierMethod::Zscore,
        "modified_zscore" | "hampel" => OutlierMethod::ModifiedZscore,
        other => {
            return Err(js_err(WireError::unknown_option(
                "method",
                other,
                &[
                    "iqr",
                    "tukey",
                    "zscore",
                    "three_sigma",
                    "modified_zscore",
                    "hampel",
                ],
            )))
        }
    };

    let r = detect_outliers_slice(&req.data, method).ok_or_else(|| {
        js_err(WireError::new(
            "insufficient_data",
            format!(
                "outlier detection needs at least 3 values, got {}",
                req.data.len()
            ),
            json!({ "parameter": "data", "min": 3, "got": req.data.len() }),
        ))
    })?;

    let method_str = match r.method {
        OutlierMethod::Iqr => "iqr",
        OutlierMethod::Zscore => "zscore",
        OutlierMethod::ModifiedZscore => "modified_zscore",
    };

    let dto = UnivariateOutlierDto {
        method: method_str.to_string(),
        indices: r.indices,
        scores: r.scores,
        count: r.count,
        pct: r.pct,
        lower_fence: r.lower_fence,
        upper_fence: r.upper_fence,
        center: r.center,
        spread: r.spread,
    };

    serde_wasm_bindgen::to_value(&dto).map_err(js_err)
}

// ── Time series ──────────────────────────────────────────────────────

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct SeasonalityInputDto {
    data: Vec<f64>,
}

#[derive(Serialize, tsify::Tsify)]
struct PeriodCandidateDto {
    period: usize,
    acf: f64,
    bin: usize,
    power: f64,
    power_share: f64,
}

#[derive(Serialize, tsify::Tsify)]
struct SeasonalityDto {
    period: Option<usize>,
    candidates: Vec<PeriodCandidateDto>,
    n: usize,
    acf_threshold: f64,
    power_threshold: f64,
}

/// Estimate the dominant period of a univariate series (AutoPeriod —
/// Vlachos, Yu & Castelli 2005: permutation-thresholded periodogram peaks
/// refined on the autocorrelation function; deterministic for a series).
///
/// # Input
/// `{ data: number[] }` — at least 8 finite values.
///
/// # Output
/// `{ period: number | null, candidates: [{ period, acf, bin, power, power_share }],
///    n, acf_threshold, power_threshold }`
///
/// `period` is `null` — explicitly, not an error — when no periodicity passes
/// both stages (a constant, a pure trend, white noise). Only periods from 2 to
/// `n / 2` are admissible.
#[wasm_bindgen(unchecked_return_type = "SeasonalityDto")]
pub fn estimate_period(
    #[wasm_bindgen(unchecked_param_type = "SeasonalityInputDto")] data: JsValue,
) -> Result<JsValue, JsValue> {
    let req: SeasonalityInputDto = from_js(data, "data")?;
    if let Some(i) = req.data.iter().position(|x| !x.is_finite()) {
        return Err(js_err(WireError::value_not_finite("data", i)));
    }
    let r = u_analytics::seasonality::estimate_period(&req.data).ok_or_else(|| {
        js_err(WireError::new(
            "insufficient_data",
            "data must have at least 8 observations".to_string(),
            json!({ "parameter": "data", "min": 8, "got": req.data.len() }),
        ))
    })?;
    let dto = SeasonalityDto {
        period: r.period,
        candidates: r
            .candidates
            .into_iter()
            .map(|c| PeriodCandidateDto {
                period: c.period,
                acf: c.acf,
                bin: c.bin,
                power: c.power,
                power_share: c.power_share,
            })
            .collect(),
        n: r.n,
        acf_threshold: r.acf_threshold,
        power_threshold: r.power_threshold,
    };
    serde_wasm_bindgen::to_value(&dto).map_err(js_err)
}

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct SpectralResidualInputDto {
    data: Vec<f64>,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    averaging_window: Option<usize>,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    judgement_window: Option<usize>,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    threshold: Option<f64>,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    min_zscore: Option<f64>,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    sensitivity: Option<f64>,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    batch_size: Option<usize>,
}

#[derive(Serialize, tsify::Tsify)]
struct SrPointDto {
    index: usize,
    value: f64,
    saliency: f64,
    score: f64,
    expected: f64,
    lower: f64,
    upper: f64,
    is_anomaly: bool,
    /// Within kappa = 5 places of an end of the batch, where the
    /// transform's own boundary handling moves the saliency most.
    near_edge: bool,
}

#[derive(Serialize, tsify::Tsify)]
struct SpectralResidualDto {
    points: Vec<SrPointDto>,
    anomalies: Vec<usize>,
}

/// Score every point of a series for anomalies by spectral residual saliency
/// (Ren et al. 2019) — spikes, steps and dropouts, without a trained model
/// and without assuming a period.
///
/// # Input
/// `{ data: number[], averaging_window?, judgement_window?, threshold?,
///    min_zscore?, sensitivity?, batch_size? }` — at least 12 finite values;
/// the options default to the paper's (3, 40, 3, 1.5, 70, none).
///
/// # Output
/// `{ points: [{ index, value, saliency, score, expected, lower, upper, is_anomaly,
/// near_edge }],
///    anomalies: number[] }`
#[wasm_bindgen(unchecked_return_type = "SpectralResidualDto")]
pub fn spectral_residual(
    #[wasm_bindgen(unchecked_param_type = "SpectralResidualInputDto")] data: JsValue,
) -> Result<JsValue, JsValue> {
    let req: SpectralResidualInputDto = from_js(data, "data")?;
    if let Some(i) = req.data.iter().position(|x| !x.is_finite()) {
        return Err(js_err(WireError::value_not_finite("data", i)));
    }
    let mut sr = u_analytics::detection::SpectralResidual::new();
    if let Some(q) = req.averaging_window {
        sr = sr.with_averaging_window(q);
    }
    if let Some(z) = req.judgement_window {
        sr = sr.with_judgement_window(z);
    }
    if let Some(t) = req.threshold {
        sr = sr.with_threshold(t);
    }
    if let Some(z) = req.min_zscore {
        sr = sr.with_min_zscore(z);
    }
    if let Some(s) = req.sensitivity {
        sr = sr.with_sensitivity(s);
    }
    if req.batch_size.is_some() {
        sr = sr.with_batch_size(req.batch_size);
    }
    // The crate names the one condition that failed; repeating the whole
    // rulebook here is what left consumers re-validating the options.
    let points = sr.analyze(&req.data).map_err(js_err)?;
    let dto = SpectralResidualDto {
        anomalies: points
            .iter()
            .filter(|p| p.is_anomaly)
            .map(|p| p.index)
            .collect(),
        points: points
            .into_iter()
            .map(|p| SrPointDto {
                index: p.index,
                value: p.value,
                saliency: p.saliency,
                score: p.score,
                expected: p.expected,
                lower: p.lower,
                upper: p.upper,
                is_anomaly: p.is_anomaly,
                near_edge: p.near_edge,
            })
            .collect(),
    };
    serde_wasm_bindgen::to_value(&dto).map_err(js_err)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{parse_linkage, resolve_bin_method, WireError};
    use crate::clustering::Linkage;
    use crate::distribution::BinMethod;
    use crate::error::InsightError;
    use serde_json::json;

    fn refused_parameter<T: std::fmt::Debug>(r: Result<T, WireError>) -> String {
        match r {
            Err(e) if e.code() == "unknown_option" => e.fields["parameter"]
                .as_str()
                .expect("a parameter name")
                .to_string(),
            other => panic!("expected unknown_option, got {other:?}"),
        }
    }

    /// A refused name carries what was given and what would have been read.
    #[test]
    fn an_unknown_option_names_what_was_given_and_what_is_known() {
        let e = parse_linkage("centroid").expect_err("unknown");
        assert_eq!(
            e.fields,
            json!({
                "code": "unknown_option",
                "parameter": "linkage",
                "got": "centroid",
                "expected": ["single", "complete", "average", "ward"],
            })
        );
    }

    /// The crate's own errors keep their reason and the values behind it.
    #[test]
    fn crate_errors_become_codes_with_their_values() {
        let e = WireError::from(InsightError::InsufficientData {
            min_required: 3,
            actual: 1,
        });
        assert_eq!(
            e.fields,
            json!({ "code": "insufficient_data", "min": 3, "got": 1 })
        );
        let e = WireError::from(InsightError::InvalidParameter {
            name: "threshold".into(),
            message: "must be positive".into(),
        });
        assert_eq!(
            e.fields,
            json!({ "code": "invalid_option", "parameter": "threshold" })
        );
        let e = WireError::from(
            u_analytics::detection::SpectralResidualError::ValueNotFinite { index: 4 },
        );
        assert_eq!(
            e.fields,
            json!({ "code": "value_not_finite", "parameter": "data", "index": 4 })
        );
    }

    #[test]
    fn bins_overrides_bin_method() {
        assert_eq!(
            resolve_bin_method("sturges", Some(5)).unwrap(),
            BinMethod::Fixed(5)
        );
        assert_eq!(
            resolve_bin_method("freedman_diaconis", Some(20)).unwrap(),
            BinMethod::Fixed(20)
        );
    }

    #[test]
    fn bin_method_used_when_bins_absent() {
        assert_eq!(
            resolve_bin_method("sturges", None).unwrap(),
            BinMethod::Sturges
        );
        assert_eq!(resolve_bin_method("Scott", None).unwrap(), BinMethod::Scott);
        assert_eq!(
            resolve_bin_method("freedman_diaconis", None).unwrap(),
            BinMethod::FreedmanDiaconis
        );
    }

    /// An unknown name used to be read as Freedman-Diaconis; it is refused,
    /// and so is a misspelt name beside an explicit `bins`.
    #[test]
    fn unknown_bin_method_is_refused() {
        assert_eq!(
            refused_parameter(resolve_bin_method("not_a_method", None)),
            "bin_method"
        );
        assert_eq!(
            refused_parameter(resolve_bin_method("sturgess", Some(10))),
            "bin_method"
        );
    }

    #[test]
    fn linkage_names_are_read_and_unknown_ones_refused() {
        assert_eq!(parse_linkage("single").unwrap(), Linkage::Single);
        assert_eq!(parse_linkage("Complete").unwrap(), Linkage::Complete);
        assert_eq!(parse_linkage("average").unwrap(), Linkage::Average);
        assert_eq!(parse_linkage("ward").unwrap(), Linkage::Ward);
        // It used to be read as "ward".
        assert_eq!(refused_parameter(parse_linkage("centroid")), "linkage");
    }
}

#[cfg(test)]
mod dto_strictness_tests {
    use serde_json::json;

    fn assert_rejects_unknown<T: serde::de::DeserializeOwned>(v: serde_json::Value) {
        match serde_json::from_value::<T>(v) {
            Ok(_) => panic!("unknown key must be rejected"),
            Err(e) => assert!(e.to_string().contains("unknown field"), "{e}"),
        }
    }

    /// `auto_scale` is optional and defaults to standardised PCA, the same
    /// default as the C# binding; `false` gives covariance PCA.
    #[test]
    fn pca_config_defaults_to_standardised() {
        let dto: super::PcaConfigDto =
            serde_json::from_value(json!({ "n_components": 2 })).unwrap();
        let cfg = super::pca_config(&dto);
        assert_eq!(cfg.n_components, 2);
        assert!(cfg.auto_scale);
        let dto: super::PcaConfigDto =
            serde_json::from_value(json!({ "n_components": 2, "auto_scale": false })).unwrap();
        assert!(!super::pca_config(&dto).auto_scale);
        assert_rejects_unknown::<super::PcaConfigDto>(json!({
            "n_components": 2, "scale": true
        }));
    }

    #[test]
    fn cluster_configs_reject_unknown_keys() {
        assert_rejects_unknown::<super::DbscanConfigDto>(json!({
            "epsilon": 0.5, "min_samples": 5, "metric": "euclidean"
        }));
        assert_rejects_unknown::<super::HierarchicalConfigDto>(json!({
            "linkage": "ward", "n_clusters": 3, "criterion": "maxclust"
        }));
    }

    #[test]
    fn outlier_configs_reject_unknown_keys() {
        assert_rejects_unknown::<super::IsolationForestConfigDto>(json!({
            "n_estimators": 50, "n_trees": 50
        }));
        assert_rejects_unknown::<super::LofConfigDto>(json!({
            "k": 10, "metric": "euclidean"
        }));
        assert_rejects_unknown::<super::OutlierInputDto>(json!({
            "data": [1.0, 2.0], "threshold": 3.0
        }));
    }

    #[test]
    fn distribution_config_rejects_unknown_keys() {
        // The defect class that motivated this guard: a consumer typo
        // (`fit` instead of `fit_distributions`) was silently ignored.
        assert_rejects_unknown::<super::DistributionConfigDto>(json!({
            "fit": true
        }));
    }

    #[test]
    fn analysis_inputs_reject_unknown_keys() {
        assert_rejects_unknown::<super::RegressionInputDto>(json!({
            "predictors": { "x": [1.0, 2.0] }, "target": [1.0, 2.0],
            "target_name": "y", "intercept": true
        }));
        assert_rejects_unknown::<super::FeatureImportanceInputDto>(json!({
            "features": { "f": [1.0, 2.0] }, "target": [0.0, 1.0], "repeats": 5
        }));
    }

    #[test]
    fn a_class_label_that_is_not_a_whole_number_is_refused() {
        assert_eq!(
            super::class_labels(&[0.0, 1.0, 2.0]).expect("labels"),
            vec![0, 1, 2]
        );
        for (target, index) in [
            (vec![0.0, 1.7], 1),
            (vec![-2.0], 0),
            (vec![1.0, f64::NAN], 1),
        ] {
            let err = super::class_labels(&target).expect_err("not a label");
            assert_eq!(err.code(), "not_a_class_label");
            assert_eq!(err.fields["index"], index);
            assert_eq!(err.fields["parameter"], "target");
        }
    }
}
