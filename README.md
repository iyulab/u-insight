# u-insight

[![Crates.io](https://img.shields.io/crates/v/u-insight.svg)](https://crates.io/crates/u-insight)
[![NuGet](https://img.shields.io/nuget/v/UInsight.svg)](https://www.nuget.org/packages/UInsight)
[![docs.rs](https://docs.rs/u-insight/badge.svg)](https://docs.rs/u-insight)
[![CI](https://github.com/iyulab/u-insight/actions/workflows/ci.yml/badge.svg)](https://github.com/iyulab/u-insight/actions/workflows/ci.yml)

A statistical analysis and data profiling engine in Rust with C FFI bindings.

## What's New in 0.18.0

- **Univariate time-series primitives**, pure Rust on every transport
  (`u-analytics` 0.11 / `u-numflow` 0.5): `insight_estimate_period` — the
  dominant period of a series (AutoPeriod: permutation-thresholded periodogram
  peaks refined on the autocorrelation function, deterministic) — and
  `insight_spectral_residual` — one-shot anomaly scoring by spectral residual
  saliency (Ren et al. 2019) with an expected value and a coverage band per
  point. C# `EstimatePeriod` / `SpectralResidual(data, options?)`; WASM
  `estimate_period` / `spectral_residual`.

## What's New in 0.14.0

- **26 new FFI functions** exposing `u-analytics` domains that were already a
  dependency but not previously reachable from outside this crate: Mann-Kendall
  trend test, kernel density estimation, the full SPC control chart family
  (X-bar-R/S, Individual-MR, P/NP/C/U, Laney P'/U', G/T), process capability
  (Cp/Cpk/Pp/Ppk/Cpm, Box-Cox, percentile-based, sigma↔PPM), and Weibull
  reliability analysis (MLE/MRR fitting, survival, hazard, MTBF, B-life). See
  the C FFI section below and `CHANGELOG.md` for the full function list.
- Matching C# bindings for all of the above, including the binding's first
  `double?`-based optional parameters/return values (`ProcessCapability`,
  `PpmToSigma`, etc.), converting to/from `NaN` only at the P/Invoke boundary.

## What's New in 0.9.1

- **BREAKING — Rust**: `InsightError::NonNumericColumn` variant removed. The 0.9.0 audit redirected all internal call sites to `DegenerateData`, leaving the variant unused. Removed per `Delete over deprecate` policy. External `match` arms over `InsightError` must drop the corresponding branch.

## What's New in 0.9.0

- **Kendall tau-b correlation** added to `CorrelationMethod` (Pearson / Spearman / Kendall)
- **Outlier fences exposed** on `OutlierResult` — `lower_fence`, `upper_fence`, `center`, `spread`
- **`detect_outliers_slice(&[f64], method)`** helper for raw-slice input
- **`vif_analysis()` and `condition_number()`** standalone multicollinearity diagnostics
- **WASM**: `correlation_matrix` accepts optional `_method` field (`"pearson"` | `"spearman"` | `"kendall"`); new `detect_univariate_outliers`, `vif_diagnostic`, `condition_number_diagnostic`
- **C#**: `CorrelationMethodKind` enum + `Correlate(data, method)` parameter
- **BREAKING — Rust**: Non-finite numeric inputs now return `InsightError::DegenerateData` (was `NonNumericColumn`); audit covered 11 call sites
- **BREAKING — Rust**: `OutlierResult` gained 4 new fields (exhaustive pattern matches must be updated)
- **BREAKING — FFI**: `insight_correlation` signature gained a `method: u32` parameter (use `INSIGHT_CORR_PEARSON` = 0 to keep prior behaviour)
- **BREAKING — C#**: `Correlate(...)` now takes an optional `CorrelationMethodKind` parameter (default `Pearson` keeps existing call-sites compiling)

## Overview

u-insight transforms raw tabular data into actionable statistical insights. It operates in two distinct layers with **opposite assumptions about input data quality**:

```
CSV (raw)
  │
  ├─→ Profiling ─→ "What is the state of this data?"
  │     Tolerates dirty data (missing values, type mismatches expected)
  │
  │   (external preprocessing)
  │
  └─→ Analysis  ─→ "What can we learn from this data?"
        Requires clean numeric data (no NaN, no missing)
```

Built on `u-analytics` (statistical algorithms), `u-numflow` (math primitives).

## Modules

### Data Layer

| Module | Description |
|--------|-------------|
| `dataframe` | Column-major tabular data model (DataFrame, Column, DataType) |
| `csv_parser` | CSV parsing with automatic type inference |
| `error` | Error types (InsightError) |

### Profiling Layer (dirty data tolerated)

| Module | Description |
|--------|-------------|
| `profiling` | Column-level and dataset-level data profiling — descriptive stats, missing analysis, outlier flagging (IQR/Z-score/Modified Z-score), diagnostic flags |

### Analysis Layer (clean data required)

| Module | Description |
|--------|-------------|
| `analysis` | Correlation (Pearson/Spearman), regression (simple/multiple OLS), Cramer's V contingency analysis |
| `clustering` | K-Means++ (auto-K, Gap Statistic), Mini-Batch K-Means, DBSCAN, Hierarchical Agglomerative (Single/Complete/Average/Ward), HDBSCAN |
| `distribution` | ECDF, histogram bins (Sturges/Scott/FD), QQ-plot, normality tests (KS, Jarque-Bera, Shapiro-Wilk, Anderson-Darling), Grubbs test, distribution fitting |
| `pca` | Principal Component Analysis with auto-scaling option |
| `isolation_forest` | Isolation Forest anomaly detection (Liu et al. 2008) |
| `lof` | Local Outlier Factor (LOF) density-based anomaly detection |
| `mahalanobis` | Mahalanobis distance multivariate outlier detection |
| `feature_importance` | Variance threshold, correlation filter, VIF, condition number, composite importance, ANOVA F-test selection, Mutual Information, Permutation Importance |

### FFI Layer

| Module | Description |
|--------|-------------|
| `ffi` | C FFI bindings — 32 functions, 20 `#[repr(C)]` structs, auto-generated C header via cbindgen |

## Quick Start

```rust
use u_insight::csv_parser::CsvParser;
use u_insight::profiling::profile_dataframe;

// 1. Parse CSV
let csv = "name,value,active\nAlice,1.5,true\nBob,2.3,false\nCharlie,3.1,true\n";
let df = CsvParser::new().parse_str(csv).unwrap();

// 2. Profile
let profiles = profile_dataframe(&df);
```

### Clustering

```rust
use u_insight::clustering::{kmeans, dbscan, KMeansConfig, DbscanConfig};

let data = vec![
    vec![0.0, 0.0], vec![0.5, 0.5],
    vec![10.0, 10.0], vec![10.5, 10.5],
];

// K-Means
let km = kmeans(&data, &KMeansConfig::new(2)).unwrap();
assert_eq!(km.k, 2);

// DBSCAN
let db = dbscan(&data, &DbscanConfig::new(1.5, 2)).unwrap();
assert_eq!(db.n_clusters, 2);
```

### Distribution Analysis

```rust
use u_insight::distribution::{distribution_analysis, DistributionConfig};

let data: Vec<f64> = (0..50).map(|i| (i as f64 - 25.0) * 0.2).collect();
let result = distribution_analysis(&data, &DistributionConfig::default()).unwrap();
println!("Normal: {}", result.normality.is_normal);
```

## C FFI

u-insight builds as `cdylib` + `staticlib` for cross-language interop. A C header (`u_insight.h`) is auto-generated by cbindgen at build time.

### Profiling

| Function | Description |
|----------|-------------|
| `insight_profile_csv` | Profile a CSV string → opaque context |
| `insight_profile_json` | Profile a JSON string → opaque context |
| `insight_profile_free` | Free profile context |
| `insight_profile_row_count` | Row count from profile |
| `insight_profile_col_count` | Column count from profile |
| `insight_profile_column` | Get column summary |

### Clustering

| Function | Description |
|----------|-------------|
| `insight_kmeans` | K-Means++ clustering |
| `insight_mini_batch_kmeans` | Mini-Batch K-Means clustering |
| `insight_dbscan` | DBSCAN density-based clustering |
| `insight_hierarchical` | Hierarchical Agglomerative clustering (4 linkages) |
| `insight_hdbscan` | HDBSCAN clustering with membership probabilities |
| `insight_gap_statistic` | Gap statistic for optimal K selection |
| `insight_silhouette` | Silhouette score for cluster validation |

### Dimensionality Reduction

| Function | Description |
|----------|-------------|
| `insight_pca` | Principal Component Analysis |

### Anomaly Detection

| Function | Description |
|----------|-------------|
| `insight_isolation_forest` | Isolation Forest anomaly detection |
| `insight_lof` | Local Outlier Factor detection |
| `insight_mahalanobis` | Mahalanobis distance outlier detection |

### Statistical Analysis

| Function | Description |
|----------|-------------|
| `insight_correlation` | Pearson correlation matrix |
| `insight_regression` | Simple linear regression |
| `insight_cramers_v` | Cramer's V contingency analysis |

### Distribution

| Function | Description |
|----------|-------------|
| `insight_distribution` | Normality testing (KS, JB, SW, AD) |

### Changepoint Detection

| Function | Description |
|----------|-------------|
| `insight_pelt` | PELT changepoint detection (univariate) |
| `insight_pelt_multi` | PELT changepoint detection (multivariate) |

### Time Series

| Function | Description |
|----------|-------------|
| `insight_estimate_period` | Dominant period of a series (AutoPeriod); `period` 0 = none, candidates listed |
| `insight_free_period_estimate` | Frees the candidates of a `CPeriodEstimate` |
| `insight_spectral_residual` | Spectral residual anomaly scoring (Ren et al. 2019); null options = paper defaults |
| `insight_free_spectral_residual_result` | Frees the points of a `CSpectralResidualResult` |

### Trend & Density Estimation

| Function | Description |
|----------|-------------|
| `insight_mann_kendall` | Mann-Kendall trend test with Sen's slope |
| `insight_kde` | Gaussian kernel density estimation (Silverman/Scott/manual bandwidth) |

### SPC — Variables Control Charts

| Function | Description |
|----------|-------------|
| `insight_xbar_r_chart` | X-bar-R control chart (subgroup mean + range) |
| `insight_xbar_s_chart` | X-bar-S control chart (subgroup mean + std dev) |
| `insight_individual_mr_chart` | Individual-MR control chart |

### SPC — Attributes Control Charts

| Function | Description |
|----------|-------------|
| `insight_p_chart` | P chart (proportion nonconforming) |
| `insight_np_chart` | NP chart (count nonconforming, constant sample size) |
| `insight_c_chart` | C chart (defect count, constant area) |
| `insight_u_chart` | U chart (defects per unit, variable area) |
| `insight_laney_p_chart` | Laney P' chart (overdispersion-adjusted) |
| `insight_laney_u_chart` | Laney U' chart (overdispersion-adjusted) |
| `insight_g_chart` | G chart (rare-event, geometric distribution) |
| `insight_t_chart` | T chart (rare-event, exponential distribution) |

### Process Capability

| Function | Description |
|----------|-------------|
| `insight_process_capability` | Standard capability indices (Cp/Cpk/Pp/Ppk/Cpm) |
| `insight_boxcox_capability` | Non-normal capability via Box-Cox transformation |
| `insight_percentile_capability` | Percentile-based capability (ISO 22514-2) |
| `insight_sigma_to_ppm` | Sigma quality level → PPM defect rate |
| `insight_ppm_to_sigma` | PPM defect rate → sigma quality level |

### Weibull Reliability

| Function | Description |
|----------|-------------|
| `insight_weibull_mle` | Weibull parameter fitting (Maximum Likelihood Estimation) |
| `insight_weibull_mrr` | Weibull parameter fitting (Median Rank Regression) |
| `insight_weibull_reliability` | Reliability (survival) function R(t) |
| `insight_weibull_hazard_rate` | Hazard (instantaneous failure) rate |
| `insight_weibull_mtbf` | Mean Time Between Failures |
| `insight_weibull_time_to_reliability` | Time at which reliability drops to a given level |
| `insight_weibull_b_life` | B-life (time at which a given fraction has failed) |

### Feature Importance

| Function | Description |
|----------|-------------|
| `insight_feature_importance` | Composite feature importance scores |
| `insight_anova_select` | ANOVA F-test feature selection |
| `insight_mutual_info` | Mutual information feature ranking |
| `insight_permutation_importance` | Permutation importance for regression |

### Memory Management

| Function | Description |
|----------|-------------|
| `insight_free_labels` | Free u32 label arrays |
| `insight_free_i32_array` | Free i32 arrays |
| `insight_free_f64_array` | Free f64 arrays |
| `insight_free_anova_features` | Free ANOVA feature arrays |
| `insight_free_mi_features` | Free MI feature arrays |
| `insight_free_perm_features` | Free permutation importance arrays |
| `insight_free_pelt_result` | Free PELT changepoint results |
| `insight_free_kde_result` | Free KDE results |
| `insight_free_variables_chart_result` | Free variables control chart results |
| `insight_free_attribute_chart_result` | Free attributes control chart results |
| `insight_free_laney_chart_result` | Free Laney P'/U' chart results |
| `insight_free_rare_event_chart_result` | Free G/T chart results |

### Error & Version

| Function | Description |
|----------|-------------|
| `insight_last_error` | Last error message (thread-local) |
| `insight_clear_error` | Clear error state |
| `insight_version` | Library version string |

All FFI functions that accept data pointers use `catch_unwind` to prevent panics from crossing the FFI boundary. A handful of pure closed-form scalar conversions (e.g. `insight_sigma_to_ppm`, `insight_weibull_reliability`) skip the `catch_unwind`/error-code ceremony and return the value directly, since they cannot panic and have no data to validate.

## C# Binding (UInsight)

Install via NuGet — native libraries are bundled automatically:

```bash
dotnet add package UInsight
```

```csharp
using UInsight;

using var client = new InsightClient();
Console.WriteLine(client.GetVersion());

var data = new double[,] { {0,0}, {1,1}, {10,10}, {11,11} };
var result = client.KMeans(data, k: 2);
Console.WriteLine($"K={result.K}, WCSS={result.Wcss:F2}");
```

The binding is in `bindings/csharp/UInsight/` with:

- `Interop/NativeLibrary.cs` — `[LibraryImport]` declarations for all 67 FFI functions
- `Interop/NativeStructs.cs` — `[StructLayout]` mappings for all 35 C structs
- `InsightClient.cs` — High-level managed API (automatic memory management)
- `InsightException.cs` — Error code to exception conversion

## Test Status

```
474 lib tests + 53 doc-tests = 527 total
0 clippy warnings
Build: lib + cdylib + staticlib
C header: auto-generated via cbindgen (35 structs, 67 functions)
```

## Scope & Non-Goals

**In Scope:**
- Data profiling (dirty data → quality report + diagnostic flags)
- Statistical analysis (clean data → patterns + relationships)
- Correlation, regression, clustering, PCA, anomaly detection
- Feature importance and selection (ANOVA, MI, Permutation)
- Distribution analysis and normality testing
- C FFI for cross-language use
- C# binding (UInsight NuGet package)

**Out of Scope:**
- Visualization / charting
- Data cleaning / transformation / imputation
- ML model training / deployment
- Deep learning

## Requirements

- Rust 1.85+
- Dependencies: `u-analytics`, `u-numflow`

## WebAssembly / npm

Available as an npm package via [wasm-pack](https://rustwasm.github.io/wasm-pack/).

```bash
npm install @iyulab/u-insight
```

### Quick Start

```javascript
import init, { describe, kmeans } from '@iyulab/u-insight';

await init();
const stats = describe({ col1: [1, 2, 3], col2: [4, 5, 6] });
```

### Functions

#### `describe(data) -> [ColumnResult]`

Descriptive statistics per column. Input: column-major `{ "col1": [1,2,3] }`.

**Output:** Array of `{ name, data_type, numeric: { count, min, max, mean, median, std_dev, variance, skewness, kurtosis, q1, q3, iqr, p5, p95, ... } }`.

#### `correlation_matrix(data) -> CorrelationResult`

Pearson correlation matrix. Input: column-major `{ "col1": [1,2,3], "col2": [4,5,6] }`.

**Output:**
```json
{ "names": ["col1","col2"], "matrix": [1,0.99,0.99,1], "n": 2, "high_pairs": [{ "col_a": "col1", "col_b": "col2", "r": 0.99, "p_value": 0.01 }] }
```

#### `kmeans(data, k) -> KMeansResult`

K-Means++ clustering on row-major data `[[x,y,...], ...]`.

**Output:**
```json
{ "k": 3, "labels": [0,0,1,1,2,2], "centroids": [[...]], "wcss": 5.2, "iterations": 12, "cluster_sizes": [2,2,2] }
```

#### `silhouette(data, labels, k) -> SilhouetteResult`

Silhouette analysis for an existing clustering assignment. Works with any clustering output (`kmeans`, `dbscan`, `hierarchical`, etc.). `data` is row-major `[[x,y,...], ...]`, `labels` is one cluster id per row (each `< k`), `k` is the number of distinct clusters. O(n²) — use sparingly on very large inputs.

**Output:**
```json
{ "avg": 0.74, "per_sample": [0.81, 0.79, 0.62, ...] }
```

`avg` ranges from -1 (wrong cluster) to +1 (well-separated); singleton-cluster points report 0.0 in `per_sample`.

#### `pca(data, n_components) -> PcaResult`

Principal Component Analysis on row-major data.

**Output:**
```json
{ "n_components": 2, "n_features": 4, "eigenvalues": [3.1,0.9], "explained_variance_ratio": [0.77,0.23], "cumulative_variance_ratio": [0.77,1.0], "loadings": [[...]], "scores": [[...]], "means": [...], "stds": [...] }
```

#### `dbscan(data, config) -> DbscanResult`

DBSCAN density-based clustering. `config`: `{ "epsilon": 1.5, "min_samples": 3 }`.

**Output:**
```json
{ "labels": [0,0,null,1,1], "n_clusters": 2, "noise_count": 1, "cluster_sizes": [2,2], "core_points": [true,true,false,true,true] }
```

#### `hierarchical(data, config) -> HierarchicalResult`

Hierarchical agglomerative clustering (nearest-neighbor-chain, **O(n²)** time / O(n²) memory). `config`: `{ "linkage": "ward", "n_clusters": 3 }` or `{ "linkage": "single", "distance_threshold": 5.0 }`.

**Config fields:**
- `linkage` — `"single" | "complete" | "average" | "ward"` (default `"ward"`).
- `n_clusters` — flat clusters to extract (mutually exclusive with `distance_threshold`).
- `distance_threshold` — dendrogram cut height (mutually exclusive with `n_clusters`).
- `max_points` — memory guard; inputs with more points are rejected before allocating the O(n²) distance matrix. Omit for the default (`10000`, ≈400 MB matrix); set `0` to disable. Raise it for large native batches; lower it for tight memory (e.g. a browser tab).

```js
// large dataset on a memory-constrained page: cap it explicitly
hierarchical(data, { linkage: "ward", n_clusters: 3, max_points: 5000 });
```

**Output:**
```json
{ "merges": [{ "cluster_a": 0, "cluster_b": 1, "distance": 1.2, "size": 2 }], "labels": [0,0,1,1,2], "n_clusters": 3 }
```

#### `isolation_forest(data, config) -> IsolationForestResult`

Isolation Forest anomaly detection. `config`: `{ "n_estimators": 100, "contamination": 0.1, "seed": 42 }`.

**Output:**
```json
{ "scores": [0.45, 0.82], "anomalies": [false, true], "threshold": 0.65, "anomaly_count": 1, "anomaly_fraction": 0.5 }
```

#### `lof(data, config) -> LofResult`

Local Outlier Factor anomaly detection. `config`: `{ "k": 20, "threshold": 1.5 }`.

**Output:**
```json
{ "scores": [1.0, 2.3], "anomalies": [false, true], "threshold": 1.5, "anomaly_count": 1, "anomaly_fraction": 0.5 }
```

#### `distribution_analysis(data, config) -> DistributionResult`

Distribution analysis on a 1-D array. `config`: `{ "bin_method": "freedman_diaconis", "bins": null, "significance_level": 0.05, "compute_ecdf": true, "compute_histogram": true, "compute_qq_plot": true, "fit_distributions": false }`.

- `bin_method`: `"sturges" | "scott" | "freedman_diaconis"` — automatic bin count rule (default `"freedman_diaconis"`).
- `bins` (optional, integer >= 1): explicit histogram bin count. When set it takes precedence over `bin_method`, and the histogram `method` field echoes `"Fixed(n)"`.

**Output:**
```json
{ "n": 100, "ecdf": { "values": [...], "probabilities": [...] }, "histogram": { "n_bins": 10, "bin_width": 0.5, "edges": [...], "counts": [...] }, "qq_plot": { "theoretical": [...], "sample": [...] }, "normality": { "shapiro_wilk": { "statistic": 0.98, "p_value": 0.45, "rejected": false }, "is_normal": true }, "fits": [] }
```

#### `regression(data) -> RegressionResult`

OLS regression analysis.

**Input:**
```json
{ "predictors": { "x1": [1,2,3,4,5] }, "target": [2.1, 3.9, 6.1, 7.9, 10.1], "target_name": "y" }
```

**Output:**
```json
{ "target_name": "y", "predictor_names": ["x1"], "r_squared": 0.99, "adj_r_squared": 0.99, "coefficients": [0.1, 2.0], "p_values": [0.9, 0.0001], "vif": [1.0], "f_p_value": 0.0001 }
```

#### `feature_importance(data) -> FeatureImportanceResult`

Feature importance via permutation, ANOVA, or mutual information.

**Input:**
```json
{ "features": { "f1": [1,2,3], "f2": [5,4,3] }, "target": [0,0,1], "method": "permutation", "n_repeats": 5, "seed": 42 }
```

**Output:**
```json
{ "method": "permutation", "features": [{ "name": "f1", "index": 0, "score": 0.8, "std_dev": 0.1 }], "baseline_score": 0.5 }
```

#### `estimate_period(data) -> PeriodEstimate`

Dominant period of a univariate series — AutoPeriod (Vlachos, Yu & Castelli
2005): peaks of the detrended, zero-padded periodogram above a permutation
threshold (100 seeded shuffles, so the estimate is deterministic), each refined
on the autocorrelation function to the integer lag that is a local maximum
above the `1.96/√n` bound. At least 8 finite values.

**Input:** `{ "data": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1] }`

**Output:**
```json
{ "period": 7, "candidates": [{ "period": 7, "acf": 0.71, "bin": 18, "power": 21.3, "power_share": 0.62 }],
  "n": 16, "acf_threshold": 0.49, "power_threshold": 6.8 }
```

`period` is `null` — explicitly, not an error — when no periodicity passes both
stages (a constant, a pure trend, white noise). Only periods from 2 to `n/2`
are admissible.

#### `spectral_residual(data) -> SpectralResidualResult`

Score every point for anomalies by spectral residual saliency (Ren et al.
2019) — spikes, steps and dropouts, without a trained model and without
assuming a period. At least 12 finite values; the options default to the
paper's.

**Input:**
```json
{ "data": [1, 1.1, 0.9, 1, 6, 1, 1.1, 0.9, 1, 1, 1.1, 0.9],
  "averaging_window": 3, "judgement_window": 40, "threshold": 3.0,
  "min_zscore": 1.5, "sensitivity": 70, "batch_size": null }
```

**Output:**
```json
{ "points": [{ "index": 4, "value": 6, "saliency": 2.1, "score": 5.3,
               "expected": 1.0, "lower": 0.9, "upper": 1.1, "is_anomaly": true }],
  "anomalies": [4] }
```

`expected` is the low-frequency reconstruction of the series with its anomalies
replaced by their neighbours and `lower`/`upper` the band of `sensitivity`
percent coverage around it — chart information; the anomaly decision is the
`score` against `threshold`, gated by `min_zscore` against the level of the
window before the point.

## npm (WebAssembly)

```bash
npm install @iyulab/u-insight
```

The package resolves per environment via a conditional `exports` map:

| Environment | Entry |
|---|---|
| Bundlers (webpack, Vite, …) | ESM + WebAssembly ESM-integration (`default` condition) |
| Node.js — `require()`, ESM `import`, CJS TS runners (`tsx`, `ts-node`) | CJS glue loading the wasm from the filesystem (`node` condition) — no loader hooks or flags |

## Related

- [u-analytics](https://github.com/iyulab/u-analytics) -- Statistical analytics
- [u-numflow](https://github.com/iyulab/u-numflow) -- Mathematical primitives

## License

MIT License
