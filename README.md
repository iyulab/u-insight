# u-insight

A high-performance statistical analysis and data profiling engine in Rust with C FFI bindings.

## Overview

u-insight transforms raw tabular data into actionable statistical insights. It operates in two distinct layers with **opposite assumptions about input data quality**:

```
CSV (raw)
  │
  ├─→ u-insight Profiling ─→ "What is the state of this data?"
  │     Tolerates dirty data.    (Basis for preprocessing decisions)
  │
  │   (external preprocessing)
  │
  └─→ u-insight Analysis  ─→ "What can we learn from this data?"
        Requires clean data.     (Patterns, relationships, structure)
```

Built on Rust's numerical ecosystem (`u-analytics`, `u-numflow`, `u-metaheur`).

## Two Layers

### Profiling — before preprocessing (dirty data tolerated)

Profiling accepts raw, messy data and reports what's in it. Missing values, type mismatches, and anomalies are **expected inputs**, not errors. The output provides the basis for preprocessing decisions.

| Capability | What it tells you |
|---|---|
| **Column Summary** | count, min, max, mean, median, std_dev, skewness, kurtosis, quantiles — computed over non-missing values |
| **Missing Analysis** | count, percentage, pattern (scattered, block, leading/trailing) per column |
| **Type Inference** | storage type → semantic type (numeric, categorical, boolean, datetime, text) |
| **Cardinality & Frequency** | distinct count, top-N values, constant column detection, imbalance ratio |
| **Outlier Flagging** | IQR, Z-score, Modified Z-score — univariate, per-column |
| **String Patterns** | email, URL, phone, date format detection; length statistics |
| **Data Quality Score** | composite score across completeness, uniqueness, validity |
| **Diagnostic Flags** | `CONSTANT_COLUMN`, `HIGH_MISSING`, `EXTREME_SKEWNESS`, `HIGH_CARDINALITY`, `ZERO_VARIANCE`, ... |

**Key contract:** Profiling never fails on missing or malformed data. It reports what it finds.

```rust
use u_insight::profiling::Profile;

let profile = Profile::from_csv("raw_data.csv")?;
for col in profile.columns() {
    println!("{}: missing={:.1}%, skew={:.2}", col.name, col.missing_pct, col.skewness);
    for flag in &col.flags {
        println!("  ⚠ {:?}", flag); // EXTREME_SKEWNESS, HIGH_MISSING, etc.
    }
}
// "X_Accel: missing=2.3%, skew=412.6"
// "  ⚠ EXTREME_SKEWNESS"
// "Z_Current: missing=0.0%, skew=0.0"
// "  ⚠ CONSTANT_COLUMN"
```

### Analysis — after preprocessing (clean data required)

Analysis accepts preprocessed, numeric data and discovers patterns and relationships. Missing values, non-numeric columns, and unscaled features are **caller's responsibility**.

| Capability | What it tells you |
|---|---|
| **Correlation** | Pearson, Spearman, Kendall matrices with p-values; multicollinearity (VIF) |
| **Regression** | Simple/multiple linear regression with R², ANOVA, coefficient significance |
| **Clustering** | K-Means, DBSCAN, HDBSCAN, hierarchical; optimal K selection |
| **Dimensionality Reduction** | PCA with explained variance ratios and component loadings |
| **Anomaly Detection** | Isolation Forest, LOF, Mahalanobis — multivariate, pattern-based |
| **Feature Importance** | Permutation importance, mutual information, ANOVA F-test, chi-squared |
| **Distribution Fitting** | Normality tests, KDE, best-fit distribution selection |

**Input contract for Analysis:**
- No missing values (returns `Err(InsightError::MissingValues)`)
- Numeric columns only (categorical must be encoded beforehand)
- Scaling is caller's choice — but K-Means and PCA are scale-dependent (auto-scaling option provided)

```rust
use u_insight::analysis::Analyzer;

let analysis = Analyzer::new(&clean_data)
    .correlation(Method::Pearson)?
    .clustering(KMeansConfig::auto(2..=10))?
    .pca(3)?
    .run()?;

println!("Correlation: {} high pairs", analysis.correlation.high_pairs(0.7).len());
println!("Clusters: K={}, silhouette={:.3}", analysis.clusters.k, analysis.clusters.silhouette);
println!("PCA: {:.1}% explained", analysis.pca.explained_variance_pct());
```

### Outlier Detection — spans both layers

Outlier detection serves different purposes in each layer:

| | Profiling layer | Analysis layer |
|---|---|---|
| **Purpose** | "How many suspicious values are there?" (preprocessing decision) | "Which observations are structurally anomalous?" (pattern discovery) |
| **Methods** | IQR, Z-score, Modified Z-score | Isolation Forest, LOF, Mahalanobis, MCD |
| **Scope** | Univariate, per-column | Multivariate, cross-column |
| **Missing data** | Tolerates (skips NaN) | Rejects |
| **Output** | Count, percentage, indices | Anomaly score, ranking |

### Typical Workflow

```rust
// 1. Profile — understand raw data
let profile = Profile::from_csv("raw_data.csv")?;
// "X_Accel: kurtosis=412.6 → extreme outliers present"
// "Z_Current: constant column → no analytical value"
// "SensorB: 34% missing → consider imputation or drop"

// 2. Preprocess externally (FilePrepper, custom code, etc.)
//    - Drop constant columns
//    - Handle missing values
//    - Encode categoricals
//    - Scale if needed

// 3. Analyze — discover patterns in clean data
let analysis = Analyzer::new(&clean_data)
    .correlation(Method::Pearson)?
    .clustering(KMeansConfig::auto(2..=10))?
    .run()?;
```

## C FFI Bindings

u-insight exposes both layers through a complete C FFI.

### Profiling FFI

```c
#include "u_insight.h"

// Profiling — accepts raw CSV
InsightResult insight_profile(const char* csv_path, ProfileOutput** out);
InsightResult insight_column_flags(ProfileOutput* profile, uint32_t col_idx, FlagList** out);

void insight_free_profile(ProfileOutput* ptr);
```

### Analysis FFI

```c
// Analysis — requires clean numeric data (no NaN)
InsightResult insight_correlation(const double* data, uint32_t rows, uint32_t cols,
                                  CorrType type, CorrMatrix** out);
InsightResult insight_kmeans(const double* data, uint32_t rows, uint32_t cols,
                             uint32_t k_min, uint32_t k_max, ClusterOutput** out);
InsightResult insight_pca(const double* data, uint32_t rows, uint32_t cols,
                          uint32_t n_components, PCAOutput** out);
InsightResult insight_outliers(const double* data, uint32_t rows, uint32_t cols,
                               OutlierMethod method, OutlierOutput** out);

void insight_free_correlation(CorrMatrix* ptr);
// ... etc.
```

### C# / .NET Example

```csharp
using UInsight.Interop;

// Profiling
var profile = Insight.Profile("raw_data.csv");
foreach (var col in profile.Columns)
{
    Console.WriteLine($"{col.Name}: missing={col.MissingPct:F1}%");
    foreach (var flag in col.Flags)
        Console.WriteLine($"  ⚠ {flag}");
}

// Analysis (after preprocessing)
var corr = Insight.Correlation(cleanData, CorrelationType.Pearson);
var clusters = Insight.KMeans(cleanData, kRange: (2, 10));
```

## Installation

### As Rust Crate

```toml
[dependencies]
u-insight = "0.1"
```

### Pre-built FFI Libraries

| Platform | Library |
|----------|---------|
| Windows x64 | `u_insight.dll` |
| Linux x64 | `libu_insight.so` |
| macOS x64 | `libu_insight.dylib` |
| macOS ARM64 | `libu_insight.dylib` |

### NuGet (C# Bindings)

```bash
dotnet add package UInsight.Interop
```

## Scope & Non-Goals

**In Scope:**
- Data profiling (dirty data → quality report + flags)
- Statistical analysis (clean data → patterns + relationships)
- Correlation, regression, clustering, PCA
- Outlier detection (univariate profiling + multivariate analysis)
- Feature importance and selection
- Complete C FFI for cross-language use

**Out of Scope:**
- Visualization / charting
- Data cleaning / transformation / imputation
- ML model training / deployment
- Deep learning
- Interactive notebooks / REPL

## Requirements

- Rust 1.75+
- Dependencies: `u-analytics`, `u-numflow`, `u-metaheur`

## Related

- [u-analytics](https://github.com/iyulab/u-analytics) -- Statistical analytics
- [u-numflow](https://github.com/iyulab/u-numflow) -- Mathematical primitives
- [u-metaheur](https://github.com/iyulab/u-metaheur) -- Metaheuristic optimization

## License

MIT License
