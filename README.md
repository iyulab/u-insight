# u-insight

A statistical analysis and data profiling engine in Rust with C FFI bindings.

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
| `analysis` | Correlation (Pearson/Spearman) and regression (simple/multiple OLS) |
| `clustering` | K-Means++ with auto-K selection, DBSCAN density-based clustering |
| `distribution` | ECDF, histogram bins (Sturges/Scott/FD), QQ-plot, KS and Jarque-Bera normality tests |
| `pca` | Principal Component Analysis with auto-scaling option |
| `isolation_forest` | Isolation Forest anomaly detection (Liu et al. 2008) |
| `feature_importance` | Variance threshold, correlation filter, VIF, condition number, composite importance scores |

### FFI Layer

| Module | Description |
|--------|-------------|
| `ffi` | C FFI bindings — profiling, K-Means, DBSCAN, PCA, distribution analysis, feature importance |

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

u-insight builds as `cdylib` + `staticlib` for cross-language interop.

| Function | Description |
|----------|-------------|
| `insight_profile_csv` | Profile a CSV string |
| `insight_kmeans` | K-Means clustering |
| `insight_dbscan` | DBSCAN clustering |
| `insight_pca` | Principal Component Analysis |
| `insight_distribution` | Normality testing |
| `insight_feature_importance` | Feature importance scores |
| `insight_last_error` | Last error message |
| `insight_version` | Library version |

All FFI functions use `catch_unwind` to prevent panics from crossing the FFI boundary.

## Test Status

```
221 lib tests + 32 doc-tests = 253 total
0 clippy warnings
Build: lib + cdylib + staticlib
```

## Scope & Non-Goals

**In Scope:**
- Data profiling (dirty data → quality report + diagnostic flags)
- Statistical analysis (clean data → patterns + relationships)
- Correlation, regression, clustering, PCA, anomaly detection
- Feature importance and selection
- Distribution analysis and normality testing
- C FFI for cross-language use

**Out of Scope:**
- Visualization / charting
- Data cleaning / transformation / imputation
- ML model training / deployment
- Deep learning

## Requirements

- Rust 1.75+
- Dependencies: `u-analytics`, `u-numflow`

## Related

- [u-analytics](https://github.com/iyulab/u-analytics) -- Statistical analytics
- [u-numflow](https://github.com/iyulab/u-numflow) -- Mathematical primitives

## License

MIT License
