//! # u-insight
//!
//! Statistical analysis and data profiling engine with C FFI bindings.
//!
//! u-insight transforms raw tabular data into actionable statistical
//! insights. It operates in two distinct layers:
//!
//! - **Profiling** — tolerates dirty data, reports data quality and statistics
//! - **Analysis** — requires clean data, discovers patterns and relationships
//!
//! ## Modules
//!
//! - [`dataframe`] — Column-major tabular data model (DataFrame, Column, DataType)
//! - [`csv_parser`] — CSV parsing with automatic type inference
//! - [`profiling`] — Column-level and dataset-level data profiling
//! - [`analysis`] — Correlation (Pearson/Spearman), regression (simple/multiple OLS), Cramér's V
//! - [`clustering`] — K-Means++ (auto-K, Gap Statistic), Mini-Batch K-Means, DBSCAN, Hierarchical (4 linkages), HDBSCAN
//! - [`distribution`] — ECDF, histogram, QQ-plot, normality tests (KS, JB, Shapiro-Wilk, Anderson-Darling), Grubbs, distribution fitting
//! - [`pca`] — Principal Component Analysis (dimensionality reduction)
//! - [`isolation_forest`] — Isolation Forest anomaly detection (Liu et al. 2008)
//! - [`lof`] — Local Outlier Factor (LOF) density-based anomaly detection
//! - [`mahalanobis`] — Mahalanobis distance multivariate outlier detection
//! - [`feature_importance`] — Composite importance, ANOVA F-test, Mutual Information, Permutation Importance
//! - [`ffi`] — C FFI bindings (32 functions, 20 structs, auto-generated C header via cbindgen)
//! - [`error`] — Error types
//!
//! ## Quick Start
//!
//! ```
//! use u_insight::csv_parser::CsvParser;
//! use u_insight::dataframe::DataType;
//!
//! let csv = "name,value,active\nAlice,1.5,true\nBob,2.3,false\nCharlie,3.1,true\n";
//! let df = CsvParser::new().parse_str(csv).unwrap();
//!
//! assert_eq!(df.row_count(), 3);
//! assert_eq!(df.column_count(), 3);
//!
//! // Type inference: name=Text, value=Numeric, active=Boolean
//! let schema = df.schema();
//! assert_eq!(schema[1].1, DataType::Numeric);
//! assert_eq!(schema[2].1, DataType::Boolean);
//! ```

pub mod analysis;
pub mod clustering;
pub mod csv_parser;
pub mod dataframe;
pub mod distribution;
pub mod error;
pub mod feature_importance;
pub mod ffi;
pub mod isolation_forest;
pub mod lof;
pub mod mahalanobis;
pub mod pca;
pub mod profiling;
