//! Column-level and dataset-level data profiling.
//!
//! The profiling module analyzes raw data and reports statistics, missing
//! patterns, cardinality, and diagnostic flags. It tolerates dirty data —
//! missing values are expected input, not errors.
//!
//! # Example
//!
//! ```
//! use u_insight::csv_parser::CsvParser;
//! use u_insight::profiling::profile_dataframe;
//!
//! let csv = "x,y\n1.0,A\n2.0,B\nNA,A\n4.0,A\n5.0,B\n";
//! let df = CsvParser::new().parse_str(csv).unwrap();
//! let profiles = profile_dataframe(&df);
//!
//! assert_eq!(profiles.len(), 2);
//! let x_prof = &profiles[0];
//! assert_eq!(x_prof.name, "x");
//! assert_eq!(x_prof.null_count, 1);
//! assert!(x_prof.numeric.is_some());
//! ```

use crate::dataframe::{Column, DataFrame, DataType};
use std::collections::{HashMap, HashSet};

// ── Numeric Profile ───────────────────────────────────────────────────

/// Descriptive statistics for a numeric column (computed over valid values only).
#[derive(Debug, Clone)]
pub struct NumericProfile {
    /// Number of valid (non-null) values.
    pub valid_count: usize,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// Sample standard deviation.
    pub std_dev: f64,
    /// Sample variance.
    pub variance: f64,
    /// Skewness (Fisher's definition, bias-corrected G1).
    pub skewness: f64,
    /// Excess kurtosis (Fisher's definition, bias-corrected G2).
    pub kurtosis: f64,
    /// 5th percentile.
    pub p5: f64,
    /// 25th percentile (Q1).
    pub q1: f64,
    /// 75th percentile (Q3).
    pub q3: f64,
    /// 95th percentile.
    pub p95: f64,
    /// Interquartile range (Q3 - Q1).
    pub iqr: f64,
    /// Count of zero values.
    pub zero_count: usize,
    /// Count of negative values.
    pub negative_count: usize,
    /// Count of infinite values (±∞).
    pub infinity_count: usize,
    /// Number of distinct values.
    pub distinct_count: usize,
}

// ── Boolean Profile ───────────────────────────────────────────────────

/// Statistics for a boolean column.
#[derive(Debug, Clone)]
pub struct BooleanProfile {
    /// Number of valid (non-null) values.
    pub valid_count: usize,
    /// Count of `true` values.
    pub true_count: usize,
    /// Count of `false` values.
    pub false_count: usize,
    /// Proportion of `true` among valid values.
    pub true_ratio: f64,
}

// ── Categorical Profile ───────────────────────────────────────────────

/// Statistics for a categorical column.
#[derive(Debug, Clone)]
pub struct CategoricalProfile {
    /// Number of valid (non-null) values.
    pub valid_count: usize,
    /// Number of distinct categories.
    pub distinct_count: usize,
    /// Top-N most frequent categories (category, count), sorted by count desc.
    pub top_values: Vec<(String, usize)>,
    /// Ratio of the most frequent category to valid count.
    pub mode_ratio: f64,
    /// Whether the column is constant (all same value).
    pub is_constant: bool,
}

// ── Text Profile ──────────────────────────────────────────────────────

/// Statistics for a text column.
#[derive(Debug, Clone)]
pub struct TextProfile {
    /// Number of valid (non-null) values.
    pub valid_count: usize,
    /// Number of distinct values.
    pub distinct_count: usize,
    /// Minimum string length among valid values.
    pub min_length: usize,
    /// Maximum string length among valid values.
    pub max_length: usize,
    /// Mean string length.
    pub mean_length: f64,
    /// Count of empty strings (length 0) among valid values.
    pub empty_count: usize,
}

// ── Column Profile ────────────────────────────────────────────────────

/// Complete profile for a single column.
#[derive(Debug, Clone)]
pub struct ColumnProfile {
    /// Column name.
    pub name: String,
    /// Inferred data type.
    pub data_type: DataType,
    /// Total number of rows.
    pub row_count: usize,
    /// Number of null (missing) values.
    pub null_count: usize,
    /// Missing value percentage (0.0 to 100.0).
    pub missing_pct: f64,
    /// Numeric-specific statistics (if numeric column).
    pub numeric: Option<NumericProfile>,
    /// Boolean-specific statistics (if boolean column).
    pub boolean: Option<BooleanProfile>,
    /// Categorical-specific statistics (if categorical column).
    pub categorical: Option<CategoricalProfile>,
    /// Text-specific statistics (if text column).
    pub text: Option<TextProfile>,
}

// ── Profiling functions ───────────────────────────────────────────────

/// Profiles all columns in a DataFrame.
///
/// Returns one [`ColumnProfile`] per column, in the same order as
/// the DataFrame's columns.
///
/// ```
/// use u_insight::csv_parser::CsvParser;
/// use u_insight::profiling::profile_dataframe;
///
/// let csv = "temp,status\n20.5,OK\n21.3,OK\nNA,FAIL\n19.8,OK\n";
/// let df = CsvParser::new().parse_str(csv).unwrap();
/// let profiles = profile_dataframe(&df);
///
/// let temp = &profiles[0];
/// assert_eq!(temp.null_count, 1);
/// assert!(temp.numeric.is_some());
/// ```
pub fn profile_dataframe(df: &DataFrame) -> Vec<ColumnProfile> {
    df.iter()
        .map(|(name, col)| profile_column(name, col))
        .collect()
}

/// Profiles a single column.
pub fn profile_column(name: &str, col: &Column) -> ColumnProfile {
    let row_count = col.len();
    let null_count = col.null_count();
    let missing_pct = if row_count > 0 {
        (null_count as f64 / row_count as f64) * 100.0
    } else {
        0.0
    };

    let (numeric, boolean, categorical, text) = match col {
        Column::Numeric { values, validity } => {
            (Some(profile_numeric(values, validity)), None, None, None)
        }
        Column::Boolean { values, validity } => {
            (None, Some(profile_boolean(values, validity)), None, None)
        }
        Column::Categorical {
            dictionary,
            indices,
            validity,
        } => (
            None,
            None,
            Some(profile_categorical(dictionary, indices, validity)),
            None,
        ),
        Column::Text { values, validity } => {
            (None, None, None, Some(profile_text(values, validity)))
        }
    };

    ColumnProfile {
        name: name.to_string(),
        data_type: col.data_type(),
        row_count,
        null_count,
        missing_pct,
        numeric,
        boolean,
        categorical,
        text,
    }
}

// ── Internal profiling helpers ────────────────────────────────────────

fn profile_numeric(
    values: &[f64],
    validity: &crate::dataframe::ValidityBitmap,
) -> NumericProfile {
    // Extract valid values
    let valid: Vec<f64> = validity
        .valid_indices()
        .map(|i| values[i])
        .collect();

    let valid_count = valid.len();
    if valid_count == 0 {
        return NumericProfile {
            valid_count: 0,
            min: f64::NAN,
            max: f64::NAN,
            mean: f64::NAN,
            median: f64::NAN,
            std_dev: f64::NAN,
            variance: f64::NAN,
            skewness: f64::NAN,
            kurtosis: f64::NAN,
            p5: f64::NAN,
            q1: f64::NAN,
            q3: f64::NAN,
            p95: f64::NAN,
            iqr: f64::NAN,
            zero_count: 0,
            negative_count: 0,
            infinity_count: 0,
            distinct_count: 0,
        };
    }

    // Use u-numflow stats
    let mean = u_numflow::stats::mean(&valid).unwrap_or(f64::NAN);
    let variance = u_numflow::stats::variance(&valid).unwrap_or(f64::NAN);
    let std_dev = u_numflow::stats::std_dev(&valid).unwrap_or(f64::NAN);
    let min = u_numflow::stats::min(&valid).unwrap_or(f64::NAN);
    let max = u_numflow::stats::max(&valid).unwrap_or(f64::NAN);
    let median = u_numflow::stats::median(&valid).unwrap_or(f64::NAN);
    let skewness = u_numflow::stats::skewness(&valid).unwrap_or(f64::NAN);
    let kurtosis = u_numflow::stats::kurtosis(&valid).unwrap_or(f64::NAN);

    let p5 = u_numflow::stats::quantile(&valid, 0.05).unwrap_or(f64::NAN);
    let q1 = u_numflow::stats::quantile(&valid, 0.25).unwrap_or(f64::NAN);
    let q3 = u_numflow::stats::quantile(&valid, 0.75).unwrap_or(f64::NAN);
    let p95 = u_numflow::stats::quantile(&valid, 0.95).unwrap_or(f64::NAN);
    let iqr = q3 - q1;

    let zero_count = valid.iter().filter(|&&v| v == 0.0).count();
    let negative_count = valid.iter().filter(|&&v| v < 0.0).count();
    let infinity_count = valid.iter().filter(|&&v| v.is_infinite()).count();

    // Distinct count using bitwise representation for exact f64 comparison
    let mut distinct_bits: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for &v in &valid {
        distinct_bits.insert(v.to_bits());
    }
    let distinct_count = distinct_bits.len();

    NumericProfile {
        valid_count,
        min,
        max,
        mean,
        median,
        std_dev,
        variance,
        skewness,
        kurtosis,
        p5,
        q1,
        q3,
        p95,
        iqr,
        zero_count,
        negative_count,
        infinity_count,
        distinct_count,
    }
}

fn profile_boolean(
    values: &[bool],
    validity: &crate::dataframe::ValidityBitmap,
) -> BooleanProfile {
    let mut true_count = 0usize;
    let mut false_count = 0usize;

    for idx in validity.valid_indices() {
        if values[idx] {
            true_count += 1;
        } else {
            false_count += 1;
        }
    }

    let valid_count = true_count + false_count;
    let true_ratio = if valid_count > 0 {
        true_count as f64 / valid_count as f64
    } else {
        0.0
    };

    BooleanProfile {
        valid_count,
        true_count,
        false_count,
        true_ratio,
    }
}

fn profile_categorical(
    dictionary: &[String],
    indices: &[u32],
    validity: &crate::dataframe::ValidityBitmap,
) -> CategoricalProfile {
    let mut freq: HashMap<u32, usize> = HashMap::new();

    for idx in validity.valid_indices() {
        *freq.entry(indices[idx]).or_insert(0) += 1;
    }

    let valid_count: usize = freq.values().sum();
    let distinct_count = freq.len();

    // Top values sorted by frequency descending
    let mut freq_vec: Vec<(u32, usize)> = freq.into_iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(&a.1));

    let top_n = 10.min(freq_vec.len());
    let top_values: Vec<(String, usize)> = freq_vec[..top_n]
        .iter()
        .map(|&(dict_idx, count)| {
            let label = dictionary
                .get(dict_idx as usize)
                .cloned()
                .unwrap_or_default();
            (label, count)
        })
        .collect();

    let mode_count = freq_vec.first().map_or(0, |&(_, c)| c);
    let mode_ratio = if valid_count > 0 {
        mode_count as f64 / valid_count as f64
    } else {
        0.0
    };

    let is_constant = distinct_count <= 1;

    CategoricalProfile {
        valid_count,
        distinct_count,
        top_values,
        mode_ratio,
        is_constant,
    }
}

fn profile_text(
    values: &[String],
    validity: &crate::dataframe::ValidityBitmap,
) -> TextProfile {
    let mut distinct: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut lengths: Vec<usize> = Vec::new();
    let mut empty_count = 0usize;

    for idx in validity.valid_indices() {
        let s = &values[idx];
        distinct.insert(s.as_str());
        lengths.push(s.len());
        if s.is_empty() {
            empty_count += 1;
        }
    }

    let valid_count = lengths.len();
    let distinct_count = distinct.len();
    let min_length = lengths.iter().copied().min().unwrap_or(0);
    let max_length = lengths.iter().copied().max().unwrap_or(0);
    let mean_length = if valid_count > 0 {
        lengths.iter().sum::<usize>() as f64 / valid_count as f64
    } else {
        0.0
    };

    TextProfile {
        valid_count,
        distinct_count,
        min_length,
        max_length,
        mean_length,
        empty_count,
    }
}

// ── Dataset Profile ───────────────────────────────────────────────────

/// Summary statistics for an entire dataset.
#[derive(Debug, Clone)]
pub struct DatasetProfile {
    /// Number of rows.
    pub row_count: usize,
    /// Number of columns.
    pub column_count: usize,
    /// Estimated memory usage in bytes.
    pub memory_bytes: usize,
    /// Count of columns by data type.
    pub type_counts: TypeCounts,
    /// Total number of null values across all columns.
    pub total_nulls: usize,
    /// Overall missing rate as percentage (0-100).
    pub sparsity_pct: f64,
    /// Number of duplicate rows.
    pub duplicate_count: usize,
    /// Duplicate rows as percentage of total rows.
    pub duplicate_pct: f64,
    /// Data quality score (0.0–1.0, higher is better).
    pub quality_score: DataQualityScore,
    /// Per-column profiles.
    pub columns: Vec<ColumnProfile>,
}

/// Composite data quality score across multiple dimensions.
///
/// Each dimension is scored from 0.0 (worst) to 1.0 (best).
/// The `overall` score is the weighted average of all dimensions.
#[derive(Debug, Clone)]
pub struct DataQualityScore {
    /// Completeness: proportion of non-null values.
    pub completeness: f64,
    /// Uniqueness: proportion of non-duplicate rows.
    pub uniqueness: f64,
    /// Validity: proportion of columns without diagnostic flags.
    pub validity: f64,
    /// Consistency: proportion of numeric columns without extreme values (inf, extreme skew/kurtosis).
    pub consistency: f64,
    /// Overall weighted score.
    pub overall: f64,
}

/// Counts of columns by data type.
#[derive(Debug, Clone, Default)]
pub struct TypeCounts {
    pub numeric: usize,
    pub boolean: usize,
    pub categorical: usize,
    pub text: usize,
}

/// Profiles an entire DataFrame at both column and dataset level.
///
/// ```
/// use u_insight::csv_parser::CsvParser;
/// use u_insight::profiling::profile_dataset;
///
/// let csv = "x,y,ok\n1.0,A,true\n2.0,B,false\n3.0,A,true\n";
/// let df = CsvParser::new().parse_str(csv).unwrap();
/// let ds = profile_dataset(&df);
///
/// assert_eq!(ds.row_count, 3);
/// assert_eq!(ds.column_count, 3);
/// assert_eq!(ds.type_counts.numeric, 1);
/// assert_eq!(ds.sparsity_pct, 0.0);
/// ```
pub fn profile_dataset(df: &DataFrame) -> DatasetProfile {
    let columns = profile_dataframe(df);
    let row_count = df.row_count();
    let column_count = df.column_count();

    let mut type_counts = TypeCounts::default();
    let mut total_nulls = 0usize;
    let mut memory_bytes = 0usize;

    for (_, col) in df.iter() {
        total_nulls += col.null_count();
        match col {
            Column::Numeric { values, .. } => {
                type_counts.numeric += 1;
                // f64 values + bitmap
                memory_bytes += values.len() * 8 + values.len().div_ceil(64) * 8;
            }
            Column::Boolean { values, .. } => {
                type_counts.boolean += 1;
                memory_bytes += values.len() + values.len().div_ceil(64) * 8;
            }
            Column::Categorical {
                dictionary,
                indices,
                ..
            } => {
                type_counts.categorical += 1;
                let dict_size: usize = dictionary.iter().map(|s| s.len() + 24).sum(); // String overhead
                memory_bytes += dict_size + indices.len() * 4 + indices.len().div_ceil(64) * 8;
            }
            Column::Text { values, .. } => {
                type_counts.text += 1;
                let text_size: usize = values.iter().map(|s| s.len() + 24).sum();
                memory_bytes += text_size + values.len().div_ceil(64) * 8;
            }
        }
    }

    let total_cells = row_count * column_count;
    let sparsity_pct = if total_cells > 0 {
        (total_nulls as f64 / total_cells as f64) * 100.0
    } else {
        0.0
    };

    // Duplicate row detection
    let duplicate_count = count_duplicate_rows(df);
    let duplicate_pct = if row_count > 0 {
        (duplicate_count as f64 / row_count as f64) * 100.0
    } else {
        0.0
    };

    // Compute quality score
    let quality_score = compute_quality_score(
        row_count,
        column_count,
        total_nulls,
        duplicate_count,
        &columns,
    );

    DatasetProfile {
        row_count,
        column_count,
        memory_bytes,
        type_counts,
        total_nulls,
        sparsity_pct,
        duplicate_count,
        duplicate_pct,
        quality_score,
        columns,
    }
}

/// Computes the composite data quality score from profiling results.
fn compute_quality_score(
    row_count: usize,
    column_count: usize,
    total_nulls: usize,
    duplicate_count: usize,
    columns: &[ColumnProfile],
) -> DataQualityScore {
    let total_cells = row_count * column_count;

    // Completeness: proportion of non-null values
    let completeness = if total_cells > 0 {
        1.0 - (total_nulls as f64 / total_cells as f64)
    } else {
        1.0
    };

    // Uniqueness: proportion of non-duplicate rows
    let uniqueness = if row_count > 0 {
        1.0 - (duplicate_count as f64 / row_count as f64)
    } else {
        1.0
    };

    let thresholds = FlagThresholds::default();

    // Validity: proportion of columns without diagnostic flags
    let validity = if columns.is_empty() {
        1.0
    } else {
        let flagged = columns
            .iter()
            .filter(|c| !compute_flags(c, &thresholds).is_empty())
            .count();
        1.0 - (flagged as f64 / columns.len() as f64)
    };

    // Consistency: proportion of numeric columns without extreme values
    let consistency = {
        let numeric_cols: Vec<&ColumnProfile> = columns
            .iter()
            .filter(|c| c.numeric.is_some())
            .collect();
        if numeric_cols.is_empty() {
            1.0
        } else {
            let problematic = numeric_cols
                .iter()
                .filter(|c| {
                    let flags = compute_flags(c, &thresholds);
                    flags.iter().any(|f| {
                        matches!(
                            f,
                            DiagnosticFlag::ContainsInfinity
                                | DiagnosticFlag::ExtremeSkewness
                                | DiagnosticFlag::ExtremeKurtosis
                        )
                    })
                })
                .count();
            1.0 - (problematic as f64 / numeric_cols.len() as f64)
        }
    };

    // Overall: weighted average
    // Completeness and validity are most important
    let overall = completeness * 0.30 + uniqueness * 0.25 + validity * 0.25 + consistency * 0.20;

    DataQualityScore {
        completeness,
        uniqueness,
        validity,
        consistency,
        overall,
    }
}

/// Counts duplicate rows in a DataFrame.
///
/// A row is considered a duplicate if there is an earlier row with
/// identical values in all columns (including null status). The first
/// occurrence is not counted as a duplicate.
///
/// Returns the number of duplicate rows (total rows - unique rows).
fn count_duplicate_rows(df: &DataFrame) -> usize {
    let n = df.row_count();
    if n <= 1 {
        return 0;
    }

    let cols: Vec<(&str, &Column)> = df.iter().collect();
    let mut seen = HashSet::with_capacity(n);
    let mut dups = 0usize;

    for row_idx in 0..n {
        let key = row_key(&cols, row_idx);
        if !seen.insert(key) {
            dups += 1;
        }
    }

    dups
}

/// Produces a string key for a row, suitable for hash-based duplicate detection.
fn row_key(cols: &[(&str, &Column)], row_idx: usize) -> String {
    use std::fmt::Write;

    let mut key = String::new();
    for (i, (_, col)) in cols.iter().enumerate() {
        if i > 0 {
            key.push('\x1F'); // unit separator
        }
        if !col.is_valid(row_idx) {
            key.push_str("\x00NULL");
            continue;
        }
        match col {
            Column::Numeric { values, .. } => {
                // Use bits for exact comparison (avoids floating-point formatting issues)
                let _ = write!(key, "{}", values[row_idx].to_bits());
            }
            Column::Boolean { values, .. } => {
                key.push(if values[row_idx] { 'T' } else { 'F' });
            }
            Column::Categorical {
                dictionary,
                indices,
                ..
            } => {
                let idx = indices[row_idx] as usize;
                if idx < dictionary.len() {
                    key.push_str(&dictionary[idx]);
                }
            }
            Column::Text { values, .. } => {
                key.push_str(&values[row_idx]);
            }
        }
    }
    key
}

// ── Outlier Detection ─────────────────────────────────────────────────

/// Method for univariate outlier detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlierMethod {
    /// Tukey fences: outlier if value < Q1 - k*IQR or > Q3 + k*IQR.
    /// Default k = 1.5.
    Iqr,
    /// Z-score: outlier if |z| > threshold. Default threshold = 3.0.
    Zscore,
    /// Modified Z-score using MAD (Median Absolute Deviation).
    /// Outlier if |M_i| > threshold. Default threshold = 3.5.
    /// Reference: Iglewicz & Hoaglin (1993).
    ModifiedZscore,
}

/// Result of outlier detection on a numeric column.
#[derive(Debug, Clone)]
pub struct OutlierResult {
    /// Method used.
    pub method: OutlierMethod,
    /// Indices of detected outliers.
    pub indices: Vec<usize>,
    /// Outlier scores (absolute deviation from center).
    pub scores: Vec<f64>,
    /// Number of outliers detected.
    pub count: usize,
    /// Percentage of outliers among valid values.
    pub pct: f64,
}

/// Detects outliers in a numeric column using the specified method.
///
/// Null values are skipped. Returns `None` if the column is not numeric
/// or has insufficient valid data.
///
/// ```
/// use u_insight::dataframe::{Column, ValidityBitmap};
/// use u_insight::profiling::{detect_outliers, OutlierMethod};
///
/// let col = Column::numeric(
///     vec![1.0, 2.0, 3.0, 2.5, 100.0, 2.0, 3.0, 2.0],
///     ValidityBitmap::all_valid(8),
/// );
/// let result = detect_outliers(&col, OutlierMethod::Iqr).unwrap();
/// assert!(result.count >= 1); // 100.0 is an outlier
/// assert!(result.indices.contains(&4));
/// ```
pub fn detect_outliers(col: &Column, method: OutlierMethod) -> Option<OutlierResult> {
    let (values, validity) = match col {
        Column::Numeric { values, validity } => (values, validity),
        _ => return None,
    };

    let valid: Vec<(usize, f64)> = validity
        .valid_indices()
        .map(|i| (i, values[i]))
        .filter(|(_, v)| v.is_finite()) // skip NaN/Inf
        .collect();

    if valid.len() < 3 {
        return Some(OutlierResult {
            method,
            indices: Vec::new(),
            scores: Vec::new(),
            count: 0,
            pct: 0.0,
        });
    }

    let vals: Vec<f64> = valid.iter().map(|&(_, v)| v).collect();

    match method {
        OutlierMethod::Iqr => detect_iqr(&valid, &vals),
        OutlierMethod::Zscore => detect_zscore(&valid, &vals),
        OutlierMethod::ModifiedZscore => detect_modified_zscore(&valid, &vals),
    }
}

fn detect_iqr(valid: &[(usize, f64)], vals: &[f64]) -> Option<OutlierResult> {
    let q1 = u_numflow::stats::quantile(vals, 0.25)?;
    let q3 = u_numflow::stats::quantile(vals, 0.75)?;
    let iqr = q3 - q1;
    let k = 1.5;
    let lower = q1 - k * iqr;
    let upper = q3 + k * iqr;

    let mut indices = Vec::new();
    let mut scores = Vec::new();
    for &(idx, v) in valid {
        if v < lower || v > upper {
            let score = if v < lower {
                (lower - v) / iqr.max(1e-15)
            } else {
                (v - upper) / iqr.max(1e-15)
            };
            indices.push(idx);
            scores.push(score);
        }
    }

    let count = indices.len();
    let pct = if !valid.is_empty() {
        (count as f64 / valid.len() as f64) * 100.0
    } else {
        0.0
    };

    Some(OutlierResult {
        method: OutlierMethod::Iqr,
        indices,
        scores,
        count,
        pct,
    })
}

fn detect_zscore(valid: &[(usize, f64)], vals: &[f64]) -> Option<OutlierResult> {
    let mean = u_numflow::stats::mean(vals)?;
    let std = u_numflow::stats::std_dev(vals)?;
    let threshold = 3.0;

    if std < 1e-15 {
        // Zero variance: no outliers possible
        return Some(OutlierResult {
            method: OutlierMethod::Zscore,
            indices: Vec::new(),
            scores: Vec::new(),
            count: 0,
            pct: 0.0,
        });
    }

    let mut indices = Vec::new();
    let mut scores = Vec::new();
    for &(idx, v) in valid {
        let z = ((v - mean) / std).abs();
        if z > threshold {
            indices.push(idx);
            scores.push(z);
        }
    }

    let count = indices.len();
    let pct = if !valid.is_empty() {
        (count as f64 / valid.len() as f64) * 100.0
    } else {
        0.0
    };

    Some(OutlierResult {
        method: OutlierMethod::Zscore,
        indices,
        scores,
        count,
        pct,
    })
}

fn detect_modified_zscore(valid: &[(usize, f64)], vals: &[f64]) -> Option<OutlierResult> {
    let median = u_numflow::stats::median(vals)?;
    let threshold = 3.5;

    // MAD = median(|x_i - median(x)|)
    let mut abs_devs: Vec<f64> = vals.iter().map(|&v| (v - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = u_numflow::stats::median(&abs_devs)?;

    if mad < 1e-15 {
        return Some(OutlierResult {
            method: OutlierMethod::ModifiedZscore,
            indices: Vec::new(),
            scores: Vec::new(),
            count: 0,
            pct: 0.0,
        });
    }

    // Modified Z-score: M_i = 0.6745 * (x_i - median) / MAD
    let factor = 0.6745;
    let mut indices = Vec::new();
    let mut scores = Vec::new();
    for &(idx, v) in valid {
        let m = (factor * (v - median) / mad).abs();
        if m > threshold {
            indices.push(idx);
            scores.push(m);
        }
    }

    let count = indices.len();
    let pct = if !valid.is_empty() {
        (count as f64 / valid.len() as f64) * 100.0
    } else {
        0.0
    };

    Some(OutlierResult {
        method: OutlierMethod::ModifiedZscore,
        indices,
        scores,
        count,
        pct,
    })
}

// ── Diagnostic Flags ──────────────────────────────────────────────────

/// Diagnostic flags raised during profiling.
///
/// Each flag indicates a potential data quality issue that may need
/// attention during preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticFlag {
    /// Column has zero variance (all same value).
    ConstantColumn,
    /// Near-zero variance (std_dev < threshold).
    ZeroVariance,
    /// High missing rate (> threshold %).
    HighMissing,
    /// Extreme skewness (|skewness| > threshold).
    ExtremeSkewness,
    /// Extreme kurtosis (|kurtosis| > threshold).
    ExtremeKurtosis,
    /// High cardinality categorical (many unique values).
    HighCardinality,
    /// All values are unique (potential ID column).
    AllUnique,
    /// Most frequent category dominates (imbalanced).
    Imbalanced,
    /// Column contains ±infinity values.
    ContainsInfinity,
}

impl std::fmt::Display for DiagnosticFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstantColumn => write!(f, "CONSTANT_COLUMN"),
            Self::ZeroVariance => write!(f, "ZERO_VARIANCE"),
            Self::HighMissing => write!(f, "HIGH_MISSING"),
            Self::ExtremeSkewness => write!(f, "EXTREME_SKEWNESS"),
            Self::ExtremeKurtosis => write!(f, "EXTREME_KURTOSIS"),
            Self::HighCardinality => write!(f, "HIGH_CARDINALITY"),
            Self::AllUnique => write!(f, "ALL_UNIQUE"),
            Self::Imbalanced => write!(f, "IMBALANCED"),
            Self::ContainsInfinity => write!(f, "CONTAINS_INFINITY"),
        }
    }
}

/// Configurable thresholds for diagnostic flags.
#[derive(Debug, Clone)]
pub struct FlagThresholds {
    /// Missing rate threshold (%). Default: 50.0.
    pub high_missing_pct: f64,
    /// Near-zero variance threshold. Default: 1e-10.
    pub zero_variance_threshold: f64,
    /// Extreme skewness threshold. Default: 10.0.
    pub extreme_skewness: f64,
    /// Extreme kurtosis threshold. Default: 100.0.
    pub extreme_kurtosis: f64,
    /// High cardinality threshold (distinct count). Default: 100.
    pub high_cardinality: usize,
    /// Imbalance threshold (mode ratio). Default: 0.9.
    pub imbalance_ratio: f64,
}

impl Default for FlagThresholds {
    fn default() -> Self {
        Self {
            high_missing_pct: 50.0,
            zero_variance_threshold: 1e-10,
            extreme_skewness: 10.0,
            extreme_kurtosis: 100.0,
            high_cardinality: 100,
            imbalance_ratio: 0.9,
        }
    }
}

/// Computes diagnostic flags for a column profile.
///
/// ```
/// use u_insight::dataframe::{Column, ValidityBitmap};
/// use u_insight::profiling::{profile_column, compute_flags, FlagThresholds, DiagnosticFlag};
///
/// // Constant column
/// let col = Column::numeric(vec![5.0, 5.0, 5.0], ValidityBitmap::all_valid(3));
/// let prof = profile_column("x", &col);
/// let flags = compute_flags(&prof, &FlagThresholds::default());
/// assert!(flags.contains(&DiagnosticFlag::ConstantColumn));
/// ```
pub fn compute_flags(profile: &ColumnProfile, thresholds: &FlagThresholds) -> Vec<DiagnosticFlag> {
    let mut flags = Vec::new();

    // High missing
    if profile.missing_pct > thresholds.high_missing_pct {
        flags.push(DiagnosticFlag::HighMissing);
    }

    // Type-specific flags
    if let Some(np) = &profile.numeric {
        if np.valid_count > 0 && np.distinct_count <= 1 {
            flags.push(DiagnosticFlag::ConstantColumn);
        } else if np.std_dev.is_finite() && np.std_dev < thresholds.zero_variance_threshold {
            flags.push(DiagnosticFlag::ZeroVariance);
        }

        if np.skewness.is_finite() && np.skewness.abs() > thresholds.extreme_skewness {
            flags.push(DiagnosticFlag::ExtremeSkewness);
        }

        if np.kurtosis.is_finite() && np.kurtosis.abs() > thresholds.extreme_kurtosis {
            flags.push(DiagnosticFlag::ExtremeKurtosis);
        }

        if np.infinity_count > 0 {
            flags.push(DiagnosticFlag::ContainsInfinity);
        }

        if np.valid_count > 1 && np.distinct_count == np.valid_count {
            flags.push(DiagnosticFlag::AllUnique);
        }
    }

    if let Some(cp) = &profile.categorical {
        if cp.is_constant {
            flags.push(DiagnosticFlag::ConstantColumn);
        }

        if cp.distinct_count > thresholds.high_cardinality {
            flags.push(DiagnosticFlag::HighCardinality);
        }

        if cp.mode_ratio > thresholds.imbalance_ratio {
            flags.push(DiagnosticFlag::Imbalanced);
        }

        if cp.valid_count > 1 && cp.distinct_count == cp.valid_count {
            flags.push(DiagnosticFlag::AllUnique);
        }
    }

    if let Some(bp) = &profile.boolean {
        if bp.valid_count > 0 && (bp.true_count == 0 || bp.false_count == 0) {
            flags.push(DiagnosticFlag::ConstantColumn);
        }

        let ratio = bp.true_ratio.max(1.0 - bp.true_ratio);
        if ratio > thresholds.imbalance_ratio {
            flags.push(DiagnosticFlag::Imbalanced);
        }
    }

    if let Some(tp) = &profile.text {
        if tp.valid_count > 0 && tp.distinct_count <= 1 {
            flags.push(DiagnosticFlag::ConstantColumn);
        }

        if tp.distinct_count > thresholds.high_cardinality {
            flags.push(DiagnosticFlag::HighCardinality);
        }

        if tp.valid_count > 1 && tp.distinct_count == tp.valid_count {
            flags.push(DiagnosticFlag::AllUnique);
        }
    }

    flags
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csv_parser::CsvParser;
    use crate::dataframe::ValidityBitmap;

    // ── Numeric profiling ────────────────────────────────────────

    #[test]
    fn numeric_basic_stats() {
        let col = Column::Numeric {
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            validity: ValidityBitmap::all_valid(5),
        };
        let prof = profile_column("x", &col);
        assert_eq!(prof.null_count, 0);
        assert_eq!(prof.missing_pct, 0.0);

        let np = prof.numeric.expect("numeric profile");
        assert_eq!(np.valid_count, 5);
        assert_eq!(np.min, 1.0);
        assert_eq!(np.max, 5.0);
        assert!((np.mean - 3.0).abs() < 1e-10);
        assert!((np.median - 3.0).abs() < 1e-10);
        assert_eq!(np.zero_count, 0);
        assert_eq!(np.negative_count, 0);
        assert_eq!(np.distinct_count, 5);
    }

    #[test]
    fn numeric_with_nulls() {
        let mut validity = ValidityBitmap::all_valid(5);
        validity.set_invalid(2);
        validity.set_invalid(4);
        let col = Column::Numeric {
            values: vec![10.0, 20.0, 0.0, 40.0, 0.0],
            validity,
        };
        let prof = profile_column("x", &col);
        assert_eq!(prof.null_count, 2);
        assert!((prof.missing_pct - 40.0).abs() < 1e-10);

        let np = prof.numeric.expect("numeric profile");
        assert_eq!(np.valid_count, 3);
        assert_eq!(np.min, 10.0);
        assert_eq!(np.max, 40.0);
    }

    #[test]
    fn numeric_all_null() {
        let col = Column::Numeric {
            values: vec![0.0, 0.0, 0.0],
            validity: ValidityBitmap::all_invalid(3),
        };
        let prof = profile_column("x", &col);
        let np = prof.numeric.expect("numeric profile");
        assert_eq!(np.valid_count, 0);
        assert!(np.mean.is_nan());
    }

    #[test]
    fn numeric_zeros_negatives_infinity() {
        let col = Column::Numeric {
            values: vec![0.0, -1.0, f64::INFINITY, -2.0, f64::NEG_INFINITY, 0.0, 3.0],
            validity: ValidityBitmap::all_valid(7),
        };
        let np = profile_column("x", &col).numeric.expect("numeric");
        assert_eq!(np.zero_count, 2);
        assert_eq!(np.negative_count, 3); // -1.0, -2.0, NEG_INFINITY (all < 0.0)
        assert_eq!(np.infinity_count, 2);
    }

    #[test]
    fn numeric_quantiles() {
        // 0..100 gives us well-known quantiles
        let values: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        let n = values.len();
        let col = Column::Numeric {
            values,
            validity: ValidityBitmap::all_valid(n),
        };
        let np = profile_column("x", &col).numeric.expect("numeric");
        assert!((np.q1 - 25.0).abs() < 1.0);
        assert!((np.median - 50.0).abs() < 1e-10);
        assert!((np.q3 - 75.0).abs() < 1.0);
        assert!((np.iqr - 50.0).abs() < 1.0);
    }

    #[test]
    fn numeric_distinct_count() {
        let col = Column::Numeric {
            values: vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0],
            validity: ValidityBitmap::all_valid(6),
        };
        let np = profile_column("x", &col).numeric.expect("numeric");
        assert_eq!(np.distinct_count, 3);
    }

    // ── Boolean profiling ────────────────────────────────────────

    #[test]
    fn boolean_basic() {
        let col = Column::Boolean {
            values: vec![true, false, true, true, false],
            validity: ValidityBitmap::all_valid(5),
        };
        let bp = profile_column("flag", &col).boolean.expect("boolean");
        assert_eq!(bp.valid_count, 5);
        assert_eq!(bp.true_count, 3);
        assert_eq!(bp.false_count, 2);
        assert!((bp.true_ratio - 0.6).abs() < 1e-10);
    }

    #[test]
    fn boolean_with_nulls() {
        let mut validity = ValidityBitmap::all_valid(4);
        validity.set_invalid(1);
        let col = Column::Boolean {
            values: vec![true, false, false, true],
            validity,
        };
        let bp = profile_column("flag", &col).boolean.expect("boolean");
        assert_eq!(bp.valid_count, 3);
        assert_eq!(bp.true_count, 2);
        assert_eq!(bp.false_count, 1);
    }

    // ── Categorical profiling ────────────────────────────────────

    #[test]
    fn categorical_basic() {
        let dict = vec!["A".into(), "B".into(), "C".into()];
        let indices = vec![0, 1, 0, 2, 0, 1, 0];
        let col = Column::Categorical {
            dictionary: dict,
            indices,
            validity: ValidityBitmap::all_valid(7),
        };
        let cp = profile_column("cat", &col).categorical.expect("categorical");
        assert_eq!(cp.valid_count, 7);
        assert_eq!(cp.distinct_count, 3);
        assert!(!cp.is_constant);

        // Top value should be "A" with count 4
        assert_eq!(cp.top_values[0].0, "A");
        assert_eq!(cp.top_values[0].1, 4);
        assert!((cp.mode_ratio - 4.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn categorical_constant() {
        let dict = vec!["X".into()];
        let indices = vec![0, 0, 0];
        let col = Column::Categorical {
            dictionary: dict,
            indices,
            validity: ValidityBitmap::all_valid(3),
        };
        let cp = profile_column("cat", &col).categorical.expect("categorical");
        assert!(cp.is_constant);
        assert_eq!(cp.distinct_count, 1);
        assert!((cp.mode_ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn categorical_with_nulls() {
        let dict = vec!["A".into(), "B".into()];
        let indices = vec![0, 0, 1, 0];
        let mut validity = ValidityBitmap::all_valid(4);
        validity.set_invalid(2);
        let col = Column::Categorical {
            dictionary: dict,
            indices,
            validity,
        };
        let cp = profile_column("cat", &col).categorical.expect("categorical");
        assert_eq!(cp.valid_count, 3); // 3 valid, 1 null
        assert_eq!(cp.distinct_count, 1); // only "A" in valid values
        assert!(cp.is_constant);
    }

    // ── Text profiling ───────────────────────────────────────────

    #[test]
    fn text_basic() {
        let values = vec![
            "hello".into(),
            "world".into(),
            "hi".into(),
            "".into(),
            "hello".into(),
        ];
        let col = Column::Text {
            values,
            validity: ValidityBitmap::all_valid(5),
        };
        let tp = profile_column("txt", &col).text.expect("text");
        assert_eq!(tp.valid_count, 5);
        assert_eq!(tp.distinct_count, 4); // hello, world, hi, ""
        assert_eq!(tp.min_length, 0);
        assert_eq!(tp.max_length, 5);
        assert_eq!(tp.empty_count, 1);
    }

    #[test]
    fn text_with_nulls() {
        let values = vec!["abc".into(), String::new(), "de".into()];
        let mut validity = ValidityBitmap::all_valid(3);
        validity.set_invalid(1);
        let col = Column::Text {
            values,
            validity,
        };
        let tp = profile_column("txt", &col).text.expect("text");
        assert_eq!(tp.valid_count, 2);
        assert_eq!(tp.min_length, 2);
        assert_eq!(tp.max_length, 3);
    }

    // ── DataFrame profiling ──────────────────────────────────────

    #[test]
    fn profile_from_csv() {
        let csv = "x,y,ok\n1.0,A,true\n2.0,B,false\nNA,A,true\n4.0,A,NA\n5.0,B,false\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let profiles = profile_dataframe(&df);

        assert_eq!(profiles.len(), 3);

        // x: numeric with 1 null
        let x = &profiles[0];
        assert_eq!(x.name, "x");
        assert_eq!(x.data_type, DataType::Numeric);
        assert_eq!(x.null_count, 1);
        assert!(x.numeric.is_some());
        assert_eq!(x.numeric.as_ref().unwrap().valid_count, 4);

        // y: categorical (2 unique / 5 = 0.4 < 0.5)
        let y = &profiles[1];
        assert_eq!(y.name, "y");
        assert_eq!(y.data_type, DataType::Categorical);
        assert_eq!(y.null_count, 0);
        assert!(y.categorical.is_some());
        assert_eq!(y.categorical.as_ref().unwrap().distinct_count, 2);

        // ok: boolean with 1 null
        let ok = &profiles[2];
        assert_eq!(ok.name, "ok");
        assert_eq!(ok.data_type, DataType::Boolean);
        assert_eq!(ok.null_count, 1);
        assert!(ok.boolean.is_some());
    }

    #[test]
    fn empty_dataframe_profile() {
        let df = DataFrame::new();
        let profiles = profile_dataframe(&df);
        assert!(profiles.is_empty());
    }

    #[test]
    fn numeric_skewness_kurtosis() {
        // Symmetric data should have near-zero skewness
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let n = values.len();
        let col = Column::Numeric {
            values,
            validity: ValidityBitmap::all_valid(n),
        };
        let np = profile_column("x", &col).numeric.expect("numeric");
        assert!(np.skewness.abs() < 0.5, "skewness should be near zero for uniform-like data");
    }

    #[test]
    fn negative_count_excludes_neg_infinity() {
        // NEG_INFINITY is negative AND infinite
        let col = Column::Numeric {
            values: vec![-1.0, f64::NEG_INFINITY, 0.0],
            validity: ValidityBitmap::all_valid(3),
        };
        let np = profile_column("x", &col).numeric.expect("numeric");
        // Both -1.0 and NEG_INFINITY are < 0.0
        assert_eq!(np.negative_count, 2);
        assert_eq!(np.infinity_count, 1);
    }

    // ── Dataset profiling ────────────────────────────────────────

    #[test]
    fn dataset_profile_basic() {
        let csv = "x,y,ok\n1.0,A,true\n2.0,B,false\n3.0,A,true\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let ds = profile_dataset(&df);

        assert_eq!(ds.row_count, 3);
        assert_eq!(ds.column_count, 3);
        assert_eq!(ds.type_counts.numeric, 1);
        assert_eq!(ds.type_counts.boolean, 1);
        // y: 2 unique / 3 rows = 0.67 → Text (not categorical)
        assert_eq!(ds.total_nulls, 0);
        assert!((ds.sparsity_pct - 0.0).abs() < 1e-10);
        assert!(ds.memory_bytes > 0);
        assert_eq!(ds.columns.len(), 3);
    }

    #[test]
    fn dataset_profile_with_nulls() {
        let csv = "x,y\n1.0,A\nNA,B\n3.0,NA\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let ds = profile_dataset(&df);

        assert_eq!(ds.total_nulls, 2);
        // 2 nulls / (3 rows * 2 cols) = 33.33%
        assert!((ds.sparsity_pct - 33.333).abs() < 1.0);
    }

    #[test]
    fn dataset_empty() {
        let df = DataFrame::new();
        let ds = profile_dataset(&df);
        assert_eq!(ds.row_count, 0);
        assert_eq!(ds.column_count, 0);
        assert_eq!(ds.total_nulls, 0);
        assert!((ds.sparsity_pct - 0.0).abs() < 1e-10);
    }

    // ── Outlier detection ────────────────────────────────────────

    #[test]
    fn iqr_outlier_detection() {
        let data = vec![1.0, 2.0, 2.5, 3.0, 2.0, 3.0, 2.5, 100.0];
        let col = Column::numeric(data, ValidityBitmap::all_valid(8));
        let result = detect_outliers(&col, OutlierMethod::Iqr).unwrap();

        assert!(result.count >= 1);
        assert!(result.indices.contains(&7)); // 100.0 is an outlier
    }

    #[test]
    fn zscore_outlier_detection() {
        let mut data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        data.push(500.0); // extreme outlier
        let n = data.len();
        let col = Column::numeric(data, ValidityBitmap::all_valid(n));
        let result = detect_outliers(&col, OutlierMethod::Zscore).unwrap();

        assert!(result.count >= 1);
        assert!(result.indices.contains(&50)); // 500.0
    }

    #[test]
    fn modified_zscore_outlier_detection() {
        let data = vec![1.0, 2.0, 2.5, 3.0, 2.0, 3.0, 2.5, 50.0];
        let col = Column::numeric(data, ValidityBitmap::all_valid(8));
        let result = detect_outliers(&col, OutlierMethod::ModifiedZscore).unwrap();

        assert!(result.count >= 1);
        assert!(result.indices.contains(&7)); // 50.0
    }

    #[test]
    fn outlier_with_nulls() {
        let mut validity = ValidityBitmap::all_valid(5);
        validity.set_invalid(2);
        let col = Column::numeric(vec![1.0, 2.0, 0.0, 3.0, 100.0], validity);
        let result = detect_outliers(&col, OutlierMethod::Iqr).unwrap();
        // null at index 2 should be skipped, 100.0 should be detected
        assert!(result.count >= 1);
        assert!(result.indices.contains(&4));
        assert!(!result.indices.contains(&2)); // null, not an outlier
    }

    #[test]
    fn outlier_not_numeric() {
        let col = Column::text(vec!["a".into()], ValidityBitmap::all_valid(1));
        assert!(detect_outliers(&col, OutlierMethod::Iqr).is_none());
    }

    #[test]
    fn outlier_too_few_values() {
        let col = Column::numeric(vec![1.0, 2.0], ValidityBitmap::all_valid(2));
        let result = detect_outliers(&col, OutlierMethod::Iqr).unwrap();
        assert_eq!(result.count, 0);
    }

    #[test]
    fn no_outliers_in_uniform_data() {
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let col = Column::numeric(data, ValidityBitmap::all_valid(10));
        let result = detect_outliers(&col, OutlierMethod::Zscore).unwrap();
        assert_eq!(result.count, 0);
    }

    #[test]
    fn outlier_zero_variance() {
        let data = vec![5.0; 10];
        let col = Column::numeric(data, ValidityBitmap::all_valid(10));
        let result = detect_outliers(&col, OutlierMethod::Zscore).unwrap();
        assert_eq!(result.count, 0); // zero variance, no outliers
    }

    // ── Diagnostic flags ─────────────────────────────────────────

    #[test]
    fn flag_constant_numeric() {
        let col = Column::numeric(vec![5.0, 5.0, 5.0], ValidityBitmap::all_valid(3));
        let prof = profile_column("x", &col);
        let flags = compute_flags(&prof, &FlagThresholds::default());
        assert!(flags.contains(&DiagnosticFlag::ConstantColumn));
    }

    #[test]
    fn flag_high_missing() {
        let mut validity = ValidityBitmap::all_valid(10);
        for i in 0..6 {
            validity.set_invalid(i);
        }
        let col = Column::numeric(vec![0.0; 10], validity);
        let prof = profile_column("x", &col);
        let flags = compute_flags(&prof, &FlagThresholds::default());
        assert!(flags.contains(&DiagnosticFlag::HighMissing));
    }

    #[test]
    fn flag_extreme_skewness() {
        // Highly skewed data
        let mut data = vec![1.0; 100];
        data.push(10000.0);
        let n = data.len();
        let col = Column::numeric(data, ValidityBitmap::all_valid(n));
        let prof = profile_column("x", &col);
        let flags = compute_flags(&prof, &FlagThresholds::default());
        assert!(
            flags.contains(&DiagnosticFlag::ExtremeSkewness),
            "skewness = {}", prof.numeric.as_ref().unwrap().skewness
        );
    }

    #[test]
    fn flag_contains_infinity() {
        let col = Column::numeric(
            vec![1.0, f64::INFINITY, 3.0],
            ValidityBitmap::all_valid(3),
        );
        let prof = profile_column("x", &col);
        let flags = compute_flags(&prof, &FlagThresholds::default());
        assert!(flags.contains(&DiagnosticFlag::ContainsInfinity));
    }

    #[test]
    fn flag_all_unique_numeric() {
        let col = Column::numeric(vec![1.0, 2.0, 3.0, 4.0], ValidityBitmap::all_valid(4));
        let prof = profile_column("x", &col);
        let flags = compute_flags(&prof, &FlagThresholds::default());
        assert!(flags.contains(&DiagnosticFlag::AllUnique));
    }

    #[test]
    fn flag_imbalanced_boolean() {
        // 95% true → imbalanced (threshold 90%)
        let mut values = vec![true; 95];
        values.extend(vec![false; 5]);
        let n = values.len();
        let col = Column::boolean(values, ValidityBitmap::all_valid(n));
        let prof = profile_column("flag", &col);
        let flags = compute_flags(&prof, &FlagThresholds::default());
        assert!(flags.contains(&DiagnosticFlag::Imbalanced));
    }

    #[test]
    fn flag_constant_categorical() {
        let dict = vec!["X".into()];
        let indices = vec![0, 0, 0, 0, 0];
        let col = Column::categorical(dict, indices, ValidityBitmap::all_valid(5));
        let prof = profile_column("cat", &col);
        let flags = compute_flags(&prof, &FlagThresholds::default());
        assert!(flags.contains(&DiagnosticFlag::ConstantColumn));
    }

    #[test]
    fn no_flags_for_normal_data() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let col = Column::numeric(data, ValidityBitmap::all_valid(100));
        let prof = profile_column("x", &col);
        let flags = compute_flags(&prof, &FlagThresholds::default());
        // Normal 1-100 range should not trigger extreme skewness or kurtosis
        assert!(!flags.contains(&DiagnosticFlag::ExtremeSkewness));
        assert!(!flags.contains(&DiagnosticFlag::ExtremeKurtosis));
        assert!(!flags.contains(&DiagnosticFlag::ConstantColumn));
    }

    #[test]
    fn custom_thresholds() {
        let col = Column::numeric(vec![1.0, 2.0, 3.0], ValidityBitmap::all_valid(3));
        let prof = profile_column("x", &col);

        // Default thresholds: no flags expected
        let flags = compute_flags(&prof, &FlagThresholds::default());
        assert!(!flags.contains(&DiagnosticFlag::ExtremeSkewness));

        // Very strict threshold: skewness > 0.01
        let strict = FlagThresholds {
            extreme_skewness: 0.0001,
            ..FlagThresholds::default()
        };
        let flags2 = compute_flags(&prof, &strict);
        // [1, 2, 3] has skewness = 0 (symmetric), so even strict threshold shouldn't trigger
        // But this tests that custom thresholds work
        assert!(!flags2.contains(&DiagnosticFlag::ExtremeSkewness));
    }

    // ── Duplicate row detection ───────────────────────────────

    // ── Data quality score ──────────────────────────────────────

    #[test]
    fn quality_score_perfect() {
        let csv = "x,y\n1.0,A\n2.0,B\n3.0,C\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let ds = profile_dataset(&df);
        let q = &ds.quality_score;

        assert!((q.completeness - 1.0).abs() < 1e-10); // no nulls
        assert!((q.uniqueness - 1.0).abs() < 1e-10); // no duplicates
        // validity may not be 1.0 (ALL_UNIQUE flag on 3-value numeric column)
        assert!(q.overall > 0.5, "overall should be reasonable: {}", q.overall);
    }

    #[test]
    fn quality_score_with_nulls() {
        let csv = "x,y\n1.0,A\nNA,B\n3.0,NA\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let ds = profile_dataset(&df);
        let q = &ds.quality_score;

        // 2 nulls out of 6 cells = completeness 4/6
        assert!((q.completeness - 4.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn quality_score_with_duplicates() {
        let csv = "x,y\n1.0,A\n1.0,A\n2.0,B\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let ds = profile_dataset(&df);
        let q = &ds.quality_score;

        // 1 duplicate out of 3 rows = uniqueness 2/3
        assert!((q.uniqueness - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn quality_score_empty() {
        let df = DataFrame::new();
        let ds = profile_dataset(&df);
        let q = &ds.quality_score;
        assert!((q.overall - 1.0).abs() < 1e-10);
    }

    #[test]
    fn quality_score_dimensions_in_range() {
        let csv = "a,b,c\n1.0,x,true\nNA,y,false\n3.0,x,true\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let ds = profile_dataset(&df);
        let q = &ds.quality_score;

        for score in [q.completeness, q.uniqueness, q.validity, q.consistency, q.overall] {
            assert!(
                (0.0..=1.0).contains(&score),
                "score {score} out of range"
            );
        }
    }

    // ── Duplicate row detection ───────────────────────────────

    #[test]
    fn duplicate_rows_none() {
        let csv = "x,y\n1.0,A\n2.0,B\n3.0,C\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let ds = profile_dataset(&df);
        assert_eq!(ds.duplicate_count, 0);
        assert!((ds.duplicate_pct - 0.0).abs() < 1e-10);
    }

    #[test]
    fn duplicate_rows_all_same() {
        let csv = "x,y\n1.0,A\n1.0,A\n1.0,A\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let ds = profile_dataset(&df);
        assert_eq!(ds.duplicate_count, 2); // first is original, 2 duplicates
        assert!((ds.duplicate_pct - 200.0 / 3.0).abs() < 0.1);
    }

    #[test]
    fn duplicate_rows_partial() {
        let csv = "x,y\n1.0,A\n2.0,B\n1.0,A\n3.0,C\n2.0,B\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let ds = profile_dataset(&df);
        assert_eq!(ds.duplicate_count, 2); // row 3 = dup of row 1, row 5 = dup of row 2
    }

    #[test]
    fn duplicate_rows_empty() {
        let df = DataFrame::new();
        let ds = profile_dataset(&df);
        assert_eq!(ds.duplicate_count, 0);
        assert!((ds.duplicate_pct - 0.0).abs() < 1e-10);
    }

    #[test]
    fn duplicate_rows_with_nulls() {
        // Two rows with nulls in same positions should be considered duplicates
        let csv = "x,y\n1.0,A\nNA,B\nNA,B\n2.0,C\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let ds = profile_dataset(&df);
        assert_eq!(ds.duplicate_count, 1); // row 3 = dup of row 2 (both have NA,B)
    }
}
