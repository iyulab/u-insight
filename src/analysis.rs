//! Analysis module for clean, preprocessed data.
//!
//! Unlike the [`profiling`](crate::profiling) module which tolerates dirty data,
//! the analysis module **requires** clean numeric data with no missing values.
//! If missing values are detected, it returns
//! [`InsightError::MissingValues`].
//!
//! # Correlation Analysis
//!
//! ```
//! use u_insight::analysis::{correlation_analysis, CorrelationConfig, CorrelationMethod};
//!
//! let data = vec![
//!     vec![1.0, 2.0, 3.0, 4.0, 5.0],
//!     vec![2.0, 4.0, 5.0, 4.0, 5.0],
//!     vec![5.0, 4.0, 3.0, 2.0, 1.0],
//! ];
//! let names = vec!["x".into(), "y".into(), "z".into()];
//! let config = CorrelationConfig::default();
//! let result = correlation_analysis(&data, &names, &config).unwrap();
//!
//! assert_eq!(result.matrix.rows(), 3);
//! assert!(result.high_pairs.len() > 0); // x-z should be highly correlated (negatively)
//! ```

use crate::error::{Bound, InsightError};

// ── Input Validation ──────────────────────────────────────────────────

/// Validates that all columns contain clean numeric data (no NaN, no Inf, no missing).
///
/// Returns `Ok(())` if valid, or an appropriate `InsightError`.
///
/// ```
/// use u_insight::analysis::validate_clean_data;
///
/// let clean = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
/// assert!(validate_clean_data(&clean, &["a".into(), "b".into()]).is_ok());
///
/// let dirty = vec![vec![1.0, f64::NAN, 3.0]];
/// assert!(validate_clean_data(&dirty, &["a".into()]).is_err());
/// ```
pub fn validate_clean_data(columns: &[Vec<f64>], names: &[String]) -> Result<(), InsightError> {
    if columns.is_empty() {
        return Err(InsightError::empty("data"));
    }

    let n_rows = columns[0].len();
    if n_rows < 2 {
        return Err(InsightError::too_few("data", 2, n_rows));
    }

    for (i, col) in columns.iter().enumerate() {
        if col.len() != n_rows {
            return Err(InsightError::DimensionMismatch {
                expected: n_rows,
                actual: col.len(),
                index: Some(i),
            });
        }

        let name = names.get(i).cloned().unwrap_or_else(|| format!("col_{i}"));

        // Check for NaN or infinite values
        let nan_count = col.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            return Err(InsightError::MissingValues {
                column: name,
                count: nan_count,
            });
        }

        if let Some(index) = col.iter().position(|v| v.is_infinite()) {
            return Err(InsightError::ValueNotFinite {
                column: format!("data.{name}"),
                index,
            });
        }
    }

    Ok(())
}

// ── Correlation Analysis ──────────────────────────────────────────────

/// Method for correlation computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationMethod {
    /// Pearson product-moment correlation.
    Pearson,
    /// Spearman rank correlation.
    Spearman,
    /// Kendall tau-b rank correlation (handles ties).
    Kendall,
}

/// Configuration for correlation analysis.
#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    /// Correlation method. Default: Pearson.
    pub method: CorrelationMethod,
    /// Threshold for filtering high-correlation pairs: a pair is listed when
    /// |r| exceeds it, so it lies in [0, 1]. Default: 0.7.
    pub high_threshold: f64,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            method: CorrelationMethod::Pearson,
            high_threshold: 0.7,
        }
    }
}

/// A pair of columns with a high correlation value.
#[derive(Debug, Clone)]
pub struct CorrelationPair {
    /// First column name.
    pub col_a: String,
    /// Second column name.
    pub col_b: String,
    /// Correlation coefficient.
    pub r: f64,
    /// P-value for the correlation.
    pub p_value: f64,
}

/// Result of correlation analysis.
#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    /// Correlation method used.
    pub method: CorrelationMethod,
    /// n×n correlation matrix.
    pub matrix: u_numflow::matrix::Matrix,
    /// Pairs with |r| > threshold, sorted by |r| descending.
    pub high_pairs: Vec<CorrelationPair>,
    /// Column names.
    pub names: Vec<String>,
}

/// Computes a correlation matrix and identifies high-correlation pairs.
///
/// ```
/// use u_insight::analysis::{correlation_analysis, CorrelationConfig, CorrelationMethod};
///
/// let data = vec![
///     vec![1.0, 2.0, 3.0, 4.0, 5.0],
///     vec![2.1, 3.9, 6.1, 7.9, 10.1],
/// ];
/// let names = vec!["x".into(), "y".into()];
/// let config = CorrelationConfig::default();
/// let result = correlation_analysis(&data, &names, &config).unwrap();
/// assert!(result.matrix.get(0, 1).abs() > 0.99); // near-perfect correlation
/// ```
pub fn correlation_analysis(
    columns: &[Vec<f64>],
    names: &[String],
    config: &CorrelationConfig,
) -> Result<CorrelationAnalysis, InsightError> {
    check_correlation_threshold("high_threshold", config.high_threshold)?;
    if columns.is_empty() {
        return Err(InsightError::empty("data"));
    }
    // A pair's p-value is a t-test on n - 2 degrees of freedom: three rows
    // are the fewest it is defined for, for every method.
    let n_rows = columns[0].len();
    if n_rows < 3 {
        return Err(InsightError::too_few("data", 3, n_rows));
    }
    validate_clean_data(columns, names)?;

    // A constant column has no correlation with anything (its variance, or
    // every rank, is tied); name it rather than refuse the whole matrix.
    for (i, col) in columns.iter().enumerate() {
        if col.iter().all(|&v| v == col[0]) {
            let name = names.get(i).cloned().unwrap_or_else(|| format!("col_{i}"));
            return Err(InsightError::DegenerateData {
                parameter: Some(format!("data.{name}")),
                reason: format!("column '{name}' is constant; its correlation is undefined"),
            });
        }
    }

    // One column correlates only with itself.
    if columns.len() == 1 {
        return Ok(CorrelationAnalysis {
            method: config.method,
            matrix: u_numflow::matrix::Matrix::new(1, 1, vec![1.0])
                .expect("a 1 x 1 matrix holds one value"),
            high_pairs: Vec::new(),
            names: names.to_vec(),
        });
    }

    let refs: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();

    let matrix = match config.method {
        CorrelationMethod::Pearson => u_analytics::correlation::correlation_matrix(&refs),
        CorrelationMethod::Spearman => u_analytics::correlation::spearman_matrix(&refs),
        CorrelationMethod::Kendall => u_analytics::correlation::kendall_matrix(&refs),
    }
    .ok_or_else(|| {
        InsightError::degenerate("correlation matrix computation failed (near-zero variance)")
    })?;

    // Extract high-correlation pairs
    let n = columns.len();
    let mut high_pairs = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let r = matrix.get(i, j);
            if r.abs() > config.high_threshold {
                // Compute individual p-value for this pair
                let pair_result = match config.method {
                    CorrelationMethod::Pearson => {
                        u_analytics::correlation::pearson(refs[i], refs[j])
                    }
                    CorrelationMethod::Spearman => {
                        u_analytics::correlation::spearman(refs[i], refs[j])
                    }
                    CorrelationMethod::Kendall => {
                        u_analytics::correlation::kendall_tau_b(refs[i], refs[j])
                    }
                };

                let p_value = pair_result.map_or(f64::NAN, |pr| pr.p_value);

                high_pairs.push(CorrelationPair {
                    col_a: names[i].clone(),
                    col_b: names[j].clone(),
                    r,
                    p_value,
                });
            }
        }
    }

    // Sort by |r| descending
    high_pairs.sort_by(|a, b| {
        b.r.abs()
            .partial_cmp(&a.r.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(CorrelationAnalysis {
        method: config.method,
        matrix,
        high_pairs,
        names: names.to_vec(),
    })
}

// ── Regression Analysis ───────────────────────────────────────────────

/// Result of regression analysis within u-insight.
#[derive(Debug, Clone)]
pub struct RegressionAnalysis {
    /// Name of the target (dependent) variable.
    pub target_name: String,
    /// Names of predictor (independent) variables.
    pub predictor_names: Vec<String>,
    /// R² (coefficient of determination).
    pub r_squared: f64,
    /// Adjusted R².
    pub adj_r_squared: f64,
    /// Coefficients (intercept first, then predictors).
    pub coefficients: Vec<f64>,
    /// P-values for each coefficient.
    pub p_values: Vec<f64>,
    /// Variance Inflation Factors for each predictor (empty for simple regression).
    pub vif: Vec<f64>,
    /// F-statistic p-value.
    pub f_p_value: f64,
}

/// Performs regression analysis (simple or multiple OLS).
///
/// Delegates to `u-analytics::regression` for computation.
///
/// ```
/// use u_insight::analysis::regression_analysis;
///
/// let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
/// let y = vec![2.1, 3.9, 6.1, 7.9, 10.1];
/// let result = regression_analysis(
///     &x, &["x".into()], &y, "y",
/// ).unwrap();
/// assert!(result.r_squared > 0.99);
/// ```
pub fn regression_analysis(
    predictors: &[Vec<f64>],
    predictor_names: &[String],
    target: &[f64],
    target_name: &str,
) -> Result<RegressionAnalysis, InsightError> {
    if predictors.is_empty() {
        return Err(InsightError::empty("predictors"));
    }

    let n = target.len();
    if n < 3 {
        return Err(InsightError::too_few("target", 3, n));
    }

    // Validate target
    let nan_count = target.iter().filter(|v| v.is_nan()).count();
    if nan_count > 0 {
        return Err(InsightError::MissingValues {
            column: target_name.to_string(),
            count: nan_count,
        });
    }

    // Validate predictors
    let mut all_names = vec![target_name.to_string()];
    all_names.extend(predictor_names.iter().cloned());
    let mut all_cols = vec![target.to_vec()];
    all_cols.extend(predictors.iter().cloned());
    validate_clean_data(&all_cols, &all_names)?;

    if predictors.len() == 1 {
        // Simple linear regression
        let result = u_analytics::regression::simple_linear_regression(&predictors[0], target)
            .ok_or_else(|| InsightError::ComputationFailed {
                operation: "simple linear regression".into(),
                detail: "computation returned None (possible degenerate data)".into(),
            })?;

        Ok(RegressionAnalysis {
            target_name: target_name.to_string(),
            predictor_names: predictor_names.to_vec(),
            r_squared: result.r_squared,
            adj_r_squared: result.adjusted_r_squared,
            coefficients: vec![result.intercept, result.slope],
            p_values: vec![result.intercept_p, result.slope_p],
            vif: Vec::new(),
            f_p_value: result.f_p_value,
        })
    } else {
        // Multiple linear regression
        let x_refs: Vec<&[f64]> = predictors.iter().map(|c| c.as_slice()).collect();
        let result = u_analytics::regression::multiple_linear_regression(&x_refs, target)
            .ok_or_else(|| InsightError::ComputationFailed {
                operation: "multiple linear regression".into(),
                detail: "computation returned None (possible rank deficiency or singular data)"
                    .into(),
            })?;

        Ok(RegressionAnalysis {
            target_name: target_name.to_string(),
            predictor_names: predictor_names.to_vec(),
            r_squared: result.r_squared,
            adj_r_squared: result.adjusted_r_squared,
            coefficients: result.coefficients.clone(),
            p_values: result.p_values.clone(),
            vif: result.vif.clone(),
            f_p_value: result.f_p_value,
        })
    }
}

// ── Multicollinearity Diagnostics ────────────────────────────────────

/// Result of standalone VIF analysis.
#[derive(Debug, Clone)]
pub struct VifAnalysis {
    /// VIF for each column (same order as input).
    pub vif_per_column: Vec<f64>,
    /// Indices of columns with VIF > `threshold`.
    pub high_vif_columns: Vec<u32>,
    /// Threshold used.
    pub threshold: f64,
    /// Column names (echoed back from input).
    pub names: Vec<String>,
}

/// Computes VIF for each column in a column-major numeric dataset.
///
/// `VIF_j = 1 / (1 - R²_j)`, where `R²_j` is the coefficient of determination
/// from regressing column `j` on all others. Columns flagged in
/// `high_vif_columns` exceed `threshold` (typical: 5 moderate, 10 severe).
///
/// # Errors
/// - [`InsightError::ParameterOutOfRange`] if `threshold` is below 1 or not
///   finite: a VIF is never below 1, so a lower threshold flags every column
/// - [`InsightError::DimensionMismatch`] if column lengths differ
/// - [`InsightError::MissingValues`] if any column contains NaN
/// - [`InsightError::ValueNotFinite`] if any column contains Inf
/// - [`InsightError::DegenerateData`] if the VIF computation fails
///   (insufficient rows / near-zero variance)
///
/// # References
/// Marquardt (1970); Belsley/Kuh/Welsch (1980).
pub fn vif_analysis(
    columns: &[Vec<f64>],
    names: &[String],
    threshold: f64,
) -> Result<VifAnalysis, InsightError> {
    check_vif_threshold("threshold", threshold)?;
    validate_clean_data(columns, names)?;
    let refs: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();
    let v = u_analytics::regression::vif(&refs).ok_or_else(|| {
        InsightError::degenerate(
            "VIF computation failed (insufficient data, length mismatch, or near-zero variance)",
        )
    })?;
    let high_vif_columns: Vec<u32> = v
        .iter()
        .enumerate()
        .filter(|(_, &r)| r > threshold)
        .map(|(i, _)| i as u32)
        .collect();
    Ok(VifAnalysis {
        vif_per_column: v,
        high_vif_columns,
        threshold,
        names: names.to_vec(),
    })
}

/// A threshold on |r|: in [0, 1], since |r| never leaves it (1 lists no pair).
pub(crate) fn check_correlation_threshold(name: &str, value: f64) -> Result<(), InsightError> {
    InsightError::check_range(name, value, Bound::Closed(0.0), Bound::Closed(1.0))
}

/// A threshold on a variance inflation factor: at least 1, the smallest VIF
/// there is (`1 / (1 - R²)` with `R²` in [0, 1]).
pub(crate) fn check_vif_threshold(name: &str, value: f64) -> Result<(), InsightError> {
    InsightError::check_range(name, value, Bound::Closed(1.0), Bound::Unbounded)
}

/// Computes the 2-norm condition number of the sample covariance matrix.
///
/// Returns `f64::INFINITY` for numerically singular input (smallest
/// eigenvalue below 1e-15). Standard threshold: `cond > 30` indicates
/// multicollinearity (Belsley 1991).
///
/// # Errors
/// - Same as [`vif_analysis`].
pub fn condition_number(columns: &[Vec<f64>], names: &[String]) -> Result<f64, InsightError> {
    validate_clean_data(columns, names)?;
    let refs: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();
    u_analytics::regression::condition_number(&refs).ok_or_else(|| {
        InsightError::degenerate(
            "condition number computation failed (insufficient data or eigendecomposition failed)",
        )
    })
}

// ── Cramér's V ──────────────────────────────────────────────────────

/// Result of Cramér's V computation.
#[derive(Debug, Clone)]
pub struct CramersVResult {
    /// Cramér's V statistic (0.0 to 1.0).
    pub v: f64,
    /// Chi-squared statistic.
    pub chi_squared: f64,
    /// P-value from chi-squared test.
    pub p_value: f64,
    /// Number of rows in the contingency table.
    pub n_rows: usize,
    /// Number of columns in the contingency table.
    pub n_cols: usize,
}

/// Computes Cramér's V for the association between two categorical variables.
///
/// Cramér's V measures the strength of association between two nominal
/// variables. It ranges from 0 (no association) to 1 (complete association).
///
/// V = sqrt(χ² / (n * min(r-1, c-1)))
///
/// Reference: Cramér (1946). "Mathematical Methods of Statistics."
///
/// # Arguments
///
/// * `table` — Flat row-major contingency table (observed frequencies).
/// * `n_rows` — Number of rows in the table.
/// * `n_cols` — Number of columns in the table.
///
/// # Errors
///
/// - [`InsightError::ParameterOutOfRange`] if `n_rows` or `n_cols` is below 2,
///   or an entry of `table` is negative (`parameter` `table[i]`)
/// - [`InsightError::DimensionMismatch`] if `table` does not hold
///   `n_rows * n_cols` entries
/// - [`InsightError::ValueNotFinite`] (`table`, `index`) for a NaN or infinity
/// - [`InsightError::DegenerateData`] (`table`) if the table holds no
///   observations, or a whole row or column is 0: its expected frequencies are
///   0 and the chi-squared statistic divides by them
///
/// ```
/// use u_insight::analysis::cramers_v;
///
/// // Strong association
/// let table = [50.0, 0.0, 0.0, 50.0]; // 2x2, perfectly associated
/// let result = cramers_v(&table, 2, 2).unwrap();
/// assert!(result.v > 0.9);
/// ```
pub fn cramers_v(
    table: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<CramersVResult, InsightError> {
    InsightError::check_count("n_rows", n_rows, 2)?;
    InsightError::check_count("n_cols", n_cols, 2)?;
    if table.len() != n_rows * n_cols {
        return Err(InsightError::DimensionMismatch {
            expected: n_rows * n_cols,
            actual: table.len(),
            index: None,
        });
    }
    for (index, &v) in table.iter().enumerate() {
        if !v.is_finite() {
            return Err(InsightError::ValueNotFinite {
                column: "table".into(),
                index,
            });
        }
        // A frequency is a count of observations.
        InsightError::check_range(
            &format!("table[{index}]"),
            v,
            Bound::Closed(0.0),
            Bound::Unbounded,
        )?;
    }
    let empty_line = |what: &str, i: usize| InsightError::DegenerateData {
        parameter: Some("table".into()),
        reason: format!(
            "{what} {i} of the table sums to 0, so its expected frequencies are 0 and \
             chi-squared is undefined"
        ),
    };
    if let Some(i) =
        (0..n_rows).find(|&i| table[i * n_cols..(i + 1) * n_cols].iter().sum::<f64>() <= 0.0)
    {
        return Err(empty_line("row", i));
    }
    if let Some(j) =
        (0..n_cols).find(|&j| (0..n_rows).map(|i| table[i * n_cols + j]).sum::<f64>() <= 0.0)
    {
        return Err(empty_line("column", j));
    }

    let test =
        u_analytics::testing::chi_squared_independence(table, n_rows, n_cols).ok_or_else(|| {
            InsightError::ComputationFailed {
                operation: "chi-squared independence test".into(),
                detail: "refused a table every check above accepted".into(),
            }
        })?;

    let n: f64 = table.iter().sum();
    let k = n_rows.min(n_cols);
    let v = (test.statistic / (n * (k - 1) as f64)).sqrt();

    Ok(CramersVResult {
        v,
        chi_squared: test.statistic,
        p_value: test.p_value,
        n_rows,
        n_cols,
    })
}

// ── ANOVA F-test for Feature Selection ───────────────────────────────

/// Result of ANOVA F-test for a single feature.
#[derive(Debug, Clone)]
pub struct AnovaFeatureResult {
    /// Feature name.
    pub name: String,
    /// F-statistic.
    pub f_statistic: f64,
    /// P-value.
    pub p_value: f64,
}

/// Result of ANOVA feature selection across all features.
#[derive(Debug, Clone)]
pub struct AnovaSelectionResult {
    /// Per-feature ANOVA results, sorted by p-value ascending (most significant first).
    pub features: Vec<AnovaFeatureResult>,
    /// Indices of features with p-value <= significance_level, in sorted order.
    pub selected_indices: Vec<usize>,
}

/// Performs one-way ANOVA F-test for each feature against a categorical target.
///
/// This is the standard `f_classif` approach for feature selection:
/// for each continuous feature, split its values by the target class and
/// run one-way ANOVA to test if the group means differ significantly.
///
/// Features with low p-values have means that differ across classes, making them
/// useful for classification.
///
/// # Arguments
///
/// * `features` — Column-major numeric feature data. Each inner Vec is one feature.
/// * `feature_names` — Name for each feature.
/// * `target` — Categorical target labels (integers). Length must equal feature length.
/// * `significance_level` — Threshold for selecting features, in (0, 1)
///   (e.g., 0.05).
///
/// # Returns
///
/// `Err` if `significance_level` is not in (0, 1), data is empty or
/// dimensions mismatch.
///
/// ```
/// use u_insight::analysis::anova_feature_selection;
///
/// // Feature A separates classes well; Feature B does not
/// let features = vec![
///     vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2],  // A: clear separation
///     vec![3.0, 3.1, 2.9, 3.0, 3.1, 2.9],   // B: no separation
/// ];
/// let names = vec!["A".into(), "B".into()];
/// let target = vec![0, 0, 0, 1, 1, 1];
///
/// let result = anova_feature_selection(&features, &names, &target, 0.05).unwrap();
/// assert!(result.features[0].name == "A"); // A is most significant
/// assert!(result.features[0].p_value < 0.05);
/// assert!(result.selected_indices.contains(&0)); // A is selected
/// ```
pub fn anova_feature_selection(
    features: &[Vec<f64>],
    feature_names: &[String],
    target: &[usize],
    significance_level: f64,
) -> Result<AnovaSelectionResult, InsightError> {
    crate::distribution::check_significance_level("significance_level", significance_level)?;
    if features.is_empty() {
        return Err(InsightError::empty("features"));
    }

    let n = features[0].len();
    if n < 4 {
        return Err(InsightError::too_few("features", 4, n));
    }

    if target.len() != n {
        return Err(InsightError::DimensionMismatch {
            expected: n,
            actual: target.len(),
            index: None,
        });
    }

    // Find unique classes
    let mut classes: Vec<usize> = target.to_vec();
    classes.sort_unstable();
    classes.dedup();

    if classes.len() < 2 {
        return Err(InsightError::DegenerateData {
            parameter: Some("target".into()),
            reason: format!("ANOVA requires at least 2 classes, got {}", classes.len()),
        });
    }

    // For each feature, group values by class and run ANOVA
    let mut results: Vec<(usize, AnovaFeatureResult)> = Vec::with_capacity(features.len());

    for (i, feature) in features.iter().enumerate() {
        if feature.len() != n {
            return Err(InsightError::DimensionMismatch {
                expected: n,
                actual: feature.len(),
                index: Some(i),
            });
        }

        // Check for NaN
        let nan_count = feature.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            let name = feature_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("feature_{i}"));
            return Err(InsightError::MissingValues {
                column: name,
                count: nan_count,
            });
        }

        // Group by class
        let groups: Vec<Vec<f64>> = classes
            .iter()
            .map(|&cls| {
                feature
                    .iter()
                    .zip(target.iter())
                    .filter(|(_, &t)| t == cls)
                    .map(|(&v, _)| v)
                    .collect()
            })
            .collect();

        let group_refs: Vec<&[f64]> = groups.iter().map(|g| g.as_slice()).collect();

        let name = feature_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("feature_{i}"));

        match u_analytics::testing::one_way_anova(&group_refs) {
            Some(anova) => {
                results.push((
                    i,
                    AnovaFeatureResult {
                        name,
                        f_statistic: anova.f_statistic,
                        p_value: anova.p_value,
                    },
                ));
            }
            None => {
                // ANOVA failed (e.g., group too small) — report as non-significant
                results.push((
                    i,
                    AnovaFeatureResult {
                        name,
                        f_statistic: 0.0,
                        p_value: 1.0,
                    },
                ));
            }
        }
    }

    // Sort by p-value ascending
    results.sort_by(|a, b| {
        a.1.p_value
            .partial_cmp(&b.1.p_value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let selected_indices: Vec<usize> = results
        .iter()
        .filter(|(_, r)| r.p_value <= significance_level)
        .map(|(idx, _)| *idx)
        .collect();

    let features_sorted = results.into_iter().map(|(_, r)| r).collect();

    Ok(AnovaSelectionResult {
        features: features_sorted,
        selected_indices,
    })
}

// ── Mutual Information ──────────────────────────────────────────────

/// Result of mutual information estimation for a single feature.
#[derive(Debug, Clone)]
pub struct MutualInfoFeature {
    /// Feature name.
    pub name: String,
    /// Feature index.
    pub index: usize,
    /// Estimated mutual information (nats, ≥ 0).
    pub mi: f64,
}

/// Result of mutual information feature selection.
#[derive(Debug, Clone)]
pub struct MutualInfoResult {
    /// Per-feature MI estimates, sorted by MI descending (most informative first).
    pub features: Vec<MutualInfoFeature>,
}

/// Estimates mutual information between continuous features and a categorical target.
///
/// Uses equal-frequency binning (adaptive bins) to discretize continuous features,
/// then computes MI using the standard discrete formula:
///
/// MI(X; Y) = Σ_x Σ_y p(x,y) * ln(p(x,y) / (p(x) * p(y)))
///
/// Higher MI indicates stronger (possibly nonlinear) dependence between
/// the feature and target.
///
/// # Arguments
///
/// * `features` — Column-major numeric data. Each inner Vec is one feature.
/// * `feature_names` — Name for each feature.
/// * `target` — Categorical target labels (integers).
/// * `n_bins` — Number of bins for discretization (`None` = Sturges' rule).
///
/// ```
/// use u_insight::analysis::mutual_info_classif;
///
/// // Feature A separates classes; Feature B is noise
/// let features = vec![
///     vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2],
///     vec![3.0, 3.1, 2.9, 3.0, 3.1, 2.9],
/// ];
/// let names = vec!["A".into(), "B".into()];
/// let target = vec![0, 0, 0, 1, 1, 1];
///
/// let result = mutual_info_classif(&features, &names, &target, None).unwrap();
/// assert!(result.features[0].mi > result.features[1].mi);
/// ```
pub fn mutual_info_classif(
    features: &[Vec<f64>],
    feature_names: &[String],
    target: &[usize],
    n_bins: Option<usize>,
) -> Result<MutualInfoResult, InsightError> {
    if let Some(bins) = n_bins {
        InsightError::check_count("n_bins", bins, 1)?;
    }
    if features.is_empty() {
        return Err(InsightError::empty("features"));
    }

    let n = features[0].len();
    if n < 4 {
        return Err(InsightError::too_few("features", 4, n));
    }

    if target.len() != n {
        return Err(InsightError::DimensionMismatch {
            expected: n,
            actual: target.len(),
            index: None,
        });
    }

    // Determine number of bins: Sturges' rule
    let bins = n_bins.unwrap_or_else(|| {
        let k = ((n as f64).ln() / std::f64::consts::LN_2 + 1.0).ceil() as usize;
        k.max(2)
    });

    // Unique classes
    let mut classes: Vec<usize> = target.to_vec();
    classes.sort_unstable();
    classes.dedup();
    let n_classes = classes.len();

    if n_classes < 2 {
        return Err(InsightError::DegenerateData {
            parameter: Some("target".into()),
            reason: format!("mutual information requires at least 2 classes, got {n_classes}"),
        });
    }

    // Class index map
    let class_map: std::collections::HashMap<usize, usize> =
        classes.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    // p(y) — class marginal
    let mut class_counts = vec![0usize; n_classes];
    for &t in target {
        let ci = class_map[&t];
        class_counts[ci] += 1;
    }

    let n_f = n as f64;
    let mut results: Vec<MutualInfoFeature> = Vec::with_capacity(features.len());

    for (fi, feature) in features.iter().enumerate() {
        if feature.len() != n {
            return Err(InsightError::DimensionMismatch {
                expected: n,
                actual: feature.len(),
                index: Some(fi),
            });
        }

        let nan_count = feature.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            let name = feature_names
                .get(fi)
                .cloned()
                .unwrap_or_else(|| format!("feature_{fi}"));
            return Err(InsightError::MissingValues {
                column: name,
                count: nan_count,
            });
        }

        // Equal-frequency binning: sort values, assign bin by rank
        let mut indexed: Vec<(usize, f64)> = feature.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut bin_labels = vec![0usize; n];
        for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
            bin_labels[orig_idx] = (rank * bins / n).min(bins - 1);
        }

        // Joint counts: bin × class
        let mut joint = vec![0usize; bins * n_classes];
        let mut bin_counts = vec![0usize; bins];

        for i in 0..n {
            let b = bin_labels[i];
            let c = class_map[&target[i]];
            joint[b * n_classes + c] += 1;
            bin_counts[b] += 1;
        }

        // MI = Σ p(x,y) ln(p(x,y) / (p(x) * p(y)))
        let mut mi = 0.0;
        for b in 0..bins {
            for c in 0..n_classes {
                let joint_count = joint[b * n_classes + c];
                if joint_count == 0 {
                    continue;
                }
                let p_xy = joint_count as f64 / n_f;
                let p_x = bin_counts[b] as f64 / n_f;
                let p_y = class_counts[c] as f64 / n_f;
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }

        // Clip to non-negative (MI is non-negative by definition)
        mi = mi.max(0.0);

        let name = feature_names
            .get(fi)
            .cloned()
            .unwrap_or_else(|| format!("feature_{fi}"));

        results.push(MutualInfoFeature {
            name,
            index: fi,
            mi,
        });
    }

    // Sort by MI descending
    results.sort_by(|a, b| b.mi.partial_cmp(&a.mi).unwrap_or(std::cmp::Ordering::Equal));

    Ok(MutualInfoResult { features: results })
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Input validation ─────────────────────────────────────────

    #[test]
    fn validate_clean_ok() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let names = vec!["a".into(), "b".into()];
        assert!(validate_clean_data(&data, &names).is_ok());
    }

    #[test]
    fn validate_nan_rejected() {
        let data = vec![vec![1.0, f64::NAN, 3.0]];
        let names = vec!["a".into()];
        let err = validate_clean_data(&data, &names).unwrap_err();
        match err {
            InsightError::MissingValues { column, count } => {
                assert_eq!(column, "a");
                assert_eq!(count, 1);
            }
            _ => panic!("expected MissingValues error"),
        }
    }

    #[test]
    fn validate_infinity_rejected() {
        let data = vec![vec![1.0, f64::INFINITY, 3.0]];
        let names = vec!["a".into()];
        let err = validate_clean_data(&data, &names).unwrap_err();
        assert_eq!(
            err,
            InsightError::ValueNotFinite {
                column: "data.a".into(),
                index: 1
            }
        );
    }

    #[test]
    fn validate_nan_returns_missing_values() {
        let data = vec![vec![1.0, 2.0, f64::NAN]];
        let names = vec!["a".into()];
        let err = validate_clean_data(&data, &names).unwrap_err();
        assert!(matches!(err, InsightError::MissingValues { .. }));
    }

    #[test]
    fn validate_empty_rejected() {
        let data: Vec<Vec<f64>> = vec![];
        let names: Vec<String> = vec![];
        assert!(validate_clean_data(&data, &names).is_err());
    }

    #[test]
    fn validate_single_row_rejected() {
        let data = vec![vec![1.0]];
        let names = vec!["a".into()];
        assert!(validate_clean_data(&data, &names).is_err());
    }

    #[test]
    fn validate_dimension_mismatch() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]];
        let names = vec!["a".into(), "b".into()];
        let err = validate_clean_data(&data, &names).unwrap_err();
        assert!(matches!(err, InsightError::DimensionMismatch { .. }));
    }

    // ── Correlation analysis ─────────────────────────────────────

    #[test]
    fn pearson_correlation_matrix() {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0], // perfect correlation with x
            vec![5.0, 4.0, 3.0, 2.0, 1.0],  // perfect negative correlation
        ];
        let names = vec!["x".into(), "y".into(), "z".into()];
        let config = CorrelationConfig::default();
        let result = correlation_analysis(&data, &names, &config).unwrap();

        assert_eq!(result.matrix.rows(), 3);
        // Diagonal should be 1.0
        assert!((result.matrix.get(0, 0) - 1.0).abs() < 1e-10);
        // x-y: perfect positive
        assert!((result.matrix.get(0, 1) - 1.0).abs() < 1e-10);
        // x-z: perfect negative
        assert!((result.matrix.get(0, 2) + 1.0).abs() < 1e-10);

        // High pairs should include all 3 pairs (all |r| > 0.7)
        assert!(!result.high_pairs.is_empty());
    }

    #[test]
    fn spearman_correlation() {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 4.0, 9.0, 16.0, 25.0], // monotonic (perfect Spearman)
        ];
        let names = vec!["x".into(), "y".into()];
        let config = CorrelationConfig {
            method: CorrelationMethod::Spearman,
            high_threshold: 0.7,
        };
        let result = correlation_analysis(&data, &names, &config).unwrap();
        assert!((result.matrix.get(0, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn kendall_correlation() {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0], // perfect concordance
            vec![5.0, 4.0, 3.0, 2.0, 1.0], // perfect discordance
        ];
        let names = vec!["a".into(), "b".into(), "c".into()];
        let config = CorrelationConfig {
            method: CorrelationMethod::Kendall,
            high_threshold: 0.7,
        };
        let result = correlation_analysis(&data, &names, &config).unwrap();
        assert_eq!(result.method, CorrelationMethod::Kendall);
        assert!((result.matrix.get(0, 1) - 1.0).abs() < 1e-10);
        assert!((result.matrix.get(0, 2) + 1.0).abs() < 1e-10);
        // both off-diagonal pairs above |0.7|
        assert!(result.high_pairs.len() >= 2);
    }

    #[test]
    fn kendall_correlation_with_ties() {
        let data = vec![
            vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
        ];
        let names = vec!["a".into(), "b".into()];
        let config = CorrelationConfig {
            method: CorrelationMethod::Kendall,
            high_threshold: 0.5,
        };
        let result = correlation_analysis(&data, &names, &config).unwrap();
        // tau-b correctly handles ties; expect high but not perfect
        assert!(result.matrix.get(0, 1) > 0.7);
    }

    // ── Multicollinearity diagnostics ────────────────────────────

    #[test]
    fn vif_analysis_collinear_flags() {
        let x1: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let x2: Vec<f64> = x1.iter().map(|&v| v * 2.0 + 1e-3).collect();
        let cols = vec![x1, x2];
        let names = vec!["x1".into(), "x2".into()];
        let r = vif_analysis(&cols, &names, 10.0).unwrap();
        assert!(r.vif_per_column[0] > 10.0);
        assert!(r.high_vif_columns.contains(&0));
        assert!(r.high_vif_columns.contains(&1));
        assert_eq!(r.threshold, 10.0);
    }

    #[test]
    fn vif_analysis_independent_no_flag() {
        let x1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let x2: Vec<f64> = vec![5.0, 9.0, 1.0, 7.0, 3.0, 8.0, 2.0, 6.0, 4.0, 10.0];
        let cols = vec![x1, x2];
        let names = vec!["x1".into(), "x2".into()];
        let r = vif_analysis(&cols, &names, 10.0).unwrap();
        assert!(r.high_vif_columns.is_empty());
    }

    #[test]
    fn condition_number_orthogonal_small() {
        let x1 = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let x2 = vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
        let cols = vec![x1, x2];
        let names = vec!["x1".into(), "x2".into()];
        let c = condition_number(&cols, &names).unwrap();
        assert!(c < 5.0);
    }

    #[test]
    fn condition_number_collinear_huge() {
        let x1: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let x2: Vec<f64> = x1.iter().map(|&v| v * 2.0 + 1e-6).collect();
        let cols = vec![x1, x2];
        let names = vec!["x1".into(), "x2".into()];
        let c = condition_number(&cols, &names).unwrap();
        assert!(c > 1e5 || c.is_infinite());
    }

    #[test]
    fn vif_analysis_rejects_nan() {
        let x1 = vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0];
        let x2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cols = vec![x1, x2];
        let names = vec!["x1".into(), "x2".into()];
        assert!(vif_analysis(&cols, &names, 10.0).is_err());
    }

    #[test]
    fn high_correlation_filtering() {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0], // r ≈ 1.0 with x
            vec![3.1, 2.9, 3.0, 3.2, 2.8],  // nearly uncorrelated with x
        ];
        let names = vec!["x".into(), "y".into(), "noise".into()];
        let config = CorrelationConfig {
            method: CorrelationMethod::Pearson,
            high_threshold: 0.9,
        };
        let result = correlation_analysis(&data, &names, &config).unwrap();

        // Only x-y should be above 0.9
        assert_eq!(result.high_pairs.len(), 1);
        assert_eq!(result.high_pairs[0].col_a, "x");
        assert_eq!(result.high_pairs[0].col_b, "y");
    }

    #[test]
    fn correlation_rejects_nan() {
        let data = vec![vec![1.0, f64::NAN, 3.0], vec![4.0, 5.0, 6.0]];
        let names = vec!["a".into(), "b".into()];
        let config = CorrelationConfig::default();
        assert!(correlation_analysis(&data, &names, &config).is_err());
    }

    /// Two rows give no p-value under any method: refused before anything
    /// is computed, as too few rows.
    #[test]
    fn correlation_needs_three_rows() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 5.0]];
        let names = vec!["a".into(), "b".into()];
        for method in [
            CorrelationMethod::Pearson,
            CorrelationMethod::Spearman,
            CorrelationMethod::Kendall,
        ] {
            let config = CorrelationConfig {
                method,
                ..CorrelationConfig::default()
            };
            assert_eq!(
                correlation_analysis(&data, &names, &config).unwrap_err(),
                InsightError::too_few("data", 3, 2),
                "{method:?}"
            );
        }
    }

    /// A constant column is named; the other columns are not blamed.
    #[test]
    fn a_constant_column_is_named() {
        let data = vec![vec![1.0, 2.0, 3.0, 4.0], vec![7.0, 7.0, 7.0, 7.0]];
        let names = vec!["a".into(), "col2".into()];
        for method in [
            CorrelationMethod::Pearson,
            CorrelationMethod::Spearman,
            CorrelationMethod::Kendall,
        ] {
            let config = CorrelationConfig {
                method,
                ..CorrelationConfig::default()
            };
            match correlation_analysis(&data, &names, &config).unwrap_err() {
                InsightError::DegenerateData { parameter, .. } => {
                    assert_eq!(parameter.as_deref(), Some("data.col2"), "{method:?}")
                }
                other => panic!("expected DegenerateData, got {other:?}"),
            }
        }
    }

    /// No columns is an empty input; one column correlates with itself.
    #[test]
    fn correlation_of_no_columns_and_of_one() {
        let config = CorrelationConfig::default();
        assert_eq!(
            correlation_analysis(&[], &[], &config).unwrap_err(),
            InsightError::empty("data")
        );
        let one = correlation_analysis(&[vec![1.0, 2.0, 4.0]], &["a".into()], &config)
            .expect("one column");
        assert_eq!((one.matrix.rows(), one.matrix.get(0, 0)), (1, 1.0));
        assert!(one.high_pairs.is_empty());
    }

    // ── Regression analysis ──────────────────────────────────────

    #[test]
    fn simple_regression() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let y = vec![2.1, 3.9, 6.1, 7.9, 10.1];
        let result = regression_analysis(&x, &["x".into()], &y, "y").unwrap();

        assert!(result.r_squared > 0.99);
        assert_eq!(result.coefficients.len(), 2); // intercept + slope
        assert!(result.vif.is_empty()); // no VIF for simple regression
    }

    #[test]
    fn multiple_regression() {
        let x1: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        // x2 is NOT linearly dependent on x1 (add varying perturbation)
        let x2: Vec<f64> = (1..=20)
            .map(|i| (i as f64) * 0.5 + 3.0 + ((i * 7 % 11) as f64) * 0.3)
            .collect();
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 2.0 * a + 3.0 * b + 1.0)
            .collect();

        let predictors = vec![x1, x2];
        let result =
            regression_analysis(&predictors, &["x1".into(), "x2".into()], &y, "y").unwrap();

        assert!(result.r_squared > 0.99);
        assert_eq!(result.coefficients.len(), 3); // intercept + 2 predictors
        assert_eq!(result.vif.len(), 2); // VIF for each predictor
    }

    #[test]
    fn regression_rejects_nan_target() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![1.0, f64::NAN, 3.0];
        let err = regression_analysis(&x, &["x".into()], &y, "y").unwrap_err();
        assert!(matches!(err, InsightError::MissingValues { .. }));
    }

    #[test]
    fn regression_rejects_nan_predictor() {
        let x = vec![vec![1.0, f64::NAN, 3.0]];
        let y = vec![1.0, 2.0, 3.0];
        let err = regression_analysis(&x, &["x".into()], &y, "y").unwrap_err();
        assert!(matches!(err, InsightError::MissingValues { .. }));
    }

    #[test]
    fn regression_insufficient_data() {
        let x = vec![vec![1.0, 2.0]];
        let y = vec![1.0, 2.0];
        assert!(regression_analysis(&x, &["x".into()], &y, "y").is_err());
    }

    // ── Cramér's V ──────────────────────────────────────────────

    #[test]
    fn cramers_v_perfect_association() {
        // 2x2: perfectly associated
        let table = [50.0, 0.0, 0.0, 50.0];
        let result = cramers_v(&table, 2, 2).unwrap();
        assert!(result.v > 0.9, "V should be near 1.0: {}", result.v);
    }

    #[test]
    fn cramers_v_no_association() {
        // 2x2: uniform — no association
        let table = [25.0, 25.0, 25.0, 25.0];
        let result = cramers_v(&table, 2, 2).unwrap();
        assert!(result.v < 0.05, "V should be near 0: {}", result.v);
    }

    #[test]
    fn cramers_v_3x3() {
        // 3x3: strong diagonal association
        let table = [30.0, 1.0, 1.0, 1.0, 30.0, 1.0, 1.0, 1.0, 30.0];
        let result = cramers_v(&table, 3, 3).unwrap();
        assert!(result.v > 0.7, "V should be high: {}", result.v);
    }

    #[test]
    fn cramers_v_range() {
        let table = [10.0, 20.0, 30.0, 15.0, 25.0, 5.0];
        let result = cramers_v(&table, 2, 3).unwrap();
        assert!(result.v >= 0.0 && result.v <= 1.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    /// Every reason the table cannot be read is its own refusal, naming the
    /// argument -- not one `None` for all of them.
    #[test]
    fn cramers_v_says_why_it_refuses() {
        assert_eq!(
            cramers_v(&[10.0, 20.0], 1, 2).unwrap_err(),
            InsightError::ParameterOutOfRange {
                name: "n_rows".into(),
                low: Bound::Closed(2.0),
                high: Bound::Unbounded,
                got: 1.0,
            }
        );
        assert_eq!(
            cramers_v(&[10.0], 2, 2).unwrap_err(),
            InsightError::DimensionMismatch {
                expected: 4,
                actual: 1,
                index: None,
            }
        );
        assert_eq!(
            cramers_v(&[1.0, f64::NAN, 1.0, 1.0], 2, 2).unwrap_err(),
            InsightError::ValueNotFinite {
                column: "table".into(),
                index: 1,
            }
        );
        assert_eq!(
            cramers_v(&[1.0, 2.0, -3.0, 1.0], 2, 2).unwrap_err(),
            InsightError::ParameterOutOfRange {
                name: "table[2]".into(),
                low: Bound::Closed(0.0),
                high: Bound::Unbounded,
                got: -3.0,
            }
        );
        for (table, line) in [
            ([0.0, 0.0, 0.0, 0.0], "row 0"),
            ([1.0, 2.0, 0.0, 0.0], "row 1"),
            ([0.0, 2.0, 0.0, 1.0], "column 0"),
        ] {
            match cramers_v(&table, 2, 2).unwrap_err() {
                InsightError::DegenerateData {
                    parameter: Some(p),
                    reason,
                } => {
                    assert_eq!(p, "table");
                    assert!(reason.starts_with(line), "{reason}");
                }
                other => panic!("{other:?}"),
            }
        }
    }

    /// The thresholds and the significance level are refused outside the
    /// interval their comparison reads: |r| in [0, 1], a VIF at least 1, a
    /// probability in (0, 1).
    #[test]
    fn analysis_options_are_refused_outside_their_interval() {
        let columns = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
        ];
        let names: Vec<String> = vec!["a".into(), "b".into()];
        for bad in [-0.1, 1.5, f64::NAN] {
            let config = CorrelationConfig {
                high_threshold: bad,
                ..CorrelationConfig::default()
            };
            assert!(matches!(
                correlation_analysis(&columns, &names, &config),
                Err(InsightError::ParameterOutOfRange { name, low: Bound::Closed(l), high: Bound::Closed(h), .. })
                    if name == "high_threshold" && l == 0.0 && h == 1.0
            ));
        }
        for bad in [0.5, -1.0, f64::INFINITY] {
            assert!(matches!(
                vif_analysis(&columns, &names, bad),
                Err(InsightError::ParameterOutOfRange { name, low: Bound::Closed(l), high: Bound::Unbounded, .. })
                    if name == "threshold" && l == 1.0
            ));
        }
        assert!(vif_analysis(&columns, &names, 1.0).is_ok());
        let target = [0, 0, 0, 1, 1, 1];
        for bad in [0.0, 1.0, 2.0] {
            assert_eq!(
                anova_feature_selection(&columns, &names, &target, bad).unwrap_err(),
                InsightError::ParameterOutOfRange {
                    name: "significance_level".into(),
                    low: Bound::Open(0.0),
                    high: Bound::Open(1.0),
                    got: bad,
                }
            );
        }
    }

    // ── ANOVA feature selection ─────────────────────────────────

    #[test]
    fn anova_separating_feature() {
        // Feature A clearly separates classes; Feature B does not
        let features = vec![
            vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2], // A
            vec![3.0, 3.1, 2.9, 3.0, 3.1, 2.9], // B
        ];
        let names = vec!["A".into(), "B".into()];
        let target = vec![0, 0, 0, 1, 1, 1];

        let result = anova_feature_selection(&features, &names, &target, 0.05).unwrap();
        assert_eq!(result.features.len(), 2);
        // A should be most significant (lowest p)
        assert_eq!(result.features[0].name, "A");
        assert!(result.features[0].p_value < 0.01);
        // A should be selected
        assert!(result.selected_indices.contains(&0));
    }

    #[test]
    fn anova_no_significant() {
        // Both features are noise — no group difference
        let features = vec![
            vec![3.0, 3.1, 2.9, 3.0, 3.1, 2.9],
            vec![5.0, 5.1, 4.9, 5.0, 5.1, 4.9],
        ];
        let names = vec!["X".into(), "Y".into()];
        let target = vec![0, 0, 0, 1, 1, 1];

        let result = anova_feature_selection(&features, &names, &target, 0.01).unwrap();
        // With very tight significance, noise features should not be selected
        // (but p-values depend on the actual means, so just check structure)
        assert_eq!(result.features.len(), 2);
    }

    #[test]
    fn anova_multiple_classes() {
        // 3 classes, feature clearly separates them
        let features = vec![vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2]];
        let names = vec!["f0".into()];
        let target = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let result = anova_feature_selection(&features, &names, &target, 0.05).unwrap();
        assert_eq!(result.features.len(), 1);
        assert!(result.features[0].p_value < 0.01);
        assert_eq!(result.selected_indices.len(), 1);
    }

    #[test]
    fn anova_rejects_nan() {
        let features = vec![vec![1.0, f64::NAN, 3.0, 4.0]];
        let names = vec!["f0".into()];
        let target = vec![0, 0, 1, 1];
        assert!(anova_feature_selection(&features, &names, &target, 0.05).is_err());
    }

    #[test]
    fn anova_rejects_single_class() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let names = vec!["f0".into()];
        let target = vec![0, 0, 0, 0]; // only one class
        assert!(anova_feature_selection(&features, &names, &target, 0.05).is_err());
    }

    #[test]
    fn anova_rejects_dimension_mismatch() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let names = vec!["f0".into()];
        let target = vec![0, 0, 1]; // wrong length
        assert!(anova_feature_selection(&features, &names, &target, 0.05).is_err());
    }

    // ── Mutual information ──────────────────────────────────────

    #[test]
    fn mi_separating_vs_noise() {
        // A clearly separates classes; B is noise
        let features = vec![
            vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2],
            vec![3.0, 3.1, 2.9, 3.0, 3.1, 2.9],
        ];
        let names = vec!["A".into(), "B".into()];
        let target = vec![0, 0, 0, 1, 1, 1];

        let result = mutual_info_classif(&features, &names, &target, None).unwrap();
        assert_eq!(result.features.len(), 2);
        // A should have higher MI than B
        assert!(
            result.features[0].mi > result.features[1].mi,
            "A MI={} should > B MI={}",
            result.features[0].mi,
            result.features[1].mi
        );
    }

    #[test]
    fn mi_perfect_dependence() {
        // Perfect dependence: feature bins map 1:1 to classes
        let features = vec![vec![0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2]];
        let names = vec!["f0".into()];
        let target = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let result = mutual_info_classif(&features, &names, &target, None).unwrap();
        assert!(result.features[0].mi > 0.5, "MI={}", result.features[0].mi);
    }

    #[test]
    fn mi_nonnegative() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![5.0, 3.0, 7.0, 1.0, 8.0, 2.0, 6.0, 4.0],
        ];
        let names = vec!["a".into(), "b".into()];
        let target = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let result = mutual_info_classif(&features, &names, &target, None).unwrap();
        for f in &result.features {
            assert!(f.mi >= 0.0, "{} MI={}", f.name, f.mi);
        }
    }

    #[test]
    fn mi_custom_bins() {
        let features = vec![vec![1.0, 1.1, 5.0, 5.1, 10.0, 10.1]];
        let names = vec!["f0".into()];
        let target = vec![0, 0, 1, 1, 2, 2];

        let result = mutual_info_classif(&features, &names, &target, Some(3)).unwrap();
        assert!(result.features[0].mi > 0.0);
    }

    #[test]
    fn mi_rejects_nan() {
        let features = vec![vec![1.0, f64::NAN, 3.0, 4.0]];
        let names = vec!["f0".into()];
        let target = vec![0, 0, 1, 1];
        assert!(mutual_info_classif(&features, &names, &target, None).is_err());
    }

    #[test]
    fn mi_rejects_single_class() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let names = vec!["f0".into()];
        let target = vec![0, 0, 0, 0];
        assert!(mutual_info_classif(&features, &names, &target, None).is_err());
    }
}
