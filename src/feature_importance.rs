//! Feature importance analysis and selection.
//!
//! Provides tools for understanding feature relevance and detecting
//! multicollinearity in clean numeric datasets.
//!
//! - **Variance threshold**: filters near-constant features
//! - **Correlation filter**: detects highly correlated feature pairs
//! - **VIF (Variance Inflation Factor)**: multicollinearity detection
//! - **Condition number**: overall collinearity diagnostic
//! - **Feature ranking**: composite importance score
//!
//! All functions operate on the Analysis layer (clean data, no NaN/Inf).
//!
//! # Example
//!
//! ```
//! use u_insight::feature_importance::{feature_analysis, FeatureConfig};
//!
//! let features = vec![
//!     vec![1.0, 2.0, 3.0, 4.0, 5.0],   // varying
//!     vec![10.0, 20.0, 30.0, 40.0, 50.0], // varying (correlated with first)
//!     vec![5.0, 5.0, 5.0, 5.0, 5.0],   // constant
//! ];
//! let names = vec!["x1".to_string(), "x2".to_string(), "const".to_string()];
//! let result = feature_analysis(&features, &names, &FeatureConfig::default()).unwrap();
//!
//! assert_eq!(result.low_variance.len(), 1); // "const" flagged
//! assert_eq!(result.low_variance[0].name, "const");
//! ```

use crate::error::InsightError;
use u_numflow::matrix::Matrix;

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for feature importance analysis.
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Minimum variance threshold. Features with variance below this
    /// are flagged as low-variance. Default: 1e-10.
    pub variance_threshold: f64,
    /// Correlation threshold for flagging highly correlated pairs.
    /// Default: 0.9.
    pub correlation_threshold: f64,
    /// VIF threshold for flagging multicollinearity.
    /// VIF > threshold indicates problematic collinearity.
    /// Default: 10.0.
    pub vif_threshold: f64,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            variance_threshold: 1e-10,
            correlation_threshold: 0.9,
            vif_threshold: 10.0,
        }
    }
}

impl FeatureConfig {
    /// Sets the variance threshold.
    pub fn variance_threshold(mut self, t: f64) -> Self {
        self.variance_threshold = t;
        self
    }

    /// Sets the correlation threshold.
    pub fn correlation_threshold(mut self, t: f64) -> Self {
        self.correlation_threshold = t;
        self
    }

    /// Sets the VIF threshold.
    pub fn vif_threshold(mut self, t: f64) -> Self {
        self.vif_threshold = t;
        self
    }
}

// ── Result types ──────────────────────────────────────────────────────

/// A feature flagged as low-variance.
#[derive(Debug, Clone)]
pub struct LowVarianceFeature {
    /// Feature name.
    pub name: String,
    /// Feature index.
    pub index: usize,
    /// Computed variance.
    pub variance: f64,
}

/// A pair of features with high correlation.
#[derive(Debug, Clone)]
pub struct HighCorrelationPair {
    /// First feature name.
    pub feature_a: String,
    /// Second feature name.
    pub feature_b: String,
    /// First feature index.
    pub index_a: usize,
    /// Second feature index.
    pub index_b: usize,
    /// Pearson correlation coefficient.
    pub correlation: f64,
}

/// VIF result for a single feature.
#[derive(Debug, Clone)]
pub struct VifResult {
    /// Feature name.
    pub name: String,
    /// Feature index.
    pub index: usize,
    /// Variance Inflation Factor.
    pub vif: f64,
    /// Whether VIF exceeds the threshold.
    pub is_collinear: bool,
}

/// Individual feature importance score.
#[derive(Debug, Clone)]
pub struct FeatureScore {
    /// Feature name.
    pub name: String,
    /// Feature index.
    pub index: usize,
    /// Variance of the feature.
    pub variance: f64,
    /// Maximum absolute correlation with any other feature.
    pub max_abs_correlation: f64,
    /// VIF value.
    pub vif: f64,
    /// Composite importance score (0.0–1.0, higher = more important/less problematic).
    pub importance: f64,
}

/// Complete feature analysis result.
#[derive(Debug, Clone)]
pub struct FeatureAnalysisResult {
    /// Features flagged as low-variance.
    pub low_variance: Vec<LowVarianceFeature>,
    /// Highly correlated feature pairs.
    pub high_correlations: Vec<HighCorrelationPair>,
    /// VIF results for each feature.
    pub vif_results: Vec<VifResult>,
    /// Condition number of the feature matrix (sqrt(max_eigenvalue/min_eigenvalue)).
    pub condition_number: f64,
    /// Per-feature importance scores, sorted by importance descending.
    pub feature_scores: Vec<FeatureScore>,
    /// Number of features analyzed.
    pub n_features: usize,
    /// Number of observations.
    pub n_observations: usize,
}

// ── Feature analysis ──────────────────────────────────────────────────

/// Runs comprehensive feature importance analysis.
///
/// `features` is a list of feature vectors (one per feature, each of length n).
/// `names` provides a name for each feature.
///
/// ```
/// use u_insight::feature_importance::{feature_analysis, FeatureConfig};
///
/// let features = vec![
///     vec![1.0, 2.0, 3.0, 4.0],
///     vec![2.0, 4.0, 6.0, 8.0], // perfectly correlated with first
///     vec![5.0, 3.0, 7.0, 1.0], // independent
/// ];
/// let names: Vec<String> = vec!["a", "b", "c"].into_iter().map(String::from).collect();
/// let result = feature_analysis(&features, &names, &FeatureConfig::default()).unwrap();
///
/// // a and b should be flagged as highly correlated
/// assert!(!result.high_correlations.is_empty());
/// assert_eq!(result.n_features, 3);
/// ```
pub fn feature_analysis(
    features: &[Vec<f64>],
    names: &[String],
    config: &FeatureConfig,
) -> Result<FeatureAnalysisResult, InsightError> {
    let p = features.len();
    if p == 0 {
        return Err(InsightError::InvalidParameter {
            name: "features".into(),
            message: "at least 1 feature required".into(),
        });
    }
    if names.len() != p {
        return Err(InsightError::DimensionMismatch {
            expected: p,
            actual: names.len(),
        });
    }

    let n = features[0].len();
    if n < 2 {
        return Err(InsightError::InsufficientData {
            min_required: 2,
            actual: n,
        });
    }

    // Validate dimensions and data
    for (i, feat) in features.iter().enumerate() {
        if feat.len() != n {
            return Err(InsightError::DimensionMismatch {
                expected: n,
                actual: feat.len(),
            });
        }
        for &v in feat {
            if v.is_nan() || v.is_infinite() {
                return Err(InsightError::NonNumericColumn {
                    column: names[i].clone(),
                });
            }
        }
    }

    // Step 1: Compute variances
    let variances: Vec<f64> = features.iter().map(|f| compute_variance(f)).collect();

    let low_variance: Vec<LowVarianceFeature> = variances
        .iter()
        .enumerate()
        .filter(|(_, &v)| v < config.variance_threshold)
        .map(|(i, &v)| LowVarianceFeature {
            name: names[i].clone(),
            index: i,
            variance: v,
        })
        .collect();

    // Step 2: Correlation matrix
    let means: Vec<f64> = features
        .iter()
        .map(|f| f.iter().sum::<f64>() / n as f64)
        .collect();
    let stds: Vec<f64> = variances.iter().map(|v| v.sqrt()).collect();

    let mut correlation_matrix = vec![vec![0.0; p]; p];
    for i in 0..p {
        correlation_matrix[i][i] = 1.0;
        for j in (i + 1)..p {
            let r = if stds[i] > 1e-15 && stds[j] > 1e-15 {
                let cov: f64 = features[i]
                    .iter()
                    .zip(features[j].iter())
                    .map(|(&a, &b)| (a - means[i]) * (b - means[j]))
                    .sum::<f64>()
                    / (n - 1) as f64;
                cov / (stds[i] * stds[j])
            } else {
                0.0
            };
            correlation_matrix[i][j] = r;
            correlation_matrix[j][i] = r;
        }
    }

    let high_correlations: Vec<HighCorrelationPair> = {
        let mut pairs = Vec::new();
        for i in 0..p {
            for j in (i + 1)..p {
                let r = correlation_matrix[i][j];
                if r.abs() > config.correlation_threshold {
                    pairs.push(HighCorrelationPair {
                        feature_a: names[i].clone(),
                        feature_b: names[j].clone(),
                        index_a: i,
                        index_b: j,
                        correlation: r,
                    });
                }
            }
        }
        pairs.sort_by(|a, b| {
            b.correlation
                .abs()
                .partial_cmp(&a.correlation.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs
    };

    // Step 3: VIF (using simple regression approach)
    let vifs = compute_vif_values(features, n);
    let vif_results: Vec<VifResult> = vifs
        .iter()
        .enumerate()
        .map(|(i, &vif)| VifResult {
            name: names[i].clone(),
            index: i,
            vif,
            is_collinear: vif > config.vif_threshold,
        })
        .collect();

    // Step 4: Condition number via eigenvalues of correlation matrix
    let condition_number = compute_condition_number(&correlation_matrix, p);

    // Step 5: Feature importance scores
    let max_abs_correlations: Vec<f64> = (0..p)
        .map(|i| {
            (0..p)
                .filter(|&j| j != i)
                .map(|j| correlation_matrix[i][j].abs())
                .fold(0.0f64, f64::max)
        })
        .collect();

    let mut feature_scores: Vec<FeatureScore> = (0..p)
        .map(|i| {
            // Importance = high variance × (low correlation + low VIF) / 2
            // Zero-variance features get importance = 0.
            let has_variance = variances[i] > config.variance_threshold;
            let corr_score = 1.0 - max_abs_correlations[i].min(1.0);
            let vif_score = if vifs[i].is_finite() && vifs[i] > 0.0 {
                (1.0 / vifs[i]).min(1.0)
            } else {
                0.0
            };
            let importance = if has_variance {
                (corr_score + vif_score) / 2.0
            } else {
                0.0
            };

            FeatureScore {
                name: names[i].clone(),
                index: i,
                variance: variances[i],
                max_abs_correlation: max_abs_correlations[i],
                vif: vifs[i],
                importance,
            }
        })
        .collect();

    feature_scores.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(FeatureAnalysisResult {
        low_variance,
        high_correlations,
        vif_results,
        condition_number,
        feature_scores,
        n_features: p,
        n_observations: n,
    })
}

// ── Internal helpers ──────────────────────────────────────────────────

fn compute_variance(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let sum_sq: f64 = data.iter().map(|&x| (x - mean) * (x - mean)).sum();
    sum_sq / (n - 1) as f64
}

/// Compute VIF for each feature by regressing each on all others.
fn compute_vif_values(features: &[Vec<f64>], n: usize) -> Vec<f64> {
    let p = features.len();
    if p <= 1 {
        return vec![1.0; p];
    }

    let mut vifs = Vec::with_capacity(p);

    for target in 0..p {
        // Collect other features as predictors
        let predictors: Vec<&[f64]> = (0..p)
            .filter(|&i| i != target)
            .map(|i| features[i].as_slice())
            .collect();

        // Compute R² of target regressed on predictors
        let r2 = compute_r2_multiple(&features[target], &predictors, n);

        if r2 < 1.0 - 1e-15 {
            vifs.push(1.0 / (1.0 - r2));
        } else {
            vifs.push(f64::INFINITY);
        }
    }

    vifs
}

/// Compute R² for multiple regression using normal equations (Cholesky).
fn compute_r2_multiple(target: &[f64], predictors: &[&[f64]], n: usize) -> f64 {
    let p = predictors.len();
    if p == 0 || n < p + 1 {
        return 0.0;
    }

    // Center target
    let y_mean = target.iter().sum::<f64>() / n as f64;

    // Build X'X and X'y (with intercept)
    let p1 = p + 1; // predictors + intercept
    let mut xtx_data = vec![0.0; p1 * p1];
    let mut xty = vec![0.0; p1];

    for i in 0..n {
        let y = target[i];

        // Row of X: [1, x1, x2, ...]
        let mut x_row = Vec::with_capacity(p1);
        x_row.push(1.0);
        for pred in predictors {
            x_row.push(pred[i]);
        }

        for j in 0..p1 {
            xty[j] += x_row[j] * y;
            for k in j..p1 {
                xtx_data[j * p1 + k] += x_row[j] * x_row[k];
                if j != k {
                    xtx_data[k * p1 + j] += x_row[j] * x_row[k];
                }
            }
        }
    }

    let xtx = match Matrix::new(p1, p1, xtx_data) {
        Ok(m) => m,
        Err(_) => return 0.0,
    };

    let coeffs = match xtx.cholesky_solve(&xty) {
        Ok(c) => c,
        Err(_) => return 0.0, // singular → return 0
    };

    // Compute R² = 1 - SS_res / SS_tot
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for i in 0..n {
        let mut y_hat = coeffs[0]; // intercept
        for (j, pred) in predictors.iter().enumerate() {
            y_hat += coeffs[j + 1] * pred[i];
        }
        let y = target[i];
        ss_res += (y - y_hat) * (y - y_hat);
        ss_tot += (y - y_mean) * (y - y_mean);
    }

    if ss_tot < 1e-15 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Compute condition number from correlation matrix eigenvalues.
fn compute_condition_number(corr: &[Vec<f64>], p: usize) -> f64 {
    if p <= 1 {
        return 1.0;
    }

    // Build Matrix from correlation data
    let mut data = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            data[i * p + j] = corr[i][j];
        }
    }

    let mat = match Matrix::new(p, p, data) {
        Ok(m) => m,
        Err(_) => return f64::INFINITY,
    };

    match mat.eigen_symmetric() {
        Ok((eigenvalues, _)) => {
            let max_ev = eigenvalues
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let min_ev = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);

            if min_ev.abs() < 1e-10 || min_ev <= 0.0 {
                // Near-zero eigenvalue → near-singular → infinite condition number
                f64::INFINITY
            } else {
                (max_ev / min_ev).sqrt()
            }
        }
        Err(_) => f64::INFINITY,
    }
}

// ── Permutation Importance ────────────────────────────────────────────

/// Result of permutation importance for a single feature.
#[derive(Debug, Clone)]
pub struct PermutationImportanceFeature {
    /// Feature name.
    pub name: String,
    /// Feature index.
    pub index: usize,
    /// Mean decrease in score when feature is permuted.
    pub importance: f64,
    /// Standard deviation of importance across repetitions.
    pub std_dev: f64,
}

/// Result of permutation importance analysis.
#[derive(Debug, Clone)]
pub struct PermutationImportanceResult {
    /// Baseline score (R² before any permutation).
    pub baseline_score: f64,
    /// Per-feature results, sorted by importance descending.
    pub features: Vec<PermutationImportanceFeature>,
}

/// Computes permutation importance for regression features.
///
/// Measures each feature's importance by randomly shuffling it and
/// measuring the decrease in R² score. Model-agnostic approach using
/// simple OLS regression as the scoring model.
///
/// Reference: Breiman (2001). "Random Forests", Machine Learning.
///
/// # Arguments
///
/// * `features` — Column-major numeric features.
/// * `feature_names` — Name for each feature.
/// * `target` — Target variable.
/// * `n_repeats` — Number of permutation repeats (default: 5).
/// * `seed` — Random seed.
///
/// ```
/// use u_insight::feature_importance::permutation_importance;
///
/// let features = vec![
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
///     vec![5.0, 3.0, 7.0, 1.0, 8.0, 2.0, 6.0, 4.0],
/// ];
/// let names = vec!["signal".into(), "noise".into()];
/// let target: Vec<f64> = features[0].iter().map(|&x| 2.0 * x + 1.0).collect();
///
/// let result = permutation_importance(&features, &names, &target, 5, 42).unwrap();
/// assert!(result.features[0].importance > result.features[1].importance);
/// ```
pub fn permutation_importance(
    features: &[Vec<f64>],
    feature_names: &[String],
    target: &[f64],
    n_repeats: usize,
    seed: u64,
) -> Result<PermutationImportanceResult, InsightError> {
    if features.is_empty() {
        return Err(InsightError::InvalidParameter {
            name: "features".into(),
            message: "at least 1 feature required".into(),
        });
    }

    let n = target.len();
    let p = features.len();
    if n < p + 2 {
        return Err(InsightError::InsufficientData {
            min_required: p + 2,
            actual: n,
        });
    }

    for (i, f) in features.iter().enumerate() {
        if f.len() != n {
            return Err(InsightError::DimensionMismatch {
                expected: n,
                actual: f.len(),
            });
        }
        let nan_count = f.iter().filter(|v| v.is_nan()).count();
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
    }

    // Baseline score: R² using all features
    let pred_refs: Vec<&[f64]> = features.iter().map(|f| f.as_slice()).collect();
    let baseline_score = compute_r2_multiple(target, &pred_refs, n).max(0.0);

    let n_repeats = n_repeats.max(1);
    let mut rng_state = seed;
    let mut results: Vec<PermutationImportanceFeature> = Vec::with_capacity(p);

    for fi in 0..p {
        let mut decreases = Vec::with_capacity(n_repeats);

        for _ in 0..n_repeats {
            // Create permuted copy of feature fi
            let mut permuted = features[fi].clone();
            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = (rng_state >> 33) as usize % (i + 1);
                permuted.swap(i, j);
            }

            // Build features with fi replaced
            let mut perm_refs: Vec<&[f64]> = features.iter().map(|f| f.as_slice()).collect();
            perm_refs[fi] = &permuted;

            let perm_score = compute_r2_multiple(target, &perm_refs, n).max(0.0);
            decreases.push(baseline_score - perm_score);
        }

        let mean_decrease = decreases.iter().sum::<f64>() / decreases.len() as f64;
        let variance = if decreases.len() > 1 {
            decreases
                .iter()
                .map(|&d| (d - mean_decrease).powi(2))
                .sum::<f64>()
                / (decreases.len() - 1) as f64
        } else {
            0.0
        };

        let name = feature_names
            .get(fi)
            .cloned()
            .unwrap_or_else(|| format!("feature_{fi}"));

        results.push(PermutationImportanceFeature {
            name,
            index: fi,
            importance: mean_decrease,
            std_dev: variance.sqrt(),
        });
    }

    // Sort by importance descending
    results.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(PermutationImportanceResult {
        baseline_score,
        features: results,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: &[&str]) -> Vec<String> {
        n.iter().map(|s| s.to_string()).collect()
    }

    // ── Variance threshold ────────────────────────────────────────

    #[test]
    fn constant_feature_flagged() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 5.0, 5.0, 5.0], // constant
        ];
        let result = feature_analysis(
            &features,
            &names(&["vary", "const"]),
            &FeatureConfig::default(),
        )
        .unwrap();

        assert_eq!(result.low_variance.len(), 1);
        assert_eq!(result.low_variance[0].name, "const");
        assert_eq!(result.low_variance[0].index, 1);
    }

    #[test]
    fn all_varying_no_flags() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]];
        let result =
            feature_analysis(&features, &names(&["a", "b"]), &FeatureConfig::default()).unwrap();

        assert!(result.low_variance.is_empty());
    }

    #[test]
    fn custom_variance_threshold() {
        let features = vec![
            vec![1.0, 1.001, 1.002, 1.003], // very low variance
            vec![1.0, 2.0, 3.0, 4.0],
        ];
        let config = FeatureConfig::default().variance_threshold(0.01);
        let result = feature_analysis(&features, &names(&["low", "high"]), &config).unwrap();

        assert_eq!(result.low_variance.len(), 1);
        assert_eq!(result.low_variance[0].name, "low");
    }

    // ── Correlation filter ────────────────────────────────────────

    #[test]
    fn perfectly_correlated_detected() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0], // perfect correlation
            vec![5.0, 3.0, 7.0, 1.0, 9.0],  // independent
        ];
        let result = feature_analysis(
            &features,
            &names(&["a", "b", "c"]),
            &FeatureConfig::default(),
        )
        .unwrap();

        assert!(!result.high_correlations.is_empty());
        let pair = &result.high_correlations[0];
        assert!((pair.correlation.abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn uncorrelated_no_pairs() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 3.0, 1.0, 4.0, 2.0], // shuffled
        ];
        let result =
            feature_analysis(&features, &names(&["a", "b"]), &FeatureConfig::default()).unwrap();

        // Low correlation should not be flagged
        assert!(result.high_correlations.is_empty());
    }

    // ── VIF ───────────────────────────────────────────────────────

    #[test]
    fn independent_features_low_vif() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![8.0, 3.0, 6.0, 1.0, 7.0, 2.0, 5.0, 4.0],
            vec![2.0, 7.0, 4.0, 5.0, 3.0, 8.0, 1.0, 6.0],
        ];
        let result = feature_analysis(
            &features,
            &names(&["x1", "x2", "x3"]),
            &FeatureConfig::default(),
        )
        .unwrap();

        for vr in &result.vif_results {
            assert!(
                vr.vif < 5.0,
                "VIF for {} = {}, expected < 5.0 for independent features",
                vr.name,
                vr.vif
            );
            assert!(!vr.is_collinear);
        }
    }

    #[test]
    fn collinear_features_high_vif() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], // x1 * 2
            vec![5.0, 3.0, 7.0, 1.0, 9.0, 2.0, 6.0, 4.0],     // independent
        ];
        let result = feature_analysis(
            &features,
            &names(&["x1", "x2", "x3"]),
            &FeatureConfig::default(),
        )
        .unwrap();

        // x1 and x2 are perfectly collinear → VIF should be very high or infinite
        assert!(
            result.vif_results[0].vif > 10.0 || result.vif_results[0].vif.is_infinite(),
            "VIF for x1 = {}, expected > 10",
            result.vif_results[0].vif
        );
    }

    // ── Condition number ──────────────────────────────────────────

    #[test]
    fn well_conditioned_features() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![8.0, 3.0, 6.0, 1.0, 7.0, 2.0, 5.0, 4.0],
        ];
        let result =
            feature_analysis(&features, &names(&["a", "b"]), &FeatureConfig::default()).unwrap();

        assert!(
            result.condition_number < 100.0,
            "condition number = {}, expected reasonable for independent features",
            result.condition_number
        );
    }

    #[test]
    fn ill_conditioned_features() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.001, 2.002, 3.003, 4.004, 5.005], // nearly identical
        ];
        let result =
            feature_analysis(&features, &names(&["a", "b"]), &FeatureConfig::default()).unwrap();

        assert!(
            result.condition_number > 10.0,
            "condition number = {}, expected high for near-identical features",
            result.condition_number
        );
    }

    // ── Feature scores ────────────────────────────────────────────

    #[test]
    fn scores_sorted_descending() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            vec![5.0, 5.0, 5.0, 5.0, 5.0], // constant → lowest importance
        ];
        let result = feature_analysis(
            &features,
            &names(&["a", "b", "const"]),
            &FeatureConfig::default(),
        )
        .unwrap();

        for i in 1..result.feature_scores.len() {
            assert!(
                result.feature_scores[i].importance <= result.feature_scores[i - 1].importance,
                "scores not sorted: {} > {}",
                result.feature_scores[i].importance,
                result.feature_scores[i - 1].importance
            );
        }

        // Constant feature should be ranked last
        let last = result.feature_scores.last().expect("non-empty");
        assert_eq!(last.name, "const");
    }

    #[test]
    fn scores_in_range() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 3.0, 7.0, 1.0]];
        let result =
            feature_analysis(&features, &names(&["a", "b"]), &FeatureConfig::default()).unwrap();

        for score in &result.feature_scores {
            assert!(
                (0.0..=1.0).contains(&score.importance),
                "{}: importance {} out of range",
                score.name,
                score.importance
            );
        }
    }

    // ── Error cases ──────────────────────────────────────────────

    #[test]
    fn empty_features() {
        let features: Vec<Vec<f64>> = vec![];
        assert!(feature_analysis(&features, &[], &FeatureConfig::default()).is_err());
    }

    #[test]
    fn name_count_mismatch() {
        let features = vec![vec![1.0, 2.0]];
        assert!(
            feature_analysis(&features, &names(&["a", "b"]), &FeatureConfig::default()).is_err()
        );
    }

    #[test]
    fn insufficient_data() {
        let features = vec![vec![1.0]];
        assert!(feature_analysis(&features, &names(&["a"]), &FeatureConfig::default()).is_err());
    }

    #[test]
    fn nan_rejected() {
        let features = vec![vec![1.0, f64::NAN, 3.0]];
        assert!(feature_analysis(&features, &names(&["a"]), &FeatureConfig::default()).is_err());
    }

    // ── Single feature ────────────────────────────────────────────

    #[test]
    fn single_feature_analysis() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let result =
            feature_analysis(&features, &names(&["only"]), &FeatureConfig::default()).unwrap();

        assert_eq!(result.n_features, 1);
        assert_eq!(result.vif_results.len(), 1);
        assert!((result.vif_results[0].vif - 1.0).abs() < 1e-10);
        assert!(result.high_correlations.is_empty());
    }

    // ── Result metadata ───────────────────────────────────────────

    #[test]
    fn result_metadata() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let result =
            feature_analysis(&features, &names(&["a", "b"]), &FeatureConfig::default()).unwrap();

        assert_eq!(result.n_features, 2);
        assert_eq!(result.n_observations, 4);
    }

    // ── Permutation importance ────────────────────────────────────

    #[test]
    fn perm_importance_signal_vs_noise() {
        // Signal feature has strong relationship; noise does not
        let signal: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let noise: Vec<f64> = vec![5.0, 3.0, 7.0, 1.0, 8.0, 2.0, 6.0, 4.0, 9.0, 0.0];
        let target: Vec<f64> = signal.iter().map(|&x| 2.0 * x + 1.0).collect();

        let result = permutation_importance(
            &[signal, noise],
            &["signal".into(), "noise".into()],
            &target,
            5,
            42,
        )
        .unwrap();

        assert_eq!(result.features.len(), 2);
        assert!(result.baseline_score > 0.5);
        // Signal should be more important
        assert!(
            result.features[0].importance > result.features[1].importance,
            "signal imp={} should > noise imp={}",
            result.features[0].importance,
            result.features[1].importance,
        );
    }

    #[test]
    fn perm_importance_std_dev_nonneg() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![5.0, 3.0, 7.0, 1.0, 8.0, 2.0, 6.0, 4.0],
        ];
        let target: Vec<f64> = features[0].iter().map(|&x| x * 3.0).collect();

        let result =
            permutation_importance(&features, &["a".into(), "b".into()], &target, 3, 42).unwrap();

        for f in &result.features {
            assert!(f.std_dev >= 0.0, "{} std_dev={}", f.name, f.std_dev);
        }
    }

    #[test]
    fn perm_importance_sorted_desc() {
        let f1: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let f2: Vec<f64> = vec![5.0, 3.0, 7.0, 1.0, 8.0, 2.0, 6.0, 4.0, 9.0, 0.0];
        let target: Vec<f64> = f1.iter().map(|&x| x * 2.0).collect();

        let result =
            permutation_importance(&[f1, f2], &["f1".into(), "f2".into()], &target, 3, 42).unwrap();

        for i in 1..result.features.len() {
            assert!(result.features[i].importance <= result.features[i - 1].importance);
        }
    }

    #[test]
    fn perm_importance_rejects_nan() {
        let features = vec![vec![1.0, f64::NAN, 3.0, 4.0, 5.0]];
        let target = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(permutation_importance(&features, &["a".into()], &target, 3, 42).is_err());
    }

    #[test]
    fn perm_importance_rejects_empty() {
        let features: Vec<Vec<f64>> = vec![];
        let target: Vec<f64> = vec![];
        assert!(permutation_importance(&features, &[], &target, 3, 42).is_err());
    }
}
