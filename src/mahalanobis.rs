//! Mahalanobis distance for multivariate outlier detection.
//!
//! Mahalanobis distance measures how far a point is from the center of a
//! distribution, accounting for correlations between variables. Unlike
//! Euclidean distance, it respects the shape and orientation of the data
//! cloud.
//!
//! # Algorithm
//!
//! D²(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
//!
//! where μ is the mean vector and Σ is the covariance matrix.
//! Under normality, D² follows a χ²(p) distribution.
//!
//! Reference: Mahalanobis (1936). "On the generalised distance in statistics."
//!
//! # Example
//!
//! ```
//! use u_insight::mahalanobis::{mahalanobis, MahalanobisConfig};
//!
//! // Normal cluster + one outlier
//! let data = vec![
//!     vec![1.0, 2.0], vec![1.5, 2.5], vec![2.0, 3.0],
//!     vec![1.2, 2.2], vec![1.8, 2.8], vec![1.3, 2.1],
//!     vec![50.0, -20.0], // outlier
//! ];
//!
//! let result = mahalanobis(&data, &MahalanobisConfig::default()).unwrap();
//! // The outlier should have the highest distance
//! let max_idx = result.distances.iter()
//!     .enumerate()
//!     .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
//!     .unwrap().0;
//! assert_eq!(max_idx, 6);
//! assert!(result.distances[6] > result.distances[0]);
//! ```

use crate::error::InsightError;
use u_numflow::matrix::Matrix;

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for Mahalanobis distance outlier detection.
#[derive(Debug, Clone)]
pub struct MahalanobisConfig {
    /// Significance level for the chi-squared threshold. Default: 0.975.
    ///
    /// Points with CDF(D², df=p) > this value are classified as outliers.
    /// 0.975 corresponds to a 2.5% false positive rate (upper tail).
    pub chi2_quantile: f64,
}

impl Default for MahalanobisConfig {
    fn default() -> Self {
        Self {
            chi2_quantile: 0.975,
        }
    }
}

impl MahalanobisConfig {
    /// Sets the chi-squared quantile threshold.
    pub fn chi2_quantile(mut self, q: f64) -> Self {
        self.chi2_quantile = q;
        self
    }
}

// ── Result ────────────────────────────────────────────────────────────

/// Result of Mahalanobis distance computation.
#[derive(Debug, Clone)]
pub struct MahalanobisResult {
    /// Squared Mahalanobis distance for each point.
    pub distances: Vec<f64>,
    /// Binary anomaly labels: true = outlier.
    pub anomalies: Vec<bool>,
    /// Chi-squared threshold used for classification.
    pub threshold: f64,
    /// Number of outliers detected.
    pub outlier_count: usize,
    /// Fraction of points classified as outliers.
    pub outlier_fraction: f64,
    /// Mean vector (centroid).
    pub mean: Vec<f64>,
}

// ── Main function ─────────────────────────────────────────────────────

/// Computes Mahalanobis distance for multivariate outlier detection.
///
/// # Arguments
///
/// * `data` — each inner `Vec<f64>` is a point (all must have same dimensionality).
/// * `config` — parameters for outlier classification.
///
/// # Errors
///
/// Returns `InsightError` with specific categories:
/// - [`InsightError::InsufficientData`] — fewer than `p + 1` rows (where p = dimensionality)
/// - [`InsightError::DimensionMismatch`] — points with inconsistent dimensions
/// - [`InsightError::DegenerateData`] — non-finite values, near-zero variance columns,
///   or singular covariance matrix (likely collinear columns). The `reason` field
///   includes column indices when known.
/// - [`InsightError::ComputationFailed`] — internal matrix operations failed.
///
/// # Example
///
/// ```
/// use u_insight::mahalanobis::{mahalanobis, MahalanobisConfig};
///
/// let data = vec![
///     vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0],
///     vec![1.5, 3.0], vec![2.5, 5.0], vec![1.0, 2.5],
///     vec![20.0, -10.0], // outlier
/// ];
/// let result = mahalanobis(&data, &MahalanobisConfig::default()).unwrap();
/// assert!(result.distances[6] > result.distances[0]);
/// ```
pub fn mahalanobis(
    data: &[Vec<f64>],
    config: &MahalanobisConfig,
) -> Result<MahalanobisResult, InsightError> {
    let n = data.len();
    if n == 0 {
        return Err(InsightError::InsufficientData {
            min_required: 2,
            actual: 0,
        });
    }

    let p = data[0].len();
    if p == 0 {
        return Err(InsightError::DimensionMismatch {
            expected: 1,
            actual: 0,
        });
    }

    // Need at least p+1 observations for a non-singular covariance matrix
    if n <= p {
        return Err(InsightError::InsufficientData {
            min_required: p + 1,
            actual: n,
        });
    }

    // Validate dimensions and values
    for (row_idx, point) in data.iter().enumerate() {
        if point.len() != p {
            return Err(InsightError::DimensionMismatch {
                expected: p,
                actual: point.len(),
            });
        }
        if let Some(col_idx) = point.iter().position(|v| !v.is_finite()) {
            return Err(InsightError::DegenerateData {
                reason: format!(
                    "non-finite value at row {row_idx}, column {col_idx}"
                ),
            });
        }
    }

    // Compute mean vector
    let mut mean = vec![0.0; p];
    for point in data.iter() {
        for (j, val) in point.iter().enumerate() {
            mean[j] += val;
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }

    // Compute sample covariance matrix (unbiased, / (n-1))
    let mut cov_data = vec![0.0; p * p];
    for point in data.iter() {
        for r in 0..p {
            let dr = point[r] - mean[r];
            for c in r..p {
                let dc = point[c] - mean[c];
                cov_data[r * p + c] += dr * dc;
            }
        }
    }
    let denom = (n - 1) as f64;
    for r in 0..p {
        for c in r..p {
            let val = cov_data[r * p + c] / denom;
            cov_data[r * p + c] = val;
            cov_data[c * p + r] = val; // symmetric
        }
    }

    // Pre-screen for zero-variance columns (covariance matrix diagonal).
    // A column with variance ≈ 0 makes the covariance matrix singular and
    // is the most common cause of Mahalanobis failure on real-world data.
    let zero_var_cols: Vec<usize> = (0..p)
        .filter(|&c| cov_data[c * p + c].abs() < 1e-12)
        .collect();
    if !zero_var_cols.is_empty() {
        return Err(InsightError::DegenerateData {
            reason: format!(
                "near-zero variance in column(s) {zero_var_cols:?}; covariance matrix would be singular"
            ),
        });
    }

    // Build Matrix and invert
    let cov_mat = Matrix::new(p, p, cov_data).map_err(|e| InsightError::ComputationFailed {
        operation: "covariance matrix construction".into(),
        detail: e.to_string(),
    })?;
    let inv_cov = cov_mat.inverse().map_err(|_| InsightError::DegenerateData {
        reason: "covariance matrix is singular or near-singular (likely collinear columns)".into(),
    })?;

    // Compute squared Mahalanobis distance for each point
    let mut distances = Vec::with_capacity(n);
    for point in data.iter() {
        let mut diff = vec![0.0; p];
        for j in 0..p {
            diff[j] = point[j] - mean[j];
        }

        // d² = diff^T * inv_cov * diff
        let mut d2 = 0.0;
        for (r, &dr) in diff.iter().enumerate() {
            let mut row_sum = 0.0;
            for (c, &dc) in diff.iter().enumerate() {
                row_sum += inv_cov.get(r, c) * dc;
            }
            d2 += dr * row_sum;
        }

        // Ensure non-negative (numerical noise)
        distances.push(d2.max(0.0));
    }

    // Compute chi-squared threshold
    let threshold = chi2_quantile(p as f64, config.chi2_quantile);

    // Classify outliers
    let anomalies: Vec<bool> = distances.iter().map(|&d| d > threshold).collect();
    let outlier_count = anomalies.iter().filter(|&&a| a).count();
    let outlier_fraction = if n > 0 {
        outlier_count as f64 / n as f64
    } else {
        0.0
    };

    Ok(MahalanobisResult {
        distances,
        anomalies,
        threshold,
        outlier_count,
        outlier_fraction,
        mean,
    })
}

// ── Chi-squared quantile (Wilson-Hilferty approximation) ─────────────

/// Approximate chi-squared quantile using Wilson-Hilferty normal approximation.
///
/// For df degrees of freedom and probability p:
/// χ²_p ≈ df * (1 - 2/(9*df) + z_p * sqrt(2/(9*df)))^3
///
/// where z_p = Φ⁻¹(p) is the standard normal quantile.
fn chi2_quantile(df: f64, p: f64) -> f64 {
    if df <= 0.0 || p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }

    let z = u_numflow::special::inverse_normal_cdf(p);
    let term = 2.0 / (9.0 * df);
    let cube = 1.0 - term + z * term.sqrt();

    // Handle edge case where cube might be negative
    if cube <= 0.0 {
        return 0.0;
    }

    df * cube * cube * cube
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cluster(center: &[f64], n: usize, spread: f64) -> Vec<Vec<f64>> {
        let mut points = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / n as f64;
            let point: Vec<f64> = center
                .iter()
                .enumerate()
                .map(|(j, &c)| c + spread * (t * (j as f64 + 1.0)).sin())
                .collect();
            points.push(point);
        }
        points
    }

    #[test]
    fn detects_obvious_outlier() {
        let mut data = make_cluster(&[0.0, 0.0], 30, 1.0);
        data.push(vec![50.0, 50.0]); // far outlier

        let result = mahalanobis(&data, &MahalanobisConfig::default()).unwrap();

        // Outlier should have largest distance
        let max_idx = result
            .distances
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 30);
        assert!(result.anomalies[30]);
    }

    #[test]
    fn inliers_below_threshold() {
        // Tight cluster with independent noise per dimension
        let data: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let t = i as f64 * 0.1;
                let n1 = (i as f64 * 0.73).sin() * 0.5;
                let n2 = (i as f64 * 1.41).cos() * 0.5;
                let n3 = (i as f64 * 2.17).sin() * 0.5;
                vec![t + n1, n2 + 1.0, n3 + 2.0]
            })
            .collect();

        let result = mahalanobis(&data, &MahalanobisConfig::default()).unwrap();

        // Most points should be inliers
        assert!(
            result.outlier_fraction < 0.5,
            "too many outliers in tight cluster: {}",
            result.outlier_fraction
        );
    }

    #[test]
    fn mean_computed_correctly() {
        let data = vec![
            vec![2.0, 4.0],
            vec![4.0, 7.0],
            vec![6.0, 13.0],
            vec![4.0, 8.0],
        ];

        let result = mahalanobis(&data, &MahalanobisConfig::default()).unwrap();

        assert!((result.mean[0] - 4.0).abs() < 1e-10);
        assert!((result.mean[1] - 8.0).abs() < 1e-10); // (4+7+13+8)/4 = 8
    }

    #[test]
    fn empty_data() {
        let data: Vec<Vec<f64>> = Vec::new();
        assert!(mahalanobis(&data, &MahalanobisConfig::default()).is_err());
    }

    #[test]
    fn insufficient_data() {
        // p=2 dimensions needs at least 3 points
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(mahalanobis(&data, &MahalanobisConfig::default()).is_err());
    }

    #[test]
    fn dimension_mismatch() {
        let data = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(mahalanobis(&data, &MahalanobisConfig::default()).is_err());
    }

    #[test]
    fn nan_rejected() {
        let data = vec![
            vec![1.0, 2.0],
            vec![f64::NAN, 3.0],
            vec![4.0, 5.0],
            vec![6.0, 7.0],
        ];
        let err = mahalanobis(&data, &MahalanobisConfig::default()).unwrap_err();
        // Must surface as DegenerateData with row/column locator (not Io).
        match err {
            InsightError::DegenerateData { reason } => {
                assert!(
                    reason.contains("row 1") && reason.contains("column 0"),
                    "reason should pinpoint the offending cell, got: {reason}"
                );
            }
            other => panic!("expected DegenerateData, got {other:?}"),
        }
    }

    #[test]
    fn infinity_rejected_as_degenerate() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, f64::INFINITY],
            vec![4.0, 5.0],
            vec![6.0, 7.0],
        ];
        let err = mahalanobis(&data, &MahalanobisConfig::default()).unwrap_err();
        assert!(matches!(err, InsightError::DegenerateData { .. }));
    }

    #[test]
    fn zero_variance_column_diagnosed() {
        // Column 0 is constant → zero variance → singular covariance.
        // Must be diagnosed as DegenerateData with the offending column index in reason.
        let data: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let t = i as f64 * 0.1;
                vec![5.0, t.sin(), t.cos()]
            })
            .collect();
        let err = mahalanobis(&data, &MahalanobisConfig::default()).unwrap_err();
        match err {
            InsightError::DegenerateData { reason } => {
                assert!(
                    reason.contains('0'),
                    "reason should mention column 0, got: {reason}"
                );
                assert!(
                    reason.contains("variance") || reason.contains("singular"),
                    "reason should explain the cause, got: {reason}"
                );
            }
            other => panic!("expected DegenerateData, got {other:?}"),
        }
    }

    #[test]
    fn collinear_columns_diagnosed_as_singular() {
        // Column 1 = 2 × column 0 → perfect collinearity → singular covariance.
        // Pre-screen passes (variance > 0), so this exercises the inverse() path.
        let data: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let t = i as f64 * 0.5;
                vec![t, 2.0 * t]
            })
            .collect();
        let err = mahalanobis(&data, &MahalanobisConfig::default()).unwrap_err();
        assert!(
            matches!(err, InsightError::DegenerateData { .. }),
            "collinear input must yield DegenerateData, got {err:?}"
        );
    }

    #[test]
    fn distances_non_negative() {
        let data: Vec<Vec<f64>> = (0..15)
            .map(|i| {
                let t = i as f64 * 0.5;
                let noise = (i as f64 * 1.37).sin() * 0.2;
                vec![t + noise, t * 1.5 - noise]
            })
            .collect();

        let result = mahalanobis(&data, &MahalanobisConfig::default()).unwrap();

        for &d in &result.distances {
            assert!(d >= 0.0, "distance should be non-negative, got {d}");
        }
    }

    #[test]
    fn high_dimensional() {
        // 5D data with non-collinear variation
        let mut data: Vec<Vec<f64>> = (0..30)
            .map(|i| {
                let t = i as f64 * 0.1;
                let n = (i as f64 * 0.97).sin() * 0.3;
                vec![t + n, t * 0.5 - n, t * 2.0 + n * 0.5, t.sin(), t.cos()]
            })
            .collect();
        data.push(vec![100.0, 100.0, 100.0, 100.0, 100.0]); // outlier

        let result = mahalanobis(&data, &MahalanobisConfig::default()).unwrap();

        // Outlier should be detected
        let max_idx = result
            .distances
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 30);
    }

    #[test]
    fn custom_quantile() {
        let mut data = make_cluster(&[0.0, 0.0], 30, 1.0);
        data.push(vec![10.0, 10.0]); // moderate outlier

        let strict = mahalanobis(&data, &MahalanobisConfig::default().chi2_quantile(0.99)).unwrap();
        let lenient =
            mahalanobis(&data, &MahalanobisConfig::default().chi2_quantile(0.95)).unwrap();

        // Stricter threshold → fewer outliers (or equal)
        assert!(strict.threshold >= lenient.threshold);
    }

    #[test]
    fn chi2_quantile_known_values() {
        // χ²(2, 0.95) ≈ 5.991
        let q = chi2_quantile(2.0, 0.95);
        assert!(
            (q - 5.991).abs() < 0.1,
            "chi2(2, 0.95) expected ~5.991, got {q}"
        );

        // χ²(1, 0.95) ≈ 3.841
        let q1 = chi2_quantile(1.0, 0.95);
        assert!(
            (q1 - 3.841).abs() < 0.1,
            "chi2(1, 0.95) expected ~3.841, got {q1}"
        );

        // χ²(5, 0.975) ≈ 12.833
        let q5 = chi2_quantile(5.0, 0.975);
        assert!(
            (q5 - 12.833).abs() < 0.2,
            "chi2(5, 0.975) expected ~12.833, got {q5}"
        );
    }
}
