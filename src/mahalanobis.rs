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
//!
//! # Threshold
//!
//! μ and Σ are estimated from the same `n` points the distances are taken
//! for, so a point is never far from a centre it helped place: `D²` is at most
//! `(n − 1)² / n`, and under normality it follows exactly
//!
//! ```text
//! n · D² / (n − 1)²  ~  Beta(p/2, (n − p − 1)/2)
//! ```
//!
//! — the χ²(p) law holds only for a point outside the estimate (or as
//! n → ∞). The threshold is that Beta quantile, scaled back. With few rows the
//! χ² quantile can exceed `(n − 1)²/n` and no point could ever be flagged
//! (p = 2, n = 6: largest possible D² 4.17, χ²₀.₉₇₅ 7.38).
//!
//! References: Mahalanobis (1936), "On the generalised distance in
//! statistics". Gnanadesikan & Kettenring (1972), *Biometrics* 28, 81–124.
//! Tracy, Young & Mason (1992), *J. Quality Technology* 24, 88–95 (Phase I T²).
//! Montgomery, *Introduction to Statistical Quality Control*, ch. 11.
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

use crate::error::{Bound, InsightError};
use u_numflow::matrix::Matrix;

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for Mahalanobis distance outlier detection.
#[derive(Debug, Clone)]
pub struct MahalanobisConfig {
    /// Probability of the threshold's quantile, in (0, 1). Default: 0.975.
    ///
    /// A point whose distance is above the `quantile` of its exact
    /// distribution (module docs) is an outlier: under normality each point is
    /// flagged with probability `1 − quantile` (2.5 % at the default).
    pub quantile: f64,
}

impl Default for MahalanobisConfig {
    fn default() -> Self {
        Self { quantile: 0.975 }
    }
}

impl MahalanobisConfig {
    /// Sets the threshold's quantile.
    pub fn quantile(mut self, q: f64) -> Self {
        self.quantile = q;
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
    /// Threshold on the squared distance: `(n − 1)²/n` times the `quantile`
    /// of Beta(p/2, (n − p − 1)/2).
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
/// - [`InsightError::InsufficientData`] — fewer than `p + 2` rows (where p = dimensionality):
///   with `p + 1` every point lies at the same distance and none can stand out
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
    // A probability outside (0, 1) has no quantile; left unchecked the
    // threshold is NaN, no distance exceeds it, and the result reports no
    // outliers as if the data had none.
    InsightError::check_range(
        "quantile",
        config.quantile,
        Bound::Open(0.0),
        Bound::Open(1.0),
    )?;
    if n == 0 {
        return Err(InsightError::empty("data"));
    }

    let p = data[0].len();
    if p == 0 {
        return Err(InsightError::DimensionMismatch {
            expected: 1,
            actual: 0,
            index: None,
        });
    }

    // p + 1 rows give a non-singular covariance, but then every point lies at
    // the same distance, (n − 1)²/n: the Beta law below needs n − p − 1 > 0.
    if n < p + 2 {
        return Err(InsightError::too_few("data", p + 2, n));
    }
    let q = config.quantile;

    // Validate dimensions and values
    for (row_idx, point) in data.iter().enumerate() {
        if point.len() != p {
            return Err(InsightError::DimensionMismatch {
                expected: p,
                actual: point.len(),
                index: Some(row_idx),
            });
        }
        if let Some(col_idx) = point.iter().position(|v| !v.is_finite()) {
            return Err(InsightError::ValueNotFinite {
                column: format!("data[{row_idx}]"),
                index: col_idx,
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
        // Judged against the column's own size: a column in small units has a
        // small variance and is not constant for that. Rounding the mean leaves
        // a variance of about (1e-16 * mean)^2, so below (1e-12 * mean)^2 there
        // is nothing but rounding.
        .filter(|&c| cov_data[c * p + c] <= 1e-24 * mean[c] * mean[c])
        .collect();
    if !zero_var_cols.is_empty() {
        return Err(InsightError::DegenerateData {
            parameter: None,
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
    let inv_cov = cov_mat.inverse().map_err(|_| {
        InsightError::degenerate(
            "covariance matrix is singular or near-singular (likely collinear columns)",
        )
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

    // The exact law of a point's distance from an estimate it helped make.
    let (nf, pf) = (n as f64, p as f64);
    let beta = u_numflow::distributions::BetaDistribution::new(pf / 2.0, (nf - pf - 1.0) / 2.0)
        .map_err(|e| InsightError::ComputationFailed {
            operation: "threshold distribution".into(),
            detail: e.to_string(),
        })?;
    let b = beta
        .quantile(q)
        .ok_or_else(|| InsightError::ComputationFailed {
            operation: "threshold quantile".into(),
            detail: format!("no Beta quantile at {q}"),
        })?;
    let threshold = (nf - 1.0) * (nf - 1.0) / nf * b;

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
        assert_eq!(
            err,
            InsightError::ValueNotFinite {
                column: "data[1]".into(),
                index: 0
            },
            "the refusal names the offending cell"
        );
    }

    #[test]
    fn infinity_rejected_where_it_sits() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, f64::INFINITY],
            vec![4.0, 5.0],
            vec![6.0, 7.0],
        ];
        let err = mahalanobis(&data, &MahalanobisConfig::default()).unwrap_err();
        assert_eq!(
            err,
            InsightError::ValueNotFinite {
                column: "data[1]".into(),
                index: 1
            }
        );
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
            InsightError::DegenerateData { reason, .. } => {
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

        let strict = mahalanobis(&data, &MahalanobisConfig::default().quantile(0.99)).unwrap();
        let lenient = mahalanobis(&data, &MahalanobisConfig::default().quantile(0.95)).unwrap();

        // Stricter threshold → fewer outliers (or equal)
        assert!(strict.threshold >= lenient.threshold);
    }

    /// The threshold is the Beta quantile, scaled: its Beta CDF -- the
    /// regularized incomplete beta, independent of the quantile search -- is
    /// the requested probability.
    #[test]
    fn threshold_is_the_scaled_beta_quantile() {
        for (dim, n, q) in [
            (1, 12, 0.975),
            (2, 6, 0.975),
            (2, 40, 0.95),
            (3, 25, 0.999),
            (5, 40, 0.975),
        ] {
            let data: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..dim)
                        .map(|j| ((i * (j + 3) * 7919) % 101) as f64 / 10.0)
                        .collect()
                })
                .collect();
            let r = mahalanobis(&data, &MahalanobisConfig::default().quantile(q)).unwrap();
            let (nf, pf) = (n as f64, dim as f64);
            let x = r.threshold * nf / ((nf - 1.0) * (nf - 1.0));
            let cdf =
                u_numflow::special::regularized_incomplete_beta(x, pf / 2.0, (nf - pf - 1.0) / 2.0);
            assert!(
                (cdf - q).abs() < 1e-8,
                "dim {dim}, n {n}, q {q}: CDF at threshold {cdf}"
            );
            let bound = (nf - 1.0) * (nf - 1.0) / nf;
            assert!(
                r.distances.iter().all(|&d| d <= bound + 1e-9),
                "D² above (n−1)²/n"
            );
        }
    }

    /// With many rows the estimate is the truth and the law is χ²(p).
    #[test]
    fn with_many_rows_the_threshold_meets_chi_squared() {
        let data: Vec<Vec<f64>> = (0..4000)
            .map(|i| vec![((i * 7919) % 101) as f64, ((i * 104_729) % 97) as f64])
            .collect();
        let r = mahalanobis(&data, &MahalanobisConfig::default()).unwrap();
        let chi2 = u_numflow::special::chi_squared_quantile(0.975, 2.0);
        assert!(
            (r.threshold / chi2 - 1.0).abs() < 0.005,
            "{} vs {chi2}",
            r.threshold
        );
    }

    /// p = 2, n = 6: the largest possible D² is 25/6 ≈ 4.17, below χ²₀.₉₇₅(2)
    /// = 7.38 -- the χ² threshold could flag nothing, even a point far off.
    #[test]
    fn a_far_point_in_a_small_sample_is_flagged() {
        let data = vec![
            vec![1.0, 2.0],
            vec![1.2, 2.1],
            vec![0.9, 1.9],
            vec![1.1, 2.2],
            vec![1.0, 1.8],
            vec![9.0, -6.0],
        ];
        let r = mahalanobis(&data, &MahalanobisConfig::default()).unwrap();
        assert!(r.threshold < 25.0 / 6.0);
        assert!(r.anomalies[5], "{:?} over {}", r.distances, r.threshold);
        assert_eq!(r.outlier_count, 1);
    }

    /// Monte Carlo: normal data, n = 10, p = 2, 3000 samples -- each point is
    /// flagged with probability 1 − quantile = 2.5 % (30 000 points, s.e.
    /// 0.09 %). The χ² threshold flagged 0.02 % here: D² = 8.1·B with
    /// B ~ Beta(1, 3.5), and P(B > 7.378/8.1) = 0.089^3.5 ≈ 2.1e-4.
    #[test]
    fn under_normality_a_point_is_flagged_at_the_stated_rate() {
        use u_numflow::distributions::{Normal, Sample};
        let z = Normal::new(0.0, 1.0).unwrap();
        let mut rng = u_numflow::random::create_rng(5);
        let (mut flagged, mut points) = (0, 0);
        for _ in 0..3000 {
            let data: Vec<Vec<f64>> = (0..10)
                .map(|_| {
                    let a = z.sample(&mut rng);
                    vec![a, 0.6 * a + 0.8 * z.sample(&mut rng)]
                })
                .collect();
            let r = mahalanobis(&data, &MahalanobisConfig::default()).unwrap();
            flagged += r.outlier_count;
            points += 10;
        }
        let rate = flagged as f64 / points as f64;
        assert!((rate - 0.025).abs() < 0.004, "flag rate {rate}");
    }

    #[test]
    fn p_plus_one_rows_are_too_few() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![0.0, 0.0]];
        match mahalanobis(&data, &MahalanobisConfig::default()) {
            Err(InsightError::InsufficientData {
                min_required,
                actual,
                ..
            }) => assert_eq!((min_required, actual), (4, 3)),
            other => panic!("{other:?}"),
        }
    }

    /// A quantile outside (0, 1) used to yield a NaN threshold that no distance
    /// exceeds -- a result claiming there were no outliers.
    #[test]
    fn quantile_outside_the_unit_interval_is_refused() {
        let data = make_cluster(&[0.0, 0.0], 30, 1.0);
        for q in [0.0, 1.0, 1.5, -0.1, f64::NAN] {
            match mahalanobis(&data, &MahalanobisConfig::default().quantile(q)) {
                Err(InsightError::ParameterOutOfRange {
                    name, low, high, ..
                }) => {
                    assert_eq!(name, "quantile");
                    assert_eq!((low, high), (Bound::Open(0.0), Bound::Open(1.0)));
                }
                other => panic!("q = {q}: {other:?}"),
            }
        }
    }
}
