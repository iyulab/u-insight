//! Local Outlier Factor (LOF) for density-based anomaly detection.
//!
//! LOF identifies local outliers by comparing the local density of a point
//! with the local densities of its neighbors. Unlike global methods (Z-score),
//! LOF detects outliers in datasets with varying density clusters.
//!
//! # Algorithm
//!
//! Reference: Breunig, Kriegel, Ng & Sander (2000). "LOF: Identifying
//! Density-based Local Outliers", ACM SIGMOD.
//!
//! 1. Find k-nearest neighbors for each point
//! 2. Compute reachability distance: reach-dist_k(p, o) = max(k-distance(o), d(p, o))
//! 3. Local reachability density: LRD_k(p) = |N_k(p)| / Σ reach-dist_k(p, o)
//! 4. LOF_k(p) = mean(LRD of neighbors) / LRD(p)
//!
//! Scores:
//! - LOF ≈ 1.0: normal (similar density to neighbors)
//! - LOF > 1.0: less dense than neighbors (potential outlier)
//! - LOF >> 1.0: strong outlier
//!
//! # Example
//!
//! ```
//! use u_insight::lof::{lof, LofConfig};
//!
//! // Dense cluster + one far outlier
//! let mut data: Vec<Vec<f64>> = Vec::new();
//! for i in 0..30 {
//!     data.push(vec![i as f64 * 0.1, 0.0]); // cluster near origin
//! }
//! data.push(vec![100.0, 100.0]); // outlier far away
//!
//! let config = LofConfig::default().k(5);
//! let result = lof(&data, &config).unwrap();
//! // The outlier (last point) should have the highest LOF
//! let max_idx = result.scores.iter()
//!     .enumerate()
//!     .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
//!     .unwrap().0;
//! assert_eq!(max_idx, 30);
//! assert!(result.scores[30] > 1.5);
//! ```

use crate::error::{Bound, InsightError};

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for Local Outlier Factor.
#[derive(Debug, Clone)]
pub struct LofConfig {
    /// Number of nearest neighbors (k). Default: 20.
    /// Also known as MinPts in the original paper.
    pub k: usize,
    /// LOF threshold for outlier classification, a finite number `> 0`.
    /// Default: 1.5. Points with LOF > threshold are classified as outliers.
    pub threshold: f64,
    /// Expected outlier fraction, in `[0, 1]`. When set it replaces
    /// `threshold`: the `ceil(n · contamination)` highest scores are outliers
    /// (ties included) and [`LofResult::threshold`] is the lowest of them,
    /// the way `isolation_forest` cuts its scores. Default: not set.
    pub contamination: Option<f64>,
}

impl Default for LofConfig {
    fn default() -> Self {
        Self {
            k: 20,
            threshold: 1.5,
            contamination: None,
        }
    }
}

impl LofConfig {
    /// Sets the number of neighbors, in `1..=n-1` for `n` points (refused otherwise).
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the outlier threshold.
    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = t;
        self
    }

    /// Sets the expected outlier fraction, which replaces the threshold.
    pub fn contamination(mut self, c: f64) -> Self {
        self.contamination = Some(c);
        self
    }
}

// ── Result ────────────────────────────────────────────────────────────

/// Result of LOF anomaly detection.
#[derive(Debug, Clone)]
pub struct LofResult {
    /// LOF score for each data point. Higher = more anomalous.
    pub scores: Vec<f64>,
    /// Binary anomaly labels: true = anomaly (LOF > threshold).
    pub anomalies: Vec<bool>,
    /// Threshold used for classification: the configured one, or with
    /// `contamination` the lowest score flagged (infinite when none is).
    pub threshold: f64,
    /// Number of anomalies detected.
    pub anomaly_count: usize,
    /// Fraction of points classified as anomalies.
    pub anomaly_fraction: f64,
}

// ── LOF algorithm ─────────────────────────────────────────────────────

/// Runs Local Outlier Factor anomaly detection.
///
/// # Arguments
///
/// * `data` — each inner `Vec<f64>` is a point (all must have same dimensionality).
/// * `config` — LOF parameters.
///
/// # Errors
///
/// Returns `InsightError` if:
/// - `k` is 0, or `threshold` is not a finite number `> 0`
///   ([`InsightError::ParameterOutOfRange`]): a LOF score is a ratio of
///   densities, always positive, so a threshold at or below 0 flags every point
/// - `data` is empty or has fewer than `k + 1` points
/// - Points have inconsistent dimensions
/// - Points contain non-finite values
pub fn lof(data: &[Vec<f64>], config: &LofConfig) -> Result<LofResult, InsightError> {
    let n = data.len();
    InsightError::check_count("k", config.k, 1)?;
    match config.contamination {
        Some(c) => {
            InsightError::check_range("contamination", c, Bound::Closed(0.0), Bound::Closed(1.0))?
        }
        None => InsightError::check_range(
            "threshold",
            config.threshold,
            Bound::Open(0.0),
            Bound::Unbounded,
        )?,
    }
    if n == 0 {
        return Err(InsightError::empty("data"));
    }

    let dim = data[0].len();
    if dim == 0 {
        return Err(InsightError::DimensionMismatch {
            expected: 1,
            actual: 0,
            index: None,
        });
    }

    // Validate dimensions and values
    for (row_idx, point) in data.iter().enumerate() {
        if point.len() != dim {
            return Err(InsightError::DimensionMismatch {
                expected: dim,
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

    // k neighbours need k + 1 points; a k the data cannot supply is refused
    // rather than lowered.
    if config.k >= n {
        return Err(InsightError::too_few("data", config.k + 1, n));
    }
    let k = config.k;

    // Step 1: Compute pairwise distances and find k-neighbors
    // For each point: sorted list of (neighbor_index, distance)
    let mut k_distances = vec![0.0_f64; n];
    let mut k_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean_distance(&data[i], &data[j])))
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let k_dist = dists[k - 1].1;
        k_distances[i] = k_dist;

        // Include all neighbors at distance ≤ k-distance (handles ties). Two
        // distances are tied when they agree to rounding *in proportion to
        // their size*: a fixed tolerance is nothing for lengths near 1e9 and
        // more than every distance for lengths near 1e-15.
        let tied = k_dist * (1.0 + 1e-12);
        k_neighbors[i] = dists
            .iter()
            .take_while(|(_, d)| *d <= tied)
            .map(|(j, _)| *j)
            .collect();

        // Ensure at least k neighbors
        if k_neighbors[i].len() < k {
            k_neighbors[i] = dists[..k].iter().map(|(j, _)| *j).collect();
        }
    }

    // Step 2 & 3: Compute Local Reachability Density (LRD)
    let mut lrd = vec![0.0_f64; n];

    for i in 0..n {
        let neighbors = &k_neighbors[i];
        let nk = neighbors.len() as f64;

        let sum_reach_dist: f64 = neighbors
            .iter()
            .map(|&j| {
                let d = euclidean_distance(&data[i], &data[j]);
                k_distances[j].max(d) // reach-dist_k(i, j) = max(k-distance(j), d(i,j))
            })
            .sum();

        lrd[i] = if sum_reach_dist > 0.0 {
            nk / sum_reach_dist
        } else {
            f64::MAX // all neighbors at distance 0 (duplicates)
        };
    }

    // Step 4: Compute LOF scores
    let mut scores = vec![0.0_f64; n];

    for i in 0..n {
        let neighbors = &k_neighbors[i];
        let nk = neighbors.len() as f64;

        let avg_neighbor_lrd: f64 = neighbors.iter().map(|&j| lrd[j]).sum::<f64>() / nk;

        scores[i] = if lrd[i] > 0.0 && lrd[i] < f64::MAX {
            avg_neighbor_lrd / lrd[i]
        } else {
            1.0 // duplicate points → treat as inlier
        };
    }

    // Classify anomalies
    let (threshold, anomalies): (f64, Vec<bool>) = match config.contamination {
        Some(c) => {
            let mut sorted = scores.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let flagged = ((n as f64 * c).ceil() as usize).min(n);
            let cut = match flagged {
                0 => f64::INFINITY,
                f => sorted[f - 1],
            };
            (cut, scores.iter().map(|&s| s >= cut).collect())
        }
        None => (
            config.threshold,
            scores.iter().map(|&s| s > config.threshold).collect(),
        ),
    };
    let anomaly_count = anomalies.iter().filter(|&&a| a).count();
    let anomaly_fraction = anomaly_count as f64 / n as f64;

    Ok(LofResult {
        scores,
        anomalies,
        threshold,
        anomaly_count,
        anomaly_fraction,
    })
}

/// Euclidean distance between two points.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster_with_outlier() -> Vec<Vec<f64>> {
        let mut data = Vec::new();
        // Dense cluster around (0, 0)
        for i in 0..20 {
            data.push(vec![i as f64 * 0.1, i as f64 * 0.05]);
        }
        // Outlier far away
        data.push(vec![100.0, 100.0]);
        data
    }

    /// An expected outlier fraction cuts the scores the way a threshold does:
    /// the `ceil(n * c)` highest are flagged, and the result reports the cut.
    #[test]
    fn contamination_flags_the_highest_scores() {
        let data = cluster_with_outlier();
        let n = data.len();
        let result = lof(&data, &LofConfig::default().k(5).contamination(0.05)).unwrap();
        // ceil(21 * 0.05) = 2 flagged, the outlier among them.
        assert_eq!(result.anomaly_count, 2, "{:?}", result.scores);
        assert!(result.anomalies[n - 1]);
        let lowest_flagged = result
            .scores
            .iter()
            .zip(&result.anomalies)
            .filter(|&(_, &a)| a)
            .map(|(&s, _)| s)
            .fold(f64::INFINITY, f64::min);
        assert_eq!(result.threshold, lowest_flagged);
    }

    #[test]
    fn contamination_zero_flags_nothing_and_one_flags_all() {
        let data = cluster_with_outlier();
        let none = lof(&data, &LofConfig::default().k(5).contamination(0.0)).unwrap();
        assert_eq!(none.anomaly_count, 0);
        assert!(none.threshold.is_infinite());
        let all = lof(&data, &LofConfig::default().k(5).contamination(1.0)).unwrap();
        assert_eq!(all.anomaly_count, data.len());
    }

    #[test]
    fn contamination_outside_the_unit_interval_is_refused() {
        let data = cluster_with_outlier();
        for c in [-0.1, 1.1, f64::NAN] {
            let err = lof(&data, &LofConfig::default().k(5).contamination(c)).unwrap_err();
            assert!(
                matches!(&err, InsightError::ParameterOutOfRange { name, .. } if name == "contamination"),
                "{c}: {err:?}"
            );
        }
        // With a contamination the threshold is not read: a bad one is not refused.
        assert!(lof(
            &data,
            &LofConfig::default().k(5).threshold(-1.0).contamination(0.1)
        )
        .is_ok());
    }

    #[test]
    fn lof_detects_single_outlier() {
        let data = cluster_with_outlier();
        let config = LofConfig::default().k(5);
        let result = lof(&data, &config).expect("should compute");

        // Outlier should have highest LOF
        let max_idx = result
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 20, "outlier at index 20 should have max LOF");
        assert!(result.scores[20] > 2.0, "outlier LOF should be high");
    }

    #[test]
    fn lof_inliers_near_one() {
        let data = cluster_with_outlier();
        let config = LofConfig::default().k(5);
        let result = lof(&data, &config).expect("should compute");

        // Points deep in cluster should have LOF close to 1
        for i in 3..17 {
            assert!(
                result.scores[i] < 1.5,
                "inlier {} has LOF = {}, expected < 1.5",
                i,
                result.scores[i]
            );
        }
    }

    #[test]
    fn lof_anomaly_classification() {
        let data = cluster_with_outlier();
        let config = LofConfig::default().k(5).threshold(2.0);
        let result = lof(&data, &config).expect("should compute");

        assert!(
            result.anomalies[20],
            "outlier should be classified as anomaly"
        );
        assert!(result.anomaly_count >= 1, "at least one anomaly");
        assert!(result.anomaly_fraction > 0.0);
    }

    #[test]
    fn lof_two_clusters() {
        let mut data = Vec::new();
        // Cluster 1: around (0, 0)
        for i in 0..15 {
            data.push(vec![i as f64 * 0.1, 0.0]);
        }
        // Cluster 2: around (10, 10)
        for i in 0..15 {
            data.push(vec![10.0 + i as f64 * 0.1, 10.0]);
        }
        // Outlier between clusters
        data.push(vec![5.0, 5.0]);

        let config = LofConfig::default().k(5);
        let result = lof(&data, &config).expect("should compute");

        // Outlier (index 30) should have highest LOF
        let max_idx = result
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 30, "between-cluster point should have max LOF");
    }

    #[test]
    fn lof_uniform_density() {
        // Uniformly spaced points → LOF ≈ 1 for interior points
        let data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let config = LofConfig::default().k(3);
        let result = lof(&data, &config).expect("should compute");

        // Interior points should have LOF close to 1
        for i in 3..17 {
            assert!(
                (result.scores[i] - 1.0).abs() < 0.5,
                "uniform point {} has LOF = {}, expected ~1.0",
                i,
                result.scores[i]
            );
        }
    }

    #[test]
    fn a_k_the_data_cannot_supply_is_refused() {
        let data = vec![vec![0.0], vec![1.0], vec![2.0]];
        // No neighbours is an option out of range; more than the data holds
        // is too few points for that k.
        assert!(matches!(
            lof(&data, &LofConfig::default().k(0)).unwrap_err(),
            InsightError::ParameterOutOfRange { ref name, .. } if name == "k"
        ));
        for k in [3, 20] {
            assert_eq!(
                lof(&data, &LofConfig::default().k(k)).unwrap_err(),
                InsightError::too_few("data", k + 1, 3)
            );
        }
        // One point and one neighbour: two points needed, one given.
        assert_eq!(
            lof(&[vec![1.0]], &LofConfig::default().k(1)).unwrap_err(),
            InsightError::too_few("data", 2, 1)
        );
        assert_eq!(
            lof(&[], &LofConfig::default()).unwrap_err(),
            InsightError::empty("data")
        );
        assert_eq!(
            lof(&data, &LofConfig::default().k(2))
                .expect("k = n - 1")
                .scores
                .len(),
            3
        );
    }

    #[test]
    fn lof_error_empty() {
        let data: Vec<Vec<f64>> = vec![];
        assert!(lof(&data, &LofConfig::default()).is_err());
    }

    #[test]
    fn lof_error_dimension_mismatch() {
        let data = vec![vec![1.0, 2.0], vec![3.0]]; // different dims
        assert!(lof(&data, &LofConfig::default()).is_err());
    }

    #[test]
    fn lof_error_nan() {
        let data = vec![vec![1.0, f64::NAN], vec![3.0, 4.0]];
        let err = lof(&data, &LofConfig::default()).unwrap_err();
        // Refused at the offending cell, as the bindings report it (not Io, not degenerate).
        assert_eq!(
            err,
            InsightError::ValueNotFinite {
                column: "data[0]".into(),
                index: 1
            }
        );
    }

    #[test]
    fn lof_multidimensional() {
        let mut data = Vec::new();
        // Dense cluster in 3D
        for i in 0..15 {
            data.push(vec![i as f64 * 0.1, i as f64 * 0.1, i as f64 * 0.1]);
        }
        // Outlier in 3D
        data.push(vec![50.0, 50.0, 50.0]);

        let config = LofConfig::default().k(5);
        let result = lof(&data, &config).expect("should compute");
        assert!(result.scores[15] > 1.5, "3D outlier should be detected");
    }
}
