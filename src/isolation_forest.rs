//! Isolation Forest for multivariate anomaly detection.
//!
//! Detects anomalies by measuring how easily points can be isolated
//! via random recursive partitioning. Anomalies require fewer splits
//! to isolate, yielding shorter path lengths and higher anomaly scores.
//!
//! # Algorithm
//!
//! Reference: Liu, Ting & Zhou (2008). "Isolation Forest", ICDM.
//!
//! 1. Build an ensemble of isolation trees, each on a random subsample
//! 2. For each point, compute average path length across all trees
//! 3. Normalize using c(n) = 2H(n-1) - 2(n-1)/n (expected BST search depth)
//! 4. Anomaly score: s(x,n) = 2^(-E(h(x))/c(n))
//!
//! Scores range from 0 to 1:
//! - Close to 1: anomaly (short paths)
//! - Close to 0.5: normal (average path length)
//! - Close to 0: very normal (long paths)
//!
//! # Example
//!
//! ```
//! use u_insight::isolation_forest::{isolation_forest, IsolationForestConfig};
//!
//! // Normal cluster + one outlier
//! let mut data = Vec::new();
//! for i in 0..50 {
//!     data.push(vec![i as f64 * 0.1, i as f64 * 0.1]);
//! }
//! data.push(vec![100.0, 100.0]); // outlier
//!
//! let result = isolation_forest(&data, &IsolationForestConfig::default()).unwrap();
//! // The outlier (last point) should have the highest anomaly score
//! let max_idx = result.scores.iter()
//!     .enumerate()
//!     .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
//!     .unwrap().0;
//! assert_eq!(max_idx, 50);
//! ```

use crate::error::InsightError;

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for Isolation Forest.
#[derive(Debug, Clone)]
pub struct IsolationForestConfig {
    /// Number of isolation trees. Default: 100.
    pub n_estimators: usize,
    /// Subsample size per tree. Default: min(256, n).
    /// Set to 0 for automatic (min(256, n)).
    pub max_samples: usize,
    /// Expected proportion of anomalies (0.0–1.0).
    /// Used to determine the anomaly threshold.
    /// Default: 0.1.
    pub contamination: f64,
    /// Random seed. Default: Some(42).
    pub seed: Option<u64>,
}

impl Default for IsolationForestConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_samples: 0, // auto
            contamination: 0.1,
            seed: Some(42),
        }
    }
}

impl IsolationForestConfig {
    /// Sets the number of trees.
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Sets the contamination rate.
    pub fn contamination(mut self, c: f64) -> Self {
        self.contamination = c;
        self
    }

    /// Sets the random seed.
    pub fn seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the max_samples per tree.
    pub fn max_samples(mut self, m: usize) -> Self {
        self.max_samples = m;
        self
    }
}

// ── Result ────────────────────────────────────────────────────────────

/// Result of Isolation Forest anomaly detection.
#[derive(Debug, Clone)]
pub struct IsolationForestResult {
    /// Anomaly score for each data point (0.0–1.0). Higher = more anomalous.
    pub scores: Vec<f64>,
    /// Binary anomaly labels: true = anomaly.
    pub anomalies: Vec<bool>,
    /// Score threshold used for classification.
    pub threshold: f64,
    /// Number of anomalies detected.
    pub anomaly_count: usize,
    /// Fraction of points classified as anomalies.
    pub anomaly_fraction: f64,
}

// ── Isolation Forest algorithm ────────────────────────────────────────

/// Runs Isolation Forest anomaly detection.
///
/// ```
/// use u_insight::isolation_forest::{isolation_forest, IsolationForestConfig};
///
/// let data = vec![
///     vec![1.0, 1.0], vec![1.1, 1.1], vec![0.9, 0.9],
///     vec![1.0, 1.2], vec![1.2, 1.0],
///     vec![50.0, 50.0], // outlier
/// ];
/// let config = IsolationForestConfig::default().contamination(0.2);
/// let result = isolation_forest(&data, &config).unwrap();
///
/// // The outlier should be detected
/// assert!(result.anomalies[5]);
/// assert!(result.scores[5] > 0.5);
/// ```
pub fn isolation_forest(
    data: &[Vec<f64>],
    config: &IsolationForestConfig,
) -> Result<IsolationForestResult, InsightError> {
    let n = data.len();
    if n < 2 {
        return Err(InsightError::InsufficientData {
            min_required: 2,
            actual: n,
        });
    }

    let d = data[0].len();
    if d == 0 {
        return Err(InsightError::DegenerateData {
            reason: "data has 0 features".into(),
        });
    }

    // Validate
    for (i, point) in data.iter().enumerate() {
        if point.len() != d {
            return Err(InsightError::DimensionMismatch {
                expected: d,
                actual: point.len(),
            });
        }
        for &v in point {
            if v.is_nan() || v.is_infinite() {
                return Err(InsightError::NonNumericColumn {
                    column: format!("point[{i}]"),
                });
            }
        }
    }

    if config.n_estimators == 0 {
        return Err(InsightError::InvalidParameter {
            name: "n_estimators".into(),
            message: "must be at least 1".into(),
        });
    }
    if !(0.0..=1.0).contains(&config.contamination) {
        return Err(InsightError::InvalidParameter {
            name: "contamination".into(),
            message: format!("must be in [0.0, 1.0], got {}", config.contamination),
        });
    }

    let max_samples = if config.max_samples == 0 {
        n.min(256)
    } else {
        config.max_samples.min(n)
    };
    let max_depth = (max_samples as f64).log2().ceil() as usize;

    // Build forest
    let mut rng_state = config.seed.unwrap_or(12345);
    let mut trees = Vec::with_capacity(config.n_estimators);

    for _ in 0..config.n_estimators {
        // Subsample indices
        let indices = sample_indices(n, max_samples, &mut rng_state);
        let subsample: Vec<&[f64]> = indices.iter().map(|&i| data[i].as_slice()).collect();
        let tree = build_itree(&subsample, d, max_depth, &mut rng_state);
        trees.push(tree);
    }

    // Score each point
    let cn = c_factor(max_samples);
    let mut scores = Vec::with_capacity(n);

    for point in data {
        let avg_path: f64 = trees
            .iter()
            .map(|tree| path_length(point, tree, 0))
            .sum::<f64>()
            / config.n_estimators as f64;

        let score = if cn > 0.0 {
            2.0f64.powf(-avg_path / cn)
        } else {
            0.5 // degenerate case
        };
        scores.push(score);
    }

    // Determine threshold from contamination
    let mut sorted_scores: Vec<f64> = scores.clone();
    sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold_idx = ((n as f64 * config.contamination).ceil() as usize)
        .min(n)
        .max(1)
        - 1;
    let threshold = sorted_scores[threshold_idx];

    let anomalies: Vec<bool> = scores.iter().map(|&s| s >= threshold).collect();
    let anomaly_count = anomalies.iter().filter(|&&a| a).count();

    Ok(IsolationForestResult {
        scores,
        anomalies,
        threshold,
        anomaly_count,
        anomaly_fraction: anomaly_count as f64 / n as f64,
    })
}

// ── Isolation Tree internals ──────────────────────────────────────────

/// Node in an isolation tree.
enum ITreeNode {
    /// Internal split node.
    Internal {
        feature: usize,
        split_value: f64,
        left: Box<ITreeNode>,
        right: Box<ITreeNode>,
    },
    /// External (leaf) node.
    External { size: usize },
}

/// Builds a single isolation tree from a subsample.
fn build_itree(data: &[&[f64]], d: usize, max_depth: usize, rng: &mut u64) -> ITreeNode {
    let n = data.len();

    // Termination conditions
    if n <= 1 || max_depth == 0 {
        return ITreeNode::External { size: n };
    }

    // Random feature
    let feature = lcg_next_usize(rng, d);

    // Find min/max for this feature
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for point in data {
        let v = point[feature];
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    // All values identical for this feature
    if (max_val - min_val).abs() < 1e-15 {
        return ITreeNode::External { size: n };
    }

    // Random split point between min and max
    let split_value = min_val + lcg_next_f64(rng) * (max_val - min_val);

    // Partition
    let mut left_data = Vec::new();
    let mut right_data = Vec::new();
    for &point in data {
        if point[feature] < split_value {
            left_data.push(point);
        } else {
            right_data.push(point);
        }
    }

    // Avoid degenerate splits where all data goes to one side
    if left_data.is_empty() || right_data.is_empty() {
        return ITreeNode::External { size: n };
    }

    ITreeNode::Internal {
        feature,
        split_value,
        left: Box::new(build_itree(&left_data, d, max_depth - 1, rng)),
        right: Box::new(build_itree(&right_data, d, max_depth - 1, rng)),
    }
}

/// Computes the path length for a point traversing the tree.
fn path_length(point: &[f64], node: &ITreeNode, current_depth: usize) -> f64 {
    match node {
        ITreeNode::External { size } => current_depth as f64 + c_factor(*size),
        ITreeNode::Internal {
            feature,
            split_value,
            left,
            right,
        } => {
            if point[*feature] < *split_value {
                path_length(point, left, current_depth + 1)
            } else {
                path_length(point, right, current_depth + 1)
            }
        }
    }
}

/// Average path length of unsuccessful search in BST of size n.
///
/// c(n) = 2*H(n-1) - 2*(n-1)/n
/// where H(i) ≈ ln(i) + γ (Euler-Mascheroni constant).
fn c_factor(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    if n == 2 {
        return 1.0;
    }
    let n_f = n as f64;
    let harmonic = (n_f - 1.0).ln() + 0.5772156649;
    2.0 * harmonic - 2.0 * (n_f - 1.0) / n_f
}

// ── RNG helpers ───────────────────────────────────────────────────────

/// LCG random: returns [0, 1).
fn lcg_next_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

/// LCG random: returns [0, max).
fn lcg_next_usize(state: &mut u64, max: usize) -> usize {
    (lcg_next_f64(state) * max as f64) as usize % max
}

/// Sample `k` unique indices from `0..n` using Fisher-Yates partial shuffle.
fn sample_indices(n: usize, k: usize, rng: &mut u64) -> Vec<usize> {
    let k = k.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + lcg_next_usize(rng, n - i);
        indices.swap(i, j);
    }
    indices.truncate(k);
    indices
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_normal_with_outliers() -> Vec<Vec<f64>> {
        let mut data = Vec::new();
        // Normal cluster around (5, 5)
        for i in 0..40 {
            let x = 5.0 + (i % 7) as f64 * 0.2 - 0.6;
            let y = 5.0 + (i % 5) as f64 * 0.3 - 0.6;
            data.push(vec![x, y]);
        }
        // Outliers far from cluster
        data.push(vec![50.0, 50.0]);
        data.push(vec![-40.0, -40.0]);
        data.push(vec![50.0, -40.0]);
        data
    }

    // ── Basic detection ───────────────────────────────────────────

    #[test]
    fn detects_obvious_outliers() {
        let data = make_normal_with_outliers();
        let result =
            isolation_forest(&data, &IsolationForestConfig::default().contamination(0.1)).unwrap();

        // The last 3 points are outliers — they should have high scores
        let n = data.len();
        for i in (n - 3)..n {
            assert!(
                result.scores[i] > 0.5,
                "outlier point {i} score = {} should be > 0.5",
                result.scores[i]
            );
        }
    }

    #[test]
    fn outlier_scores_higher_than_normal() {
        let data = make_normal_with_outliers();
        let result = isolation_forest(&data, &IsolationForestConfig::default()).unwrap();

        let normal_max = data[..40]
            .iter()
            .enumerate()
            .map(|(i, _)| result.scores[i])
            .fold(0.0f64, f64::max);

        let outlier_min = data[40..]
            .iter()
            .enumerate()
            .map(|(i, _)| result.scores[40 + i])
            .fold(1.0f64, f64::min);

        assert!(
            outlier_min > normal_max,
            "outlier min score ({outlier_min}) should exceed normal max ({normal_max})"
        );
    }

    #[test]
    fn anomaly_count_matches_contamination() {
        let data = make_normal_with_outliers();
        let n = data.len();
        let contamination = 0.1;
        let result = isolation_forest(
            &data,
            &IsolationForestConfig::default().contamination(contamination),
        )
        .unwrap();

        let expected_max = (n as f64 * contamination).ceil() as usize + 1;
        assert!(
            result.anomaly_count <= expected_max,
            "anomaly_count {} should be <= {}",
            result.anomaly_count,
            expected_max
        );
    }

    // ── Score properties ──────────────────────────────────────────

    #[test]
    fn scores_in_range() {
        let data = make_normal_with_outliers();
        let result = isolation_forest(&data, &IsolationForestConfig::default()).unwrap();

        for (i, &score) in result.scores.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&score),
                "score[{i}] = {score} out of [0, 1] range"
            );
        }
    }

    #[test]
    fn scores_reproducible_with_seed() {
        let data = make_normal_with_outliers();
        let config = IsolationForestConfig::default().seed(Some(123));
        let r1 = isolation_forest(&data, &config).unwrap();
        let r2 = isolation_forest(&data, &config).unwrap();

        for (i, (&s1, &s2)) in r1.scores.iter().zip(r2.scores.iter()).enumerate() {
            assert!((s1 - s2).abs() < 1e-10, "score[{i}] differs: {s1} vs {s2}");
        }
    }

    // ── No outliers ───────────────────────────────────────────────

    #[test]
    fn uniform_data_low_scores() {
        // All points are similar — scores should be near 0.5
        let data: Vec<Vec<f64>> = (0..50)
            .map(|i| vec![i as f64 * 0.1, i as f64 * 0.1])
            .collect();
        let result = isolation_forest(&data, &IsolationForestConfig::default()).unwrap();

        let mean_score: f64 = result.scores.iter().sum::<f64>() / result.scores.len() as f64;
        assert!(
            mean_score < 0.7,
            "mean score for uniform data should be moderate: {mean_score}"
        );
    }

    // ── Error cases ──────────────────────────────────────────────

    #[test]
    fn empty_data() {
        let data: Vec<Vec<f64>> = vec![];
        assert!(isolation_forest(&data, &IsolationForestConfig::default()).is_err());
    }

    #[test]
    fn single_point() {
        let data = vec![vec![1.0, 2.0]];
        assert!(isolation_forest(&data, &IsolationForestConfig::default()).is_err());
    }

    #[test]
    fn nan_rejected() {
        let data = vec![vec![1.0, f64::NAN], vec![2.0, 3.0]];
        assert!(isolation_forest(&data, &IsolationForestConfig::default()).is_err());
    }

    #[test]
    fn dimension_mismatch() {
        let data = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(isolation_forest(&data, &IsolationForestConfig::default()).is_err());
    }

    // ── 1D data ──────────────────────────────────────────────────

    #[test]
    fn one_dimensional_outlier() {
        let mut data: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 * 0.1]).collect();
        data.push(vec![100.0]); // outlier

        let result = isolation_forest(&data, &IsolationForestConfig::default()).unwrap();

        let outlier_score = result.scores[30];
        let normal_mean: f64 = result.scores[..30].iter().sum::<f64>() / 30.0;

        assert!(
            outlier_score > normal_mean,
            "outlier score ({outlier_score}) should exceed normal mean ({normal_mean})"
        );
    }

    // ── High-dimensional ──────────────────────────────────────────

    #[test]
    fn high_dim_detection() {
        // 10D: normal cluster at origin, one outlier at (100, 100, ...)
        let mut data = Vec::new();
        for i in 0..30 {
            let mut point = vec![0.0; 10];
            point[i % 10] = (i % 5) as f64 * 0.2;
            data.push(point);
        }
        data.push(vec![100.0; 10]); // outlier

        let result =
            isolation_forest(&data, &IsolationForestConfig::default().n_estimators(200)).unwrap();

        // Outlier should have the highest score
        let max_idx = result
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0;
        assert_eq!(max_idx, 30, "outlier should have highest anomaly score");
    }

    // ── c_factor ──────────────────────────────────────────────────

    #[test]
    fn c_factor_known_values() {
        assert_eq!(c_factor(1), 0.0);
        assert_eq!(c_factor(2), 1.0);
        // c(256) ≈ 2*(ln(255) + 0.5772) - 2*255/256 ≈ 2*6.118 - 1.992 ≈ 10.244
        let c256 = c_factor(256);
        assert!(
            (c256 - 10.244).abs() < 0.1,
            "c(256) = {c256}, expected ~10.244"
        );
    }

    // ── Config ────────────────────────────────────────────────────

    #[test]
    fn different_n_estimators() {
        let data = make_normal_with_outliers();
        // Fewer trees should still detect obvious outliers
        let result =
            isolation_forest(&data, &IsolationForestConfig::default().n_estimators(10)).unwrap();

        // Outlier at index 40 should still be detectable
        let outlier_score = result.scores[40];
        assert!(
            outlier_score > 0.5,
            "even with 10 trees, outlier score = {outlier_score}"
        );
    }
}
