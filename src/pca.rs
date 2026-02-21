//! Principal Component Analysis (PCA).
//!
//! Reduces dimensionality by projecting data onto directions of maximum variance.
//! Uses eigenvalue decomposition of the covariance matrix via the Jacobi algorithm
//! (from `u-numflow`).
//!
//! Supports optional auto-scaling (z-score standardization) per AD-7: when enabled,
//! PCA operates on the correlation matrix instead of the covariance matrix,
//! giving equal weight to all features regardless of scale.
//!
//! # Example
//!
//! ```
//! use u_insight::pca::{pca, PcaConfig};
//!
//! // 6 points in 3D with most variance along first axis
//! let data = vec![
//!     vec![1.0, 0.1, 0.01],
//!     vec![2.0, 0.2, 0.02],
//!     vec![3.0, 0.3, 0.03],
//!     vec![4.0, 0.4, 0.04],
//!     vec![5.0, 0.5, 0.05],
//!     vec![6.0, 0.6, 0.06],
//! ];
//! let result = pca(&data, &PcaConfig::new(2)).unwrap();
//!
//! assert_eq!(result.n_components, 2);
//! assert!(result.explained_variance_ratio[0] > 0.99);
//! assert_eq!(result.scores.len(), 6); // one score vector per point
//! ```

use crate::error::InsightError;
use u_numflow::matrix::Matrix;

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for PCA.
#[derive(Debug, Clone)]
pub struct PcaConfig {
    /// Number of principal components to retain.
    pub n_components: usize,
    /// If true, z-score standardize each feature before PCA (AD-7).
    /// Default: false.
    pub auto_scale: bool,
}

impl PcaConfig {
    /// Creates a PCA config retaining `n_components` principal components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            auto_scale: false,
        }
    }

    /// Enables auto-scaling (z-score standardization).
    pub fn auto_scale(mut self, auto_scale: bool) -> Self {
        self.auto_scale = auto_scale;
        self
    }
}

// ── Result ────────────────────────────────────────────────────────────

/// Result of PCA.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Number of components retained.
    pub n_components: usize,
    /// Eigenvalues (variances) for each retained component, descending.
    pub eigenvalues: Vec<f64>,
    /// Fraction of total variance explained by each component.
    pub explained_variance_ratio: Vec<f64>,
    /// Cumulative explained variance ratio.
    pub cumulative_variance_ratio: Vec<f64>,
    /// Component loadings: n_components rows × d columns.
    /// Row i is the eigenvector (loading) for component i.
    pub loadings: Vec<Vec<f64>>,
    /// Projected scores: one `Vec<f64>` per data point (n × n_components).
    pub scores: Vec<Vec<f64>>,
    /// Feature means used for centering (length d).
    pub means: Vec<f64>,
    /// Feature standard deviations (length d). All 1.0 if auto_scale=false.
    pub stds: Vec<f64>,
    /// Total number of features (original dimensionality).
    pub n_features: usize,
}

// ── PCA algorithm ─────────────────────────────────────────────────────

/// Runs Principal Component Analysis on the data.
///
/// Input: `data` is a list of n points, each a `Vec<f64>` of dimension d.
///
/// ```
/// use u_insight::pca::{pca, PcaConfig};
///
/// let data = vec![
///     vec![1.0, 0.0], vec![2.0, 0.0],
///     vec![3.0, 0.0], vec![4.0, 0.0],
/// ];
/// let result = pca(&data, &PcaConfig::new(1)).unwrap();
/// assert_eq!(result.n_components, 1);
/// // First component explains all variance (second dim is constant)
/// assert!((result.explained_variance_ratio[0] - 1.0).abs() < 1e-10);
/// ```
pub fn pca(data: &[Vec<f64>], config: &PcaConfig) -> Result<PcaResult, InsightError> {
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
    if config.n_components == 0 || config.n_components > d {
        return Err(InsightError::InvalidParameter {
            name: "n_components".into(),
            message: format!(
                "must be between 1 and {d} (number of features), got {}",
                config.n_components
            ),
        });
    }

    // Validate dimensions and data
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

    // Step 1: Compute means
    let mut means = vec![0.0; d];
    for point in data {
        for (j, &v) in point.iter().enumerate() {
            means[j] += v;
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }

    // Step 2: Compute standard deviations (for auto-scaling)
    let stds = if config.auto_scale {
        let mut vars = vec![0.0; d];
        for point in data {
            for (j, &v) in point.iter().enumerate() {
                let diff = v - means[j];
                vars[j] += diff * diff;
            }
        }
        vars.iter()
            .map(|&v| {
                let s = (v / (n - 1) as f64).sqrt();
                if s < 1e-15 {
                    1.0
                } else {
                    s
                } // avoid division by zero for constant features
            })
            .collect::<Vec<f64>>()
    } else {
        vec![1.0; d]
    };

    // Step 3: Build centered (and optionally scaled) data
    let mut centered = vec![vec![0.0; d]; n];
    for (i, point) in data.iter().enumerate() {
        for j in 0..d {
            centered[i][j] = (point[j] - means[j]) / stds[j];
        }
    }

    // Step 4: Compute covariance matrix (d × d)
    let mut cov_data = vec![0.0; d * d];
    for point in &centered {
        for i in 0..d {
            for j in i..d {
                let v = point[i] * point[j];
                cov_data[i * d + j] += v;
                if i != j {
                    cov_data[j * d + i] += v;
                }
            }
        }
    }
    let scale = 1.0 / (n - 1) as f64;
    for v in &mut cov_data {
        *v *= scale;
    }

    let cov_matrix = Matrix::new(d, d, cov_data).map_err(|e| InsightError::ComputationFailed {
        operation: "covariance matrix construction".into(),
        detail: e.to_string(),
    })?;

    // Step 5: Eigenvalue decomposition
    let (eigenvalues, eigenvectors) =
        cov_matrix
            .eigen_symmetric()
            .map_err(|e| InsightError::ComputationFailed {
                operation: "eigenvalue decomposition".into(),
                detail: e.to_string(),
            })?;

    // Step 6: Select top-k components
    let k = config.n_components;
    let total_variance: f64 = eigenvalues.iter().sum();

    let retained_eigenvalues: Vec<f64> = eigenvalues[..k].to_vec();
    let explained_variance_ratio: Vec<f64> = if total_variance > 1e-15 {
        retained_eigenvalues
            .iter()
            .map(|&ev| ev / total_variance)
            .collect()
    } else {
        vec![0.0; k]
    };

    let mut cumulative_variance_ratio = Vec::with_capacity(k);
    let mut cum = 0.0;
    for &r in &explained_variance_ratio {
        cum += r;
        cumulative_variance_ratio.push(cum);
    }

    // Extract loadings (eigenvectors as rows): component i → row i
    let mut loadings = Vec::with_capacity(k);
    for comp in 0..k {
        let mut loading = Vec::with_capacity(d);
        for feat in 0..d {
            loading.push(eigenvectors.get(feat, comp));
        }
        loadings.push(loading);
    }

    // Step 7: Project data onto components
    let mut scores = Vec::with_capacity(n);
    for point in &centered {
        let mut score = Vec::with_capacity(k);
        for comp_loading in &loadings {
            let s: f64 = point
                .iter()
                .zip(comp_loading.iter())
                .map(|(&x, &w)| x * w)
                .sum();
            score.push(s);
        }
        scores.push(score);
    }

    Ok(PcaResult {
        n_components: k,
        eigenvalues: retained_eigenvalues,
        explained_variance_ratio,
        cumulative_variance_ratio,
        loadings,
        scores,
        means,
        stds,
        n_features: d,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_3d() -> Vec<Vec<f64>> {
        // Strong linear pattern along first axis
        vec![
            vec![1.0, 0.1, 0.01],
            vec![2.0, 0.2, 0.02],
            vec![3.0, 0.3, 0.03],
            vec![4.0, 0.4, 0.04],
            vec![5.0, 0.5, 0.05],
            vec![6.0, 0.6, 0.06],
        ]
    }

    fn make_two_dim() -> Vec<Vec<f64>> {
        // Variance split across two dimensions
        vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
            vec![5.0, 50.0],
        ]
    }

    // ── Basic PCA ─────────────────────────────────────────────────

    #[test]
    fn pca_basic_3d_to_2d() {
        let data = make_linear_3d();
        let result = pca(&data, &PcaConfig::new(2)).unwrap();

        assert_eq!(result.n_components, 2);
        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.explained_variance_ratio.len(), 2);
        assert_eq!(result.loadings.len(), 2);
        assert_eq!(result.scores.len(), 6);
        assert_eq!(result.scores[0].len(), 2);
    }

    #[test]
    fn pca_first_component_dominant() {
        let data = make_linear_3d();
        let result = pca(&data, &PcaConfig::new(1)).unwrap();

        // In this perfectly correlated data, PC1 captures ~100% of variance
        assert!(
            result.explained_variance_ratio[0] > 0.99,
            "PC1 should explain >99% of variance, got {}",
            result.explained_variance_ratio[0]
        );
    }

    #[test]
    fn pca_cumulative_variance() {
        let data = make_linear_3d();
        let result = pca(&data, &PcaConfig::new(3)).unwrap();

        // Cumulative should sum to ~1.0
        let last = *result.cumulative_variance_ratio.last().expect("non-empty");
        assert!(
            (last - 1.0).abs() < 1e-10,
            "cumulative variance should sum to 1.0, got {last}"
        );

        // Cumulative should be monotonically increasing
        for i in 1..result.cumulative_variance_ratio.len() {
            assert!(result.cumulative_variance_ratio[i] >= result.cumulative_variance_ratio[i - 1]);
        }
    }

    #[test]
    fn pca_eigenvalues_descending() {
        let data = make_linear_3d();
        let result = pca(&data, &PcaConfig::new(3)).unwrap();

        for i in 1..result.eigenvalues.len() {
            assert!(
                result.eigenvalues[i] <= result.eigenvalues[i - 1] + 1e-10,
                "eigenvalues should be descending"
            );
        }
    }

    #[test]
    fn pca_loadings_orthonormal() {
        let data = make_linear_3d();
        let result = pca(&data, &PcaConfig::new(3)).unwrap();

        // Each loading should be unit length
        for (i, loading) in result.loadings.iter().enumerate() {
            let norm: f64 = loading.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "loading {i} norm = {norm}, expected 1.0"
            );
        }

        // Loadings should be orthogonal
        for i in 0..result.loadings.len() {
            for j in (i + 1)..result.loadings.len() {
                let dot: f64 = result.loadings[i]
                    .iter()
                    .zip(result.loadings[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                assert!(
                    dot.abs() < 1e-10,
                    "loadings {i} and {j} not orthogonal: dot = {dot}"
                );
            }
        }
    }

    #[test]
    fn pca_scores_dimensions() {
        let data = make_two_dim();
        let result = pca(&data, &PcaConfig::new(2)).unwrap();

        assert_eq!(result.scores.len(), 5);
        for score in &result.scores {
            assert_eq!(score.len(), 2);
        }
    }

    #[test]
    fn pca_scores_centered() {
        let data = make_two_dim();
        let result = pca(&data, &PcaConfig::new(2)).unwrap();

        // Scores should be zero-mean (centered)
        let n = result.scores.len();
        for comp in 0..result.n_components {
            let mean: f64 = result.scores.iter().map(|s| s[comp]).sum::<f64>() / n as f64;
            assert!(
                mean.abs() < 1e-10,
                "scores for component {comp} not centered: mean = {mean}"
            );
        }
    }

    // ── Auto-scaling ──────────────────────────────────────────────

    #[test]
    fn pca_auto_scale_effect() {
        // Different scales: feature 1 has 100× the range of feature 2
        let data = vec![
            vec![100.0, 1.0],
            vec![200.0, 2.0],
            vec![300.0, 3.0],
            vec![400.0, 4.0],
            vec![500.0, 5.0],
        ];

        // Without scaling: first feature dominates
        let r_no_scale = pca(&data, &PcaConfig::new(2)).unwrap();
        // With scaling: both features contribute equally
        let r_scaled = pca(&data, &PcaConfig::new(2).auto_scale(true)).unwrap();

        // Without scaling, PC1 explains nearly all variance
        assert!(r_no_scale.explained_variance_ratio[0] > 0.99);

        // With scaling on perfectly correlated data, PC1 still dominates
        // but the stds should differ
        assert!((r_scaled.stds[0] - r_scaled.stds[1]).abs() > 1.0);
    }

    #[test]
    fn pca_auto_scale_stds_stored() {
        let data = make_two_dim();
        let result = pca(&data, &PcaConfig::new(2).auto_scale(true)).unwrap();

        // Stds should not be 1.0 (they're the actual feature stds)
        assert!(result.stds[0] > 0.0);
        assert!(result.stds[1] > 0.0);
    }

    #[test]
    fn pca_no_scale_stds_ones() {
        let data = make_two_dim();
        let result = pca(&data, &PcaConfig::new(2)).unwrap();

        for &s in &result.stds {
            assert!((s - 1.0).abs() < 1e-15);
        }
    }

    // ── Error cases ──────────────────────────────────────────────

    #[test]
    fn pca_empty_data() {
        let data: Vec<Vec<f64>> = vec![];
        assert!(pca(&data, &PcaConfig::new(1)).is_err());
    }

    #[test]
    fn pca_single_point() {
        let data = vec![vec![1.0, 2.0]];
        assert!(pca(&data, &PcaConfig::new(1)).is_err());
    }

    #[test]
    fn pca_zero_components() {
        let data = make_two_dim();
        assert!(pca(&data, &PcaConfig::new(0)).is_err());
    }

    #[test]
    fn pca_too_many_components() {
        let data = make_two_dim();
        assert!(pca(&data, &PcaConfig::new(3)).is_err());
    }

    #[test]
    fn pca_nan_rejected() {
        let data = vec![vec![1.0, f64::NAN], vec![2.0, 3.0]];
        assert!(pca(&data, &PcaConfig::new(1)).is_err());
    }

    #[test]
    fn pca_inf_rejected() {
        let data = vec![vec![1.0, f64::INFINITY], vec![2.0, 3.0]];
        assert!(pca(&data, &PcaConfig::new(1)).is_err());
    }

    #[test]
    fn pca_dimension_mismatch() {
        let data = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(pca(&data, &PcaConfig::new(1)).is_err());
    }

    // ── Specific known result ─────────────────────────────────────

    #[test]
    fn pca_known_2d_result() {
        // 2D data along y = x line
        let data = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
            vec![5.0, 5.0],
        ];
        let result = pca(&data, &PcaConfig::new(2)).unwrap();

        // PC1 should explain 100% (perfectly correlated)
        assert!(
            result.explained_variance_ratio[0] > 0.999,
            "PC1 should capture all variance"
        );

        // PC1 loading should be [1/√2, 1/√2] or [-1/√2, -1/√2]
        let l = &result.loadings[0];
        let expected = 1.0 / 2.0f64.sqrt();
        assert!(
            (l[0].abs() - expected).abs() < 1e-10,
            "loading[0] = {}, expected ±{}",
            l[0],
            expected
        );
        assert!(
            (l[1].abs() - expected).abs() < 1e-10,
            "loading[1] = {}, expected ±{}",
            l[1],
            expected
        );
    }

    // ── 1D data ───────────────────────────────────────────────────

    #[test]
    fn pca_1d_data() {
        let data = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let result = pca(&data, &PcaConfig::new(1)).unwrap();

        assert_eq!(result.n_components, 1);
        assert!((result.explained_variance_ratio[0] - 1.0).abs() < 1e-10);
        // Loading should be [1.0] or [-1.0]
        assert!((result.loadings[0][0].abs() - 1.0).abs() < 1e-10);
    }

    // ── High-dimensional ──────────────────────────────────────────

    #[test]
    fn pca_high_dim_reduction() {
        // 10D data where only first 2 dimensions vary
        let mut data = Vec::new();
        for i in 0..20 {
            let mut point = vec![0.0; 10];
            point[0] = i as f64;
            point[1] = (i as f64) * 0.5 + 1.0;
            data.push(point);
        }

        let result = pca(&data, &PcaConfig::new(2)).unwrap();

        // First two components should capture all variance
        assert!(
            result.cumulative_variance_ratio[1] > 0.999,
            "first 2 PCs should explain >99.9% of variance, got {}",
            result.cumulative_variance_ratio[1]
        );
    }

    #[test]
    fn pca_means_stored() {
        let data = make_two_dim();
        let result = pca(&data, &PcaConfig::new(2)).unwrap();

        // Mean of [1,2,3,4,5] = 3.0, mean of [10,20,30,40,50] = 30.0
        assert!((result.means[0] - 3.0).abs() < 1e-10);
        assert!((result.means[1] - 30.0).abs() < 1e-10);
    }
}
