//! Clustering algorithms: K-Means, DBSCAN, Hierarchical Agglomerative, and HDBSCAN.
//!
//! Includes validation metrics: Silhouette score, Calinski-Harabasz index,
//! Davies-Bouldin index.
//!
//! - **K-Means**: K-Means++ initialization (Arthur & Vassilvitskii, 2007),
//!   Lloyd's iterative refinement, silhouette score, WCSS, auto-K selection.
//! - **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise
//!   (Ester et al., 1996). Discovers clusters of arbitrary shape based on
//!   local density. No need to specify number of clusters.
//! - **Hierarchical Agglomerative**: Bottom-up clustering with single, complete,
//!   average (UPGMA), or Ward linkage. Produces a dendrogram (merge history)
//!   that can be cut by distance threshold or number of clusters.
//!
//! # Example
//!
//! ```
//! use u_insight::clustering::{kmeans, KMeansConfig};
//!
//! // 2D data with 2 obvious clusters
//! let data = vec![
//!     vec![1.0, 1.0], vec![1.5, 1.5], vec![1.2, 1.3],
//!     vec![8.0, 8.0], vec![8.5, 8.5], vec![8.2, 8.3],
//! ];
//! let config = KMeansConfig::new(2);
//! let result = kmeans(&data, &config).unwrap();
//!
//! assert_eq!(result.k, 2);
//! assert_eq!(result.labels.len(), 6);
//! assert!(result.wcss < 10.0); // tight clusters
//! ```

use crate::error::InsightError;

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for K-Means clustering.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters.
    pub k: usize,
    /// Maximum iterations. Default: 300.
    pub max_iter: usize,
    /// Convergence tolerance (centroid movement). Default: 1e-6.
    pub tol: f64,
    /// Number of random restarts (best result kept). Default: 10.
    pub n_init: usize,
    /// Random seed (None for time-based). Default: Some(42).
    pub seed: Option<u64>,
}

impl KMeansConfig {
    /// Creates a config for a fixed number of clusters with default parameters.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iter: 300,
            tol: 1e-6,
            n_init: 10,
            seed: Some(42),
        }
    }

    /// Sets the maximum number of iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the number of random restarts.
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Sets the random seed.
    pub fn seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }
}

// ── Result ────────────────────────────────────────────────────────────

/// Result of K-Means clustering.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Number of clusters.
    pub k: usize,
    /// Cluster centroids (k × d matrix, row-major).
    pub centroids: Vec<Vec<f64>>,
    /// Cluster label for each data point (0..k).
    pub labels: Vec<usize>,
    /// Within-Cluster Sum of Squares (total).
    pub wcss: f64,
    /// Number of iterations until convergence.
    pub iterations: usize,
    /// Number of points per cluster.
    pub cluster_sizes: Vec<usize>,
}

/// Result of auto-K selection.
#[derive(Debug, Clone)]
pub struct AutoKResult {
    /// Best K according to silhouette score.
    pub best_k: usize,
    /// Best silhouette score.
    pub best_silhouette: f64,
    /// K-Means result for the best K.
    pub best_result: KMeansResult,
    /// Silhouette scores for each tested K.
    pub silhouette_scores: Vec<(usize, f64)>,
    /// WCSS for each tested K (for elbow plot).
    pub wcss_values: Vec<(usize, f64)>,
}

// ── K-Means algorithm ─────────────────────────────────────────────────

/// Runs K-Means clustering with K-Means++ initialization.
///
/// Input: `data` is a list of points, each point is a `Vec<f64>` of the same dimension.
///
/// ```
/// use u_insight::clustering::{kmeans, KMeansConfig};
///
/// let data = vec![
///     vec![0.0, 0.0], vec![0.5, 0.5],
///     vec![10.0, 10.0], vec![10.5, 10.5],
/// ];
/// let result = kmeans(&data, &KMeansConfig::new(2)).unwrap();
/// assert_eq!(result.k, 2);
/// // Points 0,1 should be in one cluster, 2,3 in another
/// assert_eq!(result.labels[0], result.labels[1]);
/// assert_eq!(result.labels[2], result.labels[3]);
/// assert_ne!(result.labels[0], result.labels[2]);
/// ```
pub fn kmeans(data: &[Vec<f64>], config: &KMeansConfig) -> Result<KMeansResult, InsightError> {
    let n = data.len();
    let k = config.k;

    if n == 0 {
        return Err(InsightError::DegenerateData {
            reason: "no data points provided".into(),
        });
    }
    if k == 0 || k > n {
        return Err(InsightError::InvalidParameter {
            name: "k".into(),
            message: format!("must be between 1 and {n} (number of data points), got {k}"),
        });
    }

    let d = data[0].len();
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

    // Run n_init restarts, keep best (lowest WCSS)
    let mut best_result: Option<KMeansResult> = None;

    for init_idx in 0..config.n_init {
        let seed = config.seed.map(|s| s.wrapping_add(init_idx as u64));
        let result = kmeans_single(data, k, d, config.max_iter, config.tol, seed);

        if best_result
            .as_ref()
            .is_none_or(|best| result.wcss < best.wcss)
        {
            best_result = Some(result);
        }
    }

    Ok(best_result.expect("n_init > 0 guarantees at least one result"))
}

/// Selects the optimal K by maximizing silhouette score over a range.
///
/// ```
/// use u_insight::clustering::{auto_kmeans, KMeansConfig};
///
/// let data = vec![
///     vec![0.0, 0.0], vec![0.5, 0.5], vec![0.2, 0.3],
///     vec![10.0, 10.0], vec![10.5, 10.5], vec![10.2, 10.3],
///     vec![20.0, 0.0], vec![20.5, 0.5], vec![20.2, 0.3],
/// ];
/// let result = auto_kmeans(&data, 2, 5, &KMeansConfig::new(2)).unwrap();
/// assert!(result.best_k >= 2 && result.best_k <= 5);
/// assert!(result.best_silhouette > 0.0);
/// ```
pub fn auto_kmeans(
    data: &[Vec<f64>],
    k_min: usize,
    k_max: usize,
    base_config: &KMeansConfig,
) -> Result<AutoKResult, InsightError> {
    if k_min < 2 {
        return Err(InsightError::InvalidParameter {
            name: "k_min".into(),
            message: format!("must be at least 2, got {k_min}"),
        });
    }
    if k_max < k_min {
        return Err(InsightError::InvalidParameter {
            name: "k_max".into(),
            message: format!("must be >= k_min ({k_min}), got {k_max}"),
        });
    }

    let mut silhouette_scores = Vec::new();
    let mut wcss_values = Vec::new();
    let mut best_k = k_min;
    let mut best_silhouette = f64::NEG_INFINITY;
    let mut best_result: Option<KMeansResult> = None;

    for k in k_min..=k_max.min(data.len()) {
        let config = KMeansConfig {
            k,
            ..base_config.clone()
        };
        let result = kmeans(data, &config)?;
        let sil = silhouette_score(data, &result.labels, k);

        wcss_values.push((k, result.wcss));
        silhouette_scores.push((k, sil));

        if sil > best_silhouette {
            best_silhouette = sil;
            best_k = k;
            best_result = Some(result);
        }
    }

    Ok(AutoKResult {
        best_k,
        best_silhouette,
        best_result: best_result.expect("at least one K tested"),
        silhouette_scores,
        wcss_values,
    })
}

// ── Silhouette Score ──────────────────────────────────────────────────

/// Computes the mean silhouette score across all data points.
///
/// Silhouette score ranges from -1 (wrong cluster) to +1 (well-separated).
/// Uses Euclidean distance. O(n²) complexity.
///
/// Reference: Rousseeuw (1987). "Silhouettes: a graphical aid to the
/// interpretation and validation of cluster analysis."
///
/// ```
/// use u_insight::clustering::silhouette_score;
///
/// let data = vec![
///     vec![0.0, 0.0], vec![0.5, 0.5],
///     vec![10.0, 10.0], vec![10.5, 10.5],
/// ];
/// let labels = vec![0, 0, 1, 1];
/// let score = silhouette_score(&data, &labels, 2);
/// assert!(score > 0.9); // well-separated clusters
/// ```
pub fn silhouette_score(data: &[Vec<f64>], labels: &[usize], k: usize) -> f64 {
    let n = data.len();
    if n < 2 || k < 2 {
        return 0.0;
    }

    // Precompute pairwise distances
    let mut dist_matrix = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_dist(&data[i], &data[j]);
            dist_matrix[i * n + j] = d;
            dist_matrix[j * n + i] = d;
        }
    }

    let mut total_sil = 0.0;
    let mut valid_count = 0;

    for i in 0..n {
        let own_cluster = labels[i];

        // a(i) = mean distance to other points in same cluster
        let mut same_sum = 0.0;
        let mut same_count = 0usize;
        for j in 0..n {
            if j != i && labels[j] == own_cluster {
                same_sum += dist_matrix[i * n + j];
                same_count += 1;
            }
        }

        if same_count == 0 {
            // Singleton cluster → silhouette = 0
            continue;
        }

        let a_i = same_sum / same_count as f64;

        // b(i) = min over other clusters of mean distance to that cluster
        let mut b_i = f64::INFINITY;
        for c in 0..k {
            if c == own_cluster {
                continue;
            }
            let mut other_sum = 0.0;
            let mut other_count = 0usize;
            for j in 0..n {
                if labels[j] == c {
                    other_sum += dist_matrix[i * n + j];
                    other_count += 1;
                }
            }
            if other_count > 0 {
                let mean_dist = other_sum / other_count as f64;
                b_i = b_i.min(mean_dist);
            }
        }

        if b_i.is_infinite() {
            continue;
        }

        let s_i = (b_i - a_i) / a_i.max(b_i);
        total_sil += s_i;
        valid_count += 1;
    }

    if valid_count == 0 {
        0.0
    } else {
        total_sil / valid_count as f64
    }
}

// ── Internal K-Means ──────────────────────────────────────────────────

fn kmeans_single(
    data: &[Vec<f64>],
    k: usize,
    d: usize,
    max_iter: usize,
    tol: f64,
    seed: Option<u64>,
) -> KMeansResult {
    let n = data.len();

    // K-Means++ initialization
    let mut centroids = kmeans_plus_plus(data, k, d, seed);
    let mut labels = vec![0usize; n];
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Assignment step
        for (i, point) in data.iter().enumerate() {
            let mut min_dist = f64::INFINITY;
            let mut best_c = 0;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist = euclidean_dist_sq(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_c = c;
                }
            }
            labels[i] = best_c;
        }

        // Update step: compute new centroids
        let mut new_centroids = vec![vec![0.0; d]; k];
        let mut counts = vec![0usize; k];

        for (i, point) in data.iter().enumerate() {
            let c = labels[i];
            counts[c] += 1;
            for (j, &v) in point.iter().enumerate() {
                new_centroids[c][j] += v;
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for val in &mut new_centroids[c] {
                    *val /= counts[c] as f64;
                }
            } else {
                // Empty cluster: keep old centroid
                new_centroids[c] = centroids[c].clone();
            }
        }

        // Check convergence
        let max_shift: f64 = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(old, new)| euclidean_dist(old, new))
            .fold(0.0f64, f64::max);

        centroids = new_centroids;

        if max_shift < tol {
            break;
        }
    }

    // Compute WCSS and cluster sizes
    let mut wcss = 0.0;
    let mut cluster_sizes = vec![0usize; k];
    for (i, point) in data.iter().enumerate() {
        let c = labels[i];
        cluster_sizes[c] += 1;
        wcss += euclidean_dist_sq(point, &centroids[c]);
    }

    KMeansResult {
        k,
        centroids,
        labels,
        wcss,
        iterations,
        cluster_sizes,
    }
}

/// K-Means++ initialization: select k initial centroids using
/// distance-proportional probability sampling.
///
/// Reference: Arthur & Vassilvitskii (2007). "k-means++: The advantages
/// of careful seeding."
fn kmeans_plus_plus(data: &[Vec<f64>], k: usize, _d: usize, seed: Option<u64>) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);

    // Simple LCG-based random number generator for reproducibility
    let mut rng_state = seed.unwrap_or(12345);
    let mut next_rand = || -> f64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64
    };

    // First centroid: random
    let first_idx = (next_rand() * n as f64) as usize % n;
    centroids.push(data[first_idx].clone());

    // Subsequent centroids: distance-proportional sampling
    let mut min_dists = vec![f64::INFINITY; n];

    for _ in 1..k {
        // Update minimum distances to nearest centroid
        let last_centroid = centroids.last().expect("at least one centroid");
        for (i, point) in data.iter().enumerate() {
            let dist = euclidean_dist_sq(point, last_centroid);
            min_dists[i] = min_dists[i].min(dist);
        }

        // Weighted random selection (distance² proportional)
        let total: f64 = min_dists.iter().sum();
        if total < 1e-15 {
            // All points coincide: pick any remaining
            let idx = (next_rand() * n as f64) as usize % n;
            centroids.push(data[idx].clone());
            continue;
        }

        let target = next_rand() * total;
        let mut cumulative = 0.0;
        let mut chosen = 0;
        for (i, &dist) in min_dists.iter().enumerate() {
            cumulative += dist;
            if cumulative >= target {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
    }

    centroids
}

// ── Distance helpers ──────────────────────────────────────────────────

#[inline]
fn euclidean_dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .sum()
}

#[inline]
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    euclidean_dist_sq(a, b).sqrt()
}

// ── DBSCAN ───────────────────────────────────────────────────────────

/// Configuration for DBSCAN clustering.
///
/// # Parameters
///
/// - `epsilon` — Maximum distance between two points to be considered neighbors.
/// - `min_samples` — Minimum number of points in a neighborhood (including the
///   point itself) to qualify as a core point. Must be >= 2.
#[derive(Debug, Clone)]
pub struct DbscanConfig {
    /// Maximum neighborhood distance.
    pub epsilon: f64,
    /// Minimum points in epsilon-neighborhood to form a core point.
    pub min_samples: usize,
}

impl DbscanConfig {
    /// Creates a DBSCAN config with the given epsilon and min_samples.
    ///
    /// # Panics
    ///
    /// Does not panic; validation is done in [`dbscan`].
    pub fn new(epsilon: f64, min_samples: usize) -> Self {
        Self {
            epsilon,
            min_samples,
        }
    }
}

/// Result of DBSCAN clustering.
#[derive(Debug, Clone)]
pub struct DbscanResult {
    /// Cluster label for each data point.
    /// `None` = noise point, `Some(id)` = cluster membership (0-indexed).
    pub labels: Vec<Option<usize>>,
    /// Number of clusters discovered.
    pub n_clusters: usize,
    /// Number of noise points.
    pub noise_count: usize,
    /// Number of points in each cluster (indexed by cluster id).
    pub cluster_sizes: Vec<usize>,
    /// Whether each point is a core point.
    pub core_points: Vec<bool>,
}

/// Runs DBSCAN density-based clustering.
///
/// Discovers clusters of arbitrary shape based on local point density.
/// Points that do not belong to any cluster are labeled as noise.
///
/// # Algorithm
///
/// Ester, Kriegel, Sander, Xu (1996).
/// "A Density-Based Algorithm for Discovering Clusters in Large Spatial
/// Databases with Noise."
///
/// Complexity: O(n²) with brute-force neighbor search.
///
/// # Arguments
///
/// * `data` — List of points, each a `Vec<f64>` of the same dimension.
/// * `config` — DBSCAN parameters (epsilon, min_samples).
///
/// # Returns
///
/// A [`DbscanResult`] with cluster labels, noise identification, and core point flags.
///
/// # Errors
///
/// - [`InsightError::DegenerateData`] if data is empty
/// - [`InsightError::InvalidParameter`] if min_samples < 2 or epsilon <= 0
/// - [`InsightError::DimensionMismatch`] if points have different dimensions
/// - [`InsightError::NonNumericColumn`] if data contains NaN or infinite values
///
/// # Example
///
/// ```
/// use u_insight::clustering::{dbscan, DbscanConfig};
///
/// let data = vec![
///     vec![0.0, 0.0], vec![0.5, 0.0], vec![0.0, 0.5],
///     vec![10.0, 10.0], vec![10.5, 10.0], vec![10.0, 10.5],
///     vec![50.0, 50.0], // noise point
/// ];
/// let result = dbscan(&data, &DbscanConfig::new(1.5, 2)).unwrap();
///
/// assert_eq!(result.n_clusters, 2);
/// assert_eq!(result.noise_count, 1);
/// assert!(result.labels[6].is_none()); // isolated point is noise
/// ```
pub fn dbscan(data: &[Vec<f64>], config: &DbscanConfig) -> Result<DbscanResult, InsightError> {
    let n = data.len();

    if n == 0 {
        return Err(InsightError::DegenerateData {
            reason: "no data points provided".into(),
        });
    }

    if config.min_samples < 2 {
        return Err(InsightError::InvalidParameter {
            name: "min_samples".into(),
            message: format!("must be at least 2, got {}", config.min_samples),
        });
    }

    if !config.epsilon.is_finite() || config.epsilon <= 0.0 {
        return Err(InsightError::NonNumericColumn {
            column: "epsilon".to_string(),
        });
    }

    let d = data[0].len();
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

    // Precompute distance matrix (O(n²) space + time)
    // For moderate n this is acceptable and avoids repeated computation.
    let eps = config.epsilon;
    let min_pts = config.min_samples;

    // Find neighbors for each point
    let neighbors: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| euclidean_dist(&data[i], &data[j]) <= eps)
                .collect()
        })
        .collect();

    // Identify core points
    let core_points: Vec<bool> = neighbors.iter().map(|nb| nb.len() >= min_pts).collect();

    // DBSCAN main loop
    let mut labels: Vec<Option<usize>> = vec![None; n];
    let mut visited = vec![false; n];
    let mut cluster_id = 0;

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;

        if !core_points[i] {
            // Will be noise or border (assigned later if reachable from a core)
            continue;
        }

        // Start a new cluster from this core point
        labels[i] = Some(cluster_id);

        // Expand cluster using iterative BFS
        let mut queue = std::collections::VecDeque::new();
        for &nb in &neighbors[i] {
            if nb != i {
                queue.push_back(nb);
            }
        }

        while let Some(q) = queue.pop_front() {
            if labels[q].is_some() {
                // Already assigned to this or another cluster
                continue;
            }

            labels[q] = Some(cluster_id);

            if !visited[q] {
                visited[q] = true;
                if core_points[q] {
                    // Expand from this core point too
                    for &nb in &neighbors[q] {
                        if labels[nb].is_none() {
                            queue.push_back(nb);
                        }
                    }
                }
            }
        }

        cluster_id += 1;
    }

    let n_clusters = cluster_id;
    let noise_count = labels.iter().filter(|l| l.is_none()).count();

    let mut cluster_sizes = vec![0usize; n_clusters];
    for c in labels.iter().flatten() {
        cluster_sizes[*c] += 1;
    }

    Ok(DbscanResult {
        labels,
        n_clusters,
        noise_count,
        cluster_sizes,
        core_points,
    })
}

// ── Hierarchical Agglomerative Clustering ────────────────────────────

/// Linkage criterion for hierarchical agglomerative clustering.
///
/// Determines how the distance between two clusters is computed.
///
/// # References
///
/// - Lance, G.N. & Williams, W.T. (1967). "A general theory of
///   classificatory sorting strategies: Hierarchical systems".
/// - Ward, J.H. (1963). "Hierarchical grouping to optimize an
///   objective function".
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Linkage {
    /// Minimum distance between any pair of points across clusters.
    /// Tends to produce elongated "chaining" clusters.
    Single,
    /// Maximum distance between any pair of points across clusters.
    /// Produces compact, balanced clusters.
    Complete,
    /// Weighted average (UPGMA) of pairwise distances.
    /// Good balance between single and complete linkage.
    Average,
    /// Minimizes total within-cluster variance.
    /// Requires Euclidean distances. Produces spherical, compact clusters.
    Ward,
}

/// Configuration for hierarchical agglomerative clustering.
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// Linkage criterion. Default: `Linkage::Ward`.
    pub linkage: Linkage,
    /// Number of flat clusters to extract. Mutually exclusive with `distance_threshold`.
    pub n_clusters: Option<usize>,
    /// Distance threshold for cutting the dendrogram. Mutually exclusive with `n_clusters`.
    pub distance_threshold: Option<f64>,
}

impl HierarchicalConfig {
    /// Creates a config that cuts the dendrogram into `k` clusters.
    pub fn with_k(k: usize) -> Self {
        Self {
            linkage: Linkage::Ward,
            n_clusters: Some(k),
            distance_threshold: None,
        }
    }

    /// Creates a config that cuts the dendrogram at the given distance threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            linkage: Linkage::Ward,
            n_clusters: None,
            distance_threshold: Some(threshold),
        }
    }

    /// Sets the linkage criterion.
    pub fn linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }
}

/// A single merge step in the dendrogram.
#[derive(Debug, Clone)]
pub struct Merge {
    /// Index of the first cluster merged (0-based; original points use 0..n,
    /// merged clusters use n, n+1, …).
    pub cluster_a: usize,
    /// Index of the second cluster merged.
    pub cluster_b: usize,
    /// Distance (height) at which the merge occurred.
    pub distance: f64,
    /// Size of the newly formed cluster.
    pub size: usize,
}

/// Result of hierarchical agglomerative clustering.
///
/// Contains the full dendrogram (merge history) and, if a cutting criterion
/// was specified, flat cluster assignments.
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    /// Merge history (dendrogram). Length = n − 1.
    /// Merges are sorted by ascending distance.
    pub merges: Vec<Merge>,
    /// Flat cluster labels (0-based). Only present when `n_clusters` or
    /// `distance_threshold` was set in the config.
    pub labels: Option<Vec<usize>>,
    /// Number of flat clusters (if labels present).
    pub n_clusters: Option<usize>,
}

/// Performs hierarchical agglomerative clustering.
///
/// # Algorithm
///
/// 1. Initialize each data point as its own cluster.
/// 2. Compute the pairwise Euclidean distance matrix.
/// 3. Repeat n − 1 times:
///    a. Find the closest pair of clusters.
///    b. Merge them and record the merge step.
///    c. Update distances using the Lance-Williams formula for the
///    chosen linkage.
/// 4. Optionally cut the dendrogram to produce flat cluster labels.
///
/// # Complexity
///
/// Time: O(n³) — naive implementation scanning for minimum each step.
/// Space: O(n²) — condensed distance matrix.
///
/// # References
///
/// - Lance & Williams (1967). "A general theory of classificatory
///   sorting strategies".
/// - Murtagh & Legendre (2014). "Ward's hierarchical agglomerative
///   clustering method: which algorithms implement Ward's criterion?"
///
/// # Examples
///
/// ```
/// use u_insight::clustering::{hierarchical, HierarchicalConfig, Linkage};
///
/// let data = vec![
///     vec![0.0, 0.0], vec![0.5, 0.5], vec![0.2, 0.3],
///     vec![10.0, 10.0], vec![10.5, 10.5], vec![10.2, 10.3],
/// ];
/// let config = HierarchicalConfig::with_k(2).linkage(Linkage::Ward);
/// let result = hierarchical(&data, &config).unwrap();
///
/// assert_eq!(result.n_clusters, Some(2));
/// let labels = result.labels.unwrap();
/// // Points in the same physical cluster should share a label
/// assert_eq!(labels[0], labels[1]);
/// assert_eq!(labels[3], labels[4]);
/// assert_ne!(labels[0], labels[3]);
/// ```
pub fn hierarchical(
    data: &[Vec<f64>],
    config: &HierarchicalConfig,
) -> Result<HierarchicalResult, InsightError> {
    let n = data.len();

    if n < 2 {
        return Err(InsightError::InsufficientData {
            min_required: 2,
            actual: n,
        });
    }

    if let Some(k) = config.n_clusters {
        if k == 0 || k > n {
            return Err(InsightError::InvalidParameter {
                name: "n_clusters".into(),
                message: format!("must be between 1 and {n} (number of data points), got {k}"),
            });
        }
    }

    if let Some(t) = config.distance_threshold {
        if !t.is_finite() || t < 0.0 {
            return Err(InsightError::InvalidParameter {
                name: "distance_threshold".into(),
                message: format!("must be a non-negative finite number, got {t}"),
            });
        }
    }

    // Validate dimensions and values
    let d = data[0].len();
    if d == 0 {
        return Err(InsightError::DimensionMismatch {
            expected: 1,
            actual: 0,
        });
    }
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

    // Compute condensed distance matrix (upper triangle, row-major).
    // Ward uses squared Euclidean distances internally.
    let use_sq = config.linkage == Linkage::Ward;
    let cond_len = n * (n - 1) / 2;
    let mut dist = vec![0.0_f64; cond_len];
    for i in 0..n {
        for j in (i + 1)..n {
            let d_sq = euclidean_dist_sq(&data[i], &data[j]);
            dist[condensed_index(n, i, j)] = if use_sq { d_sq } else { d_sq.sqrt() };
        }
    }

    // Active cluster tracking
    let mut active = vec![true; n]; // which clusters are still alive
    let mut sizes = vec![1_usize; n]; // cluster sizes
    let mut merges = Vec::with_capacity(n - 1);

    // Cluster label mapping: original index → current cluster id
    // New merged clusters get ids n, n+1, …
    for _step in 0..(n - 1) {
        // Find closest pair among active clusters
        let mut best_i = 0;
        let mut best_j = 0;
        let mut best_d = f64::INFINITY;

        let active_indices: Vec<usize> = (0..active.len()).filter(|&i| active[i]).collect();

        for ai in 0..active_indices.len() {
            for aj in (ai + 1)..active_indices.len() {
                let ci = active_indices[ai];
                let cj = active_indices[aj];
                let d_val = dist[condensed_index(active.len(), ci, cj)];
                if d_val < best_d {
                    best_d = d_val;
                    best_i = ci;
                    best_j = cj;
                }
            }
        }

        let merge_dist = if use_sq { best_d.sqrt() } else { best_d };

        merges.push(Merge {
            cluster_a: best_i,
            cluster_b: best_j,
            distance: merge_dist,
            size: sizes[best_i] + sizes[best_j],
        });

        // Update distances from new cluster to all remaining clusters
        // using Lance-Williams formula.
        let si = sizes[best_i];
        let sj = sizes[best_j];
        let d_ij = best_d;

        for &ck in &active_indices {
            if ck == best_i || ck == best_j {
                continue;
            }
            let d_ik = dist[condensed_index(active.len(), best_i.min(ck), best_i.max(ck))];
            let d_jk = dist[condensed_index(active.len(), best_j.min(ck), best_j.max(ck))];
            let sk = sizes[ck];

            let d_new = lance_williams(config.linkage, d_ik, d_jk, d_ij, si, sj, sk);

            // Store in best_i's row (we reuse best_i as the merged cluster)
            let idx = condensed_index(active.len(), best_i.min(ck), best_i.max(ck));
            dist[idx] = d_new;
        }

        // Deactivate best_j, keep best_i as the merged cluster
        active[best_j] = false;
        sizes[best_i] = si + sj;

        // Remap cluster ids for the merge record
        merges.last_mut().expect("just pushed").cluster_a = best_i;
        merges.last_mut().expect("just pushed").cluster_b = best_j;
    }

    // Assign sequential cluster IDs to the merge history.
    // Original points: 0..n, merged clusters: n, n+1, ...
    let mut id_map: Vec<usize> = (0..n).collect();
    let mut next_merge_id = n;
    for merge in &mut merges {
        let a = merge.cluster_a;
        let b = merge.cluster_b;
        merge.cluster_a = id_map[a];
        merge.cluster_b = id_map[b];
        id_map[a] = next_merge_id;
        next_merge_id += 1;
    }

    // Cut dendrogram if requested
    let (labels, n_clusters) = if let Some(k) = config.n_clusters {
        let l = cut_dendrogram_k(&merges, n, k);
        let nc = *l.iter().max().unwrap_or(&0) + 1;
        (Some(l), Some(nc))
    } else if let Some(threshold) = config.distance_threshold {
        let l = cut_dendrogram_threshold(&merges, n, threshold);
        let nc = *l.iter().max().unwrap_or(&0) + 1;
        (Some(l), Some(nc))
    } else {
        (None, None)
    };

    Ok(HierarchicalResult {
        merges,
        labels,
        n_clusters,
    })
}

/// Condensed distance matrix index for pair (i, j) where i < j.
#[inline]
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j && j < n);
    i * n - i * (i + 1) / 2 + j - i - 1
}

/// Lance-Williams formula for updating cluster distances after a merge.
///
/// d(i∪j, k) = αi·d(i,k) + αj·d(j,k) + β·d(i,j) + γ·|d(i,k)-d(j,k)|
fn lance_williams(
    linkage: Linkage,
    d_ik: f64,
    d_jk: f64,
    d_ij: f64,
    si: usize,
    sj: usize,
    sk: usize,
) -> f64 {
    match linkage {
        Linkage::Single => d_ik.min(d_jk),
        Linkage::Complete => d_ik.max(d_jk),
        Linkage::Average => {
            let ni = si as f64;
            let nj = sj as f64;
            (ni * d_ik + nj * d_jk) / (ni + nj)
        }
        Linkage::Ward => {
            // Ward uses squared distances internally.
            let ni = si as f64;
            let nj = sj as f64;
            let nk = sk as f64;
            let total = ni + nj + nk;
            ((ni + nk) * d_ik + (nj + nk) * d_jk - nk * d_ij) / total
        }
    }
}

/// Cut dendrogram to produce exactly `k` flat clusters.
fn cut_dendrogram_k(merges: &[Merge], n: usize, k: usize) -> Vec<usize> {
    if k >= n {
        // Each point is its own cluster
        return (0..n).collect();
    }

    // We process the first (n - k) merges. The remaining merges are "cut".
    let n_merges_to_apply = n - k;

    // Union-Find
    let mut parent: Vec<usize> = (0..(2 * n)).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path compression
            x = parent[x];
        }
        x
    }

    for merge in merges.iter().take(n_merges_to_apply) {
        let new_id = n + merges
            .iter()
            .position(|m| std::ptr::eq(m, merge))
            .unwrap_or(0);
        // Actually we know merge ids: cluster_a and cluster_b are already
        // in the sequential id scheme (0..n original, n.. merged)
        let ra = find(&mut parent, merge.cluster_a);
        let rb = find(&mut parent, merge.cluster_b);
        parent[ra] = new_id;
        parent[rb] = new_id;
        parent[new_id] = new_id;
    }

    // Assign flat labels
    let mut label_map = std::collections::HashMap::new();
    let mut next_label = 0_usize;
    let mut labels = Vec::with_capacity(n);

    for i in 0..n {
        let root = find(&mut parent, i);
        let label = label_map.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        labels.push(*label);
    }

    labels
}

/// Cut dendrogram at a distance threshold.
fn cut_dendrogram_threshold(merges: &[Merge], n: usize, threshold: f64) -> Vec<usize> {
    // Apply only merges with distance <= threshold
    let mut parent: Vec<usize> = (0..(2 * n)).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    for (idx, merge) in merges.iter().enumerate() {
        if merge.distance > threshold {
            break;
        }
        let new_id = n + idx;
        let ra = find(&mut parent, merge.cluster_a);
        let rb = find(&mut parent, merge.cluster_b);
        parent[ra] = new_id;
        parent[rb] = new_id;
        parent[new_id] = new_id;
    }

    let mut label_map = std::collections::HashMap::new();
    let mut next_label = 0_usize;
    let mut labels = Vec::with_capacity(n);

    for i in 0..n {
        let root = find(&mut parent, i);
        let label = label_map.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        labels.push(*label);
    }

    labels
}

// ── HDBSCAN ──────────────────────────────────────────────────────────

/// Configuration for HDBSCAN (Hierarchical DBSCAN).
///
/// # Parameters
///
/// - `min_cluster_size` — Minimum points required to form a cluster.
///   Controls which splits persist in the condensed tree.
/// - `min_samples` — Core distance parameter (k in k-NN). Controls density
///   sensitivity. Defaults to `min_cluster_size` if `None`.
#[derive(Debug, Clone)]
pub struct HdbscanConfig {
    /// Minimum number of points for a cluster to be considered significant.
    /// Default: 5.
    pub min_cluster_size: usize,
    /// Number of neighbors for core-distance computation.
    /// Defaults to `min_cluster_size` if `None`.
    pub min_samples: Option<usize>,
}

impl HdbscanConfig {
    /// Creates a config with the given minimum cluster size.
    pub fn new(min_cluster_size: usize) -> Self {
        Self {
            min_cluster_size,
            min_samples: None,
        }
    }

    /// Sets the min_samples parameter for core distance computation.
    pub fn min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = Some(min_samples);
        self
    }
}

/// Result of HDBSCAN clustering.
#[derive(Debug, Clone)]
pub struct HdbscanResult {
    /// Cluster labels (0-based). `None` means noise/outlier.
    pub labels: Vec<Option<usize>>,
    /// Membership probability for each point (0.0–1.0).
    /// Noise points have probability 0.0.
    pub probabilities: Vec<f64>,
    /// Number of clusters found (excluding noise).
    pub n_clusters: usize,
    /// Number of noise points.
    pub noise_count: usize,
    /// Cluster sizes (indexed by cluster label).
    pub cluster_sizes: Vec<usize>,
}

/// Performs HDBSCAN clustering.
///
/// # Algorithm
///
/// 1. Compute core distances (k-NN distance for each point).
/// 2. Build mutual reachability distance graph.
/// 3. Construct minimum spanning tree (Prim's algorithm).
/// 4. Build single-linkage dendrogram from MST.
/// 5. Condense the hierarchy using `min_cluster_size`.
/// 6. Extract clusters using Excess of Mass (EOM) stability.
///
/// # Complexity
///
/// Time: O(n²) — dominated by pairwise distance / MST construction.
/// Space: O(n²) — full distance matrix.
///
/// # References
///
/// - Campello, Moulavi & Sander (2013). "Density-Based Clustering Based
///   on Hierarchical Density Estimates". PAKDD.
/// - McInnes, Healy & Astels (2017). "hdbscan: Hierarchical density based
///   clustering". JOSS, 2(11), 205.
///
/// # Examples
///
/// ```
/// use u_insight::clustering::{hdbscan, HdbscanConfig};
///
/// let data = vec![
///     vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0],
///     vec![0.0, 0.2], vec![0.1, 0.0],
///     vec![10.0, 10.0], vec![10.1, 10.1], vec![10.2, 10.0],
///     vec![10.0, 10.2], vec![10.1, 10.0],
///     vec![50.0, 50.0], // outlier
/// ];
/// let config = HdbscanConfig::new(3);
/// let result = hdbscan(&data, &config).unwrap();
///
/// assert_eq!(result.n_clusters, 2);
/// assert!(result.labels[10].is_none()); // outlier is noise
/// ```
pub fn hdbscan(data: &[Vec<f64>], config: &HdbscanConfig) -> Result<HdbscanResult, InsightError> {
    let n = data.len();
    let min_cluster_size = config.min_cluster_size;
    let min_samples = config.min_samples.unwrap_or(min_cluster_size);

    if n < 2 {
        return Err(InsightError::InsufficientData {
            min_required: 2,
            actual: n,
        });
    }
    if min_cluster_size < 2 {
        return Err(InsightError::InvalidParameter {
            name: "min_cluster_size".into(),
            message: format!("must be at least 2, got {min_cluster_size}"),
        });
    }
    if min_samples < 1 {
        return Err(InsightError::InvalidParameter {
            name: "min_samples".into(),
            message: format!("must be at least 1, got {min_samples}"),
        });
    }

    // Validate dimensions and values
    let d = data[0].len();
    if d == 0 {
        return Err(InsightError::DimensionMismatch {
            expected: 1,
            actual: 0,
        });
    }
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

    // Phase 1: Compute pairwise Euclidean distances
    let mut dist_matrix = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d_val = euclidean_dist(&data[i], &data[j]);
            dist_matrix[i * n + j] = d_val;
            dist_matrix[j * n + i] = d_val;
        }
    }

    // Phase 1b: Compute core distances (distance to k-th nearest neighbor)
    let k = min_samples.min(n);
    let mut core_dists = vec![0.0_f64; n];
    for i in 0..n {
        let mut dists_from_i: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .map(|j| dist_matrix[i * n + j])
            .collect();
        dists_from_i.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // k-th nearest neighbor (0-indexed: k-1)
        core_dists[i] = if k <= 1 {
            dists_from_i.first().copied().unwrap_or(0.0)
        } else {
            dists_from_i
                .get(k - 2)
                .copied()
                .unwrap_or(dists_from_i.last().copied().unwrap_or(0.0))
        };
    }

    // Phase 2: Mutual reachability distance
    // d_mreach(a, b) = max(core(a), core(b), d(a,b))
    let mut mreach = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d_val = dist_matrix[i * n + j].max(core_dists[i]).max(core_dists[j]);
            mreach[i * n + j] = d_val;
            mreach[j * n + i] = d_val;
        }
    }

    // Phase 3: MST via Prim's algorithm on mutual reachability graph
    let mst_edges = prim_mst(&mreach, n);

    // Phase 4: Build single-linkage dendrogram from MST
    // Sort edges by weight (ascending)
    let mut sorted_edges = mst_edges;
    sorted_edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Build dendrogram using union-find
    let mut sl_parent: Vec<usize> = (0..(2 * n)).collect();
    let mut sl_size = vec![1_usize; 2 * n];
    let mut dendrogram: Vec<(usize, usize, f64, usize)> = Vec::with_capacity(n - 1);
    let mut next_cluster = n;

    for (u, v, w) in &sorted_edges {
        let ru = uf_find(&mut sl_parent, *u);
        let rv = uf_find(&mut sl_parent, *v);
        if ru != rv {
            let new_size = sl_size[ru] + sl_size[rv];
            sl_parent[ru] = next_cluster;
            sl_parent[rv] = next_cluster;
            sl_parent[next_cluster] = next_cluster;
            sl_size[next_cluster] = new_size;
            dendrogram.push((ru, rv, *w, new_size));
            next_cluster += 1;
        }
    }

    // Phase 5: Condense the dendrogram
    let condensed = condense_tree(&dendrogram, n, min_cluster_size);

    // Phase 6: Extract clusters using EOM
    let (labels, probabilities) = extract_eom_clusters(&condensed, n);

    let n_clusters = labels
        .iter()
        .flatten()
        .collect::<std::collections::HashSet<_>>()
        .len();
    let noise_count = labels.iter().filter(|l| l.is_none()).count();
    let mut cluster_sizes = vec![0_usize; n_clusters];
    for c in labels.iter().flatten() {
        if *c < n_clusters {
            cluster_sizes[*c] += 1;
        }
    }

    Ok(HdbscanResult {
        labels,
        probabilities,
        n_clusters,
        noise_count,
        cluster_sizes,
    })
}

/// Prim's algorithm for MST on dense mutual reachability graph.
/// Returns edges as (u, v, weight).
fn prim_mst(mreach: &[f64], n: usize) -> Vec<(usize, usize, f64)> {
    let mut in_mst = vec![false; n];
    let mut min_edge = vec![(f64::INFINITY, 0_usize); n]; // (weight, from)
    let mut edges = Vec::with_capacity(n - 1);

    // Start from vertex 0
    in_mst[0] = true;
    for j in 1..n {
        min_edge[j] = (mreach[j], 0); // dist from vertex 0
    }

    for _ in 0..(n - 1) {
        // Find minimum edge from MST to non-MST vertex
        let mut best_v = 0;
        let mut best_w = f64::INFINITY;
        for v in 0..n {
            if !in_mst[v] && min_edge[v].0 < best_w {
                best_w = min_edge[v].0;
                best_v = v;
            }
        }

        in_mst[best_v] = true;
        edges.push((min_edge[best_v].1, best_v, best_w));

        // Update min edges
        for v in 0..n {
            if !in_mst[v] {
                let d = mreach[best_v * n + v];
                if d < min_edge[v].0 {
                    min_edge[v] = (d, best_v);
                }
            }
        }
    }

    edges
}

/// Union-Find: find with path compression.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Node in the condensed cluster tree.
#[derive(Debug, Clone)]
struct CondensedNode {
    parent: usize,
    child: usize,
    lambda_val: f64, // 1/distance
    child_size: usize,
}

/// Build condensed tree from single-linkage dendrogram.
fn condense_tree(
    dendrogram: &[(usize, usize, f64, usize)],
    n: usize,
    min_cluster_size: usize,
) -> Vec<CondensedNode> {
    // Build parent-child relationships from dendrogram
    // dendrogram[i] = (left, right, distance, size), node id = n + i
    let n_merges = dendrogram.len();
    let total_nodes = n + n_merges;

    // Children of each internal node
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); total_nodes];
    let mut node_size = vec![1_usize; total_nodes];
    let mut node_dist = vec![0.0_f64; total_nodes];

    for (i, (left, right, dist, size)) in dendrogram.iter().enumerate() {
        let node_id = n + i;
        children[node_id].push(*left);
        children[node_id].push(*right);
        node_size[node_id] = *size;
        node_dist[node_id] = *dist;
    }

    // Condense: traverse top-down. Relabel nodes that are too small.
    let mut condensed = Vec::new();
    let root = n + n_merges - 1;

    // BFS/DFS to condense
    // relabel[old_id] = new condensed cluster id
    let mut relabel = vec![0_usize; total_nodes];
    let mut next_condensed_id = 0_usize;
    relabel[root] = next_condensed_id;
    next_condensed_id += 1;

    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if children[node].is_empty() {
            // Leaf node (original point) — already handled when parent processed it
            continue;
        }

        let lambda = if node_dist[node] > 1e-300 {
            1.0 / node_dist[node]
        } else {
            1e300 // effectively infinite density
        };

        let left = children[node][0];
        let right = children[node][1];
        let left_size = node_size[left];
        let right_size = node_size[right];

        let left_big = left_size >= min_cluster_size;
        let right_big = right_size >= min_cluster_size;

        match (left_big, right_big) {
            (true, true) => {
                // Both children are significant: true split
                let left_id = next_condensed_id;
                next_condensed_id += 1;
                let right_id = next_condensed_id;
                next_condensed_id += 1;
                relabel[left] = left_id;
                relabel[right] = right_id;

                condensed.push(CondensedNode {
                    parent: relabel[node],
                    child: left_id,
                    lambda_val: lambda,
                    child_size: left_size,
                });
                condensed.push(CondensedNode {
                    parent: relabel[node],
                    child: right_id,
                    lambda_val: lambda,
                    child_size: right_size,
                });

                stack.push(left);
                stack.push(right);
            }
            (true, false) => {
                // Right is too small: points fall out, left continues as parent
                relabel[left] = relabel[node];
                // Record falling-out points from right subtree
                add_falling_points(
                    &mut condensed,
                    right,
                    &children,
                    &node_dist,
                    relabel[node],
                    n,
                );
                stack.push(left);
            }
            (false, true) => {
                // Left is too small: points fall out, right continues as parent
                relabel[right] = relabel[node];
                add_falling_points(
                    &mut condensed,
                    left,
                    &children,
                    &node_dist,
                    relabel[node],
                    n,
                );
                stack.push(right);
            }
            (false, false) => {
                // Both too small: all points fall out
                add_falling_points(
                    &mut condensed,
                    left,
                    &children,
                    &node_dist,
                    relabel[node],
                    n,
                );
                add_falling_points(
                    &mut condensed,
                    right,
                    &children,
                    &node_dist,
                    relabel[node],
                    n,
                );
            }
        }
    }

    condensed
}

/// Recursively add "falling out" points from a subtree to the condensed tree.
fn add_falling_points(
    condensed: &mut Vec<CondensedNode>,
    node: usize,
    children: &[Vec<usize>],
    node_dist: &[f64],
    parent_cluster: usize,
    n: usize,
) {
    if node < n {
        // Leaf: single point falls out at parent's lambda
        condensed.push(CondensedNode {
            parent: parent_cluster,
            child: node, // point index
            lambda_val: if node_dist.get(node).copied().unwrap_or(0.0) > 1e-300 {
                1.0 / node_dist[node]
            } else {
                1e300
            },
            child_size: 1,
        });
    } else {
        // Internal node: recurse into children
        let lambda = if node_dist[node] > 1e-300 {
            1.0 / node_dist[node]
        } else {
            1e300
        };
        // Add all leaves with their falling-out lambda
        let mut leaf_stack = vec![node];
        while let Some(nd) = leaf_stack.pop() {
            if nd < n {
                condensed.push(CondensedNode {
                    parent: parent_cluster,
                    child: nd,
                    lambda_val: lambda,
                    child_size: 1,
                });
            } else if !children[nd].is_empty() {
                for &ch in &children[nd] {
                    leaf_stack.push(ch);
                }
            }
        }
    }
}

/// Extract clusters from condensed tree using Excess of Mass (EOM).
fn extract_eom_clusters(condensed: &[CondensedNode], n: usize) -> (Vec<Option<usize>>, Vec<f64>) {
    if condensed.is_empty() {
        return (vec![None; n], vec![0.0; n]);
    }

    // Identify all cluster IDs (non-point nodes)
    let mut cluster_ids: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for node in condensed {
        cluster_ids.insert(node.parent);
        if node.child_size > 1 {
            cluster_ids.insert(node.child);
        }
    }

    // Compute birth lambda for each cluster (minimum lambda among its condensed entries)
    let mut birth_lambda: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for node in condensed {
        let entry = birth_lambda.entry(node.parent).or_insert(1e300);
        if node.lambda_val < *entry {
            *entry = node.lambda_val;
        }
    }

    // Compute stability for each cluster
    // S(C) = Σ_p (λ_death(p) - λ_birth(C)) for individual points in C
    let mut stability: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for &cid in &cluster_ids {
        stability.insert(cid, 0.0);
    }

    for node in condensed {
        if node.child_size == 1 {
            // Individual point falling out
            let birth = birth_lambda.get(&node.parent).copied().unwrap_or(0.0);
            let death = node.lambda_val;
            let contrib = (death - birth).max(0.0);
            *stability.entry(node.parent).or_insert(0.0) += contrib;
        }
    }

    // Find parent-child relationships among clusters
    let mut cluster_children: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for node in condensed {
        if node.child_size > 1 && cluster_ids.contains(&node.child) {
            cluster_children
                .entry(node.parent)
                .or_default()
                .push(node.child);
        }
    }

    // Find leaf clusters (no children in condensed tree)
    let leaf_clusters: Vec<usize> = cluster_ids
        .iter()
        .filter(|&&cid| !cluster_children.contains_key(&cid))
        .copied()
        .collect();

    // EOM: bottom-up selection
    let mut selected = std::collections::HashSet::new();
    for &leaf in &leaf_clusters {
        selected.insert(leaf);
    }

    // Topological order: process children before parents
    // Simple approach: repeatedly process nodes whose children are all processed
    let mut processed = std::collections::HashSet::new();
    for &leaf in &leaf_clusters {
        processed.insert(leaf);
    }

    let mut changed = true;
    while changed {
        changed = false;
        for &cid in &cluster_ids {
            if processed.contains(&cid) {
                continue;
            }
            if let Some(ch) = cluster_children.get(&cid) {
                if ch.iter().all(|c| processed.contains(c)) {
                    // All children processed: compare stability
                    let s_parent = stability.get(&cid).copied().unwrap_or(0.0);
                    let s_children: f64 = ch
                        .iter()
                        .map(|c| stability.get(c).copied().unwrap_or(0.0))
                        .sum();

                    if s_parent > s_children {
                        // Parent more stable: select parent, deselect children subtrees
                        selected.insert(cid);
                        deselect_subtree(cid, &cluster_children, &mut selected);
                        selected.insert(cid); // re-insert after deselect_subtree
                                              // Propagate parent stability upward
                    } else {
                        // Children more stable: keep children selected
                        // Propagate children stability upward
                        *stability.entry(cid).or_insert(0.0) = s_children;
                    }
                    processed.insert(cid);
                    changed = true;
                }
            }
        }
    }

    // Assign labels: map selected cluster IDs to 0-based labels
    let mut selected_sorted: Vec<usize> = selected.iter().copied().collect();
    selected_sorted.sort();
    let label_map: std::collections::HashMap<usize, usize> = selected_sorted
        .iter()
        .enumerate()
        .map(|(i, &cid)| (cid, i))
        .collect();

    // Assign each point to its selected cluster
    // Point → which clusters contain it → find the deepest selected cluster
    let mut point_cluster: Vec<Option<usize>> = vec![None; n];
    let mut point_lambda: Vec<f64> = vec![0.0; n];

    for node in condensed {
        if node.child_size == 1 && node.child < n {
            // This point belongs to node.parent cluster
            // Walk up to find which selected cluster contains it
            let mut cluster = node.parent;
            while !selected.contains(&cluster) {
                // Find parent of this cluster in condensed tree
                let parent_of = condensed
                    .iter()
                    .find(|cn| cn.child == cluster && cn.child_size > 1)
                    .map(|cn| cn.parent);
                match parent_of {
                    Some(p) => cluster = p,
                    None => break,
                }
            }
            if selected.contains(&cluster) {
                if let Some(&label) = label_map.get(&cluster) {
                    point_cluster[node.child] = Some(label);
                    point_lambda[node.child] = node.lambda_val;
                }
            }
        }
    }

    // Compute probabilities
    // For each selected cluster, compute max lambda (max death lambda among its points)
    let mut cluster_max_lambda: std::collections::HashMap<usize, f64> =
        std::collections::HashMap::new();
    for (i, label) in point_cluster.iter().enumerate() {
        if let Some(l) = label {
            if let Some(&cid) = selected_sorted.get(*l) {
                let entry = cluster_max_lambda.entry(cid).or_insert(0.0_f64);
                if point_lambda[i] > *entry {
                    *entry = point_lambda[i];
                }
            }
        }
    }

    let mut probabilities = vec![0.0_f64; n];
    for (i, label) in point_cluster.iter().enumerate() {
        if let Some(l) = label {
            if let Some(&cid) = selected_sorted.get(*l) {
                let birth = birth_lambda.get(&cid).copied().unwrap_or(0.0);
                let max_lam = cluster_max_lambda.get(&cid).copied().unwrap_or(birth);
                let range = max_lam - birth;
                if range.is_finite() && range > 0.0 && point_lambda[i].is_finite() {
                    probabilities[i] = ((point_lambda[i] - birth) / range).clamp(0.0, 1.0);
                } else {
                    probabilities[i] = 1.0;
                }
            }
        }
    }

    (point_cluster, probabilities)
}

/// Deselect all descendants of a cluster.
fn deselect_subtree(
    cluster: usize,
    cluster_children: &std::collections::HashMap<usize, Vec<usize>>,
    selected: &mut std::collections::HashSet<usize>,
) {
    if let Some(ch) = cluster_children.get(&cluster) {
        for &c in ch {
            selected.remove(&c);
            deselect_subtree(c, cluster_children, selected);
        }
    }
}

// ── Cluster Validation Metrics ─────────────────────────────────────────

/// Computes the Calinski-Harabasz index (Variance Ratio Criterion).
///
/// Measures the ratio of between-cluster dispersion to within-cluster dispersion.
/// Higher values indicate better-defined clusters.
///
/// # Arguments
///
/// * `data` — data points, each inner Vec is one observation
/// * `labels` — cluster label for each point (0-indexed)
/// * `k` — number of clusters
///
/// # Returns
///
/// The CH score, or 0.0 if k < 2, n <= k, or all points are identical.
///
/// # Reference
///
/// Calinski & Harabasz (1974). "A dendrite method for cluster analysis."
///
/// ```
/// use u_insight::clustering::{kmeans, calinski_harabasz, KMeansConfig};
///
/// let data = vec![
///     vec![0.0, 0.0], vec![0.5, 0.5],
///     vec![10.0, 10.0], vec![10.5, 10.5],
/// ];
/// let result = kmeans(&data, &KMeansConfig::new(2)).unwrap();
/// let ch = calinski_harabasz(&data, &result.labels, result.k);
/// assert!(ch > 10.0); // well-separated clusters
/// ```
pub fn calinski_harabasz(data: &[Vec<f64>], labels: &[usize], k: usize) -> f64 {
    let n = data.len();
    if k < 2 || n <= k || data.is_empty() {
        return 0.0;
    }

    let dims = data[0].len();
    if dims == 0 {
        return 0.0;
    }

    // Global centroid
    let mut global = vec![0.0; dims];
    for point in data {
        for (g, &v) in global.iter_mut().zip(point.iter()) {
            *g += v;
        }
    }
    for g in &mut global {
        *g /= n as f64;
    }

    // Per-cluster centroids and sizes
    let mut centroids = vec![vec![0.0; dims]; k];
    let mut sizes = vec![0usize; k];
    for (i, point) in data.iter().enumerate() {
        let c = labels[i];
        if c >= k {
            continue;
        }
        sizes[c] += 1;
        for (cj, &v) in centroids[c].iter_mut().zip(point.iter()) {
            *cj += v;
        }
    }
    for c in 0..k {
        if sizes[c] > 0 {
            for cj in &mut centroids[c] {
                *cj /= sizes[c] as f64;
            }
        }
    }

    // BCSS (between-cluster sum of squares)
    let mut bcss = 0.0;
    for c in 0..k {
        if sizes[c] == 0 {
            continue;
        }
        let dist_sq: f64 = centroids[c]
            .iter()
            .zip(global.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        bcss += sizes[c] as f64 * dist_sq;
    }

    // WCSS (within-cluster sum of squares)
    let mut wcss = 0.0;
    for (i, point) in data.iter().enumerate() {
        let c = labels[i];
        if c >= k {
            continue;
        }
        let dist_sq: f64 = point
            .iter()
            .zip(centroids[c].iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        wcss += dist_sq;
    }

    if wcss < 1e-300 {
        return 0.0; // all points identical within clusters
    }

    (bcss / (k - 1) as f64) / (wcss / (n - k) as f64)
}

/// Computes the Davies-Bouldin index.
///
/// Measures the average similarity between each cluster and its most similar
/// cluster. Lower values indicate better clustering.
///
/// # Arguments
///
/// * `data` — data points, each inner Vec is one observation
/// * `labels` — cluster label for each point (0-indexed)
/// * `k` — number of clusters
///
/// # Returns
///
/// The DB score, or 0.0 if k < 2.
///
/// # Reference
///
/// Davies & Bouldin (1979). "A cluster separation measure."
///
/// ```
/// use u_insight::clustering::{kmeans, davies_bouldin, KMeansConfig};
///
/// let data = vec![
///     vec![0.0, 0.0], vec![0.5, 0.5],
///     vec![10.0, 10.0], vec![10.5, 10.5],
/// ];
/// let result = kmeans(&data, &KMeansConfig::new(2)).unwrap();
/// let db = davies_bouldin(&data, &result.labels, result.k);
/// assert!(db < 1.0); // well-separated clusters → low DB
/// ```
pub fn davies_bouldin(data: &[Vec<f64>], labels: &[usize], k: usize) -> f64 {
    let n = data.len();
    if k < 2 || n == 0 {
        return 0.0;
    }

    let dims = data[0].len();
    if dims == 0 {
        return 0.0;
    }

    // Per-cluster centroids and sizes
    let mut centroids = vec![vec![0.0; dims]; k];
    let mut sizes = vec![0usize; k];
    for (i, point) in data.iter().enumerate() {
        let c = labels[i];
        if c >= k {
            continue;
        }
        sizes[c] += 1;
        for (cj, &v) in centroids[c].iter_mut().zip(point.iter()) {
            *cj += v;
        }
    }
    for c in 0..k {
        if sizes[c] > 0 {
            for cj in &mut centroids[c] {
                *cj /= sizes[c] as f64;
            }
        }
    }

    // Scatter S_i: average Euclidean distance from points to centroid
    let mut scatter = vec![0.0; k];
    for (i, point) in data.iter().enumerate() {
        let c = labels[i];
        if c >= k {
            continue;
        }
        let dist: f64 = point
            .iter()
            .zip(centroids[c].iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        scatter[c] += dist;
    }
    for c in 0..k {
        if sizes[c] > 0 {
            scatter[c] /= sizes[c] as f64;
        }
    }

    // DB = (1/k) * Σ max_j≠i R_ij, where R_ij = (S_i + S_j) / d(c_i, c_j)
    let mut db_sum = 0.0;
    for i in 0..k {
        if sizes[i] == 0 {
            continue;
        }
        let mut max_ratio = 0.0f64;
        for j in 0..k {
            if i == j || sizes[j] == 0 {
                continue;
            }
            let centroid_dist: f64 = centroids[i]
                .iter()
                .zip(centroids[j].iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            let ratio = if centroid_dist < 1e-300 {
                // Identical centroids — use large ratio
                (scatter[i] + scatter[j]) * 1e10
            } else {
                (scatter[i] + scatter[j]) / centroid_dist
            };
            max_ratio = max_ratio.max(ratio);
        }
        db_sum += max_ratio;
    }

    db_sum / k as f64
}

// ── Mini-Batch K-Means ───────────────────────────────────────────────

/// Configuration for Mini-Batch K-Means.
#[derive(Debug, Clone)]
pub struct MiniBatchKMeansConfig {
    /// Number of clusters.
    pub k: usize,
    /// Batch size (number of samples per mini-batch). Default: 100.
    pub batch_size: usize,
    /// Maximum iterations. Default: 100.
    pub max_iter: usize,
    /// Convergence tolerance. Default: 1e-4.
    pub tol: f64,
    /// Random seed. Default: Some(42).
    pub seed: Option<u64>,
}

impl MiniBatchKMeansConfig {
    /// Creates a config with the given k and default batch size.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            batch_size: 100,
            max_iter: 100,
            tol: 1e-4,
            seed: Some(42),
        }
    }

    /// Sets the batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the random seed.
    pub fn seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }
}

/// Runs Mini-Batch K-Means clustering (Sculley, 2010).
///
/// Mini-Batch K-Means is a faster variant of K-Means that uses random
/// subsamples (mini-batches) instead of the full dataset for centroid
/// updates. It converges faster with slightly worse WCSS.
///
/// Reference: Sculley (2010). "Web-scale k-means clustering", WWW.
///
/// ```
/// use u_insight::clustering::{mini_batch_kmeans, MiniBatchKMeansConfig};
///
/// let data = vec![
///     vec![0.0, 0.0], vec![0.5, 0.5], vec![0.2, 0.3],
///     vec![10.0, 10.0], vec![10.5, 10.5], vec![10.2, 10.3],
/// ];
/// let config = MiniBatchKMeansConfig::new(2).batch_size(4);
/// let result = mini_batch_kmeans(&data, &config).unwrap();
/// assert_eq!(result.k, 2);
/// assert_eq!(result.labels.len(), 6);
/// ```
pub fn mini_batch_kmeans(
    data: &[Vec<f64>],
    config: &MiniBatchKMeansConfig,
) -> Result<KMeansResult, InsightError> {
    let n = data.len();
    let k = config.k;

    if n == 0 {
        return Err(InsightError::DegenerateData {
            reason: "no data points provided".into(),
        });
    }
    if k == 0 || k > n {
        return Err(InsightError::InvalidParameter {
            name: "k".into(),
            message: format!("must be between 1 and {n} (number of data points), got {k}"),
        });
    }

    let d = data[0].len();
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

    let batch_size = config.batch_size.min(n);

    // K-Means++ initialization on full dataset
    let mut centroids = kmeans_plus_plus(data, k, d, config.seed);

    // Per-centroid update count for learning rate
    let mut centroid_counts = vec![0u64; k];
    let mut rng_state = config.seed.unwrap_or(42);
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // Sample a mini-batch (Fisher-Yates partial shuffle indices)
        let batch_indices = sample_indices(n, batch_size, &mut rng_state);

        // Assign batch points to nearest centroid
        let mut batch_labels = vec![0usize; batch_size];
        for (bi, &idx) in batch_indices.iter().enumerate() {
            let point = &data[idx];
            let mut min_dist = f64::INFINITY;
            let mut best_c = 0;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist = euclidean_dist_sq(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_c = c;
                }
            }
            batch_labels[bi] = best_c;
        }

        // Update centroids using streaming average
        let old_centroids: Vec<Vec<f64>> = centroids.clone();

        for (bi, &idx) in batch_indices.iter().enumerate() {
            let c = batch_labels[bi];
            centroid_counts[c] += 1;
            let eta = 1.0 / centroid_counts[c] as f64;
            let point = &data[idx];
            for (j, &v) in point.iter().enumerate() {
                centroids[c][j] += eta * (v - centroids[c][j]);
            }
        }

        // Check convergence
        let max_shift: f64 = centroids
            .iter()
            .zip(old_centroids.iter())
            .map(|(new, old)| euclidean_dist(new, old))
            .fold(0.0f64, f64::max);

        if max_shift < config.tol {
            break;
        }
    }

    // Final assignment on full dataset
    let mut labels = vec![0usize; n];
    let mut cluster_sizes = vec![0usize; k];
    let mut wcss = 0.0;

    for (i, point) in data.iter().enumerate() {
        let mut min_dist = f64::INFINITY;
        let mut best_c = 0;
        for (c, centroid) in centroids.iter().enumerate() {
            let dist = euclidean_dist_sq(point, centroid);
            if dist < min_dist {
                min_dist = dist;
                best_c = c;
            }
        }
        labels[i] = best_c;
        cluster_sizes[best_c] += 1;
        wcss += min_dist;
    }

    Ok(KMeansResult {
        k,
        centroids,
        labels,
        wcss,
        iterations,
        cluster_sizes,
    })
}

/// Simple LCG-based pseudorandom index sampling (no external dependency).
fn sample_indices(n: usize, count: usize, state: &mut u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    // Partial Fisher-Yates shuffle
    for i in 0..count {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = i + (*state >> 33) as usize % (n - i);
        indices.swap(i, j);
    }
    indices.truncate(count);
    indices
}

// ── Gap Statistic ────────────────────────────────────────────────────

/// Result of gap statistic computation.
#[derive(Debug, Clone)]
pub struct GapStatResult {
    /// Optimal K selected by the gap criterion.
    pub best_k: usize,
    /// Gap values for each K tested.
    pub gap_values: Vec<(usize, f64)>,
    /// Standard errors for each K.
    pub std_errors: Vec<(usize, f64)>,
}

/// Computes the gap statistic for optimal K selection (Tibshirani et al. 2001).
///
/// Compares within-cluster dispersion against a null reference distribution
/// generated by uniform sampling over the data's bounding box.
///
/// Gap(k) = E*[log(W_k)] - log(W_k)
///
/// Selects the smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}.
///
/// Reference: Tibshirani, Walther & Hastie (2001). "Estimating the number of
/// clusters in a data set via the gap statistic", JRSS-B.
///
/// # Arguments
///
/// * `data` — Input data points.
/// * `k_min` — Minimum K to test (>= 1).
/// * `k_max` — Maximum K to test.
/// * `n_refs` — Number of reference datasets (default: 10).
/// * `seed` — Random seed.
///
/// ```
/// use u_insight::clustering::gap_statistic;
///
/// let data = vec![
///     vec![0.0, 0.0], vec![0.5, 0.5], vec![0.2, 0.3],
///     vec![10.0, 10.0], vec![10.5, 10.5], vec![10.2, 10.3],
/// ];
/// let result = gap_statistic(&data, 1, 4, 10, 42).unwrap();
/// assert!(result.best_k >= 1 && result.best_k <= 4);
/// ```
pub fn gap_statistic(
    data: &[Vec<f64>],
    k_min: usize,
    k_max: usize,
    n_refs: usize,
    seed: u64,
) -> Result<GapStatResult, InsightError> {
    let n = data.len();
    if n == 0 {
        return Err(InsightError::DegenerateData {
            reason: "no data points provided".into(),
        });
    }
    if k_min < 1 || k_max < k_min {
        return Err(InsightError::InvalidParameter {
            name: "k_min/k_max".into(),
            message: format!("need 1 <= k_min <= k_max, got k_min={k_min}, k_max={k_max}"),
        });
    }

    let d = data[0].len();

    // Compute bounding box
    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];
    for point in data {
        for (j, &v) in point.iter().enumerate() {
            if v < mins[j] {
                mins[j] = v;
            }
            if v > maxs[j] {
                maxs[j] = v;
            }
        }
    }

    let actual_k_max = k_max.min(n);
    let n_refs = n_refs.max(1);
    let mut rng_state = seed;

    let mut gap_values = Vec::new();
    let mut std_errors = Vec::new();

    for k in k_min..=actual_k_max {
        // WCSS on actual data
        let config = KMeansConfig {
            k,
            max_iter: 100,
            tol: 1e-6,
            n_init: 3,
            seed: Some(seed),
        };
        let actual_wcss = match kmeans(data, &config) {
            Ok(r) => r.wcss,
            Err(_) => continue,
        };

        let log_w = if actual_wcss > 0.0 {
            actual_wcss.ln()
        } else {
            0.0
        };

        // Reference datasets: uniform over bounding box
        let mut ref_log_ws = Vec::with_capacity(n_refs);
        for b in 0..n_refs {
            let ref_data = generate_uniform_data(n, d, &mins, &maxs, &mut rng_state);

            let ref_config = KMeansConfig {
                k,
                max_iter: 100,
                tol: 1e-6,
                n_init: 1,
                seed: Some(seed.wrapping_add(b as u64 * 1000)),
            };
            if let Ok(r) = kmeans(&ref_data, &ref_config) {
                let lw = if r.wcss > 0.0 { r.wcss.ln() } else { 0.0 };
                ref_log_ws.push(lw);
            }
        }

        if ref_log_ws.is_empty() {
            continue;
        }

        let b_count = ref_log_ws.len() as f64;
        let mean_ref_log_w: f64 = ref_log_ws.iter().sum::<f64>() / b_count;

        let gap = mean_ref_log_w - log_w;

        // Standard deviation
        let variance: f64 = ref_log_ws
            .iter()
            .map(|&lw| (lw - mean_ref_log_w).powi(2))
            .sum::<f64>()
            / b_count;
        let s_k = variance.sqrt() * (1.0 + 1.0 / b_count).sqrt();

        gap_values.push((k, gap));
        std_errors.push((k, s_k));
    }

    // Selection criterion: smallest k where Gap(k) >= Gap(k+1) - s_{k+1}
    let best_k = if gap_values.len() < 2 {
        gap_values.first().map_or(k_min, |(k, _)| *k)
    } else {
        let mut selected = gap_values.last().map_or(k_min, |(k, _)| *k);
        for i in 0..gap_values.len() - 1 {
            let gap_k = gap_values[i].1;
            let gap_next = gap_values[i + 1].1;
            let s_next = std_errors[i + 1].1;

            if gap_k >= gap_next - s_next {
                selected = gap_values[i].0;
                break;
            }
        }
        selected
    };

    Ok(GapStatResult {
        best_k,
        gap_values,
        std_errors,
    })
}

/// Generates uniform random data within the given bounding box.
fn generate_uniform_data(
    n: usize,
    d: usize,
    mins: &[f64],
    maxs: &[f64],
    state: &mut u64,
) -> Vec<Vec<f64>> {
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let mut point = Vec::with_capacity(d);
        for j in 0..d {
            // LCG for random float in [min, max]
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (*state >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
            point.push(mins[j] + u * (maxs[j] - mins[j]));
        }
        data.push(point);
    }
    data
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_clusters() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![0.2, 0.3],
            vec![0.1, 0.4],
            vec![10.0, 10.0],
            vec![10.5, 10.5],
            vec![10.2, 10.3],
            vec![10.1, 10.4],
        ]
    }

    fn make_three_clusters() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![0.2, 0.3],
            vec![10.0, 0.0],
            vec![10.5, 0.5],
            vec![10.2, 0.3],
            vec![5.0, 10.0],
            vec![5.5, 10.5],
            vec![5.2, 10.3],
        ]
    }

    // ── Basic K-Means ────────────────────────────────────────────

    #[test]
    fn kmeans_two_clusters() {
        let data = make_two_clusters();
        let result = kmeans(&data, &KMeansConfig::new(2)).unwrap();

        assert_eq!(result.k, 2);
        assert_eq!(result.labels.len(), 8);

        // First 4 points should be in same cluster
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[0], result.labels[2]);
        assert_eq!(result.labels[0], result.labels[3]);

        // Last 4 points should be in same cluster
        assert_eq!(result.labels[4], result.labels[5]);
        assert_eq!(result.labels[4], result.labels[6]);
        assert_eq!(result.labels[4], result.labels[7]);

        // Different clusters
        assert_ne!(result.labels[0], result.labels[4]);
    }

    #[test]
    fn kmeans_three_clusters() {
        let data = make_three_clusters();
        let result = kmeans(&data, &KMeansConfig::new(3)).unwrap();

        assert_eq!(result.k, 3);

        // Each group of 3 should share a label
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[0], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[3], result.labels[5]);
        assert_eq!(result.labels[6], result.labels[7]);
        assert_eq!(result.labels[6], result.labels[8]);

        // All three groups should have different labels
        assert_ne!(result.labels[0], result.labels[3]);
        assert_ne!(result.labels[0], result.labels[6]);
        assert_ne!(result.labels[3], result.labels[6]);
    }

    #[test]
    fn kmeans_single_cluster() {
        let data = vec![vec![1.0, 2.0], vec![1.5, 2.5], vec![1.2, 2.3]];
        let result = kmeans(&data, &KMeansConfig::new(1)).unwrap();
        assert_eq!(result.k, 1);
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn kmeans_cluster_sizes() {
        let data = make_two_clusters();
        let result = kmeans(&data, &KMeansConfig::new(2)).unwrap();
        assert_eq!(result.cluster_sizes.iter().sum::<usize>(), 8);
        // Each cluster should have 4 points
        let mut sizes = result.cluster_sizes.clone();
        sizes.sort();
        assert_eq!(sizes, vec![4, 4]);
    }

    #[test]
    fn kmeans_wcss_decreases_with_k() {
        let data = make_three_clusters();
        let r2 = kmeans(&data, &KMeansConfig::new(2)).unwrap();
        let r3 = kmeans(&data, &KMeansConfig::new(3)).unwrap();
        assert!(r3.wcss < r2.wcss, "WCSS should decrease with more clusters");
    }

    // ── Error cases ──────────────────────────────────────────────

    #[test]
    fn kmeans_empty_data() {
        let data: Vec<Vec<f64>> = vec![];
        assert!(kmeans(&data, &KMeansConfig::new(2)).is_err());
    }

    #[test]
    fn kmeans_k_zero() {
        let data = vec![vec![1.0]];
        assert!(kmeans(&data, &KMeansConfig::new(0)).is_err());
    }

    #[test]
    fn kmeans_k_exceeds_n() {
        let data = vec![vec![1.0], vec![2.0]];
        assert!(kmeans(&data, &KMeansConfig::new(3)).is_err());
    }

    #[test]
    fn kmeans_nan_rejected() {
        let data = vec![vec![1.0, f64::NAN]];
        assert!(kmeans(&data, &KMeansConfig::new(1)).is_err());
    }

    #[test]
    fn kmeans_dimension_mismatch() {
        let data = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(kmeans(&data, &KMeansConfig::new(1)).is_err());
    }

    // ── Silhouette score ─────────────────────────────────────────

    #[test]
    fn silhouette_well_separated() {
        let data = make_two_clusters();
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let score = silhouette_score(&data, &labels, 2);
        assert!(
            score > 0.9,
            "well-separated clusters should have high silhouette: {score}"
        );
    }

    #[test]
    fn silhouette_poor_clustering() {
        let data = make_two_clusters();
        // Assign alternating labels (bad clustering)
        let labels = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let score = silhouette_score(&data, &labels, 2);
        assert!(
            score < 0.5,
            "poor clustering should have low silhouette: {score}"
        );
    }

    #[test]
    fn silhouette_single_cluster() {
        let data = vec![vec![1.0], vec![2.0]];
        let labels = vec![0, 0];
        let score = silhouette_score(&data, &labels, 1);
        assert_eq!(score, 0.0);
    }

    // ── Auto-K ───────────────────────────────────────────────────

    #[test]
    fn auto_k_finds_correct_clusters() {
        let data = make_three_clusters();
        let result = auto_kmeans(&data, 2, 5, &KMeansConfig::new(2)).unwrap();

        // Should find k=3 as optimal
        assert_eq!(result.best_k, 3, "should identify 3 clusters");
        assert!(result.best_silhouette > 0.5);
        assert_eq!(result.silhouette_scores.len(), 4); // k=2,3,4,5
        assert_eq!(result.wcss_values.len(), 4);
    }

    #[test]
    fn auto_k_wcss_monotonically_decreasing() {
        let data = make_three_clusters();
        let result = auto_kmeans(&data, 2, 5, &KMeansConfig::new(2)).unwrap();

        for i in 1..result.wcss_values.len() {
            assert!(
                result.wcss_values[i].1 <= result.wcss_values[i - 1].1 + 1e-10,
                "WCSS should not increase: k={} wcss={} > k={} wcss={}",
                result.wcss_values[i].0,
                result.wcss_values[i].1,
                result.wcss_values[i - 1].0,
                result.wcss_values[i - 1].1,
            );
        }
    }

    #[test]
    fn auto_k_min_exceeds_max() {
        let data = make_two_clusters();
        assert!(auto_kmeans(&data, 5, 3, &KMeansConfig::new(2)).is_err());
    }

    // ── Reproducibility ──────────────────────────────────────────

    #[test]
    fn kmeans_reproducible_with_seed() {
        let data = make_two_clusters();
        let config = KMeansConfig::new(2).seed(Some(42));
        let r1 = kmeans(&data, &config).unwrap();
        let r2 = kmeans(&data, &config).unwrap();
        assert_eq!(r1.labels, r2.labels);
        assert!((r1.wcss - r2.wcss).abs() < 1e-10);
    }

    // ── 1-D data ─────────────────────────────────────────────────

    #[test]
    fn kmeans_1d_data() {
        let data = vec![
            vec![1.0],
            vec![1.5],
            vec![2.0],
            vec![10.0],
            vec![10.5],
            vec![11.0],
        ];
        let result = kmeans(&data, &KMeansConfig::new(2)).unwrap();
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[0], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[3], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    // ── High-dimensional ─────────────────────────────────────────

    #[test]
    fn kmeans_high_dim() {
        // 10-dimensional data, 2 clusters
        let mut data = Vec::new();
        for _ in 0..5 {
            data.push(vec![0.0; 10]);
        }
        for _ in 0..5 {
            data.push(vec![10.0; 10]);
        }
        let result = kmeans(&data, &KMeansConfig::new(2)).unwrap();
        assert_eq!(result.labels[0], result.labels[4]);
        assert_eq!(result.labels[5], result.labels[9]);
        assert_ne!(result.labels[0], result.labels[5]);
    }

    // ── DBSCAN tests ─────────────────────────────────────────────

    #[test]
    fn dbscan_two_clusters_with_noise() {
        let data = vec![
            // Cluster A
            vec![0.0, 0.0],
            vec![0.5, 0.0],
            vec![0.0, 0.5],
            vec![0.5, 0.5],
            // Cluster B
            vec![10.0, 10.0],
            vec![10.5, 10.0],
            vec![10.0, 10.5],
            vec![10.5, 10.5],
            // Noise
            vec![50.0, 50.0],
        ];
        let result = dbscan(&data, &DbscanConfig::new(1.5, 2)).unwrap();

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.noise_count, 1);

        // First 4 in same cluster
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[0], result.labels[2]);
        assert_eq!(result.labels[0], result.labels[3]);

        // Next 4 in same cluster
        assert_eq!(result.labels[4], result.labels[5]);
        assert_eq!(result.labels[4], result.labels[6]);
        assert_eq!(result.labels[4], result.labels[7]);

        // Different clusters
        assert_ne!(result.labels[0], result.labels[4]);

        // Noise point
        assert!(result.labels[8].is_none());
    }

    #[test]
    fn dbscan_all_noise() {
        // Points too far apart for any cluster
        let data = vec![vec![0.0, 0.0], vec![100.0, 0.0], vec![0.0, 100.0]];
        let result = dbscan(&data, &DbscanConfig::new(1.0, 3)).unwrap();

        assert_eq!(result.n_clusters, 0);
        assert_eq!(result.noise_count, 3);
        assert!(result.labels.iter().all(|l| l.is_none()));
    }

    #[test]
    fn dbscan_single_cluster() {
        // All points close together
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
        ];
        let result = dbscan(&data, &DbscanConfig::new(0.5, 2)).unwrap();

        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.noise_count, 0);
        assert!(result.labels.iter().all(|l| *l == Some(0)));
    }

    #[test]
    fn dbscan_core_and_border_points() {
        // With min_samples=3, only densely connected points are core.
        let data = vec![
            vec![0.0], // border: neighbors [0,1] = 2 < 3
            vec![0.5], // core: neighbors [0,1,2] = 3
            vec![1.0], // core: neighbors [1,2,3] = 3
            vec![1.5], // border: neighbors [2,3] = 2 < 3
        ];
        let result = dbscan(&data, &DbscanConfig::new(0.6, 3)).unwrap();

        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.noise_count, 0);
        // Points 1,2 are core; points 0,3 are border
        assert!(!result.core_points[0]); // border
        assert!(result.core_points[1]); // core
        assert!(result.core_points[2]); // core
        assert!(!result.core_points[3]); // border
                                         // All assigned to same cluster
        assert!(result.labels.iter().all(|l| *l == Some(0)));
    }

    #[test]
    fn dbscan_chain_clustering() {
        // Chain of points: each point is close to its neighbors
        // DBSCAN should connect them all into one cluster
        let data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5]).collect();
        let result = dbscan(&data, &DbscanConfig::new(0.6, 2)).unwrap();

        assert_eq!(result.n_clusters, 1, "chain should form single cluster");
        assert_eq!(result.noise_count, 0);
    }

    #[test]
    fn dbscan_cluster_sizes_sum_correctly() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.0],
            vec![0.0, 0.5],
            vec![10.0, 10.0],
            vec![10.5, 10.0],
            vec![50.0, 50.0], // noise
        ];
        let result = dbscan(&data, &DbscanConfig::new(1.5, 2)).unwrap();

        let assigned: usize = result.cluster_sizes.iter().sum();
        assert_eq!(assigned + result.noise_count, data.len());
    }

    #[test]
    fn dbscan_empty_data() {
        let data: Vec<Vec<f64>> = vec![];
        assert!(dbscan(&data, &DbscanConfig::new(1.0, 2)).is_err());
    }

    #[test]
    fn dbscan_min_samples_too_small() {
        let data = vec![vec![1.0]];
        assert!(dbscan(&data, &DbscanConfig::new(1.0, 1)).is_err());
    }

    #[test]
    fn dbscan_invalid_epsilon() {
        let data = vec![vec![1.0]];
        assert!(dbscan(&data, &DbscanConfig::new(0.0, 2)).is_err());
        assert!(dbscan(&data, &DbscanConfig::new(-1.0, 2)).is_err());
        assert!(dbscan(&data, &DbscanConfig::new(f64::NAN, 2)).is_err());
    }

    #[test]
    fn dbscan_nan_rejected() {
        let data = vec![vec![1.0, f64::NAN]];
        assert!(dbscan(&data, &DbscanConfig::new(1.0, 2)).is_err());
    }

    #[test]
    fn dbscan_dimension_mismatch() {
        let data = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(dbscan(&data, &DbscanConfig::new(1.0, 2)).is_err());
    }

    #[test]
    fn dbscan_single_point() {
        let data = vec![vec![5.0, 5.0]];
        let result = dbscan(&data, &DbscanConfig::new(1.0, 2)).unwrap();
        assert_eq!(result.n_clusters, 0);
        assert_eq!(result.noise_count, 1);
        assert!(result.labels[0].is_none());
    }

    // ── Hierarchical Clustering ─────────────────────────────────────

    #[test]
    fn hierarchical_two_clusters_ward() {
        let data = make_two_clusters();
        let config = HierarchicalConfig::with_k(2).linkage(Linkage::Ward);
        let result = hierarchical(&data, &config).unwrap();

        assert_eq!(result.n_clusters, Some(2));
        let labels = result.labels.unwrap();
        assert_eq!(labels.len(), 8);
        // First 4 points should share a label, last 4 another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[4], labels[6]);
        assert_eq!(labels[4], labels[7]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn hierarchical_three_clusters_ward() {
        let data = make_three_clusters();
        let config = HierarchicalConfig::with_k(3).linkage(Linkage::Ward);
        let result = hierarchical(&data, &config).unwrap();

        assert_eq!(result.n_clusters, Some(3));
        let labels = result.labels.unwrap();
        // Each group of 3 should share a label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[6], labels[8]);
        // Groups should differ
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
        assert_ne!(labels[3], labels[6]);
    }

    #[test]
    fn hierarchical_single_linkage_chaining() {
        // Single linkage should chain nearby points into one cluster
        let data = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0], // chain
            vec![100.0],
            vec![101.0], // separate cluster
        ];
        let config = HierarchicalConfig::with_k(2).linkage(Linkage::Single);
        let result = hierarchical(&data, &config).unwrap();

        let labels = result.labels.unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn hierarchical_complete_linkage() {
        let data = make_two_clusters();
        let config = HierarchicalConfig::with_k(2).linkage(Linkage::Complete);
        let result = hierarchical(&data, &config).unwrap();

        let labels = result.labels.unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn hierarchical_average_linkage() {
        let data = make_two_clusters();
        let config = HierarchicalConfig::with_k(2).linkage(Linkage::Average);
        let result = hierarchical(&data, &config).unwrap();

        let labels = result.labels.unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn hierarchical_dendrogram_structure() {
        let data = make_two_clusters();
        let config = HierarchicalConfig::with_k(2).linkage(Linkage::Ward);
        let result = hierarchical(&data, &config).unwrap();

        // n-1 merges for n points
        assert_eq!(result.merges.len(), 7);
        // Last merge should contain all points
        assert_eq!(result.merges.last().unwrap().size, 8);
        // Merge distances should be non-decreasing
        for i in 1..result.merges.len() {
            assert!(
                result.merges[i].distance >= result.merges[i - 1].distance - 1e-10,
                "merge distances should be non-decreasing: {} < {}",
                result.merges[i].distance,
                result.merges[i - 1].distance
            );
        }
    }

    #[test]
    fn hierarchical_cut_by_threshold() {
        let data = make_two_clusters();
        let config = HierarchicalConfig::with_threshold(5.0).linkage(Linkage::Single);
        let result = hierarchical(&data, &config).unwrap();

        let labels = result.labels.unwrap();
        // Within-cluster distances are < 1, between-cluster > 10
        // Threshold 5.0 should separate into 2 clusters
        assert_eq!(result.n_clusters, Some(2));
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn hierarchical_no_cut() {
        let data = make_two_clusters();
        let config = HierarchicalConfig {
            linkage: Linkage::Ward,
            n_clusters: None,
            distance_threshold: None,
        };
        let result = hierarchical(&data, &config).unwrap();

        assert!(result.labels.is_none());
        assert!(result.n_clusters.is_none());
        assert_eq!(result.merges.len(), 7);
    }

    #[test]
    fn hierarchical_k_equals_n() {
        let data = make_two_clusters();
        let config = HierarchicalConfig::with_k(8).linkage(Linkage::Ward);
        let result = hierarchical(&data, &config).unwrap();

        let labels = result.labels.unwrap();
        // Each point is its own cluster
        let unique: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique.len(), 8);
    }

    #[test]
    fn hierarchical_k_equals_one() {
        let data = make_two_clusters();
        let config = HierarchicalConfig::with_k(1).linkage(Linkage::Ward);
        let result = hierarchical(&data, &config).unwrap();

        let labels = result.labels.unwrap();
        // All points in one cluster
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn hierarchical_errors() {
        // Too few points
        assert!(hierarchical(&[vec![1.0]], &HierarchicalConfig::with_k(1)).is_err());
        // k = 0
        assert!(hierarchical(&[vec![1.0], vec![2.0]], &HierarchicalConfig::with_k(0)).is_err());
        // k > n
        assert!(hierarchical(&[vec![1.0], vec![2.0]], &HierarchicalConfig::with_k(5)).is_err());
        // NaN data
        assert!(
            hierarchical(&[vec![f64::NAN], vec![1.0]], &HierarchicalConfig::with_k(1)).is_err()
        );
        // Dimension mismatch
        assert!(
            hierarchical(&[vec![1.0, 2.0], vec![3.0]], &HierarchicalConfig::with_k(1)).is_err()
        );
    }

    #[test]
    fn hierarchical_two_points() {
        let data = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let config = HierarchicalConfig::with_k(1).linkage(Linkage::Ward);
        let result = hierarchical(&data, &config).unwrap();

        assert_eq!(result.merges.len(), 1);
        assert!((result.merges[0].distance - 5.0).abs() < 1e-10);
        assert_eq!(result.merges[0].size, 2);
    }

    // ── HDBSCAN ────────────────────────────────────────────────────

    #[test]
    fn hdbscan_two_dense_clusters_with_noise() {
        let data = vec![
            // Cluster 1: tight group around (0,0)
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            vec![0.02, 0.08],
            vec![0.08, 0.02],
            // Cluster 2: tight group around (10,10)
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
            vec![10.1, 10.1],
            vec![10.05, 10.05],
            vec![10.02, 10.08],
            vec![10.08, 10.02],
            // Noise point far away
            vec![50.0, 50.0],
        ];

        let result = hdbscan(&data, &HdbscanConfig::new(3)).unwrap();

        // Should find 2 clusters
        assert!(
            result.n_clusters >= 2,
            "expected >=2 clusters, got {}",
            result.n_clusters
        );
        // Last point should be noise
        assert!(result.labels[14].is_none(), "outlier should be noise");
        assert!(result.noise_count >= 1);
        // Points in cluster 1 should share a label
        let c0 = result.labels[0];
        assert!(c0.is_some(), "core cluster point should have label");
        // Points in cluster 2 should share a different label
        let c1 = result.labels[7];
        assert!(c1.is_some(), "core cluster point should have label");
        assert_ne!(c0, c1, "two clusters should have different labels");
    }

    #[test]
    fn hdbscan_all_clustered() {
        // Dense blob: all points should cluster
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
        ];
        let result = hdbscan(&data, &HdbscanConfig::new(2)).unwrap();

        // Should find at least 1 cluster
        assert!(result.n_clusters >= 1);
        // Most points should be assigned
        let assigned = result.labels.iter().filter(|l| l.is_some()).count();
        assert!(assigned >= 3, "at least 3 points should be in clusters");
    }

    #[test]
    fn hdbscan_probabilities_range() {
        let mut data = Vec::new();
        for i in 0..10 {
            data.push(vec![(i as f64) * 0.1, 0.0]);
        }
        for i in 0..10 {
            data.push(vec![10.0 + (i as f64) * 0.1, 0.0]);
        }
        let result = hdbscan(&data, &HdbscanConfig::new(3)).unwrap();

        for &p in &result.probabilities {
            assert!((0.0..=1.0).contains(&p), "probability {p} out of range");
        }
    }

    #[test]
    fn hdbscan_min_samples_parameter() {
        let mut data = Vec::new();
        for i in 0..10 {
            data.push(vec![(i as f64) * 0.1, 0.0]);
        }
        for i in 0..10 {
            data.push(vec![10.0 + (i as f64) * 0.1, 0.0]);
        }
        data.push(vec![50.0, 0.0]); // noise

        let r1 = hdbscan(&data, &HdbscanConfig::new(3)).unwrap();
        let r2 = hdbscan(&data, &HdbscanConfig::new(3).min_samples(5)).unwrap();

        // Both should produce valid results
        assert!(r1.n_clusters >= 1);
        assert!(r2.n_clusters >= 1);
        assert_eq!(r1.labels.len(), 21);
        assert_eq!(r2.labels.len(), 21);
    }

    #[test]
    fn hdbscan_errors() {
        // Too few points
        assert!(hdbscan(&[vec![1.0]], &HdbscanConfig::new(2)).is_err());
        // min_cluster_size < 2
        assert!(hdbscan(&[vec![1.0], vec![2.0]], &HdbscanConfig::new(1)).is_err());
        // NaN data
        assert!(hdbscan(&[vec![f64::NAN], vec![1.0]], &HdbscanConfig::new(2)).is_err());
        // Dimension mismatch
        assert!(hdbscan(&[vec![1.0, 2.0], vec![3.0]], &HdbscanConfig::new(2)).is_err());
    }

    #[test]
    fn hdbscan_cluster_sizes_correct() {
        let mut data = Vec::new();
        for i in 0..10 {
            data.push(vec![(i as f64) * 0.1, 0.0]);
        }
        for i in 0..10 {
            data.push(vec![10.0 + (i as f64) * 0.1, 0.0]);
        }
        let result = hdbscan(&data, &HdbscanConfig::new(3)).unwrap();

        let total_assigned: usize = result.cluster_sizes.iter().sum();
        let total_from_labels = result.labels.iter().filter(|l| l.is_some()).count();
        assert_eq!(total_assigned, total_from_labels);
        assert_eq!(result.noise_count + total_assigned, data.len());
    }

    #[test]
    fn dbscan_arbitrary_shape() {
        // L-shaped cluster: DBSCAN should handle non-convex shapes
        let mut data = Vec::new();
        // Horizontal arm
        for i in 0..10 {
            data.push(vec![i as f64 * 0.3, 0.0]);
        }
        // Vertical arm
        for i in 1..10 {
            data.push(vec![0.0, i as f64 * 0.3]);
        }
        // Distant noise
        data.push(vec![20.0, 20.0]);

        let result = dbscan(&data, &DbscanConfig::new(0.5, 2)).unwrap();

        // Should be 1 cluster (L-shape connected) + 1 noise point
        assert_eq!(result.n_clusters, 1, "L-shape should be 1 cluster");
        assert_eq!(result.noise_count, 1);
        assert!(result.labels.last().unwrap().is_none());
    }

    // ── Calinski-Harabasz ────────────────────────────────────────

    #[test]
    fn calinski_harabasz_well_separated() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.2, 10.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let ch = calinski_harabasz(&data, &labels, 2);
        assert!(
            ch > 50.0,
            "CH = {ch}; well-separated clusters should score high"
        );
    }

    #[test]
    fn calinski_harabasz_with_kmeans() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
        ];
        let result = kmeans(&data, &KMeansConfig::new(2)).unwrap();
        let ch = calinski_harabasz(&data, &result.labels, result.k);
        assert!(ch > 5.0, "CH = {ch}");
    }

    #[test]
    fn calinski_harabasz_single_cluster() {
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        assert_eq!(calinski_harabasz(&data, &[0, 0, 0], 1), 0.0);
    }

    #[test]
    fn calinski_harabasz_k_equals_n() {
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        assert_eq!(calinski_harabasz(&data, &[0, 1, 2], 3), 0.0);
    }

    #[test]
    fn calinski_harabasz_empty() {
        let data: Vec<Vec<f64>> = vec![];
        assert_eq!(calinski_harabasz(&data, &[], 0), 0.0);
    }

    // ── Davies-Bouldin ───────────────────────────────────────────

    #[test]
    fn davies_bouldin_well_separated() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.2, 10.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let db = davies_bouldin(&data, &labels, 2);
        assert!(
            db < 0.5,
            "DB = {db}; well-separated clusters should have low DB"
        );
        assert!(db >= 0.0);
    }

    #[test]
    fn davies_bouldin_with_kmeans() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
        ];
        let result = kmeans(&data, &KMeansConfig::new(2)).unwrap();
        let db = davies_bouldin(&data, &result.labels, result.k);
        assert!(db < 1.0, "DB = {db}");
        assert!(db >= 0.0);
    }

    #[test]
    fn davies_bouldin_single_cluster() {
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        assert_eq!(davies_bouldin(&data, &[0, 0, 0], 1), 0.0);
    }

    #[test]
    fn davies_bouldin_empty() {
        let data: Vec<Vec<f64>> = vec![];
        assert_eq!(davies_bouldin(&data, &[], 0), 0.0);
    }

    #[test]
    fn davies_bouldin_overlapping_clusters() {
        // Overlapping clusters should have higher DB
        let data = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![2.5],
            vec![3.0],
            vec![4.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let db = davies_bouldin(&data, &labels, 2);
        assert!(
            db > 0.5,
            "DB = {db}; overlapping clusters should have higher DB"
        );
    }

    // ── Mini-Batch K-Means ──────────────────────────────────────

    #[test]
    fn mini_batch_two_clusters() {
        let data = make_two_clusters();
        let config = MiniBatchKMeansConfig::new(2).batch_size(4);
        let result = mini_batch_kmeans(&data, &config).unwrap();

        assert_eq!(result.k, 2);
        assert_eq!(result.labels.len(), 8);
        // First 4 in same cluster, last 4 in another
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[4], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[4]);
    }

    #[test]
    fn mini_batch_three_clusters() {
        let data = make_three_clusters();
        let config = MiniBatchKMeansConfig::new(3).batch_size(6);
        let result = mini_batch_kmeans(&data, &config).unwrap();

        assert_eq!(result.k, 3);
        assert_eq!(result.labels.len(), 9);
        assert!(result.wcss > 0.0);
        // All cluster sizes should be non-zero
        assert!(result.cluster_sizes.iter().all(|&s| s > 0));
    }

    #[test]
    fn mini_batch_batch_larger_than_n() {
        // Batch size > n: should still work (clamped)
        let data = make_two_clusters();
        let config = MiniBatchKMeansConfig::new(2).batch_size(1000);
        let result = mini_batch_kmeans(&data, &config).unwrap();
        assert_eq!(result.k, 2);
        assert_eq!(result.labels.len(), 8);
    }

    #[test]
    fn mini_batch_single_cluster() {
        let data = vec![vec![1.0, 2.0], vec![1.5, 2.5], vec![1.2, 2.3]];
        let config = MiniBatchKMeansConfig::new(1).batch_size(2);
        let result = mini_batch_kmeans(&data, &config).unwrap();
        assert_eq!(result.k, 1);
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn mini_batch_error_empty() {
        let data: Vec<Vec<f64>> = vec![];
        assert!(mini_batch_kmeans(&data, &MiniBatchKMeansConfig::new(2)).is_err());
    }

    #[test]
    fn mini_batch_error_nan() {
        let data = vec![vec![1.0, f64::NAN]];
        assert!(mini_batch_kmeans(&data, &MiniBatchKMeansConfig::new(1)).is_err());
    }

    // ── Gap statistic ───────────────────────────────────────────

    #[test]
    fn gap_two_clusters() {
        let data = make_two_clusters();
        let result = gap_statistic(&data, 1, 4, 5, 42).unwrap();
        assert!(result.best_k >= 1 && result.best_k <= 4);
        assert!(!result.gap_values.is_empty());
        assert_eq!(result.gap_values.len(), result.std_errors.len());
    }

    #[test]
    fn gap_three_clusters() {
        let data = make_three_clusters();
        let result = gap_statistic(&data, 1, 5, 5, 42).unwrap();
        assert!(result.best_k >= 1 && result.best_k <= 5);
    }

    #[test]
    fn gap_values_structure() {
        let data = make_two_clusters();
        let result = gap_statistic(&data, 1, 3, 5, 42).unwrap();
        // Gap values should have entries for k=1,2,3
        for &(k, _) in &result.gap_values {
            assert!((1..=3).contains(&k));
        }
        // Std errors should be non-negative
        for &(_, s) in &result.std_errors {
            assert!(s >= 0.0, "std error should be non-negative: {s}");
        }
    }

    #[test]
    fn gap_error_empty() {
        let data: Vec<Vec<f64>> = vec![];
        assert!(gap_statistic(&data, 1, 3, 5, 42).is_err());
    }

    #[test]
    fn gap_error_invalid_range() {
        let data = make_two_clusters();
        assert!(gap_statistic(&data, 3, 1, 5, 42).is_err()); // k_max < k_min
    }
}
