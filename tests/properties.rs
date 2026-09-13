//! Property tests: what every clustering and projection must satisfy on any
//! input, not only on the well-separated examples the unit tests use.
//!
//! - A clustering assigns every point to exactly one cluster, or to noise,
//!   and the counts it reports are the counts of the labels it returns.
//! - K-Means returns centroids that are the means of their members, an
//!   objective that is what the labels and centroids say it is, and an
//!   objective that never goes up with more Lloyd iterations.
//! - Hierarchical clustering builds a complete dendrogram whose merge
//!   distances only rise, and under single linkage those distances are the
//!   minimum spanning tree.
//! - PCA explains variance ratios that lie in [0, 1], fall monotonically and
//!   add up to at most one; its loadings are orthonormal and its scores are
//!   the centred data projected onto them.

use proptest::prelude::*;
use u_insight::clustering::{
    dbscan, hdbscan, hierarchical, kmeans, DbscanConfig, HdbscanConfig, HierarchicalConfig,
    KMeansConfig, Linkage,
};
use u_insight::pca::{pca, PcaConfig};

const EPS: f64 = 1e-9;

/// `n` points in `d` dimensions, coordinates in [-10, 10].
fn points(
    n: impl Strategy<Value = usize>,
    d: impl Strategy<Value = usize>,
) -> impl Strategy<Value = Vec<Vec<f64>>> {
    (n, d)
        .prop_flat_map(|(n, d)| prop::collection::vec(prop::collection::vec(-10.0f64..10.0, d), n))
}

fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn mean_of(rows: &[&Vec<f64>], d: usize) -> Vec<f64> {
    let mut m = vec![0.0; d];
    for r in rows {
        for (j, v) in r.iter().enumerate() {
            m[j] += v;
        }
    }
    for v in &mut m {
        *v /= rows.len() as f64;
    }
    m
}

/// Sum of squared distances to the grand mean.
fn total_ss(data: &[Vec<f64>]) -> f64 {
    let all: Vec<&Vec<f64>> = data.iter().collect();
    let m = mean_of(&all, data[0].len());
    data.iter().map(|p| dist_sq(p, &m)).sum()
}

/// Labels of an `Option<usize>` clustering: every `Some` is a valid cluster
/// id, the reported counts are the counts of the labels, and every point is
/// counted exactly once.
fn check_optional_partition(
    labels: &[Option<usize>],
    n_clusters: usize,
    noise_count: usize,
    cluster_sizes: &[usize],
) -> Result<(), TestCaseError> {
    prop_assert_eq!(cluster_sizes.len(), n_clusters);
    let mut sizes = vec![0usize; n_clusters];
    let mut noise = 0usize;
    for l in labels {
        match l {
            Some(c) => {
                prop_assert!(*c < n_clusters, "label {c} outside 0..{n_clusters}");
                sizes[*c] += 1;
            }
            None => noise += 1,
        }
    }
    prop_assert_eq!(noise, noise_count);
    prop_assert_eq!(&sizes, cluster_sizes);
    prop_assert_eq!(sizes.iter().sum::<usize>() + noise, labels.len());
    for (c, &s) in sizes.iter().enumerate() {
        prop_assert!(s > 0, "cluster {c} is reported but empty");
    }
    Ok(())
}

/// Weight of the Euclidean minimum spanning tree (Prim).
fn mst_weight(data: &[Vec<f64>]) -> f64 {
    let n = data.len();
    let mut in_tree = vec![false; n];
    let mut best = vec![f64::INFINITY; n];
    best[0] = 0.0;
    let mut total = 0.0;
    for _ in 0..n {
        let u = (0..n)
            .filter(|&i| !in_tree[i])
            .min_by(|&a, &b| best[a].partial_cmp(&best[b]).expect("finite distances"))
            .expect("a vertex remains");
        in_tree[u] = true;
        total += best[u];
        for v in 0..n {
            if !in_tree[v] {
                best[v] = best[v].min(dist_sq(&data[u], &data[v]).sqrt());
            }
        }
    }
    total
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Every point gets a label below k, the sizes are the label counts, each
    /// non-empty centroid is the mean of its members, `wcss` is the sum of
    /// squared distances the labels and centroids imply, and no partition
    /// into cluster means has more within-cluster scatter than the total.
    #[test]
    fn kmeans_returns_means_of_its_own_partition(
        data in points(2..=30usize, 1..=3usize),
        k_frac in 0.0f64..1.0,
    ) {
        let n = data.len();
        let d = data[0].len();
        let k = 1 + ((n - 1) as f64 * k_frac) as usize;
        let result = kmeans(&data, &KMeansConfig::new(k)).unwrap();

        prop_assert_eq!(result.k, k);
        prop_assert_eq!(result.labels.len(), n);
        prop_assert_eq!(result.centroids.len(), k);
        let mut sizes = vec![0usize; k];
        for &l in &result.labels {
            prop_assert!(l < k);
            sizes[l] += 1;
        }
        prop_assert_eq!(&sizes, &result.cluster_sizes);
        prop_assert_eq!(sizes.iter().sum::<usize>(), n);

        for c in 0..k {
            let members: Vec<&Vec<f64>> = data
                .iter()
                .zip(&result.labels)
                .filter(|(_, &l)| l == c)
                .map(|(p, _)| p)
                .collect();
            if members.is_empty() {
                continue;
            }
            let m = mean_of(&members, d);
            prop_assert!(
                dist_sq(&m, &result.centroids[c]).sqrt() < 1e-6,
                "centroid {c} is not the mean of its {} members", members.len()
            );
        }

        let wcss: f64 = data
            .iter()
            .zip(&result.labels)
            .map(|(p, &l)| dist_sq(p, &result.centroids[l]))
            .sum();
        prop_assert!(
            (wcss - result.wcss).abs() < 1e-6 * (1.0 + wcss),
            "wcss {} vs recomputed {wcss}", result.wcss
        );

        let tss = total_ss(&data);
        prop_assert!(
            result.wcss <= tss + 1e-6 * (1.0 + tss),
            "wcss {} exceeds total SS {tss}", result.wcss
        );
        if k == 1 {
            prop_assert!((result.wcss - tss).abs() < 1e-6 * (1.0 + tss));
        }
    }

    /// With one restart and a fixed seed, letting Lloyd's algorithm run one
    /// more iteration never raises the objective, and the same configuration
    /// always returns the same partition.
    #[test]
    fn kmeans_objective_never_rises_with_more_iterations(
        data in points(3..=30usize, 1..=3usize),
        k in 1..=4usize,
        seed in any::<u64>(),
    ) {
        let k = k.min(data.len());
        let mut previous = f64::INFINITY;
        for max_iter in 1..=8 {
            let config = KMeansConfig { k, max_iter, tol: 0.0, n_init: 1, seed: Some(seed) };
            let a = kmeans(&data, &config).unwrap();
            let b = kmeans(&data, &config).unwrap();
            prop_assert_eq!(&a.labels, &b.labels, "same seed, different partition");
            prop_assert!(
                a.wcss <= previous + 1e-9 * (1.0 + previous),
                "wcss rose from {previous} to {} at max_iter {max_iter}", a.wcss
            );
            prop_assert!(a.iterations <= max_iter);
            previous = a.wcss;
        }
    }

    /// DBSCAN and HDBSCAN: every point is in exactly one cluster or is noise,
    /// the counts are the label counts, a DBSCAN core point is exactly a
    /// point with `min_samples` neighbours (itself included) within epsilon
    /// and is never noise, every DBSCAN cluster holds a core point, and
    /// HDBSCAN probabilities lie in [0, 1] with noise at 0.
    #[test]
    fn density_clusterings_partition_into_clusters_and_noise(
        data in points(2..=30usize, 1..=3usize),
        epsilon in 0.5f64..8.0,
        min_samples in 2..=5usize,
    ) {
        let n = data.len();
        let db = dbscan(&data, &DbscanConfig::new(epsilon, min_samples)).unwrap();
        prop_assert_eq!(db.labels.len(), n);
        prop_assert_eq!(db.core_points.len(), n);
        check_optional_partition(&db.labels, db.n_clusters, db.noise_count, &db.cluster_sizes)?;
        let mut has_core = vec![false; db.n_clusters];
        for i in 0..n {
            let neighbours = data
                .iter()
                .filter(|q| dist_sq(&data[i], q).sqrt() <= epsilon)
                .count();
            prop_assert_eq!(db.core_points[i], neighbours >= min_samples, "core flag of point {}", i);
            if db.core_points[i] {
                let c = db.labels[i];
                prop_assert!(c.is_some(), "core point {i} labelled noise");
                has_core[c.unwrap()] = true;
            }
        }
        for (c, &h) in has_core.iter().enumerate() {
            prop_assert!(h, "cluster {c} has no core point");
        }

        let hd = hdbscan(&data, &HdbscanConfig::new(min_samples)).unwrap();
        prop_assert_eq!(hd.labels.len(), n);
        prop_assert_eq!(hd.probabilities.len(), n);
        check_optional_partition(&hd.labels, hd.n_clusters, hd.noise_count, &hd.cluster_sizes)?;
        for (i, (&l, &p)) in hd.labels.iter().zip(&hd.probabilities).enumerate() {
            prop_assert!((0.0..=1.0).contains(&p), "probability {p} of point {i}");
            if l.is_none() {
                prop_assert_eq!(p, 0.0, "noise point {} has probability {}", i, p);
            }
        }
    }

    /// The dendrogram has n − 1 merges with non-decreasing distances whose
    /// last merge covers every point; cutting it at k yields exactly k
    /// non-empty clusters; and under single linkage the merge distances sum
    /// to the weight of the Euclidean minimum spanning tree.
    #[test]
    fn hierarchical_dendrogram_is_complete_monotone_and_cuts_into_k(
        data in points(2..=25usize, 1..=3usize),
        k_frac in 0.0f64..1.0,
        linkage_idx in 0..4usize,
    ) {
        let n = data.len();
        let k = 1 + ((n - 1) as f64 * k_frac) as usize;
        let linkage =
            [Linkage::Single, Linkage::Complete, Linkage::Average, Linkage::Ward][linkage_idx];
        let result =
            hierarchical(&data, &HierarchicalConfig::with_k(k).linkage(linkage)).unwrap();

        prop_assert_eq!(result.merges.len(), n - 1);
        for w in result.merges.windows(2) {
            prop_assert!(
                w[1].distance >= w[0].distance - EPS,
                "merge distance fell from {} to {}", w[0].distance, w[1].distance
            );
        }
        if let Some(last) = result.merges.last() {
            prop_assert_eq!(last.size, n);
        }

        let labels = result.labels.as_ref().expect("with_k requests flat labels");
        prop_assert_eq!(result.n_clusters, Some(k));
        prop_assert_eq!(labels.len(), n);
        let mut sizes = vec![0usize; k];
        for &l in labels {
            prop_assert!(l < k, "label {l} outside 0..{k}");
            sizes[l] += 1;
        }
        prop_assert!(sizes.iter().all(|&s| s > 0), "a requested cluster is empty: {:?}", sizes);

        if linkage == Linkage::Single {
            let merged: f64 = result.merges.iter().map(|m| m.distance).sum();
            let mst = mst_weight(&data);
            prop_assert!(
                (merged - mst).abs() < 1e-6 * (1.0 + mst),
                "single linkage {merged} vs MST {mst}"
            );
        }
    }

    /// Explained-variance ratios lie in [0, 1], fall monotonically and sum to
    /// at most one (exactly one when every component is kept); eigenvalues
    /// are the score variances; loadings are orthonormal; scores are the
    /// centred (and scaled) data projected onto the loadings; and the kept
    /// eigenvalues never exceed the trace of the covariance matrix.
    #[test]
    fn pca_explained_variance_is_a_monotone_partition_of_the_total(
        data in points(3..=30usize, 1..=4usize),
        c_frac in 0.0f64..1.0,
        auto_scale in any::<bool>(),
    ) {
        let n = data.len();
        let d = data[0].len();
        let k = 1 + ((d - 1) as f64 * c_frac) as usize;
        let result = pca(&data, &PcaConfig::new(k).auto_scale(auto_scale)).unwrap();

        prop_assert_eq!(result.n_components, k);
        prop_assert_eq!(result.n_features, d);
        prop_assert_eq!(result.eigenvalues.len(), k);
        prop_assert_eq!(result.explained_variance_ratio.len(), k);
        prop_assert_eq!(result.cumulative_variance_ratio.len(), k);
        prop_assert_eq!(result.loadings.len(), k);
        prop_assert_eq!(result.scores.len(), n);

        let mut cum = 0.0;
        for i in 0..k {
            let r = result.explained_variance_ratio[i];
            prop_assert!((-EPS..=1.0 + EPS).contains(&r), "ratio {r} outside [0, 1]");
            if i > 0 {
                prop_assert!(
                    r <= result.explained_variance_ratio[i - 1] + EPS,
                    "ratios rise at {i}"
                );
                prop_assert!(
                    result.eigenvalues[i] <= result.eigenvalues[i - 1] + EPS,
                    "eigenvalues rise at {i}"
                );
            }
            prop_assert!(
                result.eigenvalues[i] >= -EPS,
                "negative eigenvalue {}", result.eigenvalues[i]
            );
            cum += r;
            prop_assert!((cum - result.cumulative_variance_ratio[i]).abs() < EPS);
        }
        prop_assert!(cum <= 1.0 + EPS, "ratios sum to {cum}");

        // Centred / scaled data, exactly as the result reports it.
        let centred: Vec<Vec<f64>> = data
            .iter()
            .map(|p| (0..d).map(|j| (p[j] - result.means[j]) / result.stds[j]).collect())
            .collect();
        let trace: f64 = (0..d)
            .map(|j| centred.iter().map(|p| p[j] * p[j]).sum::<f64>() / (n - 1) as f64)
            .sum();
        let kept: f64 = result.eigenvalues.iter().sum();
        prop_assert!(
            kept <= trace + 1e-6 * (1.0 + trace),
            "kept eigenvalues {kept} exceed trace {trace}"
        );
        if k == d && trace > 1e-9 {
            prop_assert!(
                (kept - trace).abs() < 1e-6 * (1.0 + trace),
                "all eigenvalues {kept} vs trace {trace}"
            );
            prop_assert!((cum - 1.0).abs() < 1e-6, "ratios of every component sum to {cum}");
        }

        for i in 0..k {
            for j in i..k {
                let dot: f64 = result.loadings[i]
                    .iter()
                    .zip(&result.loadings[j])
                    .map(|(a, b)| a * b)
                    .sum();
                let want = if i == j { 1.0 } else { 0.0 };
                prop_assert!((dot - want).abs() < 1e-6, "loadings {i}·{j} = {dot}");
            }
        }

        for (p, s) in centred.iter().zip(&result.scores) {
            prop_assert_eq!(s.len(), k);
            for (i, loading) in result.loadings.iter().enumerate() {
                let proj: f64 = p.iter().zip(loading).map(|(a, b)| a * b).sum();
                prop_assert!(
                    (proj - s[i]).abs() < 1e-6 * (1.0 + proj.abs()),
                    "score {i} is not the projection"
                );
            }
        }
        for i in 0..k {
            let mean = result.scores.iter().map(|s| s[i]).sum::<f64>() / n as f64;
            prop_assert!(mean.abs() < 1e-6, "score {i} mean {mean}");
            let var = result.scores.iter().map(|s| s[i] * s[i]).sum::<f64>() / (n - 1) as f64;
            prop_assert!(
                (var - result.eigenvalues[i]).abs() < 1e-6 * (1.0 + var),
                "score {i} variance {var} vs eigenvalue {}", result.eigenvalues[i]
            );
        }
    }
}
