//! The same observations in another unit are the same observations: the
//! clusters, the outliers and the variance each principal component explains
//! do not depend on whether the columns are in volts or in picovolts, and the
//! only setting in a unit of its own (DBSCAN's radius) goes with the data.

use u_insight::clustering::{
    calinski_harabasz, davies_bouldin, dbscan, hdbscan, kmeans, silhouette_score, DbscanConfig,
    HdbscanConfig, KMeansConfig,
};
use u_insight::isolation_forest::{isolation_forest, IsolationForestConfig};
use u_insight::lof::{lof, LofConfig};
use u_insight::mahalanobis::{mahalanobis, MahalanobisConfig};
use u_insight::pca::{pca, PcaConfig};

const SCALES: [f64; 6] = [1e-12, 1e-9, 1e-6, 1e6, 1e9, 1e12];

/// Three blobs of 30 points in three columns, and a distant point.
fn data() -> Vec<Vec<f64>> {
    let mut state = 5u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let centres = [[0.0, 0.0, 0.0], [6.0, 5.0, 1.0], [-5.0, 7.0, 4.0]];
    let mut rows = Vec::new();
    for c in centres {
        for _ in 0..30 {
            rows.push(vec![
                c[0] + 1.2 * next(),
                c[1] + 1.2 * next(),
                c[2] + 1.2 * next(),
            ]);
        }
    }
    rows.push(vec![30.0, -20.0, 25.0]);
    rows
}

fn scaled(rows: &[Vec<f64>], k: f64) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|r| r.iter().map(|v| v * k).collect())
        .collect()
}

fn close(got: f64, want: f64, what: &str, k: f64) {
    assert!(
        (got - want).abs() <= 1e-7 * want.abs().max(1e-3),
        "{what} at scale {k:e}: {got} vs {want}"
    );
}

#[test]
fn clusters_are_the_same_in_any_unit() {
    let rows = data();
    let km = kmeans(&rows, &KMeansConfig::new(3).seed(Some(7))).expect("kmeans");
    let db = dbscan(&rows, &DbscanConfig::new(2.0, 4)).expect("dbscan");
    let hd = hdbscan(&rows, &HdbscanConfig::new(8)).expect("hdbscan");
    let silhouette = silhouette_score(&rows, &km.labels, 3);
    let ch = calinski_harabasz(&rows, &km.labels, 3);
    let dbi = davies_bouldin(&rows, &km.labels, 3);
    assert!(db.n_clusters >= 2 && hd.n_clusters >= 2);
    for k in SCALES {
        let d = scaled(&rows, k);
        let got = kmeans(&d, &KMeansConfig::new(3).seed(Some(7))).expect("kmeans");
        assert_eq!(got.labels, km.labels, "k-means at scale {k:e}");
        let got = dbscan(&d, &DbscanConfig::new(2.0 * k, 4)).expect("dbscan");
        assert_eq!(got.labels, db.labels, "DBSCAN at scale {k:e}");
        let got = hdbscan(&d, &HdbscanConfig::new(8)).expect("hdbscan");
        assert_eq!(got.labels, hd.labels, "HDBSCAN at scale {k:e}");
        close(
            silhouette_score(&d, &km.labels, 3),
            silhouette,
            "silhouette",
            k,
        );
        close(
            calinski_harabasz(&d, &km.labels, 3),
            ch,
            "Calinski-Harabasz",
            k,
        );
        close(davies_bouldin(&d, &km.labels, 3), dbi, "Davies-Bouldin", k);
    }
}

#[test]
fn outliers_and_components_are_the_same_in_any_unit() {
    let rows = data();
    let iso = isolation_forest(&rows, &IsolationForestConfig::default()).expect("iforest");
    let lo = lof(&rows, &LofConfig::default().k(10)).expect("lof");
    let ma = mahalanobis(&rows, &MahalanobisConfig::default()).expect("mahalanobis");
    let pc = pca(&rows, &PcaConfig::new(3)).expect("pca");
    assert!(iso.anomaly_count >= 1 && lo.anomaly_count >= 1 && ma.outlier_count >= 1);
    for k in SCALES {
        let d = scaled(&rows, k);
        let got = isolation_forest(&d, &IsolationForestConfig::default()).expect("iforest");
        assert_eq!(
            got.anomalies, iso.anomalies,
            "isolation forest at scale {k:e}"
        );
        let got = lof(&d, &LofConfig::default().k(10)).expect("lof");
        assert_eq!(got.anomalies, lo.anomalies, "LOF at scale {k:e}");
        let got = mahalanobis(&d, &MahalanobisConfig::default()).expect("mahalanobis");
        assert_eq!(got.anomalies, ma.anomalies, "Mahalanobis at scale {k:e}");
        let got = pca(&d, &PcaConfig::new(3)).expect("pca");
        for (g, w) in got
            .explained_variance_ratio
            .iter()
            .zip(&pc.explained_variance_ratio)
        {
            close(*g, *w, "explained variance ratio", k);
        }
    }
}
