//! A result that is a ratio or a standardised distance cannot depend on the
//! unit the data are in. Each function here is run on data and on the same data
//! in units a million (or a billion) times smaller and must report the same.

use u_insight::mahalanobis::{mahalanobis, MahalanobisConfig};
use u_insight::pca::{pca, PcaConfig};

/// A deterministic spread of correlated points, so nothing depends on a
/// random-number generator.
fn points(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            let a = (t * 0.7).sin() * 3.0 + t * 0.05;
            let b = 0.6 * a + (t * 1.3).cos() * 0.8;
            let c = (t * 0.31).sin() + 0.2 * b;
            vec![a, b, c]
        })
        .collect()
}

fn scaled(data: &[Vec<f64>], factor: f64) -> Vec<Vec<f64>> {
    data.iter()
        .map(|p| p.iter().map(|v| v * factor).collect())
        .collect()
}

#[test]
fn explained_variance_ratio_does_not_depend_on_the_unit() {
    let data = points(40);
    let config = PcaConfig::new(2);
    let reference = pca(&data, &config).expect("valid");
    let small = pca(&scaled(&data, 1e-9), &config).expect("valid");
    for (a, b) in reference
        .explained_variance_ratio
        .iter()
        .zip(&small.explained_variance_ratio)
    {
        assert!((a - b).abs() < 1e-9, "ratio {a} vs {b} in other units");
    }
    assert!(reference.explained_variance_ratio[0] > 0.0);
}

#[test]
fn mahalanobis_distances_do_not_depend_on_the_unit() {
    let data = points(40);
    let config = MahalanobisConfig::default();
    let reference = mahalanobis(&data, &config).expect("valid");
    let small =
        mahalanobis(&scaled(&data, 1e-7), &config).expect("a small unit is not a constant column");
    for (a, b) in reference.distances.iter().zip(&small.distances) {
        assert!(
            (a - b).abs() <= 1e-8 * a.abs().max(1.0),
            "distance {a} vs {b}"
        );
    }
}

#[test]
fn permutation_importance_does_not_depend_on_the_unit_of_the_target() {
    use u_insight::feature_importance::permutation_importance;
    let n = 60;
    let a: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 2.0).collect();
    let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.91).cos()).collect();
    let target: Vec<f64> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| 3.0 * x + 0.4 * y + 1.0)
        .collect();
    let names = vec!["a".to_string(), "b".to_string()];
    let reference =
        permutation_importance(&[a.clone(), b.clone()], &names, &target, 5, 7).expect("valid");
    let tiny: Vec<f64> = target.iter().map(|v| v * 1e-10).collect();
    let small = permutation_importance(&[a, b], &names, &tiny, 5, 7).expect("valid");
    assert!(
        reference.features[0].importance > 0.1,
        "the fixture has a strong feature"
    );
    for (r, s) in reference.features.iter().zip(&small.features) {
        assert!(
            (r.importance - s.importance).abs() < 1e-6,
            "{}: importance {} vs {} in other units",
            r.name,
            r.importance,
            s.importance
        );
    }
}

/// Two groups of the deterministic points, the second shifted away.
fn two_groups() -> Vec<Vec<f64>> {
    let mut data = points(40);
    for p in data.iter_mut().skip(20) {
        for v in p.iter_mut() {
            *v += 12.0;
        }
    }
    data
}

#[test]
fn kmeans_converges_at_the_same_iteration_in_any_unit() {
    use u_insight::clustering::{kmeans, mini_batch_kmeans, KMeansConfig, MiniBatchKMeansConfig};
    let data = two_groups();
    let base = kmeans(&data, &KMeansConfig::new(3)).expect("valid");
    let mini = mini_batch_kmeans(&data, &MiniBatchKMeansConfig::new(3)).expect("valid");
    for factor in [1e-9, 1e9] {
        let r = kmeans(&scaled(&data, factor), &KMeansConfig::new(3)).expect("valid");
        assert_eq!(r.labels, base.labels, "kmeans labels at ×{factor}");
        assert_eq!(
            r.iterations, base.iterations,
            "kmeans iterations at ×{factor}"
        );
        let m = mini_batch_kmeans(&scaled(&data, factor), &MiniBatchKMeansConfig::new(3))
            .expect("valid");
        assert_eq!(m.labels, mini.labels, "mini-batch labels at ×{factor}");
        assert_eq!(
            m.iterations, mini.iterations,
            "mini-batch iterations at ×{factor}"
        );
    }
}

#[test]
fn isolation_forest_scores_do_not_depend_on_the_unit() {
    use u_insight::isolation_forest::{isolation_forest, IsolationForestConfig};
    let mut data = points(60);
    data.push(vec![30.0, -20.0, 15.0]);
    let config = IsolationForestConfig::default();
    let base = isolation_forest(&data, &config).expect("valid");
    let small = isolation_forest(&scaled(&data, 1e-18), &config).expect("valid");
    assert_eq!(small.anomalies, base.anomalies);
    for (s, b) in small.scores.iter().zip(&base.scores) {
        assert!((s - b).abs() < 1e-12, "{s} vs {b}");
    }
}

#[test]
fn univariate_outlier_tests_do_not_depend_on_the_unit() {
    use u_insight::distribution::grubbs_test;
    use u_insight::profiling::{detect_outliers_slice, OutlierMethod};
    let data = [10.1, 9.8, 10.3, 9.9, 10.0, 10.2, 9.7, 10.1, 14.5, 10.0];
    let base = grubbs_test(&data, 0.05).expect("valid");
    for factor in [1e-18, 1e18] {
        let x: Vec<f64> = data.iter().map(|v| v * factor).collect();
        let g = grubbs_test(&x, 0.05).expect("a small unit is not constant data");
        assert_eq!(g.outlier_index, base.outlier_index);
        assert!((g.statistic - base.statistic).abs() < 1e-9 * base.statistic);
        for method in [OutlierMethod::Zscore, OutlierMethod::ModifiedZscore] {
            let a = detect_outliers_slice(&data, method).expect("valid");
            let b = detect_outliers_slice(&x, method).expect("valid");
            assert_eq!(b.indices, a.indices, "{method:?} at ×{factor}");
        }
    }
}

/// Points on a lattice have many neighbours at exactly the same distance;
/// whether two distances are "the same" must not depend on the unit they are
/// written in (a tie tolerance of a fixed size is nothing for lengths near 1e9
/// and everything for lengths near 1e-15).
#[test]
fn lof_scores_do_not_depend_on_the_unit() {
    use u_insight::lof::{lof, LofConfig};

    let mut data: Vec<Vec<f64>> = (0..6)
        .flat_map(|x| (0..6).map(move |y| vec![x as f64, y as f64]))
        .collect();
    data.push(vec![9.0, 9.5]);
    data.push(vec![-3.0, 4.0]);
    // A ring of points all at distance 5 from (20, 20) -- ties reached by
    // different sums of squares -- and one a hair beyond them: a tie only if
    // the tolerance is as large as that hair in the unit used.
    for (dx, dy) in [
        (5.0, 0.0),
        (3.0, 4.0),
        (4.0, 3.0),
        (0.0, 5.0),
        (-3.0, 4.0),
        (-5.0, 0.0),
    ] {
        data.push(vec![20.0 + dx, 20.0 + dy]);
    }
    data.push(vec![20.0, 20.0]);
    data.push(vec![20.0 - 5.0 - 4e-7, 20.0]);
    let config = LofConfig::default().k(5);
    let reference = lof(&data, &config).expect("valid");
    for factor in [1e-14, 1e-9, 1e-6, 1e6, 1e9, 1e12] {
        let other = lof(&scaled(&data, factor), &config).expect("valid");
        assert_eq!(
            reference.anomalies, other.anomalies,
            "anomalies at {factor:e}"
        );
        for (i, (a, b)) in reference.scores.iter().zip(&other.scores).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6 * a.abs().max(1.0),
                "score {i} at {factor:e}: {a} vs {b}"
            );
        }
    }
}
