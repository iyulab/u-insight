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
