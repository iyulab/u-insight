//! Anomaly and seasonality operations on data whose answer is known by
//! construction: planted points, a generated period. Isolation forests are
//! randomised, so the checks are about structure (ranking, range, seed,
//! unit) rather than a library's numbers; the numerical values of the period
//! estimate and the spectral residual are compared with NumPy in the engine
//! that computes them (`u-analytics`), and these tests check that the
//! operation reaches them.

use u_analytics::detection::SpectralResidual;
use u_analytics::seasonality::estimate_period;
use u_insight::isolation_forest::{isolation_forest, IsolationForestConfig};

/// 120 points of a smooth three-dimensional cloud, then three planted far points.
fn cloud() -> Vec<Vec<f64>> {
    let mut data: Vec<Vec<f64>> = (0..120)
        .map(|i| {
            let t = i as f64;
            vec![
                (t * 0.37).sin() * 1.5 + (t * 0.11).cos(),
                (t * 0.23).cos() * 1.2 + (t * 0.07).sin(),
                (t * 0.51).sin() + (t * 0.19).cos() * 0.8,
            ]
        })
        .collect();
    data.push(vec![15.0, -14.0, 13.0]);
    data.push(vec![-16.0, 12.0, 15.0]);
    data.push(vec![14.0, 15.0, -16.0]);
    data
}

fn scaled(data: &[Vec<f64>], factor: f64) -> Vec<Vec<f64>> {
    data.iter()
        .map(|p| p.iter().map(|v| v * factor).collect())
        .collect()
}

#[test]
fn isolation_forest_scores_rank_the_planted_points_first() {
    let data = cloud();
    let config = IsolationForestConfig::default().contamination(0.03);
    let r = isolation_forest(&data, &config).unwrap();
    assert!(
        r.scores.iter().all(|&s| s > 0.0 && s < 1.0),
        "scores leave (0, 1)"
    );
    let mut order: Vec<usize> = (0..data.len()).collect();
    order.sort_by(|&a, &b| r.scores[b].partial_cmp(&r.scores[a]).unwrap());
    let mut top: Vec<usize> = order[..3].to_vec();
    top.sort_unstable();
    assert_eq!(top, vec![120, 121, 122], "top scores");
    assert!(
        (0..120).all(|i| r.scores[i] < r.scores[120].min(r.scores[121]).min(r.scores[122])),
        "an ordinary point scores above a planted one"
    );
    assert_eq!(r.anomaly_count, 4, "3% of 123 points, rounded up");
    for i in [120usize, 121, 122] {
        assert!(r.anomalies[i]);
    }
}

#[test]
fn isolation_forest_is_reproducible_and_does_not_depend_on_the_unit() {
    let data = cloud();
    let config = IsolationForestConfig::default()
        .contamination(0.03)
        .seed(Some(5));
    let a = isolation_forest(&data, &config).unwrap();
    let b = isolation_forest(&data, &config).unwrap();
    assert_eq!(a.scores, b.scores, "same seed, same scores");
    for factor in [1e-9, 1e6] {
        let c = isolation_forest(&scaled(&data, factor), &config).unwrap();
        assert_eq!(a.anomalies, c.anomalies, "flags in units x{factor:e}");
        for (x, y) in a.scores.iter().zip(&c.scores) {
            assert!(
                (x - y).abs() < 1e-12,
                "score {x} vs {y} in units x{factor:e}"
            );
        }
    }
}

#[test]
fn the_period_estimate_is_the_generating_period() {
    for period in [12usize, 7, 24] {
        let series: Vec<f64> = (0..120)
            .map(|i| {
                10.0 + 3.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                    + 0.01 * i as f64
            })
            .collect();
        let r = estimate_period(&series).expect("enough data");
        assert_eq!(r.period, Some(period), "period {period}");
    }
}

#[test]
fn the_spectral_residual_flags_a_planted_spike() {
    let mut series: Vec<f64> = (0..96)
        .map(|i| 5.0 + (i as f64 * 0.5).sin() + 0.05 * (i as f64 * 1.7).cos())
        .collect();
    series[60] += 12.0;
    let points = SpectralResidual::new().analyze(&series).expect("valid");
    let flagged: Vec<usize> = points
        .iter()
        .filter(|p| p.is_anomaly)
        .map(|p| p.index)
        .collect();
    assert!(flagged.contains(&60), "spike not flagged: {flagged:?}");
    assert!(flagged.len() <= 3, "too many flags: {flagged:?}");
}
