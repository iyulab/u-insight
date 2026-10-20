//! The numeric profile of a column against NumPy and SciPy, in the data's unit and
//! in others. The constants were printed by the recipe with the versions in the
//! next line.
#![allow(clippy::excessive_precision, clippy::approx_constant, dead_code)]

// Generated with NumPy 2.4.6, SciPy 1.17.1 from the gamma draw (seed 20261022, n = 60).
const X: [f64; 60] = [
    3.906625, 3.259452, 5.131401, 7.406152, 5.184578, 3.249599, 3.489774, 6.958971, 5.185472,
    8.320361, 4.162021, 4.949462, 2.896424, 7.782488, 7.474037, 3.818689, 1.25774, 1.249034,
    3.653098, 6.567233, 2.322413, 4.558148, 5.371077, 5.152898, 1.908504, 3.044117, 5.707948,
    1.693146, 3.54251, 8.988404, 1.944761, 3.730683, 9.696316, 2.892772, 3.520465, 2.075394,
    1.641062, 2.57114, 2.029892, 4.625431, 2.67186, 2.096963, 2.026348, 5.505868, 2.784648,
    7.945297, 7.32625, 10.526772, 5.805938, 6.130428, 4.718404, 2.779559, 3.151611, 3.905313,
    7.08873, 5.597058, 3.277775, 19.5, 22.0, 0.3,
];
const MEAN: f64 = 4.9676419;
const STD: f64 = 3.7297351725902974;
const VAR: f64 = 13.910924457657176;
const MIN: f64 = 0.3;
const MAX: f64 = 22.0;
const PCT: [f64; 5] = [
    1.6218959000000002,
    2.78337575,
    3.905969,
    5.8870605000000005,
    9.737838799999997,
];
const SKEW_BIASED: f64 = 2.6180332300919673;
const SKEW_ADJUSTED: f64 = 2.685646155578597;
const KURT_BIASED: f64 = 8.820729655143543;
const KURT_ADJUSTED: f64 = 9.709560202317483;

use u_insight::dataframe::{Column, ValidityBitmap};
use u_insight::profiling::profile_column;

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let scale = want.abs().max(1e-300);
    assert!(
        (got - want).abs() <= rel * scale,
        "{what}: got {got}, want {want}"
    );
}

fn check(values: Vec<f64>, factor: f64, tag: &str) {
    let n = values.len();
    let profile = profile_column("x", &Column::numeric(values, ValidityBitmap::all_valid(n)));
    let s = profile.numeric.expect("numeric profile");
    assert_eq!(s.valid_count, n);
    close(
        s.mean.unwrap() / factor,
        MEAN,
        1e-12,
        &format!("{tag} mean"),
    );
    close(
        s.std_dev.unwrap() / factor,
        STD,
        1e-12,
        &format!("{tag} std"),
    );
    close(
        s.variance.unwrap() / (factor * factor),
        VAR,
        1e-12,
        &format!("{tag} variance"),
    );
    close(s.min.unwrap() / factor, MIN, 1e-12, &format!("{tag} min"));
    close(s.max.unwrap() / factor, MAX, 1e-12, &format!("{tag} max"));
    close(s.p5.unwrap() / factor, PCT[0], 1e-12, &format!("{tag} p5"));
    close(s.q1.unwrap() / factor, PCT[1], 1e-12, &format!("{tag} q1"));
    close(
        s.median.unwrap() / factor,
        PCT[2],
        1e-12,
        &format!("{tag} median"),
    );
    close(s.q3.unwrap() / factor, PCT[3], 1e-12, &format!("{tag} q3"));
    close(
        s.p95.unwrap() / factor,
        PCT[4],
        1e-12,
        &format!("{tag} p95"),
    );
    close(
        s.iqr.unwrap() / factor,
        PCT[3] - PCT[1],
        1e-12,
        &format!("{tag} iqr"),
    );
    close(
        s.skewness.unwrap(),
        SKEW_ADJUSTED,
        1e-9,
        &format!("{tag} skewness"),
    );
    close(
        s.kurtosis.unwrap(),
        KURT_ADJUSTED,
        1e-9,
        &format!("{tag} kurtosis"),
    );
}

/// Moments, quantiles (linear interpolation, NumPy's default), the adjusted
/// skewness and the adjusted excess kurtosis -- Minitab's and Excel's
/// definitions, SciPy's `bias=False` -- and the same data in other units.
#[test]
fn describe_matches_numpy_and_scipy_in_every_unit() {
    check(X.to_vec(), 1.0, "unit");
    for factor in [1e-9, 1e6] {
        check(
            X.iter().map(|v| v * factor).collect(),
            factor,
            &format!("x{factor:e}"),
        );
    }
}
