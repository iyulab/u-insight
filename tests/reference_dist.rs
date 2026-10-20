//! Normality tests, fitted distributions, univariate outlier fences and the Grubbs
//! test against SciPy and the published definitions, on a seeded gamma draw with
//! three planted extremes, rounded to six places. The constants were printed by
//! that recipe with the versions in the next line; the same data in other units
//! must flag the same points.
#![allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::needless_range_loop
)]

// Generated with NumPy 2.4.6, SciPy 1.17.1 (seed 20261022, n = 60).
const X: [f64; 60] = [
    3.906625, 3.259452, 5.131401, 7.406152, 5.184578, 3.249599, 3.489774, 6.958971, 5.185472,
    8.320361, 4.162021, 4.949462, 2.896424, 7.782488, 7.474037, 3.818689, 1.25774, 1.249034,
    3.653098, 6.567233, 2.322413, 4.558148, 5.371077, 5.152898, 1.908504, 3.044117, 5.707948,
    1.693146, 3.54251, 8.988404, 1.944761, 3.730683, 9.696316, 2.892772, 3.520465, 2.075394,
    1.641062, 2.57114, 2.029892, 4.625431, 2.67186, 2.096963, 2.026348, 5.505868, 2.784648,
    7.945297, 7.32625, 10.526772, 5.805938, 6.130428, 4.718404, 2.779559, 3.151611, 3.905313,
    7.08873, 5.597058, 3.277775, 19.5, 22.0, 0.3,
];
const SW: [f64; 2] = [0.7506717890231014, 9.635314474441058e-09];
const JB: [f64; 2] = [263.0541590614798, 7.559884126676098e-58];
const KS: [f64; 2] = [0.16108290684632587, 0.07924652308762337];
const AD: [f64; 2] = [3.267268321006469, 2.7746636208973765e-08];
const FIT_NORMAL: [f64; 3] = [4.9676419, 3.698523450878232, -163.61233235256515];
const FIT_EXPONENTIAL: [f64; 2] = [0.20130275493489172, -156.1767156427646];
const FIT_LOGNORMAL: [f64; 3] = [1.3865567879940428, 0.678271858039774, -145.037293220994];
const FIT_WEIBULL: [f64; 3] = [1.5112049036012067, 5.551981541610342, -148.12347816517936];
const FIT_GAMMA: [f64; 3] = [2.4645221512410904, 0.4961150986428974, -145.28555947982727];
const IQR_REF: [f64; 4] = [
    -1.872151375000001,
    10.542587625000001,
    3.905969,
    3.1036847500000007,
];
const IQR_IDX: [usize; 2] = [57, 58];
const Z_REF: [f64; 4] = [
    -6.221563617770892,
    16.15684741777089,
    4.9676419,
    3.7297351725902974,
];
const Z_IDX: [usize; 2] = [57, 58];
const MZ_REF: [f64; 2] = [3.905969, 1.5917275000000002];
const MZ_IDX: [usize; 2] = [57, 58];
const GRUBBS: [f64; 2] = [4.566640072777886, 3.199661829437385];
const GRUBBS_IDX: usize = 58;

use u_insight::distribution::{
    anderson_darling, fit_distributions, grubbs_test, jarque_bera, ks_test, shapiro_wilk,
};
use u_insight::profiling::{detect_outliers_slice, OutlierMethod};

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let scale = want.abs().max(1e-300);
    assert!(
        (got - want).abs() <= rel * scale,
        "{what}: got {got}, want {want}"
    );
}

#[test]
fn normality_tests_match_scipy() {
    let sw = shapiro_wilk(&X).unwrap();
    close(sw.statistic, SW[0], 1e-6, "Shapiro-Wilk W");
    close(sw.p_value, SW[1], 1e-3, "Shapiro-Wilk p");
    let jb = jarque_bera(&X).unwrap();
    close(jb.statistic, JB[0], 1e-9, "Jarque-Bera");
    close(jb.p_value, JB[1], 1e-7, "Jarque-Bera p");
    let (ks, ks_p) = ks_test(&X).unwrap();
    close(ks, KS[0], 1e-9, "KS D");
    close(ks_p, KS[1], 1e-2, "KS p"); // finite-n corrected approximation vs SciPy exact
    let ad = anderson_darling(&X).unwrap();
    // reported as the small-sample adjusted A*^2 = A^2 (1 + 0.75/n + 2.25/n^2)
    let n = X.len() as f64;
    close(
        ad.statistic,
        AD[0] * (1.0 + 0.75 / n + 2.25 / (n * n)),
        1e-9,
        "Anderson-Darling A*2",
    );
    close(ad.p_value, AD[1], 1e-6, "Anderson-Darling p");
}

#[test]
fn fitted_distributions_match_scipy_maximum_likelihood() {
    let fits = fit_distributions(&X);
    let get = |name: &str| {
        fits.iter()
            .find(|f| f.distribution == name)
            .unwrap_or_else(|| panic!("no {name} fit"))
    };
    let p = |f: &u_insight::distribution::FitResult, k: &str| {
        f.parameters
            .iter()
            .find(|(n, _)| n == k)
            .unwrap_or_else(|| panic!("no parameter {k}"))
            .1
    };
    let n = get("Normal");
    close(p(n, "mu"), FIT_NORMAL[0], 1e-10, "normal mu");
    close(p(n, "sigma"), FIT_NORMAL[1], 1e-10, "normal sigma");
    close(n.log_likelihood, FIT_NORMAL[2], 1e-10, "normal logL");
    let e = get("Exponential");
    close(
        p(e, "lambda"),
        FIT_EXPONENTIAL[0],
        1e-10,
        "exponential lambda",
    );
    close(
        e.log_likelihood,
        FIT_EXPONENTIAL[1],
        1e-10,
        "exponential logL",
    );
    let l = get("LogNormal");
    close(p(l, "mu"), FIT_LOGNORMAL[0], 1e-9, "lognormal mu");
    close(p(l, "sigma"), FIT_LOGNORMAL[1], 1e-9, "lognormal sigma");
    close(l.log_likelihood, FIT_LOGNORMAL[2], 1e-9, "lognormal logL");
    // numerical fits: the maximum is flat, so the likelihood is the fair comparison
    let w = get("Weibull");
    close(p(w, "shape"), FIT_WEIBULL[0], 1e-4, "weibull shape");
    close(p(w, "scale"), FIT_WEIBULL[1], 1e-4, "weibull scale");
    assert!(
        w.log_likelihood >= FIT_WEIBULL[2] - 1e-6,
        "weibull logL {}",
        w.log_likelihood
    );
    let g = get("Gamma");
    close(p(g, "alpha"), FIT_GAMMA[0], 1e-4, "gamma alpha");
    close(p(g, "beta"), FIT_GAMMA[1], 1e-4, "gamma rate");
    assert!(
        g.log_likelihood >= FIT_GAMMA[2] - 1e-6,
        "gamma logL {}",
        g.log_likelihood
    );
}

fn check_outliers(data: &[f64], factor: f64, tag: &str) {
    let iqr = detect_outliers_slice(data, OutlierMethod::Iqr).unwrap();
    close(
        iqr.lower_fence / factor,
        IQR_REF[0],
        1e-9,
        &format!("{tag} IQR lower"),
    );
    close(
        iqr.upper_fence / factor,
        IQR_REF[1],
        1e-9,
        &format!("{tag} IQR upper"),
    );
    close(
        iqr.center / factor,
        IQR_REF[2],
        1e-9,
        &format!("{tag} IQR median"),
    );
    close(
        iqr.spread / factor,
        IQR_REF[3],
        1e-9,
        &format!("{tag} IQR spread"),
    );
    assert_eq!(iqr.indices, IQR_IDX.to_vec(), "{tag} IQR flags");

    let z = detect_outliers_slice(data, OutlierMethod::Zscore).unwrap();
    close(
        z.lower_fence / factor,
        Z_REF[0],
        1e-9,
        &format!("{tag} z lower"),
    );
    close(
        z.upper_fence / factor,
        Z_REF[1],
        1e-9,
        &format!("{tag} z upper"),
    );
    close(z.center / factor, Z_REF[2], 1e-9, &format!("{tag} z mean"));
    close(z.spread / factor, Z_REF[3], 1e-9, &format!("{tag} z sd"));
    assert_eq!(z.indices, Z_IDX.to_vec(), "{tag} z flags");

    let m = detect_outliers_slice(data, OutlierMethod::ModifiedZscore).unwrap();
    close(
        m.center / factor,
        MZ_REF[0],
        1e-9,
        &format!("{tag} MZ median"),
    );
    close(m.spread / factor, MZ_REF[1], 1e-9, &format!("{tag} MAD"));
    assert_eq!(m.indices, MZ_IDX.to_vec(), "{tag} MZ flags");
}

#[test]
fn univariate_outlier_fences_match_the_definitions() {
    check_outliers(&X, 1.0, "unit");
}

/// Fences, centres and flags are properties of the data's shape: in another
/// unit they scale with it and the same points are flagged.
#[test]
fn univariate_outliers_and_grubbs_do_not_depend_on_the_unit() {
    for factor in [1e-9, 1e-6, 1e6] {
        let scaled: Vec<f64> = X.iter().map(|v| v * factor).collect();
        check_outliers(&scaled, factor, &format!("x{factor:e}"));
        let g = grubbs_test(&scaled, 0.05).unwrap();
        close(
            g.statistic,
            GRUBBS[0],
            1e-9,
            &format!("x{factor:e} Grubbs G"),
        );
        close(
            g.critical_value,
            GRUBBS[1],
            1e-6,
            &format!("x{factor:e} Grubbs critical"),
        );
        assert_eq!(g.outlier_index, GRUBBS_IDX);
        assert!(g.is_outlier);
    }
}
