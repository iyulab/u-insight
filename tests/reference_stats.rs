//! Correlation, regression, variance-inflation and conditioning results against
//! SciPy, statsmodels and NumPy. The data are a seeded NumPy draw rounded to six
//! places; the constants below were printed by that recipe with the versions in
//! the next line (rerun it to change the data). Kendall p-values are SciPy's
//! default (`method="auto"`, the exact distribution at 30 values without ties),
//! or -- until the declared u-analytics floor has it -- the normal approximation.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

// Generated with NumPy 2.4.6, SciPy 1.17.1, statsmodels 0.15.0 (seed 20261020, n = 30).
const X1: [f64; 30] = [
    6.167518, 9.741384, 5.793356, 13.05344, 12.781419, 9.144597, 11.085521, 9.285555, 13.417676,
    7.462985, 5.85969, 9.718885, 9.289475, 10.352114, 10.78703, 11.905239, 10.32408, 9.260227,
    10.954651, 11.818272, 5.714954, 6.603504, 8.556729, 10.330623, 10.529734, 9.318417, 10.087176,
    13.327409, 7.96746, 8.90806,
];
const X2: [f64; 30] = [
    2.920453, 8.341048, 2.703237, 9.750872, 6.443053, 10.476955, 8.134085, 4.374095, 8.13621,
    7.198217, 4.520058, 7.103566, 6.04903, 5.599387, 7.30968, 6.29422, 3.934567, 4.41938, 7.479145,
    6.604159, 1.978775, 3.023681, 8.415757, 3.721347, 8.616686, 6.037314, 6.40037, 9.389103,
    3.44459, 6.17301,
];
const X3: [f64; 30] = [
    -3.508598, -5.156075, 3.167298, -0.981445, -7.563466, -5.80148, -3.220436, -2.846998,
    -7.132545, 3.047987, -5.714872, 2.219227, -1.452602, -0.860026, -4.49694, -1.773638, -9.193231,
    -4.25338, 4.735421, -3.539308, -1.054252, -1.555355, -6.508697, -4.591932, -7.412836, -0.37103,
    -3.035813, -0.491797, 2.390734, -7.795575,
];
const X4: [f64; 30] = [
    5.714125, 6.340882, 1.887552, 4.771517, 7.71147, 5.061603, 8.456213, 7.409318, 9.918508,
    2.299746, 5.919456, 3.575869, 5.338283, 5.396334, 7.524535, 8.050568, 9.832336, 7.043232,
    2.020764, 7.126874, 1.963272, 1.559756, 8.395199, 8.23432, 8.168083, 5.200243, 6.19137,
    6.687854, 3.251703, 6.031146,
];
const Y: [f64; 30] = [
    14.472209, 20.661342, 7.087817, 24.707347, 28.077293, 20.844277, 22.11343, 15.592873,
    28.545987, 9.900802, 17.401896, 15.675233, 17.384002, 18.389842, 24.895092, 17.78868,
    25.214604, 21.133503, 14.140755, 22.230976, 14.355178, 15.426614, 19.998221, 24.340031,
    25.586435, 13.735552, 18.644991, 24.331762, 12.051542, 19.42797,
];
const PEARSON_R: [f64; 6] = [
    0.6492356344612433,
    -0.21437377615546668,
    0.5719320387573747,
    -0.1485515040131594,
    0.28896595965315974,
    -0.7747638057843599,
];
const PEARSON_P: [f64; 6] = [
    0.00010379149932074686,
    0.2553010766067823,
    0.0009601190629094478,
    0.43336968820516664,
    0.12144655643066607,
    5.023770626496475e-07,
];
const SPEARMAN_R: [f64; 6] = [
    0.5906562847608453,
    -0.14616240266963293,
    0.5448275862068965,
    -0.1839822024471635,
    0.2836484983314794,
    -0.7139043381535038,
];
const SPEARMAN_P: [f64; 6] = [
    0.00058958865870008,
    0.44088472389295164,
    0.0018513857777054268,
    0.33043686013128404,
    0.128766222673162,
    9.436240764749218e-06,
];
const KENDALL_R: [f64; 6] = [
    0.439080459770115,
    -0.08045977011494253,
    0.38390804597701156,
    -0.13563218390804596,
    0.18160919540229886,
    -0.5218390804597701,
];
const KENDALL_P: [f64; 6] = [
    0.00048749523007180005,
    0.5473088481323835,
    0.0025434639403121434,
    0.303589884403213,
    0.1653935231257671,
    2.4092178700649936e-05,
];

// SciPy `method="asymptotic"` -- what u-analytics 0.31 returns for every sample.
// TODO(upstream: u-analytics 0.32): once 0.32 is the declared floor, drop this
// table and the either-or below; the p-value is then the exact one above.
const KENDALL_P_ASYMPTOTIC: [f64; 6] = [
    0.0006552802320946897,
    0.532341233012158,
    0.0028876430068704062,
    0.2925148069039126,
    0.15870467504648134,
    5.123645060690808e-05,
];
const OLS_COEF: [f64; 5] = [
    2.47214786769358,
    1.7136791695197222,
    -0.08266308696795066,
    -1.0669663380627954,
    -0.40283175374368174,
];
const OLS_P: [f64; 5] = [
    0.10293940858976999,
    5.791853649476832e-07,
    0.6767728012461474,
    7.268693437845674e-07,
    0.18584728817854881,
];
const OLS_R2: f64 = 0.9056016789879933;
const OLS_ADJ_R2: f64 = 0.8904979476260722;
const OLS_F_P: f64 = 1.8952426523464386e-12;
const OLS_FITTED: [f64; 30] = [
    14.241606296540539,
    21.423307779743826,
    8.036877220223689,
    23.160569122651996,
    28.80633535145973,
    21.42800527562208,
    20.826451405972797,
    18.07597631977968,
    28.407870944996144,
    10.48777272454852,
    15.853165821631464,
    14.731681863162938,
    17.290743711738667,
    18.493291478290992,
    22.120377740262416,
    21.00299591072823,
    25.68715611480476,
    19.67686217373918,
    14.760093172593306,
    23.084345587950704,
    12.4361569353415,
    14.569680496712833,
    20.002671528879866,
    21.450294584442695,
    24.423335101849748,
    16.24291547408535,
    19.9742867607132,
    22.365569705381773,
    11.980355737530443,
    23.115503658621098,
];
const OLS_RESID: [f64; 30] = [
    0.23060270345946066,
    -0.7619657797438251,
    -0.9490602202236884,
    1.5467778773480028,
    -0.7290423514597286,
    -0.5837282756220787,
    1.2869785940272038,
    -2.48310331977968,
    0.13811605500385582,
    -0.5869707245485198,
    1.5487301783685368,
    0.9435511368370619,
    0.09325828826133176,
    -0.10344947829099027,
    2.7747142597375856,
    -3.2143159107282315,
    -0.4725521148047598,
    1.45664082626082,
    -0.6193381725933058,
    -0.8533695879507057,
    1.9190210646584998,
    0.8569335032871681,
    -0.004450528879864635,
    2.889736415557305,
    1.1630998981502536,
    -2.5073634740853503,
    -1.3292957607131974,
    1.966192294618228,
    0.07118626246955628,
    -3.687533658621099,
];
const VIF: [f64; 4] = [
    3.0618307189583547,
    1.8392150605568385,
    3.246843192411884,
    4.684327418100414,
];
const COND: f64 = 26.079667364364795;

use u_insight::analysis::{
    condition_number, correlation_analysis, regression_analysis, vif_analysis, CorrelationConfig,
    CorrelationMethod,
};

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let scale = want.abs().max(1e-300);
    assert!(
        (got - want).abs() <= rel * scale + 1e-300,
        "{what}: got {got}, want {want}"
    );
}

fn predictors() -> (Vec<Vec<f64>>, Vec<String>) {
    (
        vec![X1.to_vec(), X2.to_vec(), X3.to_vec(), X4.to_vec()],
        ["x1", "x2", "x3", "x4"].map(String::from).to_vec(),
    )
}

fn pair_value(result: &u_insight::analysis::CorrelationAnalysis, a: &str, b: &str) -> (f64, f64) {
    let p = result
        .high_pairs
        .iter()
        .find(|p| p.col_a == a && p.col_b == b)
        .unwrap_or_else(|| panic!("pair {a},{b} not listed"));
    (p.r, p.p_value)
}

#[test]
fn correlation_matrices_and_p_values_match_scipy() {
    let (cols, names) = predictors();
    let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for (method, r_ref, p_ref) in [
        (CorrelationMethod::Pearson, PEARSON_R, PEARSON_P),
        (CorrelationMethod::Spearman, SPEARMAN_R, SPEARMAN_P),
        (CorrelationMethod::Kendall, KENDALL_R, KENDALL_P),
    ] {
        let config = CorrelationConfig {
            method,
            high_threshold: 0.0,
        };
        let got = correlation_analysis(&cols, &names, &config).unwrap();
        for (idx, &(i, j)) in pairs.iter().enumerate() {
            close(
                got.matrix.get(i, j),
                r_ref[idx],
                1e-10,
                &format!("{method:?} r {i}{j}"),
            );
            close(
                got.matrix.get(j, i),
                r_ref[idx],
                1e-10,
                &format!("{method:?} r {j}{i}"),
            );
            let (r, p) = pair_value(&got, &names[i], &names[j]);
            close(r, r_ref[idx], 1e-10, &format!("{method:?} listed r {i}{j}"));
            if matches!(method, CorrelationMethod::Kendall) {
                // The exact value (u-analytics with the exact distribution) or
                // the approximation of the floor this crate still declares.
                let exact = (p - p_ref[idx]).abs() <= 1e-6 * p_ref[idx].abs();
                let approx =
                    (p - KENDALL_P_ASYMPTOTIC[idx]).abs() <= 1e-6 * KENDALL_P_ASYMPTOTIC[idx].abs();
                assert!(
                    exact || approx,
                    "Kendall p {i}{j}: got {p}, want {} or {}",
                    p_ref[idx],
                    KENDALL_P_ASYMPTOTIC[idx]
                );
            } else {
                close(p, p_ref[idx], 1e-6, &format!("{method:?} p {i}{j}"));
            }
        }
    }
}

#[test]
fn regression_matches_statsmodels_ols() {
    let (cols, names) = predictors();
    let got = regression_analysis(&cols, &names, &Y, "y").unwrap();
    for (i, want) in OLS_COEF.iter().enumerate() {
        close(got.coefficients[i], *want, 1e-9, &format!("coef {i}"));
        close(got.p_values[i], OLS_P[i], 1e-7, &format!("p {i}"));
    }
    close(got.r_squared, OLS_R2, 1e-10, "R2");
    close(got.adj_r_squared, OLS_ADJ_R2, 1e-10, "adjusted R2");
    close(got.f_p_value, OLS_F_P, 1e-7, "F p-value");
    for i in 0..Y.len() {
        close(got.fitted[i], OLS_FITTED[i], 1e-9, &format!("fitted {i}"));
        assert!(
            (got.residuals[i] - OLS_RESID[i]).abs() < 1e-9,
            "residual {i}: {} vs {}",
            got.residuals[i],
            OLS_RESID[i]
        );
    }
    for (i, want) in VIF.iter().enumerate() {
        close(got.vif[i], *want, 1e-9, &format!("regression vif {i}"));
    }
}

#[test]
fn vif_and_condition_number_match_statsmodels_and_numpy() {
    let (cols, names) = predictors();
    let v = vif_analysis(&cols, &names, 10.0).unwrap();
    for (i, want) in VIF.iter().enumerate() {
        close(v.vif_per_column[i], *want, 1e-9, &format!("vif {i}"));
    }
    assert!(v.high_vif_columns.is_empty());
    close(condition_number(&cols, &names).unwrap(), COND, 1e-9, "cond");
}

/// The condition number of a covariance matrix and the variance inflation
/// factors are ratios: the same data in a smaller unit must report the same.
#[test]
fn condition_number_and_vif_do_not_depend_on_the_unit() {
    let (cols, names) = predictors();
    for factor in [1e-9, 1e-6, 1e6] {
        let scaled: Vec<Vec<f64>> = cols
            .iter()
            .map(|c| c.iter().map(|v| v * factor).collect())
            .collect();
        close(
            condition_number(&scaled, &names).unwrap(),
            COND,
            1e-7,
            &format!("cond x{factor:e}"),
        );
        let v = vif_analysis(&scaled, &names, 10.0).unwrap();
        for (i, want) in VIF.iter().enumerate() {
            close(
                v.vif_per_column[i],
                *want,
                1e-7,
                &format!("vif {i} x{factor:e}"),
            );
        }
    }
}
