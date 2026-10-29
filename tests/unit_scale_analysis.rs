//! The same observations written in another unit give the same analysis, scaled
//! the way the unit scales it: normality tests, histograms, distribution fits,
//! outlier flags, correlations and regressions, from 1e-12 to 1e12.

use u_insight::analysis::{
    correlation_analysis, regression_analysis, CorrelationConfig, CorrelationMethod,
};
use u_insight::distribution::{
    anderson_darling, ecdf, fit_distributions, histogram, jarque_bera, ks_test, shapiro_wilk,
    BinMethod,
};
use u_insight::profiling::{detect_outliers_slice, OutlierMethod};

const SCALES: [f64; 6] = [1e-12, 1e-9, 1e-3, 1e3, 1e9, 1e12];

/// A deterministic skewed sample with a few repeats and one far value.
fn sample(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed;
    let mut v: Vec<f64> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            ((1.0 + 4.0 * u * u) * 16.0).round() / 16.0
        })
        .collect();
    v.push(30.0);
    v
}

fn scaled(v: &[f64], k: f64) -> Vec<f64> {
    v.iter().map(|x| x * k).collect()
}

fn close(got: f64, want: f64, what: &str, k: f64) {
    let tolerance = 1e-8 * want.abs().max(1e-12);
    assert!(
        (got - want).abs() <= tolerance,
        "{what} at scale {k:e}: {got:e} vs {want:e}"
    );
}

#[test]
fn the_normality_tests_do_not_depend_on_the_unit() {
    let x = sample(1, 60);
    for k in SCALES {
        let s = scaled(&x, k);
        let (a, b) = (ks_test(&x).unwrap(), ks_test(&s).unwrap());
        close(b.0, a.0, "ks statistic", k);
        close(b.1, a.1, "ks p", k);
        for (name, f) in [
            ("jarque_bera", jarque_bera as fn(&[f64]) -> _),
            ("shapiro_wilk", shapiro_wilk),
            ("anderson_darling", anderson_darling),
        ] {
            let (a, b) = (f(&x).unwrap(), f(&s).unwrap());
            close(b.statistic, a.statistic, &format!("{name} statistic"), k);
            close(b.p_value, a.p_value, &format!("{name} p"), k);
        }
    }
}

#[test]
fn histograms_and_the_ecdf_follow_the_unit() {
    // No value on a bin edge: a value exactly on one is in either bin by how the
    // rounding of `value * unit` falls, which no implementation can settle.
    let x: Vec<f64> = sample(2, 80)
        .iter()
        .enumerate()
        .map(|(i, v)| v + 0.0137 * (i as f64 + 1.0).sqrt())
        .collect();
    for k in SCALES {
        let s = scaled(&x, k);
        for method in [
            BinMethod::Sturges,
            BinMethod::Scott,
            BinMethod::FreedmanDiaconis,
            BinMethod::Fixed(7),
        ] {
            let (a, b) = (
                histogram(&x, method).unwrap(),
                histogram(&s, method).unwrap(),
            );
            assert_eq!(a.n_bins, b.n_bins, "{method:?} bins at scale {k:e}");
            assert_eq!(a.counts, b.counts, "{method:?} counts at scale {k:e}");
            close(b.bin_width, k * a.bin_width, "bin width", k);
        }
        let (a, b) = (ecdf(&x).unwrap(), ecdf(&s).unwrap());
        assert_eq!(a.probabilities, b.probabilities, "ecdf at scale {k:e}");
    }
}

#[test]
fn the_best_fitting_family_does_not_depend_on_the_unit() {
    let x = sample(3, 120);
    // The beta family is fitted to data in (0, 1) only, so whether it is a
    // candidate depends on the unit by definition; the others do not.
    let without_beta =
        |fits: Vec<u_insight::distribution::FitResult>| -> Vec<u_insight::distribution::FitResult> {
            fits.into_iter()
                .filter(|f| f.distribution != "Beta")
                .collect()
        };
    let base = without_beta(fit_distributions(&x));
    let order: Vec<&str> = base.iter().map(|f| f.distribution.as_str()).collect();
    for k in SCALES {
        let fits = without_beta(fit_distributions(&scaled(&x, k)));
        let got: Vec<&str> = fits.iter().map(|f| f.distribution.as_str()).collect();
        assert_eq!(got, order, "ranking at scale {k:e}");
        // A density changes by 1/k per observation: every log-likelihood by the
        // same amount, so the differences between families stay.
        let shift = x.len() as f64 * k.ln();
        for (a, b) in base.iter().zip(&fits) {
            close(
                b.log_likelihood,
                a.log_likelihood - shift,
                &format!("{} log-likelihood", a.distribution),
                k,
            );
        }
    }
}

#[test]
fn the_same_values_are_flagged_in_any_unit() {
    let x = sample(4, 100);
    for method in [
        OutlierMethod::Iqr,
        OutlierMethod::Zscore,
        OutlierMethod::ModifiedZscore,
    ] {
        let base = detect_outliers_slice(&x, method).map(|r| r.indices);
        for k in SCALES {
            let got = detect_outliers_slice(&scaled(&x, k), method).map(|r| r.indices);
            assert_eq!(got, base, "{method:?} at scale {k:e}");
        }
    }
}

#[test]
fn correlations_and_regressions_follow_the_units_of_their_columns() {
    let a = sample(5, 50);
    let b: Vec<f64> = sample(6, 50)
        .iter()
        .zip(&a)
        .map(|(n, a)| 2.0 * a + 0.5 * n)
        .collect();
    let c = sample(7, 50);
    let names: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    for method in [
        CorrelationMethod::Pearson,
        CorrelationMethod::Spearman,
        CorrelationMethod::Kendall,
    ] {
        let config = CorrelationConfig {
            method,
            ..CorrelationConfig::default()
        };
        let base = correlation_analysis(&[a.clone(), b.clone(), c.clone()], &names, &config)
            .expect("correlation");
        for k in SCALES {
            // Each column in its own unit.
            let cols = [scaled(&a, k), scaled(&b, 1.0 / k), scaled(&c, k * k)];
            let got = correlation_analysis(&cols, &names, &config).expect("correlation");
            for i in 0..3 {
                for j in 0..3 {
                    close(
                        got.matrix.get(i, j),
                        base.matrix.get(i, j),
                        &format!("{method:?} r[{i}][{j}]"),
                        k,
                    );
                }
            }
        }
    }
    // y = f(a, c): the coefficients take the units of the columns, the fit does not.
    let y: Vec<f64> = a
        .iter()
        .zip(&c)
        .map(|(a, c)| 3.0 * a - 0.5 * c + 1.0)
        .collect();
    let base = regression_analysis(
        &[a.clone(), c.clone()],
        &["a".to_string(), "c".to_string()],
        &y,
        "y",
    )
    .expect("regression");
    for k in SCALES {
        let (ka, kc, ky) = (k, 1.0 / k, k * 7.0);
        let got = regression_analysis(
            &[scaled(&a, ka), scaled(&c, kc)],
            &["a".to_string(), "c".to_string()],
            &scaled(&y, ky),
            "y",
        )
        .expect("regression");
        close(got.r_squared, base.r_squared, "r_squared", k);
        close(got.adj_r_squared, base.adj_r_squared, "adj_r_squared", k);
        close(got.f_p_value, base.f_p_value, "f_p_value", k);
        close(
            got.coefficients[0],
            ky * base.coefficients[0],
            "intercept",
            k,
        );
        close(
            got.coefficients[1],
            ky / ka * base.coefficients[1],
            "slope a",
            k,
        );
        close(
            got.coefficients[2],
            ky / kc * base.coefficients[2],
            "slope c",
            k,
        );
        for (g, w) in got.p_values.iter().zip(&base.p_values) {
            close(*g, *w, "p value", k);
        }
        for (g, w) in got.vif.iter().zip(&base.vif) {
            close(*g, *w, "vif", k);
        }
    }
}
