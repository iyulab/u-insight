//! Autocorrelation and unit-root testing of a univariate series: the request
//! rules both transports apply before handing a series to `u-analytics`.
//!
//! The statistics are `u-analytics`' own (`correlation::acf` / `pacf`,
//! `testing::adf_test`); what lives here is the part a request adds -- the
//! default and the admissible range of a lag option, the fewest values each
//! statistic is defined for, and the refusal for a series that does not vary --
//! so the WebAssembly binding and the C ABI cannot give the same request two
//! answers.

use u_analytics::correlation::{AcfResult, PacfResult};
use u_analytics::testing::{AdfLags, AdfModel, AdfResult};

use crate::error::{Bound, InsightError};

/// The fewest values `acf` takes: one lag needs a pair.
pub(crate) const ACF_MIN: usize = 2;
/// The fewest values `pacf` takes: lags below `n / 2` leave lag 1 from 4 on.
pub(crate) const PACF_MIN: usize = 4;
/// The fewest values `adf_test` takes.
pub(crate) const ADF_MIN: usize = 10;
/// The fewest rows the ADF test regression is run on.
const ADF_MIN_REGRESSION_ROWS: usize = 5;

/// `adf_test`'s `model` names, in the order of the C ABI's `INSIGHT_ADF_*`.
pub(crate) const ADF_MODELS: [(&str, AdfModel); 3] = [
    ("none", AdfModel::None),
    ("constant", AdfModel::Constant),
    ("constant_trend", AdfModel::ConstantTrend),
];

/// The columns of the ADF regression besides the lags: the lagged level, and
/// the deterministic terms of `model`.
fn adf_fixed_columns(model: AdfModel) -> usize {
    match model {
        AdfModel::None => 1,
        AdfModel::Constant => 2,
        AdfModel::ConstantTrend => 3,
    }
}

/// `data` when every value is finite and there are at least `min` of them.
fn series(data: &[f64], min: usize) -> Result<(), InsightError> {
    if let Some(index) = data.iter().position(|v| !v.is_finite()) {
        return Err(InsightError::ValueNotFinite {
            column: "data".into(),
            index,
        });
    }
    if data.len() < min {
        return Err(InsightError::too_few("data", min, data.len()));
    }
    Ok(())
}

/// The largest lag of an autocorrelation request: `max_lag` within
/// `[1, highest]`, absent = `min(⌊10·log₁₀ n⌋, highest)` (statsmodels'
/// default), never below 1.
fn max_lag(value: Option<f64>, n: usize, highest: usize) -> Result<usize, InsightError> {
    match value {
        None => Ok((((n as f64).log10() * 10.0).floor() as usize).clamp(1, highest)),
        Some(k) => {
            InsightError::check_range(
                "max_lag",
                k,
                Bound::Closed(1.0),
                Bound::Closed(highest as f64),
            )?;
            Ok(k as usize)
        }
    }
}

fn no_autocorrelation() -> InsightError {
    InsightError::DegenerateData {
        parameter: Some("data".into()),
        reason: "the series does not vary, so its autocorrelation is undefined".into(),
    }
}

/// Sample autocorrelation for lags 0 to `max_lag` (a whole number, already
/// read; `None` = the default). At least 2 finite values; `max_lag` in
/// `[1, n − 1]`.
pub(crate) fn acf(data: &[f64], max_lag_option: Option<f64>) -> Result<AcfResult, InsightError> {
    series(data, ACF_MIN)?;
    let n = data.len();
    let lag = max_lag(max_lag_option, n, n - 1)?;
    u_analytics::correlation::acf(data, lag).ok_or_else(no_autocorrelation)
}

/// Sample partial autocorrelation for lags 1 to `max_lag`. At least 4 finite
/// values; `max_lag` in `[1, n / 2 − 1]` -- beyond `n / 2` (statsmodels'
/// limit) the recursion runs on autocorrelations from too few pairs.
pub(crate) fn pacf(data: &[f64], max_lag_option: Option<f64>) -> Result<PacfResult, InsightError> {
    series(data, PACF_MIN)?;
    let n = data.len();
    let lag = max_lag(max_lag_option, n, n / 2 - 1)?;
    u_analytics::correlation::pacf(data, lag).ok_or_else(no_autocorrelation)
}

/// The augmented Dickey-Fuller test of `data` under `model`. `max_lags`
/// bounds the AIC search for the number of lagged differences (`None`:
/// Schwert's rule); `lags` fixes it instead -- at most one of the two. Both
/// are whole numbers already read, each in `[0, max]` where `max` leaves the
/// regression at least 5 rows and a residual degree of freedom. At least 10
/// finite values.
pub(crate) fn adf(
    data: &[f64],
    model: AdfModel,
    max_lags: Option<f64>,
    lags: Option<f64>,
) -> Result<AdfResult, InsightError> {
    series(data, ADF_MIN)?;
    // With p lags the regression keeps n - 1 - p rows (one value is lost to
    // differencing, p to the lags) and has fixed + p columns.
    let n = data.len();
    let max = (n - 1 - ADF_MIN_REGRESSION_ROWS).min((n - 2 - adf_fixed_columns(model)) / 2);
    let count = |name: &str, value: Option<f64>| -> Result<Option<usize>, InsightError> {
        value
            .map(|p| {
                InsightError::check_range(name, p, Bound::Closed(0.0), Bound::Closed(max as f64))
                    .map(|()| p as usize)
            })
            .transpose()
    };
    let lags = match (count("max_lags", max_lags)?, count("lags", lags)?) {
        (Some(_), Some(_)) => {
            return Err(InsightError::InvalidParameter {
                name: "lags".into(),
                message: "lags fixes the number of lagged differences and max_lags bounds the \
                          AIC search for it -- give one"
                    .into(),
            })
        }
        (_, Some(p)) => AdfLags::Fixed(p),
        (max, None) => AdfLags::Aic { max },
    };
    u_analytics::testing::adf_test(data, model, lags).ok_or_else(|| InsightError::DegenerateData {
        parameter: Some("data".into()),
        reason: "the ADF test regression is singular, or an exact fit of the differences \
                 (a deterministic series), so the t-ratio is undefined"
            .into(),
    })
}

#[cfg(test)]
mod tests {
    //! The request rules, and the statistics against the same reference values
    //! `u-analytics` checks its own against (statsmodels 0.15): what reaches a
    //! caller through either transport is the verified engine's result.

    use super::*;
    use crate::refusal::Refusal;

    /// statsmodels' reference series in `u-analytics` (`correlation`).
    const X: [f64; 24] = [
        2.3, 3.1, 4.0, 3.6, 2.9, 2.2, 2.8, 3.9, 4.6, 4.1, 3.3, 2.5, 2.7, 3.8, 4.9, 4.4, 3.5, 2.6,
        3.0, 4.2, 5.1, 4.5, 3.4, 2.9,
    ];

    fn close(got: &[f64], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-12, "{what}[{k}]: {g} vs {w}");
        }
    }

    fn fields(e: InsightError) -> serde_json::Value {
        Refusal::from(e).fields
    }

    /// statsmodels 0.15 `acf(x, nlags=8, fft=False, alpha=0.05)`; the limits
    /// are `confint` upper minus `acf`.
    #[test]
    fn acf_matches_statsmodels() {
        let r = acf(&X, Some(8.0)).expect("defined");
        close(
            &r.acf,
            &[
                1.0,
                0.486403576681892,
                -0.36186808082372074,
                -0.6951594797553611,
                -0.2634125570953007,
                0.42164395757528844,
                0.6593055663079663,
                0.26106100487729356,
                -0.3171014941549895,
            ],
            "acf",
        );
        close(
            &r.bartlett_limits,
            &[
                0.40007597302960923,
                0.48559020426940086,
                0.5269888881332575,
                0.6575826795727397,
                0.6742603293351266,
                0.7152198020271394,
                0.80665422242141,
                0.8200659919957791,
            ],
            "bartlett",
        );
        assert!((r.confidence_threshold - 0.40007597302960923).abs() < 1e-12);
    }

    /// statsmodels 0.15 `pacf(x, nlags=8, method='ldb')`.
    #[test]
    fn pacf_matches_statsmodels() {
        let r = pacf(&X, Some(8.0)).expect("defined");
        close(
            &r.pacf,
            &[
                0.486403576681892,
                -0.7839238375815395,
                0.00047345408884359627,
                0.1902495235598503,
                0.19973005383776718,
                0.1225767860916582,
                0.05567632845800997,
                0.03539504608148146,
            ],
            "pacf",
        );
        assert!((r.confidence_threshold - 0.40007597302960923).abs() < 1e-12);
    }

    /// statsmodels 0.15 `adfuller` on `u-analytics`' AR(1) reference series.
    #[test]
    fn adf_matches_statsmodels() {
        let mut x = vec![0.0_f64; 120];
        for t in 1..x.len() {
            let e = ((t * 7919) % 101) as f64 / 50.0 - 1.0;
            x[t] = 0.7 * x[t - 1] + e;
        }
        for (model, max_lags, lags, stat, n_lags, n_obs) in [
            (
                AdfModel::Constant,
                Some(8.0),
                None,
                -3.436_052_894_8,
                5,
                114,
            ),
            (AdfModel::Constant, None, None, -3.436_052_894_8, 5, 114),
            (
                AdfModel::ConstantTrend,
                Some(6.0),
                None,
                -3.505_247_069_0,
                5,
                114,
            ),
            (AdfModel::None, Some(5.0), None, -3.459_240_021_9, 5, 114),
            (
                AdfModel::Constant,
                None,
                Some(3.0),
                -4.652_930_484_4,
                3,
                116,
            ),
        ] {
            let r = adf(&x, model, max_lags, lags).expect("fits");
            assert_eq!((r.n_lags, r.n_obs), (n_lags, n_obs), "{model:?}");
            assert!(
                (r.statistic - stat).abs() < 1e-8,
                "{model:?}: {}",
                r.statistic
            );
            assert!(r.critical_values[0] < r.critical_values[1]);
            assert!(r.critical_values[1] < r.critical_values[2]);
            for (cv, rejected) in r.critical_values.iter().zip(r.rejected) {
                assert_eq!(rejected, r.statistic <= *cv);
            }
        }
    }

    #[test]
    fn the_default_lag_is_ten_log_n_within_the_range() {
        // n = 24: ⌊10·log₁₀ 24⌋ = 13, within n − 1 = 23 for acf; pacf caps at 11.
        assert_eq!(acf(&X, None).expect("defined").acf.len(), 14);
        assert_eq!(pacf(&X, None).expect("defined").pacf.len(), 11);
        // n = 2: the only lag is 1.
        assert_eq!(acf(&[1.0, 2.0], None).expect("defined").acf.len(), 2);
    }

    #[test]
    fn a_lag_outside_the_range_names_its_bounds() {
        for (got, op) in [(24.0, "acf"), (0.0, "acf"), (-1.0, "acf"), (12.0, "pacf")] {
            let e = if op == "acf" {
                acf(&X, Some(got)).expect_err("out of range")
            } else {
                pacf(&X, Some(got)).expect_err("out of range")
            };
            let max = if op == "acf" { 23 } else { 11 };
            assert_eq!(
                fields(e),
                serde_json::json!({
                    "code": "parameter_out_of_range", "parameter": "max_lag",
                    "min": 1.0, "max": max as f64, "got": got,
                }),
                "{op} {got}"
            );
        }
    }

    #[test]
    fn too_few_values_constant_and_non_finite_series_are_refused() {
        assert_eq!(
            fields(acf(&[1.0], None).expect_err("one value")),
            serde_json::json!({ "code": "insufficient_data", "parameter": "data", "min": 2, "got": 1 })
        );
        assert_eq!(
            fields(pacf(&[1.0, 2.0, 3.0], None).expect_err("three values"))["min"],
            4
        );
        for e in [
            acf(&[3.0; 10], None).expect_err("constant"),
            pacf(&[3.0; 10], None).expect_err("constant"),
        ] {
            assert_eq!(
                fields(e),
                serde_json::json!({ "code": "degenerate_data", "parameter": "data" })
            );
        }
        assert_eq!(
            fields(acf(&[1.0, f64::NAN, 2.0], None).expect_err("NaN")),
            serde_json::json!({ "code": "value_not_finite", "parameter": "data", "index": 1 })
        );
    }

    #[test]
    fn adf_lag_options_are_checked_against_the_room_the_data_leaves() {
        let x: Vec<f64> = (0..20).map(|i| ((i * 37) % 11) as f64).collect();
        // n = 20, constant: min(20 - 1 - 5, (20 - 2 - 2) / 2) = 8.
        let e = adf(&x, AdfModel::Constant, Some(9.0), None).expect_err("past the room");
        assert_eq!(
            fields(e),
            serde_json::json!({
                "code": "parameter_out_of_range", "parameter": "max_lags",
                "min": 0.0, "max": 8.0, "got": 9.0,
            })
        );
        let e = adf(&x, AdfModel::Constant, None, Some(-1.0)).expect_err("negative");
        assert_eq!(fields(e)["parameter"], "lags");
        let e = adf(&x, AdfModel::Constant, Some(2.0), Some(2.0)).expect_err("both");
        assert_eq!(
            fields(e),
            serde_json::json!({ "code": "invalid_option", "parameter": "lags" })
        );
        assert!(adf(&x, AdfModel::ConstantTrend, None, Some(0.0)).is_ok());
        assert_eq!(
            fields(adf(&x[..9], AdfModel::Constant, None, None).expect_err("nine"))["min"],
            10
        );
        let line: Vec<f64> = (0..20).map(f64::from).collect();
        assert_eq!(
            fields(adf(&line, AdfModel::Constant, None, None).expect_err("deterministic")),
            serde_json::json!({ "code": "degenerate_data", "parameter": "data" })
        );
    }
}
