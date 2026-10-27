//! Distribution analysis module.
//!
//! Provides a unified interface for analyzing the distributional properties
//! of numeric data. Wraps `u-analytics` distribution and testing functions
//! into a single coherent analysis result.
//!
//! # Features
//!
//! - **Normality testing** — KS, Jarque-Bera, Shapiro-Wilk, and Anderson-Darling tests with combined verdict
//! - **Distribution fitting** — MLE fitting of the continuous families that apply (Normal; Exponential, Gamma, LogNormal, Weibull for positive data; Beta for (0, 1)) with AIC/BIC ranking
//! - **Empirical CDF** — sorted values and cumulative probabilities
//! - **Histogram** — optimal bin computation (Sturges, Scott, Freedman-Diaconis)
//! - **QQ-plot** — theoretical vs sample quantiles for normal distribution
//!
//! # Example
//!
//! ```
//! use u_insight::distribution::{distribution_analysis, DistributionConfig, BinMethod};
//!
//! let data = vec![1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0];
//! let config = DistributionConfig::default();
//! let result = distribution_analysis(&data, &config).unwrap();
//!
//! assert!(result.ecdf.is_some());
//! assert!(result.histogram.is_some());
//! assert!(result.qq_plot.is_some());
//! assert!(result.normality.ks_test.is_some());
//! ```

use crate::error::{Bound, InsightError};

// ── Configuration ───────────────────────────────────────────────────

/// Method for computing optimal histogram bins.
///
/// Mirrors `u_analytics::distribution::BinMethod` to avoid leaking
/// the dependency to consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinMethod {
    /// Sturges' rule: k = ceil(log2(n)) + 1. Best for near-normal data.
    Sturges,
    /// Scott's rule: h = 3.49 * sigma * n^(-1/3). Width-based.
    Scott,
    /// Freedman-Diaconis rule: h = 2 * IQR * n^(-1/3). Robust to outliers.
    FreedmanDiaconis,
    /// Explicit bin count chosen by the caller. Must be >= 1.
    Fixed(usize),
}

/// Configuration for distribution analysis.
#[derive(Debug, Clone)]
pub struct DistributionConfig {
    /// Histogram bin method. Default: `FreedmanDiaconis`.
    pub bin_method: BinMethod,
    /// Significance level for normality tests, in (0, 1). Default: 0.05.
    pub significance_level: f64,
    /// Whether to compute ECDF. Default: true.
    pub compute_ecdf: bool,
    /// Whether to compute histogram. Default: true.
    pub compute_histogram: bool,
    /// Whether to compute QQ-plot. Default: true.
    pub compute_qq_plot: bool,
    /// Whether to fit distributions (the continuous families that apply: Normal; Exponential,
    /// Gamma, LogNormal, Weibull for positive data; Beta for (0, 1)). Default: false.
    pub fit_distributions: bool,
}

impl Default for DistributionConfig {
    fn default() -> Self {
        Self {
            bin_method: BinMethod::FreedmanDiaconis,
            significance_level: 0.05,
            compute_ecdf: true,
            compute_histogram: true,
            compute_qq_plot: true,
            fit_distributions: false,
        }
    }
}

// ── Result Types ────────────────────────────────────────────────────

/// ECDF (Empirical Cumulative Distribution Function) result.
#[derive(Debug, Clone)]
pub struct EcdfResult {
    /// Sorted unique values.
    pub values: Vec<f64>,
    /// Cumulative probabilities corresponding to each value.
    pub probabilities: Vec<f64>,
}

/// Histogram result.
#[derive(Debug, Clone)]
pub struct HistogramResult {
    /// Number of bins.
    pub n_bins: usize,
    /// Bin width.
    pub bin_width: f64,
    /// Bin edges (length = n_bins + 1).
    pub edges: Vec<f64>,
    /// Count of observations in each bin.
    pub counts: Vec<usize>,
    /// Bin method used.
    pub method: BinMethod,
}

/// QQ-plot data for comparing sample against a normal distribution.
#[derive(Debug, Clone)]
pub struct QQPlotResult {
    /// Theoretical quantiles from the standard normal distribution.
    pub theoretical: Vec<f64>,
    /// Sorted sample quantiles.
    pub sample: Vec<f64>,
}

/// Result of a single normality test.
#[derive(Debug, Clone, Copy)]
pub struct NormalityTestResult {
    /// Test statistic.
    pub statistic: f64,
    /// P-value.
    pub p_value: f64,
    /// Whether the null hypothesis (normality) is rejected at the given significance level.
    pub rejected: bool,
}

/// One of the four normality tests of [`NormalityAssessment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalityTest {
    /// Kolmogorov-Smirnov (`ks_test`).
    KolmogorovSmirnov,
    /// Jarque-Bera (`jarque_bera`).
    JarqueBera,
    /// Shapiro-Wilk (`shapiro_wilk`).
    ShapiroWilk,
    /// Anderson-Darling (`anderson_darling`).
    AndersonDarling,
}

impl NormalityTest {
    /// The test's name in the assessment's fields (`"shapiro_wilk"`).
    pub fn name(self) -> &'static str {
        match self {
            Self::KolmogorovSmirnov => "ks_test",
            Self::JarqueBera => "jarque_bera",
            Self::ShapiroWilk => "shapiro_wilk",
            Self::AndersonDarling => "anderson_darling",
        }
    }

    /// The sample sizes the test is defined for: `(min, max)`, `max` open-ended
    /// as `None`.
    fn sample_size(self) -> (usize, Option<usize>) {
        match self {
            Self::KolmogorovSmirnov => (5, None),
            Self::JarqueBera | Self::AndersonDarling => (8, None),
            // Royston's approximation of the W distribution, 3 to 5000 values.
            Self::ShapiroWilk => (3, Some(5000)),
        }
    }
}

/// Why a normality test was not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OmissionReason {
    /// Fewer values than the test is defined for (`min`).
    TooFewValues {
        /// The fewest values the test takes.
        min: usize,
    },
    /// More values than the test is defined for (`max`).
    TooManyValues {
        /// The most values the test takes.
        max: usize,
    },
    /// Every value is the same, so there is no spread to test.
    ConstantValues,
}

impl OmissionReason {
    /// The reason's wire name (`"too_many_values"`).
    pub fn name(self) -> &'static str {
        match self {
            Self::TooFewValues { .. } => "too_few_values",
            Self::TooManyValues { .. } => "too_many_values",
            Self::ConstantValues => "constant_values",
        }
    }
}

/// A normality test that was not run, and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OmittedTest {
    /// The test.
    pub test: NormalityTest,
    /// Why it has no result.
    pub reason: OmissionReason,
}

/// Why `test` gave no result for `n` finite values: outside the sizes it is
/// defined for, or else no spread to test.
fn omission(test: NormalityTest, n: usize) -> OmittedTest {
    let (min, max) = test.sample_size();
    let reason = if n < min {
        OmissionReason::TooFewValues { min }
    } else if max.is_some_and(|max| n > max) {
        OmissionReason::TooManyValues {
            max: max.expect("checked above"),
        }
    } else {
        OmissionReason::ConstantValues
    };
    OmittedTest { test, reason }
}

/// Combined normality assessment from multiple tests.
#[derive(Debug, Clone)]
pub struct NormalityAssessment {
    /// Kolmogorov-Smirnov test result (None if insufficient data).
    pub ks_test: Option<NormalityTestResult>,
    /// Jarque-Bera test result (None if insufficient data, requires n >= 8).
    pub jarque_bera: Option<NormalityTestResult>,
    /// Shapiro-Wilk test result (None if n < 3 or n > 5000, or all values equal).
    pub shapiro_wilk: Option<NormalityTestResult>,
    /// Anderson-Darling test result (None if n < 8).
    pub anderson_darling: Option<NormalityTestResult>,
    /// The tests with no result above, each with its reason, in the order
    /// KS, Jarque-Bera, Shapiro-Wilk, Anderson-Darling. Empty when all four ran.
    pub omitted: Vec<OmittedTest>,
    /// Overall verdict: true if data appears normally distributed.
    /// Uses conservative approach: rejects normality if any test that ran
    /// rejects (and at least one ran); a test in `omitted` takes no part.
    pub is_normal: bool,
    /// Significance level used.
    pub significance_level: f64,
}

/// Result of fitting a parametric distribution via MLE.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Name of the fitted distribution (e.g., "Normal", "Exponential").
    pub distribution: String,
    /// Estimated parameters (name, value) pairs.
    pub parameters: Vec<(String, f64)>,
    /// Log-likelihood at MLE parameters.
    pub log_likelihood: f64,
    /// Akaike Information Criterion: -2*logL + 2k.
    pub aic: f64,
    /// Bayesian Information Criterion: -2*logL + k*ln(n).
    pub bic: f64,
    /// Number of estimated parameters (k).
    pub n_params: usize,
}

/// Complete result of distribution analysis.
#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    /// Number of observations.
    pub n: usize,
    /// ECDF (None if not computed or insufficient data).
    pub ecdf: Option<EcdfResult>,
    /// Histogram (None if not computed or insufficient data).
    pub histogram: Option<HistogramResult>,
    /// QQ-plot data (None if not computed or insufficient data).
    pub qq_plot: Option<QQPlotResult>,
    /// Normality assessment.
    pub normality: NormalityAssessment,
    /// Distribution fitting results, sorted by AIC (best first). Empty if not computed.
    pub fits: Vec<FitResult>,
}

// ── Public API ──────────────────────────────────────────────────────

/// Analyzes the distributional properties of a numeric data vector.
///
/// Requires clean data: no NaN, no infinite values.
///
/// # Arguments
///
/// * `data` — Slice of numeric observations (must be finite)
/// * `config` — Analysis configuration
///
/// # Returns
///
/// A [`DistributionAnalysis`] with ECDF, histogram, QQ-plot, and normality tests.
///
/// # Errors
///
/// - [`InsightError::ParameterOutOfRange`] if `significance_level` is not in
///   (0, 1), or a fixed bin count is 0
/// - [`InsightError::InsufficientData`] if fewer than 2 observations
/// - [`InsightError::MissingValues`] if data contains NaN
/// - [`InsightError::DegenerateData`] if data contains infinite values
///
/// # Example
///
/// ```
/// use u_insight::distribution::{distribution_analysis, DistributionConfig};
///
/// let data: Vec<f64> = (0..50).map(|i| (i as f64 - 25.0) * 0.2).collect();
/// let result = distribution_analysis(&data, &DistributionConfig::default()).unwrap();
/// assert!(result.histogram.is_some());
/// assert!(result.normality.ks_test.is_some());
/// ```
pub fn distribution_analysis(
    data: &[f64],
    config: &DistributionConfig,
) -> Result<DistributionAnalysis, InsightError> {
    // ── Input validation ────────────────────────────────────────
    let n = data.len();
    // A fixed count of no bins used to give no histogram, and no reason.
    if let BinMethod::Fixed(bins) = config.bin_method {
        InsightError::check_count("bins", bins, 1)?;
    }
    check_significance_level("significance_level", config.significance_level)?;
    if n == 0 {
        return Err(InsightError::empty("data"));
    }
    if n < 2 {
        return Err(InsightError::too_few("data", 2, n));
    }

    let nan_count = data.iter().filter(|v| v.is_nan()).count();
    if nan_count > 0 {
        return Err(InsightError::MissingValues {
            column: "data".to_string(),
            count: nan_count,
        });
    }

    if let Some(index) = data.iter().position(|v| v.is_infinite()) {
        return Err(InsightError::ValueNotFinite {
            column: "data".into(),
            index,
        });
    }

    // ── ECDF ────────────────────────────────────────────────────
    let ecdf = if config.compute_ecdf {
        u_analytics::distribution::ecdf(data).map(|(values, probabilities)| EcdfResult {
            values,
            probabilities,
        })
    } else {
        None
    };

    // ── Histogram ───────────────────────────────────────────────
    let histogram = if config.compute_histogram {
        let bin_method = match config.bin_method {
            BinMethod::Sturges => u_analytics::distribution::BinMethod::Sturges,
            BinMethod::Scott => u_analytics::distribution::BinMethod::Scott,
            BinMethod::FreedmanDiaconis => u_analytics::distribution::BinMethod::FreedmanDiaconis,
            BinMethod::Fixed(k) => u_analytics::distribution::BinMethod::Fixed(k),
        };
        u_analytics::distribution::histogram_bins(data, bin_method).map(|bins| HistogramResult {
            n_bins: bins.n_bins,
            bin_width: bins.bin_width,
            edges: bins.edges,
            counts: bins.counts,
            method: config.bin_method,
        })
    } else {
        None
    };

    // ── QQ-plot ─────────────────────────────────────────────────
    let qq_plot = if config.compute_qq_plot {
        u_analytics::distribution::qq_plot_normal(data).map(|(theoretical, sample)| QQPlotResult {
            theoretical,
            sample,
        })
    } else {
        None
    };

    // ── Normality tests ─────────────────────────────────────────
    let alpha = config.significance_level;

    let ks_test = u_analytics::distribution::ks_test_normal(data).map(|(statistic, p_value)| {
        NormalityTestResult {
            statistic,
            p_value,
            rejected: p_value < alpha,
        }
    });

    let jarque_bera = u_analytics::testing::jarque_bera_test(data).map(|r| NormalityTestResult {
        statistic: r.statistic,
        p_value: r.p_value,
        rejected: r.p_value < alpha,
    });

    let shapiro_wilk = u_analytics::testing::shapiro_wilk_test(data).map(|r| NormalityTestResult {
        statistic: r.w,
        p_value: r.p_value,
        rejected: r.p_value < alpha,
    });

    let anderson_darling =
        u_analytics::testing::anderson_darling_test(data).map(|r| NormalityTestResult {
            statistic: r.statistic_star,
            p_value: r.p_value,
            rejected: r.p_value < alpha,
        });

    // Conservative verdict: normal only if no test rejects
    let tests: [Option<NormalityTestResult>; 4] =
        [ks_test, jarque_bera, shapiro_wilk, anderson_darling];
    let any_rejected = tests.iter().any(|t| t.is_some_and(|r| r.rejected));
    let any_available = tests.iter().any(|t| t.is_some());
    let is_normal = any_available && !any_rejected;

    let omitted = [
        (NormalityTest::KolmogorovSmirnov, ks_test.is_none()),
        (NormalityTest::JarqueBera, jarque_bera.is_none()),
        (NormalityTest::ShapiroWilk, shapiro_wilk.is_none()),
        (NormalityTest::AndersonDarling, anderson_darling.is_none()),
    ]
    .into_iter()
    .filter(|&(_, missing)| missing)
    .map(|(test, _)| omission(test, data.len()))
    .collect();

    let normality = NormalityAssessment {
        ks_test,
        jarque_bera,
        shapiro_wilk,
        anderson_darling,
        omitted,
        is_normal,
        significance_level: alpha,
    };

    // ── Distribution fitting ──────────────────────────────────
    let fits = if config.fit_distributions {
        u_analytics::distribution::fit_best(data)
            .into_iter()
            .map(|r| FitResult {
                distribution: r.distribution,
                parameters: r.parameters,
                log_likelihood: r.log_likelihood,
                aic: r.aic,
                bic: r.bic,
                n_params: r.n_params,
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(DistributionAnalysis {
        n,
        ecdf,
        histogram,
        qq_plot,
        normality,
        fits,
    })
}

// ── Convenience Functions ───────────────────────────────────────────

/// Computes only the ECDF for the given data.
///
/// Returns `None` if data is empty or contains non-finite values.
///
/// ```
/// use u_insight::distribution::ecdf;
///
/// let data = [3.0, 1.0, 2.0, 1.0, 4.0];
/// let result = ecdf(&data).unwrap();
/// assert_eq!(result.values, vec![1.0, 2.0, 3.0, 4.0]);
/// assert!((result.probabilities[3] - 1.0).abs() < 1e-10);
/// ```
pub fn ecdf(data: &[f64]) -> Option<EcdfResult> {
    u_analytics::distribution::ecdf(data).map(|(values, probabilities)| EcdfResult {
        values,
        probabilities,
    })
}

/// Computes histogram bins using the specified method.
///
/// Returns `None` if fewer than 2 data points, non-finite values, or zero range.
///
/// ```
/// use u_insight::distribution::{histogram, BinMethod};
///
/// let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
/// let result = histogram(&data, BinMethod::Sturges).unwrap();
/// assert!(result.n_bins >= 5);
/// assert_eq!(result.counts.iter().sum::<usize>(), 100);
/// ```
pub fn histogram(data: &[f64], method: BinMethod) -> Option<HistogramResult> {
    let bin_method = match method {
        BinMethod::Sturges => u_analytics::distribution::BinMethod::Sturges,
        BinMethod::Scott => u_analytics::distribution::BinMethod::Scott,
        BinMethod::FreedmanDiaconis => u_analytics::distribution::BinMethod::FreedmanDiaconis,
        BinMethod::Fixed(k) => u_analytics::distribution::BinMethod::Fixed(k),
    };
    u_analytics::distribution::histogram_bins(data, bin_method).map(|bins| HistogramResult {
        n_bins: bins.n_bins,
        bin_width: bins.bin_width,
        edges: bins.edges,
        counts: bins.counts,
        method,
    })
}

/// Generates QQ-plot data comparing sample against a normal distribution.
///
/// Returns `None` if fewer than 3 data points or non-finite values.
///
/// ```
/// use u_insight::distribution::qq_plot;
///
/// let data = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
/// let result = qq_plot(&data).unwrap();
/// assert_eq!(result.theoretical.len(), 7);
/// assert_eq!(result.sample.len(), 7);
/// ```
pub fn qq_plot(data: &[f64]) -> Option<QQPlotResult> {
    u_analytics::distribution::qq_plot_normal(data).map(|(theoretical, sample)| QQPlotResult {
        theoretical,
        sample,
    })
}

/// Runs the Kolmogorov-Smirnov normality test.
///
/// Returns `None` if fewer than 5 observations, non-finite values, or zero variance.
///
/// ```
/// use u_insight::distribution::ks_test;
///
/// let data = [-1.2, -0.8, -0.3, 0.1, 0.5, 0.7, 1.1, 1.4];
/// let (statistic, p_value) = ks_test(&data).unwrap();
/// assert!(p_value > 0.05);
/// ```
pub fn ks_test(data: &[f64]) -> Option<(f64, f64)> {
    u_analytics::distribution::ks_test_normal(data)
}

/// Runs the Jarque-Bera normality test.
///
/// Returns `None` if fewer than 8 observations or non-finite values.
///
/// ```
/// use u_insight::distribution::jarque_bera;
///
/// let data = [-1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5];
/// let result = jarque_bera(&data).unwrap();
/// assert!(result.p_value > 0.05);
/// ```
pub fn jarque_bera(data: &[f64]) -> Option<NormalityTestResult> {
    u_analytics::testing::jarque_bera_test(data).map(|r| NormalityTestResult {
        statistic: r.statistic,
        p_value: r.p_value,
        rejected: r.p_value < 0.05,
    })
}

/// Fits multiple parametric distributions to data and returns results sorted by AIC.
///
/// Fits the continuous families that apply: Normal; Exponential, Gamma, LogNormal and
/// Weibull for positive data; Beta for data in (0, 1). Poisson is not ranked against
/// them -- a probability mass and a density are not comparable by AIC.
/// Returns an empty vector if no distribution can be fitted.
///
/// ```
/// use u_insight::distribution::fit_distributions;
///
/// let data: Vec<f64> = (1..=50).map(|i| i as f64).collect();
/// let fits = fit_distributions(&data);
/// assert!(!fits.is_empty());
/// assert!(fits[0].aic <= fits.last().unwrap().aic); // sorted by AIC
/// ```
pub fn fit_distributions(data: &[f64]) -> Vec<FitResult> {
    u_analytics::distribution::fit_best(data)
        .into_iter()
        .map(|r| FitResult {
            distribution: r.distribution,
            parameters: r.parameters,
            log_likelihood: r.log_likelihood,
            aic: r.aic,
            bic: r.bic,
            n_params: r.n_params,
        })
        .collect()
}

/// Runs the Shapiro-Wilk normality test.
///
/// The most powerful general normality test for small to moderate samples.
/// Returns `None` if n < 3, n > 5000, non-finite values, or zero variance.
///
/// ```
/// use u_insight::distribution::shapiro_wilk;
///
/// let data = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
/// let result = shapiro_wilk(&data).unwrap();
/// assert!(result.p_value > 0.05);
/// assert!(result.statistic > 0.9); // W close to 1 suggests normality
/// ```
pub fn shapiro_wilk(data: &[f64]) -> Option<NormalityTestResult> {
    u_analytics::testing::shapiro_wilk_test(data).map(|r| NormalityTestResult {
        statistic: r.w,
        p_value: r.p_value,
        rejected: r.p_value < 0.05,
    })
}

/// Runs the Anderson-Darling normality test.
///
/// More sensitive to tail deviations than Kolmogorov-Smirnov.
/// Returns `None` if n < 8, non-finite values, or zero variance.
///
/// ```
/// use u_insight::distribution::anderson_darling;
///
/// let data = [-1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5];
/// let result = anderson_darling(&data).unwrap();
/// assert!(result.p_value > 0.05);
/// ```
pub fn anderson_darling(data: &[f64]) -> Option<NormalityTestResult> {
    u_analytics::testing::anderson_darling_test(data).map(|r| NormalityTestResult {
        statistic: r.statistic_star,
        p_value: r.p_value,
        rejected: r.p_value < 0.05,
    })
}

// ── Grubbs' Test ────────────────────────────────────────────────────

/// Result of a Grubbs' test for outliers.
#[derive(Debug, Clone, Copy)]
pub struct GrubbsResult {
    /// The Grubbs' test statistic G = max|x_i - x̄| / s.
    pub statistic: f64,
    /// The critical value at the given significance level.
    pub critical_value: f64,
    /// The index of the suspected outlier.
    pub outlier_index: usize,
    /// The value of the suspected outlier.
    pub outlier_value: f64,
    /// Whether the test rejects the null hypothesis (i.e., outlier detected).
    pub is_outlier: bool,
}

/// Runs Grubbs' test for a single outlier in univariate data.
///
/// Grubbs' test detects a single outlier in a normally-distributed dataset.
/// The test statistic is G = max|x_i - x̄| / s, compared against a critical
/// value derived from the t-distribution.
///
/// Reference: Grubbs (1969). "Procedures for detecting outlying observations
/// in samples", Technometrics.
///
/// # Arguments
///
/// * `data` — numeric data (must have n ≥ 3)
/// * `alpha` — significance level (default: 0.05)
///
/// # Returns
///
/// The result, or a refusal:
/// - [`InsightError::ParameterOutOfRange`] (`alpha`) unless `0 < alpha < 1`
/// - [`InsightError::InsufficientData`] (`data`) for fewer than 3 values
/// - [`InsightError::ValueNotFinite`] (`data`, `index`) for a NaN or infinity
/// - [`InsightError::DegenerateData`] (`data`) when every value is the same:
///   the statistic divides by the standard deviation
///
/// ```
/// use u_insight::distribution::grubbs_test;
///
/// let data = [2.0, 2.1, 2.2, 2.0, 2.1, 2.3, 50.0]; // 50.0 is a clear outlier
/// let result = grubbs_test(&data, 0.05).unwrap();
/// assert_eq!(result.outlier_index, 6);
/// assert!(result.is_outlier);
/// ```
pub fn grubbs_test(data: &[f64], alpha: f64) -> Result<GrubbsResult, InsightError> {
    check_significance_level("alpha", alpha)?;
    let n = data.len();
    if n < 3 {
        return Err(InsightError::too_few("data", 3, n));
    }
    if let Some(index) = data.iter().position(|v| !v.is_finite()) {
        return Err(InsightError::ValueNotFinite {
            column: "data".into(),
            index,
        });
    }

    // Compute mean and std
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std_dev = variance.sqrt();

    // No spread beyond the rounding of the mean, judged against the data's
    // own size -- a fixed floor would call data in a small unit constant.
    if std_dev <= 1e-12 * mean.abs() {
        return Err(InsightError::DegenerateData {
            parameter: Some("data".into()),
            reason: "every value is the same: the Grubbs statistic is undefined".into(),
        });
    }

    // Find the point with maximum deviation from mean
    let (outlier_index, outlier_value) = data
        .iter()
        .enumerate()
        .max_by(|a, b| {
            (a.1 - mean)
                .abs()
                .partial_cmp(&(b.1 - mean).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("n >= 3 values were checked above");

    let statistic = (outlier_value - mean).abs() / std_dev;

    // Critical value: G_crit = ((n-1)/sqrt(n)) * sqrt(t²/(n-2+t²))
    // where t = t_{alpha/(2n), n-2}
    let alpha_adj = alpha / (2.0 * n as f64);
    let df = (n - 2) as f64;
    let t_val = u_numflow::special::t_distribution_quantile(1.0 - alpha_adj, df);
    let t2 = t_val * t_val;
    let critical_value = ((n - 1) as f64 / (n as f64).sqrt()) * (t2 / (df + t2)).sqrt();

    Ok(GrubbsResult {
        statistic,
        critical_value,
        outlier_index,
        outlier_value: *outlier_value,
        is_outlier: statistic > critical_value,
    })
}

/// A significance level is a probability strictly between 0 and 1: at 0
/// nothing is ever significant, at 1 everything is.
pub(crate) fn check_significance_level(name: &str, value: f64) -> Result<(), InsightError> {
    InsightError::check_range(name, value, Bound::Open(0.0), Bound::Open(1.0))
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn nearly_normal_data() -> Vec<f64> {
        // Roughly normal-looking data (symmetric around 0)
        vec![
            -2.5, -2.0, -1.8, -1.5, -1.2, -1.0, -0.8, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.8, 1.0,
            1.2, 1.5, 1.8, 2.0, 2.5,
        ]
    }

    fn uniform_data() -> Vec<f64> {
        (0..50).map(|i| i as f64 * 2.0).collect()
    }

    // ── distribution_analysis full pipeline ─────────────────────

    #[test]
    fn full_analysis_normal_data() {
        let data = nearly_normal_data();
        let config = DistributionConfig::default();
        let result = distribution_analysis(&data, &config).unwrap();

        assert_eq!(result.n, 20);
        assert!(result.ecdf.is_some());
        assert!(result.histogram.is_some());
        assert!(result.qq_plot.is_some());
        assert!(result.normality.ks_test.is_some());
        assert!(result.normality.jarque_bera.is_some());
        assert!(result.normality.shapiro_wilk.is_some());
        assert!(result.normality.anderson_darling.is_some());

        // Nearly normal data should not be rejected
        let ks = result.normality.ks_test.unwrap();
        assert!(ks.p_value > 0.05, "KS p = {}", ks.p_value);
        assert!(!ks.rejected);
        assert!(result.normality.is_normal);

        // All four tests should agree
        let sw = result.normality.shapiro_wilk.unwrap();
        assert!(sw.p_value > 0.05, "SW p = {}", sw.p_value);
        let ad = result.normality.anderson_darling.unwrap();
        assert!(ad.p_value > 0.05, "AD p = {}", ad.p_value);
    }

    #[test]
    fn full_analysis_uniform_data() {
        let data = uniform_data();
        let config = DistributionConfig::default();
        let result = distribution_analysis(&data, &config).unwrap();

        assert_eq!(result.n, 50);
        assert!(result.histogram.is_some());
        let hist = result.histogram.unwrap();
        assert_eq!(hist.counts.iter().sum::<usize>(), 50);
    }

    #[test]
    fn full_analysis_rejects_nan() {
        let data = vec![1.0, f64::NAN, 3.0];
        let config = DistributionConfig::default();
        let err = distribution_analysis(&data, &config).unwrap_err();
        assert!(matches!(err, InsightError::MissingValues { .. }));
    }

    #[test]
    fn full_analysis_rejects_infinity() {
        let data = vec![1.0, f64::INFINITY, 3.0];
        let config = DistributionConfig::default();
        let err = distribution_analysis(&data, &config).unwrap_err();
        assert_eq!(
            err,
            InsightError::ValueNotFinite {
                column: "data".into(),
                index: 1
            }
        );
    }

    #[test]
    fn full_analysis_rejects_insufficient() {
        let data = vec![1.0];
        let config = DistributionConfig::default();
        let err = distribution_analysis(&data, &config).unwrap_err();
        assert!(matches!(err, InsightError::InsufficientData { .. }));
    }

    #[test]
    fn config_skip_components() {
        let data = nearly_normal_data();
        let config = DistributionConfig {
            compute_ecdf: false,
            compute_histogram: false,
            compute_qq_plot: false,
            ..Default::default()
        };
        let result = distribution_analysis(&data, &config).unwrap();

        assert!(result.ecdf.is_none());
        assert!(result.histogram.is_none());
        assert!(result.qq_plot.is_none());
        // Normality tests always run
        assert!(result.normality.ks_test.is_some());
    }

    // ── ECDF ────────────────────────────────────────────────────

    #[test]
    fn ecdf_basic() {
        let data = [3.0, 1.0, 2.0, 1.0, 4.0];
        let result = ecdf(&data).unwrap();
        assert_eq!(result.values, vec![1.0, 2.0, 3.0, 4.0]);
        assert!((result.probabilities[0] - 0.4).abs() < 1e-10);
        assert!((result.probabilities[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ecdf_empty() {
        assert!(ecdf(&[]).is_none());
    }

    #[test]
    fn ecdf_nan() {
        assert!(ecdf(&[1.0, f64::NAN]).is_none());
    }

    // ── Histogram ───────────────────────────────────────────────

    #[test]
    fn histogram_sturges() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = histogram(&data, BinMethod::Sturges).unwrap();
        assert!(result.n_bins >= 5);
        assert_eq!(result.edges.len(), result.n_bins + 1);
        assert_eq!(result.counts.iter().sum::<usize>(), 100);
        assert_eq!(result.method, BinMethod::Sturges);
    }

    #[test]
    fn histogram_scott() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = histogram(&data, BinMethod::Scott).unwrap();
        assert!(result.n_bins >= 3);
        assert_eq!(result.counts.iter().sum::<usize>(), 100);
    }

    #[test]
    fn histogram_fd() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = histogram(&data, BinMethod::FreedmanDiaconis).unwrap();
        assert!(result.n_bins >= 3);
        assert_eq!(result.counts.iter().sum::<usize>(), 100);
    }

    #[test]
    fn histogram_insufficient() {
        assert!(histogram(&[1.0], BinMethod::Sturges).is_none());
    }

    #[test]
    fn histogram_constant() {
        assert!(histogram(&[5.0, 5.0, 5.0], BinMethod::Sturges).is_none());
    }

    #[test]
    fn histogram_fixed_exact_count() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = histogram(&data, BinMethod::Fixed(7)).unwrap();
        assert_eq!(result.n_bins, 7);
        assert_eq!(result.edges.len(), 8);
        assert_eq!(result.counts.iter().sum::<usize>(), 100);
        assert_eq!(result.method, BinMethod::Fixed(7));
    }

    #[test]
    fn histogram_fixed_zero_invalid() {
        assert!(histogram(&[1.0, 2.0, 3.0], BinMethod::Fixed(0)).is_none());
    }

    #[test]
    fn distribution_analysis_fixed_bins() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let config = DistributionConfig {
            bin_method: BinMethod::Fixed(5),
            ..Default::default()
        };
        let result = distribution_analysis(&data, &config).unwrap();
        let hist = result.histogram.expect("histogram requested");
        assert_eq!(hist.n_bins, 5);
        assert_eq!(hist.method, BinMethod::Fixed(5));
    }

    // ── QQ-plot ─────────────────────────────────────────────────

    #[test]
    fn qq_plot_basic() {
        let data = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let result = qq_plot(&data).unwrap();
        assert_eq!(result.theoretical.len(), 7);
        assert_eq!(result.sample.len(), 7);
        // Sorted sample
        assert!((result.sample[0] + 1.5).abs() < 1e-10);
        assert!((result.sample[6] - 1.5).abs() < 1e-10);
        // Median theoretical should be near 0
        assert!(result.theoretical[3].abs() < 0.1);
    }

    #[test]
    fn qq_plot_insufficient() {
        assert!(qq_plot(&[1.0, 2.0]).is_none());
    }

    // ── KS test ─────────────────────────────────────────────────

    #[test]
    fn ks_test_normal_data() {
        let data = nearly_normal_data();
        let (d, p) = ks_test(&data).unwrap();
        assert!(d > 0.0 && d < 1.0);
        assert!(p > 0.05, "p = {p}");
    }

    #[test]
    fn ks_test_insufficient() {
        assert!(ks_test(&[1.0, 2.0, 3.0, 4.0]).is_none());
    }

    // ── Jarque-Bera ─────────────────────────────────────────────

    #[test]
    fn jarque_bera_normal_data() {
        let data = nearly_normal_data();
        let result = jarque_bera(&data).unwrap();
        assert!(result.p_value > 0.05, "p = {}", result.p_value);
        assert!(!result.rejected);
    }

    /// The statistic this reports is the crate's, unchanged. Pinning the value
    /// here rather than only a threshold is what makes a silent divergence in
    /// the delegation visible: a wrapper that recomputed, rescaled or rounded
    /// would still pass a `p > 0.05` assertion.
    #[test]
    fn jarque_bera_passes_the_engine_statistic_through() {
        let data = [
            2.1, 3.4, 1.9, 5.6, 2.2, 3.1, 4.8, 2.9, 3.3, 7.2, 2.5, 3.0, 4.1, 2.7, 3.8, 2.4, 6.1,
            3.6, 2.8, 3.2,
        ];
        let here = jarque_bera(&data).expect("20 finite points");
        let engine = u_analytics::testing::jarque_bera_test(&data).expect("same data");
        assert_eq!(here.statistic, engine.statistic);
        assert_eq!(here.p_value, engine.p_value);
        // And that value is the moment form the cited paper defines.
        assert!(
            (here.statistic - 5.563_714_337_9).abs() < 1e-8,
            "JB = {}",
            here.statistic
        );
    }

    #[test]
    fn jarque_bera_insufficient() {
        assert!(jarque_bera(&[1.0, 2.0, 3.0]).is_none());
    }

    // ── Shapiro-Wilk ────────────────────────────────────────────

    #[test]
    fn shapiro_wilk_normal_data() {
        let data = nearly_normal_data();
        let result = shapiro_wilk(&data).unwrap();
        assert!(result.statistic > 0.9, "W = {}", result.statistic);
        assert!(result.p_value > 0.05, "p = {}", result.p_value);
        assert!(!result.rejected);
    }

    #[test]
    fn shapiro_wilk_insufficient() {
        assert!(shapiro_wilk(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn shapiro_wilk_constant() {
        assert!(shapiro_wilk(&[5.0, 5.0, 5.0, 5.0]).is_none());
    }

    // ── Tests that were not run ─────────────────────────────────

    fn varied(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (i as f64 * 1.3).sin() * 10.0 + (i % 7) as f64)
            .collect()
    }

    fn analysed(data: &[f64]) -> NormalityAssessment {
        distribution_analysis(data, &DistributionConfig::default())
            .expect("analysis")
            .normality
    }

    fn omitted_of(a: &NormalityAssessment) -> Vec<(&'static str, OmissionReason)> {
        a.omitted
            .iter()
            .map(|o| (o.test.name(), o.reason))
            .collect()
    }

    #[test]
    fn all_four_tests_run_within_their_sizes() {
        for n in [8, 100, 5000] {
            assert!(analysed(&varied(n)).omitted.is_empty(), "n = {n}");
        }
    }

    #[test]
    fn shapiro_wilk_is_omitted_above_5000_values_and_says_why() {
        let a = analysed(&varied(5001));
        assert!(a.shapiro_wilk.is_none());
        assert_eq!(
            omitted_of(&a),
            [("shapiro_wilk", OmissionReason::TooManyValues { max: 5000 })]
        );
        // The verdict combines the three that ran.
        assert!(a.ks_test.is_some() && a.jarque_bera.is_some() && a.anderson_darling.is_some());
    }

    #[test]
    fn short_samples_omit_the_tests_that_need_more_values() {
        let a = analysed(&varied(4));
        assert_eq!(
            omitted_of(&a),
            [
                ("ks_test", OmissionReason::TooFewValues { min: 5 }),
                ("jarque_bera", OmissionReason::TooFewValues { min: 8 }),
                ("anderson_darling", OmissionReason::TooFewValues { min: 8 }),
            ]
        );
        assert!(a.shapiro_wilk.is_some());
    }

    #[test]
    fn a_constant_sample_omits_every_test_it_has_the_values_for() {
        let a = analysed(&[5.0; 10]);
        assert_eq!(a.omitted.len(), 4);
        assert!(a
            .omitted
            .iter()
            .all(|o| o.reason == OmissionReason::ConstantValues));
        assert!(!a.is_normal);
    }

    // ── Anderson-Darling ────────────────────────────────────────

    #[test]
    fn anderson_darling_normal_data() {
        let data = nearly_normal_data();
        let result = anderson_darling(&data).unwrap();
        assert!(result.p_value > 0.05, "p = {}", result.p_value);
        assert!(!result.rejected);
    }

    #[test]
    fn anderson_darling_insufficient() {
        assert!(anderson_darling(&[1.0, 2.0, 3.0]).is_none());
    }

    // ── Distribution fitting ─────────────────────────────────────

    #[test]
    fn fit_distributions_positive_data() {
        let data: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let fits = fit_distributions(&data);
        assert!(!fits.is_empty());
        // Should be sorted by AIC
        for w in fits.windows(2) {
            assert!(
                w[0].aic <= w[1].aic,
                "AIC not sorted: {} > {}",
                w[0].aic,
                w[1].aic
            );
        }
        // Each fit should have valid fields
        for f in &fits {
            assert!(!f.distribution.is_empty());
            assert!(!f.parameters.is_empty());
            assert!(f.n_params > 0);
        }
    }

    #[test]
    fn fit_distributions_in_analysis() {
        let data: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let config = DistributionConfig {
            fit_distributions: true,
            ..Default::default()
        };
        let result = distribution_analysis(&data, &config).unwrap();
        assert!(!result.fits.is_empty());
        assert!(!result.fits[0].distribution.is_empty());
    }

    #[test]
    fn fit_distributions_default_off() {
        let data = nearly_normal_data();
        let config = DistributionConfig::default();
        let result = distribution_analysis(&data, &config).unwrap();
        assert!(result.fits.is_empty());
    }

    #[test]
    fn fit_distributions_empty() {
        let fits = fit_distributions(&[]);
        assert!(fits.is_empty());
    }

    // ── Normality assessment logic ──────────────────────────────

    #[test]
    fn normality_conservative_verdict() {
        // If data has too few points for JB (n < 8), but enough for KS (n >= 5),
        // verdict should rely on KS only.
        let data = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let config = DistributionConfig::default();
        let result = distribution_analysis(&data, &config).unwrap();

        assert!(result.normality.ks_test.is_some());
        assert!(result.normality.jarque_bera.is_none()); // n=7 < 8
    }

    #[test]
    fn normality_no_tests_available() {
        // Very small data: n=2, KS needs 5, JB needs 8
        let data = vec![1.0, 2.0];
        let config = DistributionConfig::default();
        let result = distribution_analysis(&data, &config).unwrap();

        assert!(result.normality.ks_test.is_none());
        assert!(result.normality.jarque_bera.is_none());
        assert!(!result.normality.is_normal); // No evidence of normality
    }

    #[test]
    fn bin_method_selection() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();

        // Different methods should generally produce different bin counts
        let sturges = histogram(&data, BinMethod::Sturges).unwrap();
        let fd = histogram(&data, BinMethod::FreedmanDiaconis).unwrap();

        // Both valid
        assert!(sturges.n_bins >= 2);
        assert!(fd.n_bins >= 2);
        assert_eq!(sturges.counts.iter().sum::<usize>(), 100);
        assert_eq!(fd.counts.iter().sum::<usize>(), 100);
    }

    // ── Grubbs' test ──────────────────────────────────────────────

    #[test]
    fn grubbs_detects_clear_outlier() {
        let data = [2.0, 2.1, 2.2, 2.0, 2.1, 2.3, 50.0];
        let result = grubbs_test(&data, 0.05).unwrap();
        assert_eq!(result.outlier_index, 6);
        assert!(result.is_outlier);
        assert!((result.outlier_value - 50.0).abs() < 1e-10);
    }

    #[test]
    fn grubbs_no_outlier_in_uniform() {
        // Tight data — no outlier expected
        let data = [1.0, 1.1, 1.2, 0.9, 1.05, 0.95, 1.15];
        let result = grubbs_test(&data, 0.05).unwrap();
        assert!(!result.is_outlier);
    }

    #[test]
    fn grubbs_too_few_points() {
        assert_eq!(
            grubbs_test(&[1.0, 2.0], 0.05).unwrap_err(),
            InsightError::too_few("data", 3, 2)
        );
        assert_eq!(
            grubbs_test(&[], 0.05).unwrap_err(),
            InsightError::too_few("data", 3, 0)
        );
    }

    #[test]
    fn grubbs_nan_is_refused_at_its_index() {
        let data = [1.0, f64::NAN, 3.0];
        assert_eq!(
            grubbs_test(&data, 0.05).unwrap_err(),
            InsightError::ValueNotFinite {
                column: "data".into(),
                index: 1
            }
        );
    }

    #[test]
    fn grubbs_constant_data_is_degenerate() {
        let data = [5.0, 5.0, 5.0, 5.0, 5.0];
        assert!(matches!(
            grubbs_test(&data, 0.05),
            Err(InsightError::DegenerateData {
                parameter: Some(p),
                ..
            }) if p == "data"
        ));
    }

    /// A significance level is in (0, 1): the ends, and anything past them,
    /// are refused with that interval -- Grubbs' `alpha` and the normality
    /// tests' `significance_level` alike.
    #[test]
    fn a_significance_level_is_in_the_open_unit_interval() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0];
        let series: Vec<f64> = (0..20).map(|i| i as f64).collect();
        for bad in [0.0, 1.0, -0.5, 1.5] {
            assert_eq!(
                grubbs_test(&data, bad).unwrap_err(),
                InsightError::ParameterOutOfRange {
                    name: "alpha".into(),
                    low: Bound::Open(0.0),
                    high: Bound::Open(1.0),
                    got: bad,
                }
            );
            let config = DistributionConfig {
                significance_level: bad,
                ..DistributionConfig::default()
            };
            assert_eq!(
                distribution_analysis(&series, &config).unwrap_err(),
                InsightError::ParameterOutOfRange {
                    name: "significance_level".into(),
                    low: Bound::Open(0.0),
                    high: Bound::Open(1.0),
                    got: bad,
                }
            );
        }
    }

    /// The critical value is Grubbs' own, which already accounts for the mean
    /// and deviation being estimated from the sample (unlike a χ² or normal
    /// cut-off): two-sided α = 0.05 gives 2.290 at n = 10 and 2.709 at n = 20
    /// (NIST/SEMATECH e-Handbook §1.3.5.17 table; Grubbs 1969).
    #[test]
    fn grubbs_critical_value_is_the_published_one() {
        for (n, want) in [(10usize, 2.290), (20, 2.709)] {
            let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
            let r = grubbs_test(&data, 0.05).unwrap();
            assert!(
                (r.critical_value - want).abs() < 1e-3,
                "n {n}: {}",
                r.critical_value
            );
        }
    }

    #[test]
    fn grubbs_statistic_positive() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let result = grubbs_test(&data, 0.05).unwrap();
        assert!(result.statistic > 0.0);
        assert!(result.critical_value > 0.0);
    }

    #[test]
    fn grubbs_stricter_alpha() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0];
        let lenient = grubbs_test(&data, 0.10).unwrap();
        let strict = grubbs_test(&data, 0.01).unwrap();
        // Same statistic, different critical values
        assert!((lenient.statistic - strict.statistic).abs() < 1e-10);
        assert!(strict.critical_value >= lenient.critical_value);
    }
}
