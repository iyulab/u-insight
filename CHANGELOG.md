# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.11.0 onward; earlier entries list release dates only (see git history).

## [Unreleased]

### Fixed

- **The Mahalanobis outlier threshold is the exact chi-squared quantile.** It
  was the Wilson-Hilferty cube-root approximation, which is off most where
  exploratory data sits -- one or two variables: 1.9 % low at the default
  0.975 (too many points flagged) and 3.0 % high at 0.999 (too few). The
  threshold now comes from the chi-squared quantile in `u-numflow`, and
  `threshold` values change for small dimension counts.
- **A `chi2_quantile` outside (0, 1) is refused** with `InvalidParameter`.
  The core computed a NaN threshold that no distance exceeds and reported no
  outliers; the C FFI (and so the C# `Mahalanobis`) silently replaced the value
  with 0.975. Both now return the error (`INSIGHT_ERR_INVALID_PARAM` across the
  C ABI, `InsightException` in C#).

## [0.21.0] - 2026-09-20

### Added

- **Every exported WASM function declares its return type.** All 17 were
  `(data: any, config: any) => any`, with the output's field *names* in the doc
  comment above and the element types only in the README. A consumer read
  `anomalies` as a list of indices where it is a per-point boolean mask,
  wrote `as WasmAnomalyResult`, and shipped a panel that printed the same
  subgroup twelve times and a chart overlay that silently drew nothing. `as`
  is the only thing that can be written against `any`, and it is exactly the
  construct that silences this.

  The declarations are derived from the structs the binding already
  serialises, so there is no second copy to drift: `tsify` emits the interface
  and `unchecked_return_type` names it in the signature. The runtime path is
  unchanged -- same serializer, same bytes. An optional field is declared
  `T | undefined`, which is what the binding sends.

  A publish-path check (`scripts/check-typed-dts.sh`) fails the release if any
  exported function returns `any`. It runs before publishing rather than
  beside it in CI, because the two run on the same push.

  Inputs remain `any`; they are validated at the boundary.


- **A scored point says whether it sits near a batch boundary** -- `near_edge`
  on the C record, the WASM object and `SrPoint.NearEdge` in C#. The transform
  appends five points before running and is circular, so its boundary handling
  moves the saliency of the five points at *either* end of a batch about eight
  times as much as the middle. A lone flag there can now be told from a
  detection in the body of the series. It marks a position, not a verdict.

  The C struct `CSrPoint` gains a trailing `bool`, so a caller that mirrors the
  layout has to add the field.

### Changed

- **A refused `spectral_residual` reports the one condition that failed.**
  Both the C entry point and the WASM binding answered every rejected option
  with the same sentence listing all of the rules, so a consumer could not tell
  a caller which setting to change and validated the options itself instead.
  The message now names the option and what it has to satisfy -- `threshold
  must be a finite number > 0` -- or states the shortfall in observations, or
  the position of the first non-finite value.

### Fixed

- **`estimate_period` landed below the true period when the series was not a
  whole number of cycles long** -- 63 over 300 observations came back as 60.
  Carried from `u-analytics`, which selected the autocorrelation hill on the
  biased estimator; its monotone shrinkage tilted the choice toward shorter
  lags. A candidate's `acf` is now the bias-corrected value it was judged on,
  so it is comparable against the `acf_threshold` shipped beside it.

## [0.20.1] - 2026-09-16

No change to this library. 0.20.0 was accepted by nuget.org but never became
installable: the feed reports the version as already present, while every read
path -- the flat container, the search index, the registration index and the
package page -- says it is not there, so a restore of `UInsight` 0.20.0 fails
with NU1102 and resolves no higher than 0.19.0. The version cannot be pushed
again, so this release carries the same code under a version that consumers can
actually resolve. crates.io and npm are republished with it to keep the three
channels on one version.

## [0.20.0] - 2026-09-16

### Fixed

- **`distribution::jarque_bera` inherits a corrected statistic and p-value**
  (`u-analytics` `jarque_bera_test`): the moment form Jarque & Bera (1987)
  define rather than the bias-adjusted estimators, and a p-value that stays a
  number in the tail. The statistic changes for every sample; on a 20-point
  sample it moves from 7.498 to 5.564. Requires `u-analytics` 0.13. No API
  change here.

## [0.19.0] - 2026-09-15

Requires `u-analytics` 0.12 and `u-numflow` 0.6. `UInsight` NuGet **0.19.0** (lockstep).

### Changed (breaking)

- **`insight_boxcox_capability` takes the λ search range and reports when the
  search hit it.** Two parameters, `lambda_min` / `lambda_max`, follow `lsl`
  (`NaN` for both → the default `[-5, 5]`); `CBoxCoxCapabilityResult` gains a
  trailing `uint8_t lambda_at_bound`. Previously λ was searched over `[-2, 2]`
  only, and a maximum beyond it came back as `±1.9999996` with nothing to tell
  it from an interior estimate.
- Specification limits are optional: with neither, λ is estimated and every
  index is `NaN`.
- C#: `InsightClient.BoxCoxCapability(data, usl, lsl, lambdaRange)` —
  `BoxCoxCapabilityResult.LambdaAtBound`, and `Indices` is `null` without limits.

### Changed

- Normal tail probabilities throughout (Anderson-Darling, rank-test p-values,
  sigma level ↔ PPM) follow the tail-precise normal functions of the analytics
  layer; returned values change in their trailing digits.

## [0.18.0] - 2026-09-13

### Added

- **Time-series primitives on every transport.** `insight_estimate_period`
  (with `insight_free_period_estimate`) returns the dominant period of a
  univariate series by AutoPeriod — permutation-thresholded periodogram
  peaks refined on the autocorrelation function, deterministic for a series
  — as a `CPeriodEstimate` whose `period` is 0 when nothing is periodic and
  whose candidates list every validated period. `insight_spectral_residual`
  (with `insight_free_spectral_residual_result`, and a null-able
  `CSpectralResidualOptions` for the defaults of Ren et al. 2019) scores every
  point by spectral residual saliency, with an expected value and a coverage
  band per point. C# `EstimatePeriod` and `SpectralResidual(data, options?)`
  with `PeriodEstimate`, `SpectralResidualOptions`, `SrPoint` and
  `SpectralResidualResult`; WASM `estimate_period` and `spectral_residual`
  with the same JSON shapes as `@iyulab/u-analytics`. Both are pure Rust: no
  native math library is required on any platform.

### Changed

- **`u-analytics` is now required at 0.11 and `u-numflow` at 0.5** — the
  versions that carry `seasonality`, `detection::SpectralResidual` and
  `fourier`.

## [0.17.0] - 2026-09-12

### Changed

- **`u-analytics` is now required at 0.10.** Its C FFI and WASM binding share
  one JSON contract from that version, and `boxcox_capability` and
  `compute_overall` report long-term indices only; the two entries below are
  this crate's side of those changes.
- **`insight_process_capability` no longer reports short-term indices when
  `sigma_within` is `NaN`.** It used the overall sigma for both roles and
  documented the result -- "Cp == Pp and Cpk == Ppk" -- which is a long-term
  number under a short-term name. Without a within sigma `cp`, `cpk`, `cpu`,
  `cpl` and `std_dev_within` are now `NaN` (the C# `CapabilityIndices` maps
  them to `null`; `StdDevWithin` becomes `double?`), and only the long-term
  indices are reported. Callers with individual observations get the
  short-term indices by estimating the within sigma from a control chart
  (`insight_imr_chart` returns it) and passing it explicitly. **Breaking**
  for callers reading those fields without supplying `sigma_within`; the
  values they were reading were `pp`/`ppk`. Comes with **`u-analytics` 0.10**,
  which makes the same change in the crate's `compute_overall`.
- **`CVariablesChartResult` carries `sigma_hat`** -- the within sigma the
  variation chart estimates (`R-bar / d2`, `S-bar / c4`, `MR-bar / d2`), NaN
  when it could not -- and the C# `VariablesChartResult` exposes it as
  `SigmaHat`. This is what makes the capability change above whole: the
  short-term indices need a within sigma, and the charts already had it but
  did not hand it out, so a caller would have had to rebuild the estimate
  from the MR center line by hand. A test pins that feeding the I-MR chart's
  `sigma_hat` into `insight_process_capability` yields `cp` from that sigma.
  **Breaking** for C callers laying out the struct themselves (one field
  added before `in_control`); the C# binding moves in lockstep.

- `insight_boxcox_capability` now returns `NaN` for `cp`, `cpk`, `cpu` and
  `cpl`. The upstream analysis stopped reporting short-term indices on the
  Box-Cox path, where they were being computed from the overall sigma and so
  equalled `pp`/`ppk` for every input. The long-term indices are unaffected.
  Comes with `u-analytics` 0.10.


## [0.16.0] - 2026-09-12

### Changed

- **`u-analytics` is now required at 0.9.** The control charts reached through
  the C FFI take subgroup sizes up to 25, not 10. The range is u-analytics' own,
  and a size outside it is rejected with that range in the message; this crate
  no longer restates a bound of its own, which had fallen behind.
- **Breaking (C FFI):** the chart entry points reject a row the chart cannot
  use -- a non-finite value, a subgroup with nothing inspected or more
  defectives than inspected, `units_inspected` that is not positive -- with
  `INSIGHT_ERR_INVALID_PARAM` and its index. They used to skip it
  (`insight_p_chart` and `insight_np_chart` documented doing so). The chart
  points carry no index, so every point after a skipped row was out of line
  with its input.
- **Breaking (C FFI):** `insight_process_capability` reports `cpm` as NaN when
  `target` is NaN. It used to measure Cpm against the specification midpoint;
  u-analytics 0.9 computes Cpm only against a declared target, from the spread
  of the data about it rather than from the within sigma. Pass the midpoint as
  `target` if that is the target.

### Fixed

- `insight_laney_p_chart` could return a chart with every limit NaN -- which
  reads as in control, because no comparison with NaN is true -- when one
  subgroup had a sample size of zero. u-analytics 0.9 refuses such a subgroup,
  and this entry point now names it.

## [0.15.0] - 2026-09-10

### Changed

- Track `u-analytics` 0.8: its WebAssembly capability binding changed shape and
  its control charts now expose the within-subgroup sigma they already computed.
  Nothing in this crate's own surface changes -- the dependency's Rust API is
  additive -- but a caret pin cannot cross a 0.x minor, so the version moves with
  it rather than silently holding an older snapshot.

## [0.14.0] - 2026-09-07

### Added

- **`insight_mann_kendall`**: Mann-Kendall non-parametric trend test with
  Sen's slope estimator (FFI + C# `InsightClient.MannKendall`). Exposes
  `u-analytics::testing::mann_kendall_test`.
- **`insight_kde`** / **`insight_free_kde_result`**: Gaussian kernel density
  estimation with Silverman/Scott/manual bandwidth selection (FFI + C#
  `InsightClient.Kde`). Exposes `u-analytics::distribution::kde`.
- **`insight_xbar_r_chart`**, **`insight_xbar_s_chart`**,
  **`insight_individual_mr_chart`**, **`insight_free_variables_chart_result`**:
  SPC variables control charts (X-bar-R, X-bar-S, Individual-MR) with
  Nelson-rule violation detection, exposed as bit flags per point (FFI + C#
  `InsightClient.XBarRChart` / `XBarSChart` / `IndividualMrChart`). Exposes
  `u-analytics::spc::{XBarRChart, XBarSChart, IndividualMRChart}`.
- **`insight_p_chart`**, **`insight_np_chart`**, **`insight_c_chart`**,
  **`insight_u_chart`**, **`insight_free_attribute_chart_result`**: SPC
  attributes control charts (proportion/count/defect-rate) with per-point
  control limits (FFI + C# `InsightClient.PChart` / `NpChart` / `CChart` /
  `UChart`). Exposes `u-analytics::spc::{PChart, NPChart, CChart, UChart}`.
- **`insight_laney_p_chart`**, **`insight_laney_u_chart`**,
  **`insight_free_laney_chart_result`**: overdispersion-adjusted P'/U'
  charts (FFI + C# `InsightClient.LaneyPChart` / `LaneyUChart`).
- **`insight_g_chart`**, **`insight_t_chart`**,
  **`insight_free_rare_event_chart_result`**: rare-event control charts
  (geometric/exponential distributions) (FFI + C# `InsightClient.GChart` /
  `TChart`). Exposes `u-analytics::spc::{laney_p_chart, laney_u_chart,
  g_chart, t_chart}`.

SPC control charts are now fully exposed.
- **`insight_process_capability`**: standard capability indices
  (Cp/Cpk/Pp/Ppk/Cpm) (FFI + C# `InsightClient.ProcessCapability`).
- **`insight_boxcox_capability`**: non-normal process capability via
  Box-Cox transformation (FFI + C# `InsightClient.BoxCoxCapability`).
- **`insight_percentile_capability`**: percentile-based (ISO 22514-2)
  process capability (FFI + C# `InsightClient.PercentileCapability`).
- **`insight_sigma_to_ppm`** / **`insight_ppm_to_sigma`**: sigma quality
  level <-> PPM defect rate conversions (FFI + C# `InsightClient.SigmaToPpm`
  / `PpmToSigma`). Exposes `u-analytics::capability::*`.

Process capability indices are now fully exposed.
- **`insight_weibull_mle`** / **`insight_weibull_mrr`**: Weibull parameter
  fitting via Maximum Likelihood Estimation and Median Rank Regression
  (FFI + C# `InsightClient.WeibullMle` / `WeibullMrr`).
- **`insight_weibull_reliability`**, **`insight_weibull_hazard_rate`**,
  **`insight_weibull_mtbf`**, **`insight_weibull_time_to_reliability`**,
  **`insight_weibull_b_life`**: Weibull reliability analysis — survival
  function, hazard rate, MTBF, and B-life (FFI + C#
  `InsightClient.Weibull*`). Exposes `u-analytics::weibull::*`.

**All 5 originally planned `u-analytics` domains are now exposed through
this crate's FFI**: Mann-Kendall, KDE, SPC control charts, process
capability, and Weibull reliability.

### Changed

- **`rand` has been dropped** rather than updated. The dependency was declared
  but never used — a pin's presence is not evidence of use — so the 0.9-to-0.10
  ecosystem migration removed it here instead of bumping it.
- **`u-analytics` is now required at 0.7 and `u-numflow` at 0.4** (previously 0.6
  and 0.3), following those crates' own releases.
- **The minimum supported Rust version is now declared as 1.85** and is verified
  by building on that exact toolchain; 1.84 and below fail. The crate previously
  declared no `rust-version` at all.

## [0.13.0] - 2026-07-07

### Changed

- **The package now declares `rust-version = "1.85"`.** This was previously
  undeclared, so a toolchain too old to build the crate failed somewhere inside
  a dependency instead of reporting the requirement. The value is verified by
  building on that exact toolchain.
- **The `rand` dependency has been dropped.** It was declared but never used —
  the sampling this crate performs runs on its own linear congruential
  generator. Nothing in the public surface changes; builds pull one fewer
  dependency tree.
- **`getrandom` is now 0.4** on WebAssembly targets, reaching the browser
  entropy source through its `wasm_js` crate feature alone.
- **`hierarchical` is now O(n²) instead of O(n³)** (all bindings). The merge
  phase rescanned every active pair each step, so on normal dataset sizes it
  froze the calling thread — ~1.4 s at n=2000, ~7.5 s at n=3000, ~12.5 s at
  n=4000 — blocking the browser main thread (an effective DoS on ordinary
  input). Replaced the naive min-scan with the nearest-neighbor-chain algorithm
  (valid for all supported reducible linkages: single/complete/average/Ward);
  measured n=4000 drops from ~12.5 s to ~66 ms (~190×), with clean O(n²)
  scaling. Output is unchanged for inputs in general position (verified against
  a brute-force reference across every linkage); for tied distances the `merges`
  array *ordering* may differ, but the tree and flat labels are equivalent.

### Added

- **`max_points` memory guard for `hierarchical`.** Because the O(n²) rewrite
  makes time cheap, the O(n²) distance matrix is now the binding constraint;
  inputs larger than `max_points` are rejected with a clear error (stating the
  count, the limit, and the estimated matrix size) before allocating. Exposed in
  the WASM config (`max_points?: number`); the Rust/WASM default is `10000`
  (≈400 MB matrix), `0` disables it. The C-FFI keeps prior *unlimited* behavior
  (its ABI has no override, so the guard is not imposed on native callers) — a
  pure speedup with no new rejection. Only crates.io + npm ship in this release;
  the C#/NuGet package (independent version track) is unchanged.

## [0.12.2] - 2026-07-05

### Fixed

- npm: expose the `./package.json` subpath in the `exports` map so tools
  that `require('<pkg>/package.json')` (license scanners, version
  reporters) keep working alongside the conditional exports introduced in
  the previous release (`ERR_PACKAGE_PATH_NOT_EXPORTED`).

## [0.12.1] - 2026-07-05

### Fixed

- **npm packaging — Node-compatible entry.** The npm package previously
  shipped only the wasm-bindgen *bundler*-target output, whose static
  `.wasm` import fails on Node's CJS path (`tsx`/`ts-node` in non-ESM
  packages) with an opaque `SyntaxError: Invalid or unexpected token`.
  The package now additionally ships the *nodejs*-target CJS glue under
  `node/` and routes Node consumers to it via a conditional `exports`
  map (`node` → CJS with filesystem wasm loading, `default` → bundler
  ESM). `require()`, native ESM `import`, and CJS TS runners all work
  without loader hooks. A pre-publish smoke test (CJS `require` + ESM
  `import`) now guards this path in CI. Rust API unchanged.

### Changed

- `u-numflow` dependency `^0.2` → `^0.3` (compatible; 0.3.0 publishes the
  previously-unreleased `wasm` feature and input-validation hardening —
  no API used by this crate changed).


## [0.12.0] - 2026-06-12

### Changed — BREAKING (WASM)

- WASM config/input objects (`dbscan`, `hierarchical`, `isolation_forest`,
  `lof`, `distribution_analysis`, `regression`, `feature_importance`,
  `detect_univariate_outliers`) now **reject unknown keys** with an explicit
  `unknown field` error instead of silently ignoring them
  (`serde(deny_unknown_fields)`). This is the guard for the defect class where
  a config typo (e.g. `fit` instead of `fit_distributions`) silently disabled
  a feature. Column-major *data* maps (`describe`, `correlation_matrix`,
  `predictors`/`features` values) are unaffected — column names stay free-form.

### Changed

- Dependency: `u-analytics` `^0.5` → `^0.6`.

## [0.11.0] - 2026-06-11

### Added

- WASM `distribution_analysis` config gains optional `bins` (integer >= 1):
  explicit histogram bin count. When set it takes precedence over
  `bin_method`, and the histogram `method` field echoes `"Fixed(n)"`.
- Core: `distribution::BinMethod::Fixed(usize)` (mirrors u-analytics 0.5.0).

### Changed

- Dependency: `u-analytics` `^0.4` → `^0.5`.
- Adding a variant to the public `BinMethod` enum breaks exhaustive `match`
  expressions in Rust consumers (WASM/FFI consumers unaffected).

## [0.10.1] - 2026-06-10

### Changed

- WASM: dropped legacy `*_json` parameter-name suffixes from 16 exported
  functions — they take native JS objects/arrays, and JSON-string arguments
  are now rejected early with a descriptive error. C header (FFI) unchanged.

## Earlier releases

- 0.10.0 — 2026-04-30
- 0.9.1 — 2026-04-29
- 0.9.0 — 2026-04-28
- 0.8.1 — 2026-04-28
- 0.8.0 — 2026-04-27
- 0.7.0 — 2026-04-03
- 0.6.0 — 2026-03-23
