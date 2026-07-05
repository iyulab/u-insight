# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.11.0 onward; earlier entries list release dates only (see git history).

## [Unreleased]

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
