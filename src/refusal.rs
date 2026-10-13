//! The one shape a refusal takes on every transport.

use serde_json::json;

use crate::error::{Bound, InsightError};

/// A refusal as both transports report it: readable text, and the fields --
/// `code` first among them -- a program branches on. WebAssembly copies the
/// fields onto a JS `Error`; the C ABI returns them, next to `"error"`, from
/// `insight_last_error_json`. One mapping, so the two cannot drift.
#[derive(Debug)]
pub(crate) struct Refusal {
    pub(crate) message: String,
    pub(crate) fields: serde_json::Value,
}

impl Refusal {
    pub(crate) fn new(code: &str, message: String, mut extra: serde_json::Value) -> Self {
        let mut fields = serde_json::Map::new();
        fields.insert("code".into(), json!(code));
        if let Some(extra) = extra.as_object_mut() {
            fields.append(extra);
        }
        Refusal {
            message,
            fields: serde_json::Value::Object(fields),
        }
    }

    /// An argument that is not the shape the function takes: a JSON string
    /// instead of a value, a wrong type, a missing or unknown key.
    #[cfg(feature = "wasm")]
    pub(crate) fn malformed_input(parameter: &str, message: String) -> Self {
        Self::new(
            "malformed_input",
            message,
            json!({ "parameter": parameter }),
        )
    }

    /// A string option that names none of the values the function knows.
    #[cfg(feature = "wasm")]
    pub(crate) fn unknown_option(parameter: &str, got: &str, expected: &[&str]) -> Self {
        let quoted: Vec<String> = expected.iter().map(|e| format!("{e:?}")).collect();
        Self::new(
            "unknown_option",
            format!(
                "unknown {parameter} {got:?}; expected one of {}",
                quoted.join(", ")
            ),
            json!({ "parameter": parameter, "got": got, "expected": expected }),
        )
    }

    /// An input that has to hold at least one column.
    #[cfg(feature = "wasm")]
    pub(crate) fn empty_input(parameter: &str, message: &str) -> Self {
        Self::new(
            "empty_input",
            message.to_string(),
            json!({ "parameter": parameter }),
        )
    }

    /// A count below `min` -- or, with `max`, above it -- that the binding
    /// refuses before the analysis sees it (a negative count, one past what
    /// the platform's `usize` holds). The same shape the analysis gives a
    /// count it refuses itself.
    #[cfg(feature = "wasm")]
    pub(crate) fn count_out_of_range(
        parameter: &str,
        min: usize,
        max: Option<f64>,
        got: f64,
    ) -> Self {
        Refusal::from(InsightError::ParameterOutOfRange {
            name: parameter.into(),
            low: Bound::Closed(min as f64),
            high: max.map_or(Bound::Unbounded, Bound::Closed),
            got,
        })
    }

    /// A NaN or infinity at `index` of `parameter`.
    pub(crate) fn value_not_finite(parameter: &str, index: usize) -> Self {
        Self::new(
            "value_not_finite",
            format!("{parameter}[{index}] is not a finite number"),
            json!({ "parameter": parameter, "index": index }),
        )
    }

    /// The stable reason, as the `code` field carries it.
    #[cfg(all(test, feature = "wasm"))]
    pub(crate) fn code(&self) -> &str {
        self.fields["code"]
            .as_str()
            .expect("every refusal carries a code")
    }
}

impl From<InsightError> for Refusal {
    fn from(e: InsightError) -> Self {
        Refusal::from(&e)
    }
}

impl From<&InsightError> for Refusal {
    fn from(e: &InsightError) -> Self {
        let message = e.to_string();
        let (code, fields) = match e {
            InsightError::CsvParse { line, .. } => ("csv_parse", json!({ "line": line })),
            InsightError::JsonParse { .. } => ("malformed_input", json!({ "parameter": "data" })),
            InsightError::MissingValues { column, count } => (
                "missing_values",
                json!({ "column": column, "count": count }),
            ),
            InsightError::EmptyInput { parameter } => {
                ("empty_input", json!({ "parameter": parameter }))
            }
            InsightError::InsufficientData {
                parameter,
                min_required,
                actual,
            } => (
                "insufficient_data",
                json!({ "parameter": parameter, "min": min_required, "got": actual }),
            ),
            InsightError::ParameterOutOfRange { name, got, .. } if !got.is_finite() => {
                ("value_not_finite", json!({ "parameter": name }))
            }
            InsightError::ParameterOutOfRange {
                name,
                low,
                high,
                got,
            } => {
                // Each bound is sent only on the side it applies to: `min` /
                // `max` inclusive, `exclusive_min` / `exclusive_max` exclusive.
                let mut fields = json!({ "parameter": name });
                for (end, closed, open) in [
                    (low, "min", "exclusive_min"),
                    (high, "max", "exclusive_max"),
                ] {
                    match end {
                        Bound::Closed(v) => fields[closed] = json!(v),
                        Bound::Open(v) => fields[open] = json!(v),
                        Bound::Unbounded => {}
                    }
                }
                fields["got"] = json!(got);
                ("parameter_out_of_range", fields)
            }
            InsightError::InvalidParameter { name, .. } => {
                ("invalid_option", json!({ "parameter": name }))
            }
            InsightError::DegenerateData {
                parameter: Some(parameter),
                ..
            } => ("degenerate_data", json!({ "parameter": parameter })),
            InsightError::DegenerateData { .. } => ("degenerate_data", json!({})),
            InsightError::ComputationFailed { operation, .. } => {
                ("computation_failed", json!({ "operation": operation }))
            }
            InsightError::ColumnNotFound { name } => {
                ("column_not_found", json!({ "column": name }))
            }
            InsightError::DimensionMismatch {
                expected,
                actual,
                index,
            } => (
                "dimension_mismatch",
                match index {
                    Some(i) => json!({ "expected": expected, "got": actual, "index": i }),
                    None => json!({ "expected": expected, "got": actual }),
                },
            ),
            InsightError::ValueNotFinite { column, index } => (
                "value_not_finite",
                json!({ "parameter": column, "index": index }),
            ),
            InsightError::Io(_) => ("internal", json!({})),
        };
        Refusal::new(code, message, fields)
    }
}

impl From<&u_analytics::detection::SpectralResidualError> for Refusal {
    fn from(e: &u_analytics::detection::SpectralResidualError) -> Self {
        use u_analytics::detection::SpectralResidualError as E;
        let message = e.to_string();
        match e {
            E::OptionOutOfRange {
                option,
                got,
                min,
                exclusive_min,
                max,
                exclusive_max,
                ..
            } => {
                // Each bound is sent only on the side it applies to: `min` /
                // `max` inclusive, `exclusive_min` / `exclusive_max` exclusive.
                let mut fields = json!({ "parameter": option });
                for (key, bound) in [
                    ("min", min),
                    ("exclusive_min", exclusive_min),
                    ("max", max),
                    ("exclusive_max", exclusive_max),
                ] {
                    if let Some(v) = bound {
                        fields[key] = json!(v);
                    }
                }
                fields["got"] = json!(got);
                Refusal::new("parameter_out_of_range", message, fields)
            }
            E::TooFewObservations { needed, got } => Refusal::new(
                "insufficient_data",
                message,
                json!({ "parameter": "data", "min": needed, "got": got }),
            ),
            E::ValueNotFinite { index } => Refusal::value_not_finite("data", *index),
            _ => Refusal::new("invalid_input", message, json!({})),
        }
    }
}
