//! The one shape a refusal takes on every transport.

use serde_json::json;

use crate::error::InsightError;

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
            InsightError::InsufficientData {
                min_required,
                actual,
            } => (
                "insufficient_data",
                json!({ "min": min_required, "got": actual }),
            ),
            InsightError::InvalidParameter { name, .. } => {
                ("invalid_option", json!({ "parameter": name }))
            }
            InsightError::DegenerateData { .. } => ("degenerate_data", json!({})),
            InsightError::ComputationFailed { operation, .. } => {
                ("computation_failed", json!({ "operation": operation }))
            }
            InsightError::ColumnNotFound { name } => {
                ("column_not_found", json!({ "column": name }))
            }
            InsightError::DimensionMismatch { expected, actual } => (
                "dimension_mismatch",
                json!({ "expected": expected, "got": actual }),
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
            E::OptionOutOfRange { option, .. } => Refusal::new(
                "parameter_out_of_range",
                message,
                json!({ "parameter": option }),
            ),
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
