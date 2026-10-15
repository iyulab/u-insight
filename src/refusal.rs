//! The one shape a refusal takes on every transport.

use serde_json::json;

use crate::error::{Bound, InsightError};

/// The published vocabulary -- `{"refusals": [{"code", "fields"}],
/// "violations": []}`, the shape every engine package shares -- from
/// [`Code::ALL`]. This crate has no checker.
pub(crate) fn code_catalog_json() -> serde_json::Value {
    let refusals: Vec<serde_json::Value> = Code::ALL
        .into_iter()
        .map(|c| json!({ "code": c.name(), "fields": c.fields() }))
        .collect();
    json!({ "refusals": refusals, "violations": [] })
}

/// Every reason the crate's transports refuse with: the stable `code` a
/// refusal carries, and the fields that can stand beside it. A [`Refusal`] is
/// built from one of these, so the published list ([`Code::ALL`]) is the list
/// the crate refuses from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Code {
    MalformedInput,
    UnknownOption,
    MissingOption,
    InvalidOption,
    ParameterOutOfRange,
    InsufficientData,
    ValueNotFinite,
    DimensionMismatch,
    EmptyInput,
    MissingValues,
    NotAClassLabel,
    DegenerateData,
    ColumnNotFound,
    ComputationFailed,
    CsvParse,
    InvalidInput,
    Internal,
}

impl Code {
    pub(crate) const ALL: [Code; 17] = [
        Code::MalformedInput,
        Code::UnknownOption,
        Code::MissingOption,
        Code::InvalidOption,
        Code::ParameterOutOfRange,
        Code::InsufficientData,
        Code::ValueNotFinite,
        Code::DimensionMismatch,
        Code::EmptyInput,
        Code::MissingValues,
        Code::NotAClassLabel,
        Code::DegenerateData,
        Code::ColumnNotFound,
        Code::ComputationFailed,
        Code::CsvParse,
        Code::InvalidInput,
        Code::Internal,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Code::MalformedInput => "malformed_input",
            Code::UnknownOption => "unknown_option",
            Code::MissingOption => "missing_option",
            Code::InvalidOption => "invalid_option",
            Code::ParameterOutOfRange => "parameter_out_of_range",
            Code::InsufficientData => "insufficient_data",
            Code::ValueNotFinite => "value_not_finite",
            Code::DimensionMismatch => "dimension_mismatch",
            Code::EmptyInput => "empty_input",
            Code::MissingValues => "missing_values",
            Code::NotAClassLabel => "not_a_class_label",
            Code::DegenerateData => "degenerate_data",
            Code::ColumnNotFound => "column_not_found",
            Code::ComputationFailed => "computation_failed",
            Code::CsvParse => "csv_parse",
            Code::InvalidInput => "invalid_input",
            Code::Internal => "internal",
        }
    }

    /// The fields a refusal with this code can carry besides `code`.
    pub(crate) fn fields(self) -> &'static [&'static str] {
        match self {
            Code::MalformedInput => &["parameter", "column", "index"],
            Code::UnknownOption => &["parameter", "got", "expected"],
            Code::MissingOption => &["parameter", "expected"],
            Code::InvalidOption => &["parameter"],
            Code::ParameterOutOfRange => &[
                "parameter",
                "index",
                "min",
                "max",
                "exclusive_min",
                "exclusive_max",
                "got",
            ],
            Code::InsufficientData => &["parameter", "min", "got"],
            Code::ValueNotFinite => &["parameter", "index"],
            Code::DimensionMismatch => &["parameter", "index", "expected", "got"],
            Code::EmptyInput => &["parameter"],
            Code::MissingValues => &["column", "count"],
            Code::NotAClassLabel => &["parameter", "index", "got"],
            Code::DegenerateData => &["parameter"],
            Code::ColumnNotFound => &["column"],
            Code::ComputationFailed => &["operation"],
            Code::CsvParse => &["line"],
            Code::InvalidInput => &[],
            Code::Internal => &[],
        }
    }
}

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
    pub(crate) fn new(code: Code, message: String, mut extra: serde_json::Value) -> Self {
        if let Some(given) = extra.as_object() {
            debug_assert!(
                given.keys().all(|k| code.fields().contains(&k.as_str())),
                "{} refusal with undeclared fields: {:?}",
                code.name(),
                given.keys().collect::<Vec<_>>()
            );
        }
        let mut fields = serde_json::Map::new();
        fields.insert("code".into(), json!(code.name()));
        if let Some(extra) = extra.as_object_mut() {
            fields.append(extra);
        }
        Refusal {
            message,
            fields: serde_json::Value::Object(fields),
        }
    }

    /// The package failed, not the input: a result it could not hand back.
    #[cfg(feature = "wasm")]
    pub(crate) fn internal(message: String) -> Self {
        Self::new(Code::Internal, message, json!({}))
    }

    /// An argument that is not the shape the function takes: a JSON string
    /// instead of a value, a wrong type, a missing or unknown key.
    #[cfg(feature = "wasm")]
    pub(crate) fn malformed_input(parameter: &str, message: String) -> Self {
        Self::new(
            Code::MalformedInput,
            message,
            json!({ "parameter": parameter }),
        )
    }

    /// A string option that names none of the values the function knows.
    #[cfg(feature = "wasm")]
    pub(crate) fn unknown_option(parameter: &str, got: &str, expected: &[&str]) -> Self {
        let quoted: Vec<String> = expected.iter().map(|e| format!("{e:?}")).collect();
        Self::new(
            Code::UnknownOption,
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
            Code::EmptyInput,
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
            Code::ValueNotFinite,
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
            InsightError::CsvParse { line, .. } => (Code::CsvParse, json!({ "line": line })),
            InsightError::JsonParse { .. } => {
                (Code::MalformedInput, json!({ "parameter": "data" }))
            }
            InsightError::MissingValues { column, count } => (
                Code::MissingValues,
                json!({ "column": column, "count": count }),
            ),
            InsightError::EmptyInput { parameter } => {
                (Code::EmptyInput, json!({ "parameter": parameter }))
            }
            InsightError::InsufficientData {
                parameter,
                min_required,
                actual,
            } => (
                Code::InsufficientData,
                json!({ "parameter": parameter, "min": min_required, "got": actual }),
            ),
            InsightError::ParameterOutOfRange { name, got, .. } if !got.is_finite() => {
                (Code::ValueNotFinite, json!({ "parameter": name }))
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
                (Code::ParameterOutOfRange, fields)
            }
            InsightError::InvalidParameter { name, .. } => {
                (Code::InvalidOption, json!({ "parameter": name }))
            }
            InsightError::DegenerateData {
                parameter: Some(parameter),
                ..
            } => (Code::DegenerateData, json!({ "parameter": parameter })),
            InsightError::DegenerateData { .. } => (Code::DegenerateData, json!({})),
            InsightError::ComputationFailed { operation, .. } => {
                (Code::ComputationFailed, json!({ "operation": operation }))
            }
            InsightError::ColumnNotFound { name } => {
                (Code::ColumnNotFound, json!({ "column": name }))
            }
            InsightError::DimensionMismatch {
                expected,
                actual,
                index,
            } => (
                Code::DimensionMismatch,
                match index {
                    Some(i) => json!({ "expected": expected, "got": actual, "index": i }),
                    None => json!({ "expected": expected, "got": actual }),
                },
            ),
            InsightError::ValueNotFinite { column, index } => (
                Code::ValueNotFinite,
                json!({ "parameter": column, "index": index }),
            ),
            InsightError::Io(_) => (Code::Internal, json!({})),
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
                Refusal::new(Code::ParameterOutOfRange, message, fields)
            }
            E::TooFewObservations { needed, got } => Refusal::new(
                Code::InsufficientData,
                message,
                json!({ "parameter": "data", "min": needed, "got": got }),
            ),
            E::ValueNotFinite { index } => Refusal::value_not_finite("data", *index),
            _ => Refusal::new(Code::InvalidInput, message, json!({})),
        }
    }
}
