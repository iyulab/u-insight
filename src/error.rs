//! Error types for u-insight.

use std::fmt;

/// All errors produced by u-insight operations.
#[derive(Debug, Clone, PartialEq)]
pub enum InsightError {
    /// CSV parsing failed.
    CsvParse { line: usize, message: String },
    /// JSON parsing failed.
    JsonParse { message: String },
    /// Column contains missing values where none are allowed.
    MissingValues { column: String, count: usize },
    /// Insufficient data rows/samples for the requested operation.
    InsufficientData { min_required: usize, actual: usize },
    /// Invalid parameter value for the requested operation.
    InvalidParameter { name: String, message: String },
    /// Data is degenerate (constant columns, singular matrix, etc.).
    DegenerateData { reason: String },
    /// Internal computation failed (eigenvalue decomposition, matrix construction, etc.).
    ComputationFailed { operation: String, detail: String },
    /// Column not found in DataFrame.
    ColumnNotFound { name: String },
    /// An element whose length differs from the others: `expected` and `actual`
    /// lengths, and its position in its input (`None` when the mismatch is
    /// between two inputs rather than within one).
    DimensionMismatch {
        expected: usize,
        actual: usize,
        index: Option<usize>,
    },
    /// An infinite value (or a NaN where NaN is not read as missing) at
    /// `index` of the named input.
    ValueNotFinite { column: String, index: usize },
    /// I/O error during file reading.
    Io(String),
}

impl fmt::Display for InsightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CsvParse { line, message } => {
                write!(f, "CSV parse error at line {line}: {message}")
            }
            Self::JsonParse { message } => {
                write!(f, "JSON parse error: {message}")
            }
            Self::MissingValues { column, count } => {
                write!(f, "column '{column}' has {count} missing values")
            }
            Self::InsufficientData {
                min_required,
                actual,
            } => {
                write!(f, "need at least {min_required} rows, got {actual}")
            }
            Self::InvalidParameter { name, message } => {
                write!(f, "invalid parameter '{name}': {message}")
            }
            Self::DegenerateData { reason } => {
                write!(f, "degenerate data: {reason}")
            }
            Self::ComputationFailed { operation, detail } => {
                write!(f, "{operation} failed: {detail}")
            }
            Self::ColumnNotFound { name } => {
                write!(f, "column '{name}' not found")
            }
            Self::DimensionMismatch {
                expected,
                actual,
                index: Some(index),
            } => {
                write!(
                    f,
                    "element {index}: expected {expected} values, got {actual}"
                )
            }
            Self::DimensionMismatch {
                expected, actual, ..
            } => {
                write!(f, "expected {expected} elements, got {actual}")
            }
            Self::ValueNotFinite { column, index } => {
                write!(f, "{column}[{index}] is not a finite number")
            }
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for InsightError {}

impl From<std::io::Error> for InsightError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
