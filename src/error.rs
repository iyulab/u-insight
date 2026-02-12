//! Error types for u-insight.

use std::fmt;

/// All errors produced by u-insight operations.
#[derive(Debug, Clone, PartialEq)]
pub enum InsightError {
    /// CSV parsing failed.
    CsvParse { line: usize, message: String },
    /// Column contains missing values where none are allowed.
    MissingValues { column: String, count: usize },
    /// Column is not numeric where numeric data is required.
    NonNumericColumn { column: String },
    /// Insufficient data for the requested operation.
    InsufficientData { min_required: usize, actual: usize },
    /// Column not found in DataFrame.
    ColumnNotFound { name: String },
    /// Dimension mismatch.
    DimensionMismatch { expected: usize, actual: usize },
    /// I/O error during file reading.
    Io(String),
}

impl fmt::Display for InsightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CsvParse { line, message } => {
                write!(f, "CSV parse error at line {line}: {message}")
            }
            Self::MissingValues { column, count } => {
                write!(f, "column '{column}' has {count} missing values")
            }
            Self::NonNumericColumn { column } => {
                write!(f, "column '{column}' is not numeric")
            }
            Self::InsufficientData {
                min_required,
                actual,
            } => {
                write!(f, "need at least {min_required} rows, got {actual}")
            }
            Self::ColumnNotFound { name } => {
                write!(f, "column '{name}' not found")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "expected {expected} elements, got {actual}")
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
