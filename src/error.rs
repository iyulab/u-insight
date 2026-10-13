//! Error types for u-insight.

use std::fmt;

/// One end of the interval a numeric option takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bound {
    /// The bound itself is accepted.
    Closed(f64),
    /// The bound itself is refused.
    Open(f64),
    /// No limit on this side.
    Unbounded,
}

impl Bound {
    /// Whether `v` lies on the accepted side of this bound as the lower end.
    fn admits_from_below(self, v: f64) -> bool {
        match self {
            Bound::Closed(b) => v >= b,
            Bound::Open(b) => v > b,
            Bound::Unbounded => true,
        }
    }

    /// Whether `v` lies on the accepted side of this bound as the upper end.
    fn admits_from_above(self, v: f64) -> bool {
        match self {
            Bound::Closed(b) => v <= b,
            Bound::Open(b) => v < b,
            Bound::Unbounded => true,
        }
    }
}

/// The interval in words, for a message: `>= 1`, `> 0`, `in (0, 1)`.
pub(crate) fn describe_range(low: Bound, high: Bound) -> String {
    match (low, high) {
        (Bound::Unbounded, Bound::Unbounded) => "a finite number".to_owned(),
        (Bound::Closed(a), Bound::Unbounded) => format!(">= {a}"),
        (Bound::Open(a), Bound::Unbounded) => format!("> {a}"),
        (Bound::Unbounded, Bound::Closed(b)) => format!("<= {b}"),
        (Bound::Unbounded, Bound::Open(b)) => format!("< {b}"),
        (low, high) => {
            let (l, a) = match low {
                Bound::Closed(a) => ('[', a),
                Bound::Open(a) => ('(', a),
                Bound::Unbounded => unreachable!("matched above"),
            };
            let (h, b) = match high {
                Bound::Closed(b) => (']', b),
                Bound::Open(b) => (')', b),
                Bound::Unbounded => unreachable!("matched above"),
            };
            format!("in {l}{a}, {b}{h}")
        }
    }
}

/// All errors produced by u-insight operations.
#[derive(Debug, Clone, PartialEq)]
pub enum InsightError {
    /// CSV parsing failed.
    CsvParse { line: usize, message: String },
    /// JSON parsing failed.
    JsonParse { message: String },
    /// Column contains missing values where none are allowed.
    MissingValues { column: String, count: usize },
    /// An input that has to hold something holds nothing: no points, no
    /// columns, no features. `parameter` names the input.
    EmptyInput { parameter: String },
    /// Fewer rows (points, observations) in `parameter` than the operation
    /// needs: `min_required` and `actual`. Used too when an option asks for
    /// more than the data can supply -- more clusters or neighbours than
    /// there are points -- since more data, not a different option, is what
    /// would let the call run.
    InsufficientData {
        parameter: String,
        min_required: usize,
        actual: usize,
    },
    /// A numeric option outside the interval it takes: `low` and `high` are
    /// the ends of that interval, `got` the value given. A NaN or an infinity
    /// is never in range.
    ParameterOutOfRange {
        name: String,
        low: Bound,
        high: Bound,
        got: f64,
    },
    /// An option refused for a reason no interval states: a combination of
    /// options that contradict each other, or a limit the data exceeds.
    InvalidParameter { name: String, message: String },
    /// Data is degenerate (constant columns, singular matrix, etc.).
    /// `parameter` names the input at fault when one can be named -- a
    /// constant column, for one.
    DegenerateData {
        parameter: Option<String>,
        reason: String,
    },
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

impl InsightError {
    /// Refuses `value` unless it is finite and lies in `low`..`high`.
    pub(crate) fn check_range(
        name: &str,
        value: f64,
        low: Bound,
        high: Bound,
    ) -> Result<(), InsightError> {
        if value.is_finite() && low.admits_from_below(value) && high.admits_from_above(value) {
            Ok(())
        } else {
            Err(InsightError::ParameterOutOfRange {
                name: name.into(),
                low,
                high,
                got: value,
            })
        }
    }

    /// Refuses a count below `min`.
    pub(crate) fn check_count(name: &str, value: usize, min: usize) -> Result<(), InsightError> {
        Self::check_range(
            name,
            value as f64,
            Bound::Closed(min as f64),
            Bound::Unbounded,
        )
    }

    /// No points, columns or features in `parameter`.
    pub(crate) fn empty(parameter: &str) -> Self {
        InsightError::EmptyInput {
            parameter: parameter.into(),
        }
    }

    /// Fewer than `min_required` rows in `parameter`.
    pub(crate) fn too_few(parameter: &str, min_required: usize, actual: usize) -> Self {
        InsightError::InsufficientData {
            parameter: parameter.into(),
            min_required,
            actual,
        }
    }

    /// Degenerate data no single input can be blamed for.
    pub(crate) fn degenerate(reason: impl Into<String>) -> Self {
        InsightError::DegenerateData {
            parameter: None,
            reason: reason.into(),
        }
    }
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
            Self::EmptyInput { parameter } => {
                write!(f, "{parameter} must not be empty")
            }
            Self::InsufficientData {
                parameter,
                min_required,
                actual,
            } => {
                write!(
                    f,
                    "{parameter}: need at least {min_required} rows, got {actual}"
                )
            }
            Self::ParameterOutOfRange {
                name,
                low,
                high,
                got,
            } => {
                write!(
                    f,
                    "invalid parameter '{name}': must be {}, got {got}",
                    describe_range(*low, *high)
                )
            }
            Self::InvalidParameter { name, message } => {
                write!(f, "invalid parameter '{name}': {message}")
            }
            Self::DegenerateData { reason, .. } => {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_range_admits_only_finite_values_inside_it() {
        let open = |v| InsightError::check_range("eps", v, Bound::Open(0.0), Bound::Unbounded);
        assert!(open(1e-9).is_ok());
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(open(bad).is_err(), "{bad}");
        }
        let closed = |v| InsightError::check_range("c", v, Bound::Closed(0.0), Bound::Closed(1.0));
        assert!(closed(0.0).is_ok() && closed(1.0).is_ok());
        assert!(closed(1.5).is_err());
    }

    #[test]
    fn a_range_reads_as_its_interval() {
        let e = InsightError::check_range("q", 1.0, Bound::Open(0.0), Bound::Open(1.0))
            .expect_err("1 is outside (0, 1)");
        assert_eq!(
            e.to_string(),
            "invalid parameter 'q': must be in (0, 1), got 1"
        );
        let e = InsightError::check_count("k", 0, 1).expect_err("below 1");
        assert_eq!(e.to_string(), "invalid parameter 'k': must be >= 1, got 0");
    }
}
