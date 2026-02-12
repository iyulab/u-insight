//! CSV parser with automatic type inference.
//!
//! Parses CSV files into a [`DataFrame`](crate::dataframe::DataFrame)
//! with column types automatically inferred from content. The inference
//! priority is: Numeric → Boolean → Categorical → Text.
//!
//! # Features
//!
//! - RFC 4180 compliant (quoted fields, escaped quotes, commas in fields)
//! - Automatic type inference per column
//! - Standard null markers recognized: empty, `NA`, `N/A`, `null`, `NULL`, `None`, `.`
//! - Low-cardinality strings are dictionary-encoded as Categorical
//! - Configurable delimiter and null markers
//!
//! # Example
//!
//! ```
//! use u_insight::csv_parser::CsvParser;
//! use u_insight::dataframe::DataType;
//!
//! let csv = "name,value,active\nAlice,1.5,true\nBob,2.3,false\n";
//! let df = CsvParser::new().parse_str(csv).unwrap();
//! assert_eq!(df.row_count(), 2);
//! assert_eq!(df.column_count(), 3);
//! assert_eq!(df.column(0).unwrap().data_type(), DataType::Text);
//! assert_eq!(df.column(1).unwrap().data_type(), DataType::Numeric);
//! assert_eq!(df.column(2).unwrap().data_type(), DataType::Boolean);
//! ```

use crate::dataframe::{Column, DataFrame, DataType, ValidityBitmap};
use crate::error::InsightError;
use std::collections::HashMap;

/// Standard null value markers recognized during parsing.
const DEFAULT_NULL_MARKERS: &[&str] = &[
    "", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", ".",
    "NaN", "nan", "NAN", "#N/A", "#NA",
];

/// Maximum unique-value ratio for a column to be classified as Categorical
/// instead of Text. Default: 50%.
const CATEGORICAL_THRESHOLD: f64 = 0.5;

/// Maximum dictionary size for categorical columns.
const MAX_CATEGORICAL_UNIQUE: usize = 1000;

/// CSV parser configuration and entry point.
///
/// ```
/// use u_insight::csv_parser::CsvParser;
///
/// let csv = "a,b\n1,2\n3,4\n";
/// let df = CsvParser::new().parse_str(csv).unwrap();
/// assert_eq!(df.row_count(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct CsvParser {
    delimiter: u8,
    has_header: bool,
    null_markers: Vec<String>,
}

impl CsvParser {
    /// Creates a parser with default settings (comma delimiter, header row, standard null markers).
    pub fn new() -> Self {
        Self {
            delimiter: b',',
            has_header: true,
            null_markers: DEFAULT_NULL_MARKERS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        }
    }

    /// Sets the field delimiter (default: comma).
    pub fn delimiter(mut self, delim: u8) -> Self {
        self.delimiter = delim;
        self
    }

    /// Sets whether the first row is a header (default: true).
    pub fn has_header(mut self, header: bool) -> Self {
        self.has_header = header;
        self
    }

    /// Sets custom null markers (replaces defaults).
    pub fn null_markers(mut self, markers: Vec<String>) -> Self {
        self.null_markers = markers;
        self
    }

    /// Parses a CSV string into a DataFrame.
    pub fn parse_str(&self, input: &str) -> Result<DataFrame, InsightError> {
        // Strip BOM if present
        let input = input.strip_prefix('\u{feff}').unwrap_or(input);

        // Parse into raw rows
        let raw_rows = self.parse_raw(input)?;
        if raw_rows.is_empty() {
            return Ok(DataFrame::new());
        }

        // Extract header
        let (headers, data_rows) = if self.has_header {
            if raw_rows.is_empty() {
                return Ok(DataFrame::new());
            }
            let headers: Vec<String> = raw_rows[0].clone();
            (headers, &raw_rows[1..])
        } else {
            let n_cols = raw_rows[0].len();
            let headers: Vec<String> = (0..n_cols).map(|i| format!("col_{i}")).collect();
            (headers, &raw_rows[..])
        };

        if data_rows.is_empty() {
            return Ok(DataFrame::new());
        }

        let n_cols = headers.len();
        let n_rows = data_rows.len();

        // Transpose to column-major raw strings
        let mut raw_columns: Vec<Vec<String>> = vec![Vec::with_capacity(n_rows); n_cols];
        for (line_idx, row) in data_rows.iter().enumerate() {
            if row.len() != n_cols {
                return Err(InsightError::CsvParse {
                    line: if self.has_header {
                        line_idx + 2
                    } else {
                        line_idx + 1
                    },
                    message: format!(
                        "expected {n_cols} fields, got {}",
                        row.len()
                    ),
                });
            }
            for (col_idx, field) in row.iter().enumerate() {
                raw_columns[col_idx].push(field.clone());
            }
        }

        // Infer types and build columns
        let mut df = DataFrame::new();
        for (col_idx, raw_col) in raw_columns.iter().enumerate() {
            let col = self.build_column(raw_col);
            df.add_column(headers[col_idx].clone(), col)
                .expect("all columns same length");
        }

        Ok(df)
    }

    /// Parses a CSV file from disk into a DataFrame.
    pub fn parse_file(&self, path: &str) -> Result<DataFrame, InsightError> {
        let content = std::fs::read_to_string(path)?;
        self.parse_str(&content)
    }

    // ── Internal parsing ─────────────────────────────────────────

    /// Parses raw CSV text into rows of string fields.
    fn parse_raw(&self, input: &str) -> Result<Vec<Vec<String>>, InsightError> {
        let delim = self.delimiter as char;
        let mut rows: Vec<Vec<String>> = Vec::new();
        let mut current_row: Vec<String> = Vec::new();
        let mut current_field = String::new();
        let mut in_quotes = false;
        let mut chars = input.chars().peekable();
        let mut _line_num: usize = 1;

        while let Some(c) = chars.next() {
            if in_quotes {
                if c == '"' {
                    if chars.peek() == Some(&'"') {
                        // Escaped quote ""
                        chars.next();
                        current_field.push('"');
                    } else {
                        // End of quoted field
                        in_quotes = false;
                    }
                } else {
                    if c == '\n' {
                        _line_num += 1;
                    }
                    current_field.push(c);
                }
            } else if c == '"' && current_field.is_empty() {
                in_quotes = true;
            } else if c == delim {
                current_row.push(std::mem::take(&mut current_field));
            } else if c == '\n' {
                // Handle \r\n: strip trailing \r from field
                let field = if current_field.ends_with('\r') {
                    current_field.truncate(current_field.len() - 1);
                    std::mem::take(&mut current_field)
                } else {
                    std::mem::take(&mut current_field)
                };
                current_row.push(field);
                if !current_row.iter().all(|f| f.is_empty()) || !rows.is_empty() {
                    rows.push(std::mem::take(&mut current_row));
                } else {
                    current_row.clear();
                }
                _line_num += 1;
            } else if c == '\r' {
                // Standalone \r (old Mac style) - treat as newline
                if chars.peek() != Some(&'\n') {
                    current_row.push(std::mem::take(&mut current_field));
                    if !current_row.iter().all(|f| f.is_empty()) || !rows.is_empty() {
                        rows.push(std::mem::take(&mut current_row));
                    } else {
                        current_row.clear();
                    }
                    _line_num += 1;
                }
                // If \r\n, the \r is just ignored; \n handles the newline
            } else {
                current_field.push(c);
            }
        }

        // Handle last field/row (no trailing newline)
        if !current_field.is_empty() || !current_row.is_empty() {
            current_row.push(current_field);
            rows.push(current_row);
        }

        // Remove trailing empty rows
        while rows.last().is_some_and(|r| r.iter().all(|f| f.is_empty())) {
            rows.pop();
        }

        Ok(rows)
    }

    /// Checks if a trimmed value is a null marker.
    fn is_null(&self, value: &str) -> bool {
        let trimmed = value.trim();
        self.null_markers.iter().any(|m| m == trimmed)
    }

    /// Infers the column type and builds a typed Column.
    fn build_column(&self, raw_values: &[String]) -> Column {
        let n = raw_values.len();
        let trimmed: Vec<&str> = raw_values.iter().map(|s| s.trim()).collect();
        let null_flags: Vec<bool> = trimmed.iter().map(|s| self.is_null(s)).collect();

        // Count non-null values
        let non_null_count = null_flags.iter().filter(|&&is_null| !is_null).count();
        if non_null_count == 0 {
            // All null: default to numeric
            return Column::numeric(vec![0.0; n], ValidityBitmap::all_invalid(n));
        }

        // Try numeric
        let inferred = self.try_infer_type(&trimmed, &null_flags);

        match inferred {
            DataType::Numeric => self.build_numeric_column(&trimmed, &null_flags),
            DataType::Boolean => self.build_boolean_column(&trimmed, &null_flags),
            DataType::Categorical => self.build_categorical_column(&trimmed, &null_flags),
            DataType::Text => self.build_text_column(&trimmed, &null_flags),
        }
    }

    /// Determines the most specific type that fits all non-null values.
    fn try_infer_type(&self, values: &[&str], null_flags: &[bool]) -> DataType {
        let non_null: Vec<&str> = values
            .iter()
            .zip(null_flags.iter())
            .filter(|(_, &is_null)| !is_null)
            .map(|(&v, _)| v)
            .collect();

        // Try numeric
        if non_null.iter().all(|s| s.parse::<f64>().is_ok()) {
            return DataType::Numeric;
        }

        // Try boolean
        if non_null.iter().all(|s| is_boolean_str(s)) {
            return DataType::Boolean;
        }

        // Categorical vs Text: based on cardinality
        let mut unique = std::collections::HashSet::new();
        for &v in &non_null {
            unique.insert(v);
        }
        let ratio = unique.len() as f64 / non_null.len() as f64;
        if ratio < CATEGORICAL_THRESHOLD && unique.len() <= MAX_CATEGORICAL_UNIQUE {
            DataType::Categorical
        } else {
            DataType::Text
        }
    }

    fn build_numeric_column(&self, values: &[&str], null_flags: &[bool]) -> Column {
        let n = values.len();
        let mut nums = Vec::with_capacity(n);
        let mut validity = ValidityBitmap::empty();

        for (i, &val) in values.iter().enumerate() {
            if null_flags[i] {
                nums.push(0.0);
                validity.push(false);
            } else {
                nums.push(val.parse::<f64>().unwrap_or(0.0));
                validity.push(true);
            }
        }

        Column::numeric(nums, validity)
    }

    fn build_boolean_column(&self, values: &[&str], null_flags: &[bool]) -> Column {
        let n = values.len();
        let mut bools = Vec::with_capacity(n);
        let mut validity = ValidityBitmap::empty();

        for (i, &val) in values.iter().enumerate() {
            if null_flags[i] {
                bools.push(false);
                validity.push(false);
            } else {
                bools.push(parse_boolean_str(val));
                validity.push(true);
            }
        }

        Column::boolean(bools, validity)
    }

    fn build_categorical_column(&self, values: &[&str], null_flags: &[bool]) -> Column {
        let n = values.len();
        let mut dict_map: HashMap<String, u32> = HashMap::new();
        let mut dictionary: Vec<String> = Vec::new();
        let mut indices = Vec::with_capacity(n);
        let mut validity = ValidityBitmap::empty();

        for (i, &val) in values.iter().enumerate() {
            if null_flags[i] {
                indices.push(0);
                validity.push(false);
            } else {
                let idx = if let Some(&existing) = dict_map.get(val) {
                    existing
                } else {
                    let idx = dictionary.len() as u32;
                    dictionary.push(val.to_string());
                    dict_map.insert(val.to_string(), idx);
                    idx
                };
                indices.push(idx);
                validity.push(true);
            }
        }

        Column::categorical(dictionary, indices, validity)
    }

    fn build_text_column(&self, values: &[&str], null_flags: &[bool]) -> Column {
        let n = values.len();
        let mut texts = Vec::with_capacity(n);
        let mut validity = ValidityBitmap::empty();

        for (i, &val) in values.iter().enumerate() {
            if null_flags[i] {
                texts.push(String::new());
                validity.push(false);
            } else {
                texts.push(val.to_string());
                validity.push(true);
            }
        }

        Column::text(texts, validity)
    }
}

impl Default for CsvParser {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helper functions ──────────────────────────────────────────────────

/// Checks if a string represents a boolean value.
fn is_boolean_str(s: &str) -> bool {
    matches!(
        s.to_lowercase().as_str(),
        "true" | "false" | "yes" | "no" | "t" | "f" | "y" | "n"
    )
}

/// Parses a boolean string to `bool`.
fn parse_boolean_str(s: &str) -> bool {
    matches!(
        s.to_lowercase().as_str(),
        "true" | "yes" | "t" | "y"
    )
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic CSV parsing ────────────────────────────────────────

    #[test]
    fn parse_simple_csv() {
        let csv = "a,b,c\n1,2,3\n4,5,6\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        assert_eq!(df.row_count(), 2);
        assert_eq!(df.column_count(), 3);
        assert_eq!(df.column_names(), &["a", "b", "c"]);
    }

    #[test]
    fn parse_numeric_columns() {
        let csv = "x,y\n1.5,2.7\n3.1,-4.2\n0,100\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let x = df.column_by_name("x").unwrap();
        assert_eq!(x.data_type(), DataType::Numeric);
        assert_eq!(x.as_numeric().unwrap(), &[1.5, 3.1, 0.0]);
    }

    #[test]
    fn parse_boolean_column() {
        let csv = "flag\ntrue\nfalse\nyes\nno\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let flag = df.column_by_name("flag").unwrap();
        assert_eq!(flag.data_type(), DataType::Boolean);
        assert_eq!(flag.as_boolean().unwrap(), &[true, false, true, false]);
    }

    #[test]
    fn parse_categorical_column() {
        // 3 unique values / 7 rows = 0.43 < 0.5 → categorical
        let csv = "status\nA\nB\nC\nA\nB\nA\nC\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let status = df.column_by_name("status").unwrap();
        assert_eq!(status.data_type(), DataType::Categorical);
        assert_eq!(status.category_at(0), Some("A"));
        assert_eq!(status.category_at(2), Some("C"));
        assert_eq!(status.category_at(5), Some("A"));
    }

    #[test]
    fn parse_text_column() {
        // High cardinality: all unique values
        let csv = "name\nAlice\nBob\nCharlie\nDave\nEve\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let name = df.column_by_name("name").unwrap();
        assert_eq!(name.data_type(), DataType::Text);
        assert_eq!(name.text_at(0), Some("Alice"));
    }

    #[test]
    fn parse_mixed_types() {
        // 2 unique categories / 5 rows = 0.4 < 0.5 → categorical
        let csv = "id,value,active,category\n1,10.5,true,A\n2,20.3,false,B\n3,30.1,true,A\n4,40.0,false,B\n5,50.5,true,A\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        assert_eq!(
            df.column_by_name("id").unwrap().data_type(),
            DataType::Numeric
        );
        assert_eq!(
            df.column_by_name("value").unwrap().data_type(),
            DataType::Numeric
        );
        assert_eq!(
            df.column_by_name("active").unwrap().data_type(),
            DataType::Boolean
        );
        assert_eq!(
            df.column_by_name("category").unwrap().data_type(),
            DataType::Categorical
        );
    }

    // ── Null handling ────────────────────────────────────────────

    #[test]
    fn parse_null_markers() {
        let csv = "x\n1.0\nNA\n3.0\n\n5.0\nnull\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let x = df.column_by_name("x").unwrap();
        assert_eq!(x.data_type(), DataType::Numeric);
        assert_eq!(x.null_count(), 3); // NA, empty, null
        assert!(x.is_valid(0));
        assert!(!x.is_valid(1));
        assert!(x.is_valid(2));
        assert!(!x.is_valid(3));
        assert!(x.is_valid(4));
        assert!(!x.is_valid(5));
    }

    #[test]
    fn all_null_column() {
        let csv = "x\nNA\n\nnull\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let x = df.column_by_name("x").unwrap();
        assert_eq!(x.data_type(), DataType::Numeric); // defaults to numeric
        assert_eq!(x.null_count(), 3);
    }

    #[test]
    fn nan_marker_as_null() {
        let csv = "x\n1.0\nNaN\n3.0\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let x = df.column_by_name("x").unwrap();
        assert_eq!(x.null_count(), 1); // NaN treated as null
        assert!(!x.is_valid(1));
    }

    // ── Quoted fields ────────────────────────────────────────────

    #[test]
    fn parse_quoted_fields() {
        let csv = "name,desc\nAlice,\"hello, world\"\nBob,\"she said \"\"hi\"\"\"\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let desc = df.column_by_name("desc").unwrap();
        assert_eq!(desc.text_at(0), Some("hello, world"));
        assert_eq!(desc.text_at(1), Some("she said \"hi\""));
    }

    #[test]
    fn parse_quoted_newlines() {
        let csv = "name,note\nAlice,\"line1\nline2\"\nBob,simple\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        assert_eq!(df.row_count(), 2);
        let note = df.column_by_name("note").unwrap();
        assert_eq!(note.text_at(0), Some("line1\nline2"));
        assert_eq!(note.text_at(1), Some("simple"));
    }

    // ── Edge cases ───────────────────────────────────────────────

    #[test]
    fn parse_crlf_line_endings() {
        let csv = "a,b\r\n1,2\r\n3,4\r\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        assert_eq!(df.row_count(), 2);
        let a = df.column_by_name("a").unwrap();
        assert_eq!(a.as_numeric().unwrap(), &[1.0, 3.0]);
    }

    #[test]
    fn parse_no_trailing_newline() {
        let csv = "x\n1\n2\n3";
        let df = CsvParser::new().parse_str(csv).unwrap();
        assert_eq!(df.row_count(), 3);
    }

    #[test]
    fn parse_bom() {
        let csv = "\u{feff}x,y\n1,2\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        assert_eq!(df.column_names(), &["x", "y"]);
    }

    #[test]
    fn parse_empty_csv() {
        let csv = "";
        let df = CsvParser::new().parse_str(csv).unwrap();
        assert_eq!(df.row_count(), 0);
        assert_eq!(df.column_count(), 0);
    }

    #[test]
    fn parse_header_only() {
        let csv = "a,b,c\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        assert_eq!(df.row_count(), 0);
        assert_eq!(df.column_count(), 0);
    }

    #[test]
    fn parse_column_count_mismatch_error() {
        let csv = "a,b\n1,2\n3\n";
        let result = CsvParser::new().parse_str(csv);
        assert!(result.is_err());
    }

    #[test]
    fn parse_without_header() {
        let csv = "1,2\n3,4\n";
        let df = CsvParser::new().has_header(false).parse_str(csv).unwrap();
        assert_eq!(df.row_count(), 2);
        assert_eq!(df.column_names(), &["col_0", "col_1"]);
    }

    #[test]
    fn parse_tab_delimiter() {
        let csv = "a\tb\n1\t2\n3\t4\n";
        let df = CsvParser::new().delimiter(b'\t').parse_str(csv).unwrap();
        assert_eq!(df.row_count(), 2);
        assert_eq!(df.column_names(), &["a", "b"]);
    }

    #[test]
    fn parse_semicolon_delimiter() {
        let csv = "a;b\n1;2\n3;4\n";
        let df = CsvParser::new().delimiter(b';').parse_str(csv).unwrap();
        assert_eq!(df.row_count(), 2);
    }

    // ── Type inference edge cases ────────────────────────────────

    #[test]
    fn numeric_with_leading_spaces() {
        let csv = "x\n  1.5  \n  2.3  \n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let x = df.column_by_name("x").unwrap();
        assert_eq!(x.data_type(), DataType::Numeric);
        assert_eq!(x.as_numeric().unwrap(), &[1.5, 2.3]);
    }

    #[test]
    fn single_non_numeric_demotes_to_text() {
        let csv = "x\n1\n2\nthree\n4\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let x = df.column_by_name("x").unwrap();
        // Cannot be numeric because "three" doesn't parse as f64
        assert_ne!(x.data_type(), DataType::Numeric);
    }

    #[test]
    fn categorical_vs_text_threshold() {
        // 2 unique values / 4 rows = 0.5 → exactly at threshold (not categorical)
        // Actually 2/4 = 0.5 which is NOT < 0.5, so it's text
        let csv = "x\nA\nB\nA\nB\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let x = df.column_by_name("x").unwrap();
        // 2/4 = 0.5, threshold is < 0.5, so this is Text
        assert_eq!(x.data_type(), DataType::Text);
    }

    #[test]
    fn categorical_below_threshold() {
        // 2 unique values / 5 rows = 0.4 → categorical
        let csv = "x\nA\nB\nA\nB\nA\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let x = df.column_by_name("x").unwrap();
        assert_eq!(x.data_type(), DataType::Categorical);
    }

    #[test]
    fn boolean_mixed_formats() {
        let csv = "x\ntrue\nFalse\nYes\nno\nT\nf\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let x = df.column_by_name("x").unwrap();
        assert_eq!(x.data_type(), DataType::Boolean);
        assert_eq!(
            x.as_boolean().unwrap(),
            &[true, false, true, false, true, false]
        );
    }

    #[test]
    fn boolean_with_nulls() {
        let csv = "x\ntrue\nNA\nfalse\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let x = df.column_by_name("x").unwrap();
        assert_eq!(x.data_type(), DataType::Boolean);
        assert_eq!(x.null_count(), 1);
        assert!(!x.is_valid(1));
    }

    #[test]
    fn negative_and_scientific_notation() {
        let csv = "x\n-1.5\n2.3e10\n-4.5E-3\n";
        let df = CsvParser::new().parse_str(csv).unwrap();
        let x = df.column_by_name("x").unwrap();
        assert_eq!(x.data_type(), DataType::Numeric);
        assert_eq!(x.as_numeric().unwrap()[0], -1.5);
        assert!((x.as_numeric().unwrap()[1] - 2.3e10).abs() < 1.0);
        assert!((x.as_numeric().unwrap()[2] - (-4.5e-3)).abs() < 1e-10);
    }

    // ── Custom null markers ──────────────────────────────────────

    #[test]
    fn custom_null_markers() {
        let csv = "x\n1.0\n-999\n3.0\n";
        let df = CsvParser::new()
            .null_markers(vec!["-999".to_string()])
            .parse_str(csv)
            .unwrap();
        let x = df.column_by_name("x").unwrap();
        assert_eq!(x.null_count(), 1);
        assert!(!x.is_valid(1));
    }
}
