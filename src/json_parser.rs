//! JSON parser with automatic type inference.
//!
//! Parses column-major JSON into a [`DataFrame`] with column types
//! automatically inferred from JSON value types. The inference priority
//! matches [`CsvParser`](crate::csv_parser::CsvParser):
//! Numeric → Boolean → Categorical → Text.
//!
//! # Input Format
//!
//! Column-major JSON object where each key is a column name and each value
//! is an array of values:
//!
//! ```json
//! {
//!   "age": [30, 25, null, 42],
//!   "name": ["Alice", "Bob", null, "Dave"],
//!   "active": [true, false, true, null]
//! }
//! ```
//!
//! # Type Inference
//!
//! | JSON type | Column type |
//! |-----------|-------------|
//! | `number`  | Numeric     |
//! | `bool`    | Boolean     |
//! | `string`  | Categorical or Text (based on cardinality) |
//! | `null`    | Missing (validity bitmap) |
//!
//! Mixed-type columns (e.g. numbers and strings) fall back to Text.
//!
//! # Example
//!
//! ```
//! use u_insight::json_parser::JsonParser;
//! use u_insight::dataframe::DataType;
//!
//! let json = r#"{"age": [30, 25, 42], "name": ["Alice", "Bob", "Dave"]}"#;
//! let df = JsonParser::new().parse_str(json).unwrap();
//! assert_eq!(df.row_count(), 3);
//! assert_eq!(df.column_count(), 2);
//! ```

use std::collections::HashMap;

use crate::dataframe::{Column, DataFrame, ValidityBitmap};
use crate::error::InsightError;

/// Maximum unique-value ratio for a column to be classified as Categorical
/// instead of Text. Default: 50%.
const CATEGORICAL_THRESHOLD: f64 = 0.5;

/// Maximum dictionary size for categorical columns.
const MAX_CATEGORICAL_UNIQUE: usize = 1000;

/// JSON parser configuration and entry point.
///
/// ```
/// use u_insight::json_parser::JsonParser;
///
/// let json = r#"{"x": [1, 2, 3], "y": [4, 5, 6]}"#;
/// let df = JsonParser::new().parse_str(json).unwrap();
/// assert_eq!(df.row_count(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct JsonParser {
    _private: (),
}

impl JsonParser {
    /// Creates a parser with default settings.
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Parses a column-major JSON string into a DataFrame.
    pub fn parse_str(&self, input: &str) -> Result<DataFrame, InsightError> {
        let parsed: serde_json::Value = serde_json::from_str(input).map_err(|e| {
            InsightError::JsonParse {
                message: e.to_string(),
            }
        })?;

        self.parse_value(&parsed)
    }

    /// Parses a column-major JSON value into a DataFrame.
    ///
    /// Accepts either an object `{"col": [values...]}` or a 2D array
    /// `[[row0...], [row1...]]` (auto-generates column names `col_0`, `col_1`, ...).
    pub fn parse_value(&self, value: &serde_json::Value) -> Result<DataFrame, InsightError> {
        match value {
            serde_json::Value::Object(map) => self.parse_column_major(map),
            serde_json::Value::Array(rows) => self.parse_row_major(rows),
            _ => Err(InsightError::JsonParse {
                message: "expected a JSON object or array".to_string(),
            }),
        }
    }

    /// Parses column-major format: `{"col1": [v1, v2, ...], "col2": [...]}`
    fn parse_column_major(
        &self,
        map: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<DataFrame, InsightError> {
        if map.is_empty() {
            return Ok(DataFrame::new());
        }

        // Sort keys for deterministic column order
        let mut keys: Vec<&String> = map.keys().collect();
        keys.sort();

        // Validate that all values are arrays of the same length
        let mut expected_len: Option<usize> = None;
        for key in &keys {
            let arr = map[*key].as_array().ok_or_else(|| InsightError::JsonParse {
                message: format!("column '{key}' value must be an array"),
            })?;
            match expected_len {
                None => expected_len = Some(arr.len()),
                Some(n) if n != arr.len() => {
                    return Err(InsightError::DimensionMismatch {
                        expected: n,
                        actual: arr.len(),
                    });
                }
                _ => {}
            }
        }

        if expected_len == Some(0) {
            return Ok(DataFrame::new());
        }

        let mut df = DataFrame::new();
        for key in &keys {
            let arr = map[*key].as_array().expect("validated above");
            let col = build_column_from_json_array(arr);
            df.add_column((*key).clone(), col)
                .expect("all columns same length");
        }

        Ok(df)
    }

    /// Parses row-major format: `[[v1, v2], [v3, v4]]`
    /// Auto-generates column names: `col_0`, `col_1`, ...
    fn parse_row_major(
        &self,
        rows: &[serde_json::Value],
    ) -> Result<DataFrame, InsightError> {
        if rows.is_empty() {
            return Ok(DataFrame::new());
        }

        // Determine number of columns from first row
        let first_row = rows[0].as_array().ok_or_else(|| InsightError::JsonParse {
            message: "row-major format requires arrays of arrays".to_string(),
        })?;
        let n_cols = first_row.len();
        if n_cols == 0 {
            return Ok(DataFrame::new());
        }

        // Transpose to column-major
        let mut columns: Vec<Vec<serde_json::Value>> = vec![Vec::with_capacity(rows.len()); n_cols];

        for (row_idx, row_val) in rows.iter().enumerate() {
            let row = row_val.as_array().ok_or_else(|| InsightError::JsonParse {
                message: format!("row {row_idx} is not an array"),
            })?;
            if row.len() != n_cols {
                return Err(InsightError::DimensionMismatch {
                    expected: n_cols,
                    actual: row.len(),
                });
            }
            for (col_idx, val) in row.iter().enumerate() {
                columns[col_idx].push(val.clone());
            }
        }

        let mut df = DataFrame::new();
        for (col_idx, col_values) in columns.iter().enumerate() {
            let col = build_column_from_json_array(col_values);
            df.add_column(format!("col_{col_idx}"), col)
                .expect("all columns same length");
        }

        Ok(df)
    }
}

impl Default for JsonParser {
    fn default() -> Self {
        Self::new()
    }
}

// ── Column building ──────────────────────────────────────────────────

/// Inferred JSON column type based on non-null value types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonColumnType {
    AllNull,
    Numeric,
    Boolean,
    String,
    Mixed,
}

/// Determines the dominant type of a JSON array.
fn infer_json_type(values: &[serde_json::Value]) -> JsonColumnType {
    let mut has_number = false;
    let mut has_bool = false;
    let mut has_string = false;

    for v in values {
        match v {
            serde_json::Value::Null => {}
            serde_json::Value::Number(_) => has_number = true,
            serde_json::Value::Bool(_) => has_bool = true,
            serde_json::Value::String(_) => has_string = true,
            // Arrays/objects inside a column → treat as string
            _ => has_string = true,
        }
    }

    let type_count = has_number as u8 + has_bool as u8 + has_string as u8;
    if type_count == 0 {
        return JsonColumnType::AllNull;
    }
    if type_count > 1 {
        return JsonColumnType::Mixed;
    }
    if has_number {
        JsonColumnType::Numeric
    } else if has_bool {
        JsonColumnType::Boolean
    } else {
        JsonColumnType::String
    }
}

/// Builds a typed Column from a JSON array with type inference.
fn build_column_from_json_array(values: &[serde_json::Value]) -> Column {
    let n = values.len();
    if n == 0 {
        return Column::numeric(Vec::new(), ValidityBitmap::empty());
    }

    let col_type = infer_json_type(values);

    match col_type {
        JsonColumnType::AllNull => {
            Column::numeric(vec![0.0; n], ValidityBitmap::all_invalid(n))
        }
        JsonColumnType::Numeric => build_numeric_column(values),
        JsonColumnType::Boolean => build_boolean_column(values),
        JsonColumnType::String => build_string_column(values),
        JsonColumnType::Mixed => build_mixed_as_text(values),
    }
}

fn build_numeric_column(values: &[serde_json::Value]) -> Column {
    let mut nums = Vec::with_capacity(values.len());
    let mut validity = ValidityBitmap::empty();

    for v in values {
        match v {
            serde_json::Value::Null => {
                nums.push(0.0);
                validity.push(false);
            }
            serde_json::Value::Number(n) => {
                nums.push(n.as_f64().unwrap_or(0.0));
                validity.push(true);
            }
            _ => {
                // Should not happen if infer_json_type returned Numeric
                nums.push(0.0);
                validity.push(false);
            }
        }
    }

    Column::numeric(nums, validity)
}

fn build_boolean_column(values: &[serde_json::Value]) -> Column {
    let mut bools = Vec::with_capacity(values.len());
    let mut validity = ValidityBitmap::empty();

    for v in values {
        match v {
            serde_json::Value::Null => {
                bools.push(false);
                validity.push(false);
            }
            serde_json::Value::Bool(b) => {
                bools.push(*b);
                validity.push(true);
            }
            _ => {
                bools.push(false);
                validity.push(false);
            }
        }
    }

    Column::boolean(bools, validity)
}

fn build_string_column(values: &[serde_json::Value]) -> Column {
    let strings: Vec<Option<&str>> = values
        .iter()
        .map(|v| match v {
            serde_json::Value::String(s) => Some(s.as_str()),
            serde_json::Value::Null => None,
            _ => None,
        })
        .collect();

    let non_null: Vec<&str> = strings.iter().filter_map(|s| *s).collect();
    if non_null.is_empty() {
        return Column::numeric(vec![0.0; values.len()], ValidityBitmap::all_invalid(values.len()));
    }

    // Categorical vs Text: based on cardinality
    let mut unique = std::collections::HashSet::new();
    for &v in &non_null {
        unique.insert(v);
    }
    let ratio = unique.len() as f64 / non_null.len() as f64;

    if ratio < CATEGORICAL_THRESHOLD && unique.len() <= MAX_CATEGORICAL_UNIQUE {
        build_categorical_from_strings(&strings)
    } else {
        build_text_from_strings(&strings)
    }
}

fn build_categorical_from_strings(values: &[Option<&str>]) -> Column {
    let mut dict_map: HashMap<&str, u32> = HashMap::new();
    let mut dictionary: Vec<String> = Vec::new();
    let mut indices = Vec::with_capacity(values.len());
    let mut validity = ValidityBitmap::empty();

    for val in values {
        match val {
            None => {
                indices.push(0);
                validity.push(false);
            }
            Some(s) => {
                let idx = if let Some(&existing) = dict_map.get(s) {
                    existing
                } else {
                    let idx = dictionary.len() as u32;
                    dictionary.push(s.to_string());
                    dict_map.insert(s, idx);
                    idx
                };
                indices.push(idx);
                validity.push(true);
            }
        }
    }

    Column::categorical(dictionary, indices, validity)
}

fn build_text_from_strings(values: &[Option<&str>]) -> Column {
    let mut texts = Vec::with_capacity(values.len());
    let mut validity = ValidityBitmap::empty();

    for val in values {
        match val {
            None => {
                texts.push(String::new());
                validity.push(false);
            }
            Some(s) => {
                texts.push(s.to_string());
                validity.push(true);
            }
        }
    }

    Column::text(texts, validity)
}

fn build_mixed_as_text(values: &[serde_json::Value]) -> Column {
    let mut texts = Vec::with_capacity(values.len());
    let mut validity = ValidityBitmap::empty();

    for v in values {
        match v {
            serde_json::Value::Null => {
                texts.push(String::new());
                validity.push(false);
            }
            serde_json::Value::String(s) => {
                texts.push(s.clone());
                validity.push(true);
            }
            other => {
                texts.push(other.to_string());
                validity.push(true);
            }
        }
    }

    Column::text(texts, validity)
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataType;

    #[test]
    fn parse_column_major_numeric() {
        let json = r#"{"x": [1, 2, 3], "y": [4.5, 5.5, 6.5]}"#;
        let df = JsonParser::new().parse_str(json).unwrap();
        assert_eq!(df.row_count(), 3);
        assert_eq!(df.column_count(), 2);
        assert_eq!(df.column(0).unwrap().data_type(), DataType::Numeric);
        assert_eq!(df.column(1).unwrap().data_type(), DataType::Numeric);
    }

    #[test]
    fn parse_column_major_mixed_types() {
        let json = r#"{
            "age": [30, 25, null, 42],
            "name": ["Alice", "Bob", null, "Dave"],
            "active": [true, false, true, null]
        }"#;
        let df = JsonParser::new().parse_str(json).unwrap();
        assert_eq!(df.row_count(), 4);
        assert_eq!(df.column_count(), 3);

        // Columns are sorted alphabetically
        let names = df.column_names();
        assert_eq!(names, &["active", "age", "name"]);

        assert_eq!(df.column_by_name("active").unwrap().data_type(), DataType::Boolean);
        assert_eq!(df.column_by_name("age").unwrap().data_type(), DataType::Numeric);
        // 3 unique / 3 non-null = 1.0 > 0.5 → Text
        assert_eq!(df.column_by_name("name").unwrap().data_type(), DataType::Text);
    }

    #[test]
    fn parse_column_major_with_nulls() {
        let json = r#"{"val": [1.0, null, 3.0]}"#;
        let df = JsonParser::new().parse_str(json).unwrap();
        let col = df.column(0).unwrap();
        assert_eq!(col.data_type(), DataType::Numeric);
        assert!(col.validity().is_valid(0));
        assert!(!col.validity().is_valid(1));
        assert!(col.validity().is_valid(2));
    }

    #[test]
    fn parse_all_null_column() {
        let json = r#"{"x": [null, null, null]}"#;
        let df = JsonParser::new().parse_str(json).unwrap();
        let col = df.column(0).unwrap();
        assert_eq!(col.data_type(), DataType::Numeric);
        assert_eq!(col.null_count(), 3);
    }

    #[test]
    fn parse_categorical_column() {
        // Low cardinality: 2 unique / 6 non-null = 0.33 < 0.5 → Categorical
        let json = r#"{"status": ["ok", "fail", "ok", "ok", "fail", "ok"]}"#;
        let df = JsonParser::new().parse_str(json).unwrap();
        assert_eq!(df.column(0).unwrap().data_type(), DataType::Categorical);
    }

    #[test]
    fn parse_row_major() {
        let json = r#"[[1, 2], [3, 4], [5, 6]]"#;
        let df = JsonParser::new().parse_str(json).unwrap();
        assert_eq!(df.row_count(), 3);
        assert_eq!(df.column_count(), 2);
        assert_eq!(df.column_names(), &["col_0", "col_1"]);
        assert_eq!(df.column(0).unwrap().data_type(), DataType::Numeric);
    }

    #[test]
    fn parse_mixed_type_column_becomes_text() {
        let json = r#"{"data": [1, "hello", true, null]}"#;
        let df = JsonParser::new().parse_str(json).unwrap();
        assert_eq!(df.column(0).unwrap().data_type(), DataType::Text);
    }

    #[test]
    fn parse_empty_object() {
        let json = r#"{}"#;
        let df = JsonParser::new().parse_str(json).unwrap();
        assert_eq!(df.row_count(), 0);
        assert_eq!(df.column_count(), 0);
    }

    #[test]
    fn parse_empty_arrays() {
        let json = r#"{"x": [], "y": []}"#;
        let df = JsonParser::new().parse_str(json).unwrap();
        assert_eq!(df.row_count(), 0);
    }

    #[test]
    fn dimension_mismatch_error() {
        let json = r#"{"x": [1, 2], "y": [1, 2, 3]}"#;
        let result = JsonParser::new().parse_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_json_error() {
        let result = JsonParser::new().parse_str("not json");
        assert!(result.is_err());
    }

    #[test]
    fn row_major_dimension_mismatch() {
        let json = r#"[[1, 2], [3]]"#;
        let result = JsonParser::new().parse_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn boolean_column_with_nulls() {
        let json = r#"{"flag": [true, null, false, true]}"#;
        let df = JsonParser::new().parse_str(json).unwrap();
        let col = df.column(0).unwrap();
        assert_eq!(col.data_type(), DataType::Boolean);
        assert_eq!(col.null_count(), 1);
        assert_eq!(col.as_boolean().unwrap(), &[true, false, false, true]);
        assert!(!col.validity().is_valid(1));
    }

    #[test]
    fn numeric_integers_and_floats() {
        let json = r#"{"val": [1, 2.5, 3, 4.0]}"#;
        let df = JsonParser::new().parse_str(json).unwrap();
        let col = df.column(0).unwrap();
        assert_eq!(col.data_type(), DataType::Numeric);
        let nums = col.as_numeric().unwrap();
        assert_eq!(nums, &[1.0, 2.5, 3.0, 4.0]);
    }

    #[test]
    fn profiling_from_json() {
        use crate::profiling::profile_dataframe;

        let json = r#"{
            "temperature": [20.5, 21.3, 19.8, 22.1],
            "status": ["ok", "ok", "fail", "ok"],
            "active": [true, true, false, true]
        }"#;
        let df = JsonParser::new().parse_str(json).unwrap();
        let profiles = profile_dataframe(&df);
        assert_eq!(profiles.len(), 3);
    }
}
