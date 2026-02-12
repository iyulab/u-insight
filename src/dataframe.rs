//! Column-major DataFrame for tabular data.
//!
//! The [`DataFrame`] stores data in column-major order with typed columns
//! and a compact validity bitmap for tracking missing values.
//!
//! # Column Types
//!
//! | Type | Storage | Use case |
//! |------|---------|----------|
//! | [`Numeric`](Column::Numeric) | `Vec<f64>` + bitmap | Continuous/integer values |
//! | [`Boolean`](Column::Boolean) | `Vec<bool>` + bitmap | True/false values |
//! | [`Categorical`](Column::Categorical) | Dictionary + `Vec<u32>` | Low-cardinality strings |
//! | [`Text`](Column::Text) | `Vec<String>` + bitmap | High-cardinality strings |
//!
//! # Example
//!
//! ```
//! use u_insight::dataframe::{DataFrame, Column, ValidityBitmap};
//!
//! let mut df = DataFrame::new();
//! df.add_column(
//!     "temperature".to_string(),
//!     Column::numeric(vec![20.5, 21.3, 19.8], ValidityBitmap::all_valid(3)),
//! ).unwrap();
//! assert_eq!(df.row_count(), 3);
//! assert_eq!(df.column_count(), 1);
//! ```

use crate::error::InsightError;

// ── ValidityBitmap ────────────────────────────────────────────────────

/// Bit-packed validity bitmap using `Vec<u64>`.
///
/// Each bit indicates whether the corresponding row is valid (1) or
/// missing/null (0). Uses 1 bit per row instead of 1 byte, yielding
/// 8× memory savings over `Vec<bool>`.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidityBitmap {
    bits: Vec<u64>,
    len: usize,
}

impl ValidityBitmap {
    /// Creates a bitmap where all `len` positions are valid.
    pub fn all_valid(len: usize) -> Self {
        let n_words = len.div_ceil(64);
        let mut bits = vec![u64::MAX; n_words];
        let trailing = len % 64;
        if trailing != 0 && n_words > 0 {
            bits[n_words - 1] = (1u64 << trailing) - 1;
        }
        Self { bits, len }
    }

    /// Creates a bitmap where all `len` positions are invalid (null).
    pub fn all_invalid(len: usize) -> Self {
        let n_words = len.div_ceil(64);
        Self {
            bits: vec![0u64; n_words],
            len,
        }
    }

    /// Creates an empty bitmap with no rows.
    pub fn empty() -> Self {
        Self {
            bits: Vec::new(),
            len: 0,
        }
    }

    /// Returns `true` if the value at `idx` is valid (not null).
    #[inline]
    pub fn is_valid(&self, idx: usize) -> bool {
        debug_assert!(idx < self.len, "index {idx} out of bounds (len={})", self.len);
        let (word, bit) = (idx / 64, idx % 64);
        (self.bits[word] >> bit) & 1 == 1
    }

    /// Marks position `idx` as valid.
    #[inline]
    pub fn set_valid(&mut self, idx: usize) {
        debug_assert!(idx < self.len, "index {idx} out of bounds (len={})", self.len);
        let (word, bit) = (idx / 64, idx % 64);
        self.bits[word] |= 1u64 << bit;
    }

    /// Marks position `idx` as invalid (null).
    #[inline]
    pub fn set_invalid(&mut self, idx: usize) {
        debug_assert!(idx < self.len, "index {idx} out of bounds (len={})", self.len);
        let (word, bit) = (idx / 64, idx % 64);
        self.bits[word] &= !(1u64 << bit);
    }

    /// Appends a new position (valid or invalid).
    pub fn push(&mut self, valid: bool) {
        let idx = self.len;
        self.len += 1;
        let word = idx / 64;
        let bit = idx % 64;
        if word >= self.bits.len() {
            self.bits.push(0);
        }
        if valid {
            self.bits[word] |= 1u64 << bit;
        }
    }

    /// Returns the total number of tracked positions.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the bitmap tracks zero positions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Counts the number of null (invalid) positions.
    ///
    /// Uses hardware `POPCNT` instruction for fast counting.
    pub fn null_count(&self) -> usize {
        let valid_count: usize = self.bits.iter().map(|w| w.count_ones() as usize).sum();
        self.len - valid_count
    }

    /// Counts the number of valid (non-null) positions.
    pub fn valid_count(&self) -> usize {
        self.len - self.null_count()
    }

    /// Returns `true` if any position is null.
    pub fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    /// Returns an iterator over indices of valid positions.
    pub fn valid_indices(&self) -> ValidIndicesIter<'_> {
        ValidIndicesIter {
            bitmap: self,
            current: 0,
        }
    }
}

/// Iterator over valid indices in a [`ValidityBitmap`].
pub struct ValidIndicesIter<'a> {
    bitmap: &'a ValidityBitmap,
    current: usize,
}

impl<'a> Iterator for ValidIndicesIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        while self.current < self.bitmap.len {
            let idx = self.current;
            self.current += 1;
            if self.bitmap.is_valid(idx) {
                return Some(idx);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.bitmap.len - self.current))
    }
}

// ── DataType ──────────────────────────────────────────────────────────

/// Semantic data type inferred for a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Continuous or integer numeric values (stored as `f64`).
    Numeric,
    /// Boolean (true/false) values.
    Boolean,
    /// Low-cardinality strings (dictionary-encoded).
    Categorical,
    /// High-cardinality or free-form text.
    Text,
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Numeric => write!(f, "Numeric"),
            Self::Boolean => write!(f, "Boolean"),
            Self::Categorical => write!(f, "Categorical"),
            Self::Text => write!(f, "Text"),
        }
    }
}

// ── Column ────────────────────────────────────────────────────────────

/// A typed column with validity bitmap for missing values.
///
/// All variants store values in a dense array alongside a
/// [`ValidityBitmap`]. Invalid positions hold a default value
/// (0.0, false, empty string, or index 0) that should be ignored.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    /// Dense `f64` values. Null positions hold `0.0`.
    Numeric {
        values: Vec<f64>,
        validity: ValidityBitmap,
    },
    /// Boolean values. Null positions hold `false`.
    Boolean {
        values: Vec<bool>,
        validity: ValidityBitmap,
    },
    /// Dictionary-encoded categorical column.
    ///
    /// `dictionary` contains unique string values.
    /// `indices` maps each row to a dictionary index.
    /// Null positions have index `0` (ignored via validity bit).
    Categorical {
        dictionary: Vec<String>,
        indices: Vec<u32>,
        validity: ValidityBitmap,
    },
    /// Free-form text column. Null positions hold an empty string.
    Text {
        values: Vec<String>,
        validity: ValidityBitmap,
    },
}

impl Column {
    /// Creates a numeric column.
    pub fn numeric(values: Vec<f64>, validity: ValidityBitmap) -> Self {
        Self::Numeric { values, validity }
    }

    /// Creates a boolean column.
    pub fn boolean(values: Vec<bool>, validity: ValidityBitmap) -> Self {
        Self::Boolean { values, validity }
    }

    /// Creates a categorical column from a dictionary and indices.
    pub fn categorical(
        dictionary: Vec<String>,
        indices: Vec<u32>,
        validity: ValidityBitmap,
    ) -> Self {
        Self::Categorical {
            dictionary,
            indices,
            validity,
        }
    }

    /// Creates a text column.
    pub fn text(values: Vec<String>, validity: ValidityBitmap) -> Self {
        Self::Text { values, validity }
    }

    /// Returns the data type of this column.
    pub fn data_type(&self) -> DataType {
        match self {
            Self::Numeric { .. } => DataType::Numeric,
            Self::Boolean { .. } => DataType::Boolean,
            Self::Categorical { .. } => DataType::Categorical,
            Self::Text { .. } => DataType::Text,
        }
    }

    /// Returns the number of rows in this column.
    pub fn len(&self) -> usize {
        self.validity().len()
    }

    /// Returns `true` if the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the validity bitmap.
    pub fn validity(&self) -> &ValidityBitmap {
        match self {
            Self::Numeric { validity, .. }
            | Self::Boolean { validity, .. }
            | Self::Categorical { validity, .. }
            | Self::Text { validity, .. } => validity,
        }
    }

    /// Returns the number of null values.
    pub fn null_count(&self) -> usize {
        self.validity().null_count()
    }

    /// Returns the number of valid (non-null) values.
    pub fn valid_count(&self) -> usize {
        self.validity().valid_count()
    }

    /// Returns `true` if the value at `idx` is valid (not null).
    pub fn is_valid(&self, idx: usize) -> bool {
        self.validity().is_valid(idx)
    }

    /// Returns the numeric values, or `None` if not a numeric column.
    pub fn as_numeric(&self) -> Option<&[f64]> {
        match self {
            Self::Numeric { values, .. } => Some(values),
            _ => None,
        }
    }

    /// Returns the boolean values, or `None` if not a boolean column.
    pub fn as_boolean(&self) -> Option<&[bool]> {
        match self {
            Self::Boolean { values, .. } => Some(values),
            _ => None,
        }
    }

    /// Returns valid numeric values (nulls excluded) as a new `Vec<f64>`.
    pub fn valid_numeric_values(&self) -> Option<Vec<f64>> {
        match self {
            Self::Numeric { values, validity } => {
                let result: Vec<f64> = validity
                    .valid_indices()
                    .map(|i| values[i])
                    .collect();
                Some(result)
            }
            _ => None,
        }
    }

    /// Returns the category string for a given row index in a categorical column.
    pub fn category_at(&self, idx: usize) -> Option<&str> {
        match self {
            Self::Categorical {
                dictionary,
                indices,
                validity,
            } => {
                if validity.is_valid(idx) {
                    dictionary.get(indices[idx] as usize).map(|s| s.as_str())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Returns the text value for a given row index in a text column.
    pub fn text_at(&self, idx: usize) -> Option<&str> {
        match self {
            Self::Text { values, validity } => {
                if validity.is_valid(idx) {
                    Some(&values[idx])
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

// ── DataFrame ─────────────────────────────────────────────────────────

/// Column-major tabular data structure.
///
/// Stores named columns of typed data. All columns must have the same
/// number of rows. Supports numeric, boolean, categorical, and text
/// column types.
///
/// # Example
///
/// ```
/// use u_insight::dataframe::{DataFrame, Column, ValidityBitmap};
///
/// let mut df = DataFrame::new();
/// df.add_column(
///     "x".to_string(),
///     Column::numeric(vec![1.0, 2.0, 3.0], ValidityBitmap::all_valid(3)),
/// ).unwrap();
/// df.add_column(
///     "label".to_string(),
///     Column::text(
///         vec!["a".into(), "b".into(), "c".into()],
///         ValidityBitmap::all_valid(3),
///     ),
/// ).unwrap();
/// assert_eq!(df.row_count(), 3);
/// assert_eq!(df.column_count(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct DataFrame {
    names: Vec<String>,
    columns: Vec<Column>,
    row_count: usize,
}

impl DataFrame {
    /// Creates an empty DataFrame with no columns or rows.
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            columns: Vec::new(),
            row_count: 0,
        }
    }

    /// Adds a named column to the DataFrame.
    ///
    /// Returns an error if the column length doesn't match the existing
    /// row count (unless this is the first column).
    pub fn add_column(&mut self, name: String, column: Column) -> Result<(), InsightError> {
        let col_len = column.len();
        if self.columns.is_empty() {
            self.row_count = col_len;
        } else if col_len != self.row_count {
            return Err(InsightError::DimensionMismatch {
                expected: self.row_count,
                actual: col_len,
            });
        }
        self.names.push(name);
        self.columns.push(column);
        Ok(())
    }

    /// Returns the number of rows.
    #[inline]
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Returns the number of columns.
    #[inline]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Returns `true` if the DataFrame has no columns.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Returns column names.
    pub fn column_names(&self) -> &[String] {
        &self.names
    }

    /// Returns a reference to the column at `index`.
    pub fn column(&self, index: usize) -> Option<&Column> {
        self.columns.get(index)
    }

    /// Returns a reference to the column with the given `name`.
    pub fn column_by_name(&self, name: &str) -> Option<&Column> {
        self.column_index(name).map(|i| &self.columns[i])
    }

    /// Returns the index of the column with the given `name`.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// Returns an iterator over (name, column) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Column)> {
        self.names.iter().map(|s| s.as_str()).zip(self.columns.iter())
    }

    /// Returns a summary of column data types.
    pub fn schema(&self) -> Vec<(&str, DataType)> {
        self.names
            .iter()
            .zip(self.columns.iter())
            .map(|(name, col)| (name.as_str(), col.data_type()))
            .collect()
    }

    /// Returns the total number of null values across all columns.
    pub fn total_null_count(&self) -> usize {
        self.columns.iter().map(|c| c.null_count()).sum()
    }
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ValidityBitmap tests ──────────────────────────────────────

    #[test]
    fn bitmap_all_valid() {
        let bm = ValidityBitmap::all_valid(100);
        assert_eq!(bm.len(), 100);
        assert_eq!(bm.null_count(), 0);
        assert_eq!(bm.valid_count(), 100);
        for i in 0..100 {
            assert!(bm.is_valid(i));
        }
    }

    #[test]
    fn bitmap_all_invalid() {
        let bm = ValidityBitmap::all_invalid(100);
        assert_eq!(bm.null_count(), 100);
        assert_eq!(bm.valid_count(), 0);
        for i in 0..100 {
            assert!(!bm.is_valid(i));
        }
    }

    #[test]
    fn bitmap_set_operations() {
        let mut bm = ValidityBitmap::all_valid(10);
        bm.set_invalid(3);
        bm.set_invalid(7);
        assert_eq!(bm.null_count(), 2);
        assert!(!bm.is_valid(3));
        assert!(!bm.is_valid(7));
        assert!(bm.is_valid(0));
        assert!(bm.is_valid(9));

        bm.set_valid(3);
        assert!(bm.is_valid(3));
        assert_eq!(bm.null_count(), 1);
    }

    #[test]
    fn bitmap_push() {
        let mut bm = ValidityBitmap::empty();
        bm.push(true);
        bm.push(false);
        bm.push(true);
        assert_eq!(bm.len(), 3);
        assert!(bm.is_valid(0));
        assert!(!bm.is_valid(1));
        assert!(bm.is_valid(2));
        assert_eq!(bm.null_count(), 1);
    }

    #[test]
    fn bitmap_boundary_64() {
        let bm = ValidityBitmap::all_valid(64);
        assert_eq!(bm.bits.len(), 1);
        assert_eq!(bm.null_count(), 0);

        let bm65 = ValidityBitmap::all_valid(65);
        assert_eq!(bm65.bits.len(), 2);
        assert_eq!(bm65.null_count(), 0);
        assert!(bm65.is_valid(64));
    }

    #[test]
    fn bitmap_push_across_word_boundary() {
        let mut bm = ValidityBitmap::empty();
        for i in 0..128 {
            bm.push(i % 3 != 0); // every 3rd is null
        }
        assert_eq!(bm.len(), 128);
        let expected_nulls = (0..128).filter(|i| i % 3 == 0).count();
        assert_eq!(bm.null_count(), expected_nulls);
    }

    #[test]
    fn bitmap_valid_indices() {
        let mut bm = ValidityBitmap::all_valid(5);
        bm.set_invalid(1);
        bm.set_invalid(3);
        let indices: Vec<usize> = bm.valid_indices().collect();
        assert_eq!(indices, vec![0, 2, 4]);
    }

    // ── Column tests ─────────────────────────────────────────────

    #[test]
    fn numeric_column_basics() {
        let col = Column::numeric(vec![1.0, 2.0, 3.0], ValidityBitmap::all_valid(3));
        assert_eq!(col.data_type(), DataType::Numeric);
        assert_eq!(col.len(), 3);
        assert_eq!(col.null_count(), 0);
        assert_eq!(col.as_numeric(), Some(&[1.0, 2.0, 3.0][..]));
    }

    #[test]
    fn numeric_column_with_nulls() {
        let mut validity = ValidityBitmap::all_valid(4);
        validity.set_invalid(1);
        validity.set_invalid(3);
        let col = Column::numeric(vec![1.0, 0.0, 3.0, 0.0], validity);
        assert_eq!(col.null_count(), 2);
        assert_eq!(col.valid_count(), 2);
        assert!(col.is_valid(0));
        assert!(!col.is_valid(1));
        let valid = col.valid_numeric_values().expect("numeric column");
        assert_eq!(valid, vec![1.0, 3.0]);
    }

    #[test]
    fn boolean_column() {
        let col = Column::boolean(vec![true, false, true], ValidityBitmap::all_valid(3));
        assert_eq!(col.data_type(), DataType::Boolean);
        assert_eq!(col.as_boolean(), Some(&[true, false, true][..]));
    }

    #[test]
    fn categorical_column() {
        let dict = vec!["low".into(), "med".into(), "high".into()];
        let indices = vec![0, 1, 2, 1, 0];
        let col = Column::categorical(dict, indices, ValidityBitmap::all_valid(5));
        assert_eq!(col.data_type(), DataType::Categorical);
        assert_eq!(col.category_at(0), Some("low"));
        assert_eq!(col.category_at(1), Some("med"));
        assert_eq!(col.category_at(2), Some("high"));
        assert_eq!(col.category_at(3), Some("med"));
    }

    #[test]
    fn categorical_column_with_null() {
        let dict = vec!["a".into(), "b".into()];
        let indices = vec![0, 0, 1];
        let mut validity = ValidityBitmap::all_valid(3);
        validity.set_invalid(1);
        let col = Column::categorical(dict, indices, validity);
        assert_eq!(col.category_at(0), Some("a"));
        assert_eq!(col.category_at(1), None);
        assert_eq!(col.category_at(2), Some("b"));
    }

    #[test]
    fn text_column() {
        let col = Column::text(
            vec!["hello".into(), "world".into()],
            ValidityBitmap::all_valid(2),
        );
        assert_eq!(col.data_type(), DataType::Text);
        assert_eq!(col.text_at(0), Some("hello"));
        assert_eq!(col.text_at(1), Some("world"));
    }

    #[test]
    fn text_column_with_null() {
        let mut validity = ValidityBitmap::all_valid(2);
        validity.set_invalid(0);
        let col = Column::text(vec![String::new(), "world".into()], validity);
        assert_eq!(col.text_at(0), None);
        assert_eq!(col.text_at(1), Some("world"));
    }

    // ── DataFrame tests ──────────────────────────────────────────

    #[test]
    fn empty_dataframe() {
        let df = DataFrame::new();
        assert_eq!(df.row_count(), 0);
        assert_eq!(df.column_count(), 0);
        assert!(df.is_empty());
    }

    #[test]
    fn add_columns() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Column::numeric(vec![1.0, 2.0, 3.0], ValidityBitmap::all_valid(3)),
        )
        .expect("first column");

        df.add_column(
            "y".to_string(),
            Column::numeric(vec![4.0, 5.0, 6.0], ValidityBitmap::all_valid(3)),
        )
        .expect("second column");

        assert_eq!(df.row_count(), 3);
        assert_eq!(df.column_count(), 2);
        assert_eq!(df.column_names(), &["x", "y"]);
    }

    #[test]
    fn column_length_mismatch() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Column::numeric(vec![1.0, 2.0], ValidityBitmap::all_valid(2)),
        )
        .unwrap();

        let result = df.add_column(
            "y".to_string(),
            Column::numeric(vec![1.0, 2.0, 3.0], ValidityBitmap::all_valid(3)),
        );
        assert!(result.is_err());
    }

    #[test]
    fn column_by_name_lookup() {
        let mut df = DataFrame::new();
        df.add_column(
            "temp".to_string(),
            Column::numeric(vec![20.5, 21.3], ValidityBitmap::all_valid(2)),
        )
        .unwrap();

        let col = df.column_by_name("temp").expect("found");
        assert_eq!(col.data_type(), DataType::Numeric);

        assert!(df.column_by_name("missing").is_none());
    }

    #[test]
    fn dataframe_schema() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Column::numeric(vec![1.0], ValidityBitmap::all_valid(1)),
        )
        .unwrap();
        df.add_column(
            "ok".to_string(),
            Column::boolean(vec![true], ValidityBitmap::all_valid(1)),
        )
        .unwrap();
        df.add_column(
            "label".to_string(),
            Column::text(vec!["a".into()], ValidityBitmap::all_valid(1)),
        )
        .unwrap();

        let schema = df.schema();
        assert_eq!(schema[0], ("x", DataType::Numeric));
        assert_eq!(schema[1], ("ok", DataType::Boolean));
        assert_eq!(schema[2], ("label", DataType::Text));
    }

    #[test]
    fn total_null_count() {
        let mut df = DataFrame::new();
        let mut v1 = ValidityBitmap::all_valid(3);
        v1.set_invalid(1);
        let mut v2 = ValidityBitmap::all_valid(3);
        v2.set_invalid(0);
        v2.set_invalid(2);
        df.add_column("a".into(), Column::numeric(vec![1.0, 0.0, 3.0], v1))
            .unwrap();
        df.add_column("b".into(), Column::numeric(vec![0.0, 5.0, 0.0], v2))
            .unwrap();
        assert_eq!(df.total_null_count(), 3);
    }

    #[test]
    fn dataframe_iter() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".into(),
            Column::numeric(vec![1.0], ValidityBitmap::all_valid(1)),
        )
        .unwrap();
        df.add_column(
            "y".into(),
            Column::numeric(vec![2.0], ValidityBitmap::all_valid(1)),
        )
        .unwrap();

        let pairs: Vec<(&str, DataType)> = df.iter().map(|(n, c)| (n, c.data_type())).collect();
        assert_eq!(pairs, vec![("x", DataType::Numeric), ("y", DataType::Numeric)]);
    }
}
