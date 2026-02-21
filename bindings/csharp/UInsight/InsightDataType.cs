namespace UInsight;

/// <summary>
/// Data type detected by u-insight profiling.
/// Matches the Rust FFI DataType enum ordinals in ffi.rs.
/// </summary>
public enum InsightDataType : uint
{
    /// <summary>Numeric (integer or floating-point) values.</summary>
    Numeric = 0,

    /// <summary>Boolean (true/false) values.</summary>
    Boolean = 1,

    /// <summary>Categorical (discrete label) values.</summary>
    Categorical = 2,

    /// <summary>Free-form text values.</summary>
    Text = 3
}
