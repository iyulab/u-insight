namespace UInsight;

/// <summary>
/// Broad category of a u-insight error, derived from the native error code.
/// </summary>
public enum InsightErrorCategory
{
    /// <summary>Unrecognised error code (null pointer, panic, or future codes).</summary>
    Unknown,
    /// <summary>Invalid input data (missing values, non-numeric columns, etc.).</summary>
    InvalidInput,
    /// <summary>CSV or data parsing failure.</summary>
    ParseFailed,
    /// <summary>I/O error or unclassified analysis failure.</summary>
    AnalysisFailed,
    /// <summary>Too few rows/samples for the requested operation.</summary>
    InsufficientData,
    /// <summary>Invalid parameter value supplied by the caller.</summary>
    InvalidParameter,
    /// <summary>Degenerate data (constant columns, singular matrix, etc.).</summary>
    DegenerateData,
    /// <summary>Internal computation failure (e.g. eigenvalue decomposition).</summary>
    ComputationFailed,
}

/// <summary>
/// Exception thrown when a u-insight native operation fails.
/// </summary>
public class InsightException : Exception
{
    /// <summary>
    /// The native error code.
    /// </summary>
    public int ErrorCode { get; }

    /// <summary>
    /// Broad error category derived from <see cref="ErrorCode"/>.
    /// </summary>
    public InsightErrorCategory Category => ErrorCode switch
    {
        Interop.NativeLibrary.INSIGHT_ERR_INVALID_INPUT => InsightErrorCategory.InvalidInput,
        Interop.NativeLibrary.INSIGHT_ERR_PARSE_FAILED => InsightErrorCategory.ParseFailed,
        Interop.NativeLibrary.INSIGHT_ERR_ANALYSIS_FAILED => InsightErrorCategory.AnalysisFailed,
        Interop.NativeLibrary.INSIGHT_ERR_INSUFFICIENT_DATA => InsightErrorCategory.InsufficientData,
        Interop.NativeLibrary.INSIGHT_ERR_INVALID_PARAM => InsightErrorCategory.InvalidParameter,
        Interop.NativeLibrary.INSIGHT_ERR_DEGENERATE_DATA => InsightErrorCategory.DegenerateData,
        Interop.NativeLibrary.INSIGHT_ERR_COMPUTATION_FAILED => InsightErrorCategory.ComputationFailed,
        _ => InsightErrorCategory.Unknown
    };

    /// <summary>
    /// Creates a new InsightException instance.
    /// </summary>
    /// <param name="errorCode">The native error code.</param>
    /// <param name="message">The error message.</param>
    public InsightException(int errorCode, string message)
        : base(message)
    {
        ErrorCode = errorCode;
    }

    /// <summary>
    /// Creates a new InsightException instance with an inner exception.
    /// </summary>
    /// <param name="errorCode">The native error code.</param>
    /// <param name="message">The error message.</param>
    /// <param name="innerException">The inner exception.</param>
    public InsightException(int errorCode, string message, Exception innerException)
        : base(message, innerException)
    {
        ErrorCode = errorCode;
    }

    /// <summary>
    /// Creates an InsightException from an error code and optional native error detail.
    /// </summary>
    internal static InsightException FromCode(int code, string? nativeError)
    {
        var baseMsg = Interop.NativeLibrary.GetErrorMessage(code);
        var msg = nativeError is not null ? $"{baseMsg}: {nativeError}" : baseMsg;
        return new InsightException(code, msg);
    }
}
