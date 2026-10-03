using System.Text.Json;

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
    /// The name of the argument or option the error is about, as the native
    /// library spells it (<c>chi2_quantile</c>, <c>threshold</c>,
    /// <c>batch_size</c>, ...), or <c>null</c> when the error is not about one
    /// named parameter. Set with <see cref="InsightErrorCategory.InvalidParameter"/>,
    /// so a caller can branch on it or map it to its own name without parsing
    /// <see cref="Exception.Message"/>.
    /// </summary>
    public string? Parameter { get; }

    /// <summary>
    /// Stable, machine-readable reason -- the same <c>code</c> the WebAssembly
    /// binding puts on its <c>Error</c>: <c>value_not_finite</c>,
    /// <c>parameter_out_of_range</c>, <c>insufficient_data</c>,
    /// <c>invalid_option</c>, <c>unknown_option</c>, <c>missing_values</c>,
    /// <c>degenerate_data</c>, <c>malformed_input</c>, <c>internal</c>, ...
    /// <c>null</c> when the native library returned no body.
    /// </summary>
    public string? Reason { get; }

    /// <summary>
    /// The whole error body: <c>error</c>, <c>code</c> and the values behind the
    /// reason (<c>parameter</c>, <c>index</c>, <c>min</c>, <c>got</c>,
    /// <c>column</c>, ...), so a caller can say which value was refused without
    /// parsing <see cref="Exception.Message"/>. <c>null</c> when there is no body.
    /// </summary>
    public JsonElement? Details { get; }

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
    /// <remarks>
    /// The native side already produces a fully-formatted message via Rust's Display impl
    /// (e.g. "degenerate data: correlation matrix computation failed (...)").
    /// We surface that message verbatim and rely on <see cref="Category"/> for typed classification,
    /// avoiding "Degenerate data: degenerate data: ..." prefix duplication.
    /// </remarks>
    internal static InsightException FromCode(
        int code, string? nativeError, string? parameter = null, string? body = null)
    {
        var msg = nativeError ?? Interop.NativeLibrary.GetErrorMessage(code);
        string? reason = null;
        JsonElement? details = null;
        if (body is not null)
        {
            try
            {
                using var doc = JsonDocument.Parse(body);
                var root = doc.RootElement;
                if (root.TryGetProperty("code", out var c) && c.ValueKind == JsonValueKind.String)
                    reason = c.GetString();
                details = root.Clone();
            }
            catch (JsonException)
            {
                // No readable body: the message and category still stand.
            }
        }
        return new InsightException(code, msg, parameter, reason, details);
    }

    private InsightException(
        int errorCode, string message, string? parameter, string? reason, JsonElement? details)
        : base(message)
    {
        ErrorCode = errorCode;
        Parameter = parameter;
        Reason = reason;
        Details = details;
    }
}
