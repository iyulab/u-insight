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
/// A u-insight native operation that failed or was refused. <see cref="Exception.Message"/> is
/// readable text; <see cref="ErrorCode"/>,
/// <see cref="Parameter"/>, <see cref="Index"/> and <see cref="Details"/> are for programs —
/// the same members, with the same meaning, on every sibling package's exception.
/// </summary>
public class InsightException : Exception
{
    /// <summary>The native status (<c>INSIGHT_ERR_*</c>); <see cref="Category"/> names
    /// it.</summary>
    public int Status { get; }

    /// <summary>
    /// Stable, machine-readable reason: the <c>code</c> of the engine's error body, the same
    /// name the WebAssembly binding puts on its <c>Error</c> — <c>value_not_finite</c>,
    /// <c>parameter_out_of_range</c>, <c>insufficient_data</c>, <c>empty_input</c>,
    /// <c>invalid_option</c>, <c>unknown_option</c>, <c>missing_values</c>,
    /// <c>degenerate_data</c>, <c>malformed_input</c>, <c>internal</c>, ….
    /// <c>null</c> when the engine returned no readable body.
    /// </summary>
    /// <remarks>
    /// A numeric option outside the interval it takes (<c>k = 0</c>, a
    /// non-positive DBSCAN <c>epsilon</c>, a <c>chi2_quantile</c> outside (0, 1))
    /// is <c>parameter_out_of_range</c>, with the interval in
    /// <see cref="Details"/>; an option asking for more than the data holds (more
    /// clusters than points) is <c>insufficient_data</c>, with the rows needed as
    /// <c>min</c> and the rows given as <c>got</c>. <c>invalid_option</c> is left
    /// for options that contradict each other.
    /// </remarks>
    public string? ErrorCode { get; }

    /// <summary>
    /// The argument or option the refusal is about, as a path into the request
    /// (<c>chi2_quantile</c>, <c>threshold</c>, <c>batch_size</c>, …) — the body's
    /// <c>parameter</c>. <c>null</c> when the refusal is not
    /// about one value.
    /// </summary>
    public string? Parameter { get; }

    /// <summary>
    /// Zero-based position of the offending element in the array <see cref="Parameter"/>
    /// names — the body's <c>index</c>. <c>null</c> when the value is not an array element.
    /// </summary>
    public int? Index { get; }

    /// <summary>
    /// The whole error body: <c>error</c>, <c>code</c> and the values behind the
    /// reason (<c>parameter</c>, <c>index</c>, <c>min</c>, <c>max</c>,
    /// <c>exclusive_min</c>, <c>exclusive_max</c>, <c>got</c>, <c>column</c>, ...).
    /// For <c>parameter_out_of_range</c>, <c>min</c> / <c>max</c> are inclusive bounds and
    /// <c>exclusive_min</c> / <c>exclusive_max</c> exclusive ones, each present only on the
    /// side it applies to. <c>null</c> when there is no body.
    /// </summary>
    public JsonElement? Details { get; }

    /// <summary>Broad error category derived from <see cref="Status"/>.</summary>
    public InsightErrorCategory Category => Status switch
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

    /// <summary>A failure with no readable body.</summary>
    /// <param name="status">The native status, see <see cref="Status"/>.</param>
    /// <param name="message">What went wrong, in words.</param>
    public InsightException(int status, string message)
        : base(message)
    {
        Status = status;
    }

    /// <summary>A failure with no readable body, caused by <paramref
    /// name="innerException"/>.</summary>
    /// <param name="status">The native status, see <see cref="Status"/>.</param>
    /// <param name="message">What went wrong, in words.</param>
    /// <param name="innerException">The cause.</param>
    public InsightException(int status, string message, Exception innerException)
        : base(message, innerException)
    {
        Status = status;
    }

    /// <summary>
    /// Creates an InsightException from a status and optional native error detail.
    /// </summary>
    /// <remarks>
    /// The native side already produces a fully-formatted message via Rust's Display impl
    /// (e.g. "degenerate data: correlation matrix computation failed (...)").
    /// We surface that message verbatim and rely on <see cref="Category"/> for typed
    /// classification,
    /// avoiding "Degenerate data: degenerate data: ..." prefix duplication.
    /// </remarks>
    internal static InsightException FromCode(
        int status, string? nativeError, string? parameter = null, string? body = null)
    {
        var msg = nativeError ?? Interop.NativeLibrary.GetErrorMessage(status);
        string? errorCode = null;
        JsonElement? details = null;
        if (body is not null)
        {
            try
            {
                using var doc = JsonDocument.Parse(body);
                var root = doc.RootElement;
                if (root.TryGetProperty("code", out var c) && c.ValueKind == JsonValueKind.String)
                    errorCode = c.GetString();
                details = root.Clone();
            }
            catch (JsonException)
            {
                // No readable body: the message and category still stand.
            }
        }
        return new InsightException(status, msg, parameter, errorCode, details);
    }

    private InsightException(
        int status, string message, string? parameter, string? errorCode, JsonElement? details)
        : base(message)
    {
        Status = status;
        ErrorCode = errorCode;
        Details = details;
        Parameter = parameter ?? ParameterIn(details);
        Index = IndexIn(details);
    }

    /// <summary>The body's <c>parameter</c>, when it is a string.</summary>
    private static string? ParameterIn(JsonElement? details) =>
        details is { ValueKind: JsonValueKind.Object } d
        && d.TryGetProperty("parameter", out var p) && p.ValueKind == JsonValueKind.String
            ? p.GetString()
            : null;

    /// <summary>The body's <c>index</c>, when it is a whole number.</summary>
    private static int? IndexIn(JsonElement? details) =>
        details is { ValueKind: JsonValueKind.Object } d
        && d.TryGetProperty("index", out var i) && i.ValueKind == JsonValueKind.Number
        && i.TryGetInt32(out var n)
            ? n
            : null;
}
