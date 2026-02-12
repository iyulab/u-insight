namespace UInsight;

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
