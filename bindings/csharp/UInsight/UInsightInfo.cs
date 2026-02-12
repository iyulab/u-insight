namespace UInsight;

/// <summary>
/// Provides information about the U-Insight library.
/// </summary>
public static class UInsightInfo
{
    /// <summary>
    /// Gets the native library version.
    /// </summary>
    public static string Version => Interop.NativeLibrary.GetVersion();
}
