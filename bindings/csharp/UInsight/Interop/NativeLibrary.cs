using System.Reflection;
using System.Runtime.InteropServices;

namespace UInsight.Interop;

/// <summary>
/// P/Invoke declarations for the u-insight native library.
/// </summary>
internal static partial class NativeLibrary
{
    private const string LibraryName = "u_insight";

    #region Error Codes

    public const int INSIGHT_OK = 0;
    public const int INSIGHT_ERR_NULL_PTR = -1;
    public const int INSIGHT_ERR_INVALID_INPUT = -2;
    public const int INSIGHT_ERR_PARSE_FAILED = -3;
    public const int INSIGHT_ERR_ANALYSIS_FAILED = -4;
    public const int INSIGHT_ERR_PANIC = -99;
    // New granular error codes — 1:1 with InsightError variants
    public const int INSIGHT_ERR_INSUFFICIENT_DATA = -5;
    public const int INSIGHT_ERR_INVALID_PARAM = -6;
    public const int INSIGHT_ERR_DEGENERATE_DATA = -7;
    public const int INSIGHT_ERR_COMPUTATION_FAILED = -8;

    #endregion

    #region Resolver

    static NativeLibrary()
    {
        System.Runtime.InteropServices.NativeLibrary.SetDllImportResolver(
            typeof(NativeLibrary).Assembly, DllImportResolver);
    }

    private static IntPtr DllImportResolver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != LibraryName)
            return IntPtr.Zero;

        if (System.Runtime.InteropServices.NativeLibrary.TryLoad(libraryName, assembly, searchPath, out var handle))
            return handle;

        return IntPtr.Zero;
    }

    #endregion

    #region Version & Error

    [LibraryImport(LibraryName)]
    public static partial IntPtr insight_version();

    [LibraryImport(LibraryName)]
    public static partial IntPtr insight_last_error();

    [LibraryImport(LibraryName)]
    public static partial void insight_clear_error();

    #endregion

    #region Profiling

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial IntPtr insight_profile_csv(string csvData);

    [LibraryImport(LibraryName)]
    public static partial void insight_profile_free(IntPtr ctx);

    [LibraryImport(LibraryName)]
    public static partial long insight_profile_row_count(IntPtr ctx);

    [LibraryImport(LibraryName)]
    public static partial long insight_profile_col_count(IntPtr ctx);

    [LibraryImport(LibraryName)]
    public static partial int insight_profile_column(IntPtr ctx, uint colIdx, ref NativeStructs.CColumnSummary result);

    #endregion

    #region Clustering

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_kmeans(
        double* data, uint nRows, uint nCols, uint k,
        ref NativeStructs.CKMeansResult result);

    [LibraryImport(LibraryName)]
    public static partial void insight_free_labels(IntPtr labels, uint count);

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_mini_batch_kmeans(
        double* data, uint nRows, uint nCols, uint k,
        uint batchSize, uint maxIter, ulong seed,
        ref NativeStructs.CKMeansResult result);

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_dbscan(
        double* data, uint nRows, uint nCols, double epsilon, uint minSamples,
        ref NativeStructs.CDbscanResult result);

    [LibraryImport(LibraryName)]
    public static partial void insight_free_i32_array(IntPtr ptr, uint count);

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_hierarchical(
        double* data, uint nRows, uint nCols, uint linkage, uint nClusters,
        ref NativeStructs.CHierarchicalResult result);

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_hdbscan(
        double* data, uint nRows, uint nCols, uint minClusterSize, uint minSamples,
        ref NativeStructs.CHdbscanResult result);

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_gap_statistic(
        double* data, uint nRows, uint nCols,
        uint kMin, uint kMax, uint nRefs, ulong seed,
        ref NativeStructs.CGapStatResult result);

    #endregion

    #region PCA

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_pca(
        double* data, uint nRows, uint nCols, uint nComponents, int autoScale,
        ref NativeStructs.CPcaResult result);

    [LibraryImport(LibraryName)]
    public static partial void insight_free_f64_array(IntPtr ptr, uint count);

    #endregion

    #region Anomaly Detection

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_isolation_forest(
        double* data, uint nRows, uint nCols,
        uint nEstimators, double contamination, ulong seed,
        ref NativeStructs.CAnomalyResult result);

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_lof(
        double* data, uint nRows, uint nCols, uint k, double threshold,
        ref NativeStructs.CAnomalyResult result);

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_mahalanobis(
        double* data, uint nRows, uint nCols, double chi2Quantile,
        ref NativeStructs.CMahalanobisResult result);

    #endregion

    #region Statistical Analysis

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_correlation(
        double* data, uint nRows, uint nCols,
        ref NativeStructs.CCorrelationResult result);

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_regression(
        double* x, double* y, uint n,
        ref NativeStructs.CRegressionResult result);

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_cramers_v(
        double* table, uint nRows, uint nCols,
        ref NativeStructs.CCramersVResult result);

    #endregion

    #region Distribution

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_distribution(
        double* data, uint n, double significanceLevel,
        ref NativeStructs.CDistributionResult result);

    #endregion

    #region Feature Importance

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_feature_importance(
        double* data, uint nRows, uint nCols,
        ref NativeStructs.CFeatureImportanceResult result);

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_anova_select(
        double* data, uint nRows, uint nFeatures, uint* target, double significanceLevel,
        ref NativeStructs.CAnovaSelectionResult result);

    [LibraryImport(LibraryName)]
    public static partial void insight_free_anova_features(IntPtr ptr, uint count);

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_mutual_info(
        double* data, uint nRows, uint nFeatures, uint* target, uint nBins,
        ref NativeStructs.CMutualInfoResult result);

    [LibraryImport(LibraryName)]
    public static partial void insight_free_mi_features(IntPtr ptr, uint count);

    [LibraryImport(LibraryName)]
    public static unsafe partial int insight_permutation_importance(
        double* data, uint nRows, uint nFeatures, double* target,
        uint nRepeats, ulong seed,
        ref NativeStructs.CPermImportanceResult result);

    [LibraryImport(LibraryName)]
    public static partial void insight_free_perm_features(IntPtr ptr, uint count);

    #endregion

    #region Helper Methods

    /// <summary>
    /// Gets the library version.
    /// </summary>
    public static string GetVersion()
    {
        var ptr = insight_version();
        return Marshal.PtrToStringUTF8(ptr) ?? "unknown";
    }

    /// <summary>
    /// Converts an error code to a descriptive message.
    /// </summary>
    public static string GetErrorMessage(int errorCode) => errorCode switch
    {
        INSIGHT_OK => "Success",
        INSIGHT_ERR_NULL_PTR => "Null pointer passed",
        INSIGHT_ERR_INVALID_INPUT => "Invalid input",
        INSIGHT_ERR_PARSE_FAILED => "Parse failed",
        INSIGHT_ERR_ANALYSIS_FAILED => "Analysis failed",
        INSIGHT_ERR_PANIC => "Internal panic",
        INSIGHT_ERR_INSUFFICIENT_DATA => "Insufficient data",
        INSIGHT_ERR_INVALID_PARAM => "Invalid parameter",
        INSIGHT_ERR_DEGENERATE_DATA => "Degenerate data",
        INSIGHT_ERR_COMPUTATION_FAILED => "Computation failed",
        _ => $"Unknown error code: {errorCode}"
    };

    /// <summary>
    /// Gets the last error message from the native library.
    /// </summary>
    public static string? GetLastError()
    {
        var ptr = insight_last_error();
        return ptr != IntPtr.Zero ? Marshal.PtrToStringUTF8(ptr) : null;
    }

    /// <summary>
    /// Throws an InsightException if the error code indicates failure.
    /// </summary>
    public static void ThrowIfFailed(int code)
    {
        if (code == INSIGHT_OK) return;
        throw InsightException.FromCode(code, GetLastError());
    }

    #endregion
}
