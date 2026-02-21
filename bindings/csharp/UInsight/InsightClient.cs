using System.Runtime.InteropServices;
using Native = UInsight.Interop.NativeLibrary;
using NativeStructs = UInsight.Interop.NativeStructs;

namespace UInsight;

/// <summary>
/// High-level managed API for u-insight statistical analysis.
/// Wraps native P/Invoke calls with automatic memory management.
/// </summary>
public sealed class InsightClient : IDisposable
{
    private bool _disposed;

    /// <summary>
    /// Gets the native library version.
    /// </summary>
    public string GetVersion() => Native.GetVersion();

    #region Profiling

    /// <summary>
    /// Profiles a CSV string and returns column summaries.
    /// </summary>
    public ProfileResult ProfileCsv(string csvData)
    {
        var ctx = Native.insight_profile_csv(csvData);
        if (ctx == IntPtr.Zero)
            throw new InsightException(-1, "Failed to profile CSV data");

        try
        {
            var rows = Native.insight_profile_row_count(ctx);
            var cols = Native.insight_profile_col_count(ctx);

            var columns = new ColumnSummary[(int)cols];
            for (uint i = 0; i < (uint)cols; i++)
            {
                var native = new NativeStructs.CColumnSummary();
                Native.ThrowIfFailed(Native.insight_profile_column(ctx, i, ref native));
                var dataType = Enum.IsDefined(typeof(InsightDataType), native.DataType)
                    ? (InsightDataType)native.DataType
                    : InsightDataType.Text;

                columns[i] = new ColumnSummary
                {
                    Index = native.Index,
                    ValidCount = native.ValidCount,
                    NullCount = native.NullCount,
                    DataType = dataType,
                    Mean = native.Mean,
                    StdDev = native.StdDev,
                    Min = native.Min,
                    Max = native.Max
                };
            }

            return new ProfileResult { RowCount = rows, ColumnCount = cols, Columns = columns };
        }
        finally
        {
            Native.insight_profile_free(ctx);
        }
    }

    #endregion

    #region Clustering

    /// <summary>
    /// Runs K-Means++ clustering.
    /// </summary>
    public KMeansResult KMeans(double[,] data, uint k)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CKMeansResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(Native.insight_kmeans(ptr, nRows, nCols, k, ref native));
            }
        }

        try
        {
            return new KMeansResult
            {
                K = native.K,
                Wcss = native.Wcss,
                Iterations = native.Iterations,
                Labels = CopyU32Array(native.Labels, native.NLabels)
            };
        }
        finally
        {
            Native.insight_free_labels(native.Labels, native.NLabels);
        }
    }

    /// <summary>
    /// Runs Mini-Batch K-Means clustering.
    /// </summary>
    public KMeansResult MiniBatchKMeans(double[,] data, uint k, uint batchSize = 100, uint maxIter = 100, ulong seed = 42)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CKMeansResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_mini_batch_kmeans(ptr, nRows, nCols, k, batchSize, maxIter, seed, ref native));
            }
        }

        try
        {
            return new KMeansResult
            {
                K = native.K,
                Wcss = native.Wcss,
                Iterations = native.Iterations,
                Labels = CopyU32Array(native.Labels, native.NLabels)
            };
        }
        finally
        {
            Native.insight_free_labels(native.Labels, native.NLabels);
        }
    }

    /// <summary>
    /// Runs DBSCAN density-based clustering.
    /// </summary>
    public DbscanResult Dbscan(double[,] data, double epsilon, uint minSamples)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CDbscanResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_dbscan(ptr, nRows, nCols, epsilon, minSamples, ref native));
            }
        }

        try
        {
            return new DbscanResult
            {
                NClusters = native.NClusters,
                NoiseCount = native.NoiseCount,
                Labels = CopyI32Array(native.Labels, native.NLabels)
            };
        }
        finally
        {
            Native.insight_free_i32_array(native.Labels, native.NLabels);
        }
    }

    /// <summary>
    /// Runs Hierarchical Agglomerative clustering.
    /// Linkage: 0=Single, 1=Complete, 2=Average, 3=Ward.
    /// </summary>
    public HierarchicalResult Hierarchical(double[,] data, uint linkage, uint nClusters)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CHierarchicalResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_hierarchical(ptr, nRows, nCols, linkage, nClusters, ref native));
            }
        }

        try
        {
            return new HierarchicalResult
            {
                NClusters = native.NClusters,
                Labels = CopyU32Array(native.Labels, native.NLabels),
                MergeDistances = CopyF64Array(native.MergeDistances, native.NMerges),
                MergeSizes = CopyU32Array(native.MergeSizes, native.NMerges)
            };
        }
        finally
        {
            Native.insight_free_labels(native.Labels, native.NLabels);
            Native.insight_free_f64_array(native.MergeDistances, native.NMerges);
            Native.insight_free_labels(native.MergeSizes, native.NMerges);
        }
    }

    /// <summary>
    /// Runs HDBSCAN clustering.
    /// </summary>
    public HdbscanResult Hdbscan(double[,] data, uint minClusterSize, uint minSamples)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CHdbscanResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_hdbscan(ptr, nRows, nCols, minClusterSize, minSamples, ref native));
            }
        }

        try
        {
            return new HdbscanResult
            {
                NClusters = native.NClusters,
                NoiseCount = native.NoiseCount,
                Labels = CopyI32Array(native.Labels, native.NLabels),
                Probabilities = CopyF64Array(native.Probabilities, native.NLabels)
            };
        }
        finally
        {
            Native.insight_free_i32_array(native.Labels, native.NLabels);
            Native.insight_free_f64_array(native.Probabilities, native.NLabels);
        }
    }

    /// <summary>
    /// Computes the Gap Statistic for optimal K selection.
    /// </summary>
    public GapStatResult GapStatistic(double[,] data, uint kMin, uint kMax, uint nRefs = 10, ulong seed = 42)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CGapStatResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_gap_statistic(ptr, nRows, nCols, kMin, kMax, nRefs, seed, ref native));
            }
        }

        try
        {
            return new GapStatResult
            {
                BestK = native.BestK,
                GapValues = CopyF64Array(native.GapValues, native.NValues),
                StdErrors = CopyF64Array(native.StdErrors, native.NValues)
            };
        }
        finally
        {
            Native.insight_free_f64_array(native.GapValues, native.NValues);
            Native.insight_free_f64_array(native.StdErrors, native.NValues);
        }
    }

    #endregion

    #region PCA

    /// <summary>
    /// Runs Principal Component Analysis.
    /// </summary>
    public PcaResult Pca(double[,] data, uint nComponents, bool autoScale = true)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CPcaResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_pca(ptr, nRows, nCols, nComponents, autoScale ? 1 : 0, ref native));
            }
        }

        try
        {
            return new PcaResult
            {
                NComponents = native.NComponents,
                ExplainedVariance = CopyF64Array(native.ExplainedVariance, native.NVariance)
            };
        }
        finally
        {
            Native.insight_free_f64_array(native.ExplainedVariance, native.NVariance);
        }
    }

    #endregion

    #region Anomaly Detection

    /// <summary>
    /// Runs Isolation Forest anomaly detection.
    /// </summary>
    public AnomalyResult IsolationForest(double[,] data, uint nEstimators = 100, double contamination = 0.1, ulong seed = 42)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CAnomalyResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_isolation_forest(ptr, nRows, nCols, nEstimators, contamination, seed, ref native));
            }
        }

        try
        {
            return new AnomalyResult
            {
                Scores = CopyF64Array(native.Scores, native.N),
                Anomalies = CopyByteArray(native.Anomalies, native.N),
                AnomalyCount = native.AnomalyCount,
                Threshold = native.Threshold
            };
        }
        finally
        {
            Native.insight_free_f64_array(native.Scores, native.N);
            Native.insight_free_f64_array(native.Anomalies, native.N);
        }
    }

    /// <summary>
    /// Runs Local Outlier Factor anomaly detection.
    /// </summary>
    public AnomalyResult Lof(double[,] data, uint k = 20, double threshold = 1.5)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CAnomalyResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_lof(ptr, nRows, nCols, k, threshold, ref native));
            }
        }

        try
        {
            return new AnomalyResult
            {
                Scores = CopyF64Array(native.Scores, native.N),
                Anomalies = CopyByteArray(native.Anomalies, native.N),
                AnomalyCount = native.AnomalyCount,
                Threshold = native.Threshold
            };
        }
        finally
        {
            Native.insight_free_f64_array(native.Scores, native.N);
            Native.insight_free_f64_array(native.Anomalies, native.N);
        }
    }

    /// <summary>
    /// Runs Mahalanobis distance outlier detection.
    /// </summary>
    public MahalanobisResult Mahalanobis(double[,] data, double chi2Quantile = 0.975)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CMahalanobisResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_mahalanobis(ptr, nRows, nCols, chi2Quantile, ref native));
            }
        }

        try
        {
            return new MahalanobisResult
            {
                Distances = CopyF64Array(native.Distances, native.N),
                Anomalies = CopyByteArray(native.Anomalies, native.N),
                Threshold = native.Threshold,
                OutlierCount = native.OutlierCount
            };
        }
        finally
        {
            Native.insight_free_f64_array(native.Distances, native.N);
            Native.insight_free_f64_array(native.Anomalies, native.N);
        }
    }

    #endregion

    #region Statistical Analysis

    /// <summary>
    /// Computes Pearson correlation matrix.
    /// </summary>
    public CorrelationResult Correlation(double[,] data)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CCorrelationResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_correlation(ptr, nRows, nCols, ref native));
            }
        }

        try
        {
            var matrixSize = native.NVars * native.NVars;
            var flatMatrix = CopyF64Array(native.Matrix, matrixSize);
            var matrix = new double[native.NVars, native.NVars];
            for (uint i = 0; i < native.NVars; i++)
                for (uint j = 0; j < native.NVars; j++)
                    matrix[i, j] = flatMatrix[i * native.NVars + j];

            return new CorrelationResult
            {
                NVars = native.NVars,
                Matrix = matrix,
                NHighPairs = native.NHighPairs
            };
        }
        finally
        {
            Native.insight_free_f64_array(native.Matrix, native.NVars * native.NVars);
        }
    }

    /// <summary>
    /// Runs simple linear regression (y = a + bx).
    /// </summary>
    public RegressionResult Regression(double[] x, double[] y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("x and y must have the same length");

        var native = new NativeStructs.CRegressionResult();

        unsafe
        {
            fixed (double* xPtr = x)
            fixed (double* yPtr = y)
            {
                Native.ThrowIfFailed(
                    Native.insight_regression(xPtr, yPtr, (uint)x.Length, ref native));
            }
        }

        return new RegressionResult
        {
            Intercept = native.Intercept,
            Slope = native.Slope,
            RSquared = native.RSquared,
            AdjRSquared = native.AdjRSquared,
            FPValue = native.FPValue
        };
    }

    /// <summary>
    /// Computes Cramer's V contingency analysis.
    /// </summary>
    public CramersVResult CramersV(double[,] table)
    {
        var (nRows, nCols, flat) = Flatten(table);
        var native = new NativeStructs.CCramersVResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_cramers_v(ptr, nRows, nCols, ref native));
            }
        }

        return new CramersVResult
        {
            V = native.V,
            ChiSquared = native.ChiSquared,
            PValue = native.PValue
        };
    }

    #endregion

    #region Distribution

    /// <summary>
    /// Runs normality testing (KS, JB, SW, AD).
    /// </summary>
    public DistributionResult Distribution(double[] data, double significanceLevel = 0.05)
    {
        var native = new NativeStructs.CDistributionResult();

        unsafe
        {
            fixed (double* ptr = data)
            {
                Native.ThrowIfFailed(
                    Native.insight_distribution(ptr, (uint)data.Length, significanceLevel, ref native));
            }
        }

        return new DistributionResult
        {
            N = native.N,
            KsStatistic = native.KsStatistic,
            KsPValue = native.KsPValue,
            JbStatistic = native.JbStatistic,
            JbPValue = native.JbPValue,
            SwStatistic = native.SwStatistic,
            SwPValue = native.SwPValue,
            AdStatistic = native.AdStatistic,
            AdPValue = native.AdPValue,
            IsNormal = native.IsNormal != 0
        };
    }

    #endregion

    #region Feature Importance

    /// <summary>
    /// Computes composite feature importance scores.
    /// </summary>
    public FeatureImportanceResult FeatureImportance(double[,] data)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CFeatureImportanceResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_feature_importance(ptr, nRows, nCols, ref native));
            }
        }

        try
        {
            return new FeatureImportanceResult
            {
                Scores = CopyF64Array(native.Scores, native.NScores),
                ConditionNumber = native.ConditionNumber,
                NLowVariance = native.NLowVariance,
                NHighCorrPairs = native.NHighCorrPairs
            };
        }
        finally
        {
            Native.insight_free_f64_array(native.Scores, native.NScores);
        }
    }

    /// <summary>
    /// Runs ANOVA F-test feature selection.
    /// </summary>
    public AnovaSelectionResult AnovaSelect(double[,] data, uint[] target, double significanceLevel = 0.05)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CAnovaSelectionResult();

        unsafe
        {
            fixed (double* ptr = flat)
            fixed (uint* tgt = target)
            {
                Native.ThrowIfFailed(
                    Native.insight_anova_select(ptr, nRows, nCols, tgt, significanceLevel, ref native));
            }
        }

        try
        {
            var features = new AnovaFeature[native.NFeatures];
            for (uint i = 0; i < native.NFeatures; i++)
            {
                var f = Marshal.PtrToStructure<NativeStructs.CAnovaFeature>(
                    native.Features + (int)i * Marshal.SizeOf<NativeStructs.CAnovaFeature>());
                features[i] = new AnovaFeature { Index = f.Index, FStatistic = f.FStatistic, PValue = f.PValue };
            }

            return new AnovaSelectionResult
            {
                Features = features,
                NSelected = native.NSelected
            };
        }
        finally
        {
            Native.insight_free_anova_features(native.Features, native.NFeatures);
        }
    }

    /// <summary>
    /// Computes Mutual Information feature ranking.
    /// </summary>
    public MutualInfoResult MutualInfo(double[,] data, uint[] target, uint nBins = 10)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CMutualInfoResult();

        unsafe
        {
            fixed (double* ptr = flat)
            fixed (uint* tgt = target)
            {
                Native.ThrowIfFailed(
                    Native.insight_mutual_info(ptr, nRows, nCols, tgt, nBins, ref native));
            }
        }

        try
        {
            var features = new MutualInfoFeature[native.NFeatures];
            for (uint i = 0; i < native.NFeatures; i++)
            {
                var f = Marshal.PtrToStructure<NativeStructs.CMutualInfoFeature>(
                    native.Features + (int)i * Marshal.SizeOf<NativeStructs.CMutualInfoFeature>());
                features[i] = new MutualInfoFeature { Index = f.Index, Mi = f.Mi };
            }

            return new MutualInfoResult { Features = features };
        }
        finally
        {
            Native.insight_free_mi_features(native.Features, native.NFeatures);
        }
    }

    /// <summary>
    /// Computes Permutation Importance for regression.
    /// </summary>
    public PermImportanceResult PermutationImportance(double[,] data, double[] target, uint nRepeats = 5, ulong seed = 42)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CPermImportanceResult();

        unsafe
        {
            fixed (double* ptr = flat)
            fixed (double* tgt = target)
            {
                Native.ThrowIfFailed(
                    Native.insight_permutation_importance(ptr, nRows, nCols, tgt, nRepeats, seed, ref native));
            }
        }

        try
        {
            var features = new PermImportanceFeature[native.NFeatures];
            for (uint i = 0; i < native.NFeatures; i++)
            {
                var f = Marshal.PtrToStructure<NativeStructs.CPermImportanceFeature>(
                    native.Features + (int)i * Marshal.SizeOf<NativeStructs.CPermImportanceFeature>());
                features[i] = new PermImportanceFeature { Index = f.Index, Importance = f.Importance, StdDev = f.StdDev };
            }

            return new PermImportanceResult
            {
                BaselineScore = native.BaselineScore,
                Features = features
            };
        }
        finally
        {
            Native.insight_free_perm_features(native.Features, native.NFeatures);
        }
    }

    #endregion

    #region Helpers

    private static (uint nRows, uint nCols, double[] flat) Flatten(double[,] data)
    {
        var nRows = (uint)data.GetLength(0);
        var nCols = (uint)data.GetLength(1);
        var flat = new double[nRows * nCols];
        Buffer.BlockCopy(data, 0, flat, 0, flat.Length * sizeof(double));
        return (nRows, nCols, flat);
    }

    private static uint[] CopyU32Array(IntPtr ptr, uint count)
    {
        var result = new uint[count];
        if (count > 0 && ptr != IntPtr.Zero)
            Marshal.Copy(ptr, (int[])(object)result, 0, (int)count);
        return result;
    }

    private static int[] CopyI32Array(IntPtr ptr, uint count)
    {
        var result = new int[count];
        if (count > 0 && ptr != IntPtr.Zero)
            Marshal.Copy(ptr, result, 0, (int)count);
        return result;
    }

    private static double[] CopyF64Array(IntPtr ptr, uint count)
    {
        var result = new double[count];
        if (count > 0 && ptr != IntPtr.Zero)
            Marshal.Copy(ptr, result, 0, (int)count);
        return result;
    }

    private static byte[] CopyByteArray(IntPtr ptr, uint count)
    {
        var result = new byte[count];
        if (count > 0 && ptr != IntPtr.Zero)
            Marshal.Copy(ptr, result, 0, (int)count);
        return result;
    }

    #endregion

    #region IDisposable

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_disposed)
        {
            Native.insight_clear_error();
            _disposed = true;
        }
    }

    #endregion
}

#region Result Models

/// <summary>Profile result with column summaries.</summary>
public class ProfileResult
{
    /// <summary>Total row count.</summary>
    public long RowCount { get; init; }
    /// <summary>Total column count.</summary>
    public long ColumnCount { get; init; }
    /// <summary>Per-column summaries.</summary>
    public ColumnSummary[] Columns { get; init; } = [];
}

/// <summary>Column-level summary statistics.</summary>
public class ColumnSummary
{
    /// <summary>Column index.</summary>
    public uint Index { get; init; }
    /// <summary>Count of valid (non-null) values.</summary>
    public ulong ValidCount { get; init; }
    /// <summary>Count of null/missing values.</summary>
    public ulong NullCount { get; init; }
    /// <summary>Detected data type of the column.</summary>
    public InsightDataType DataType { get; init; }
    /// <summary>Mean (numeric columns only).</summary>
    public double Mean { get; init; }
    /// <summary>Standard deviation (numeric columns only).</summary>
    public double StdDev { get; init; }
    /// <summary>Minimum value (numeric columns only).</summary>
    public double Min { get; init; }
    /// <summary>Maximum value (numeric columns only).</summary>
    public double Max { get; init; }
}

/// <summary>K-Means clustering result.</summary>
public class KMeansResult
{
    /// <summary>Number of clusters.</summary>
    public uint K { get; init; }
    /// <summary>Within-cluster sum of squares.</summary>
    public double Wcss { get; init; }
    /// <summary>Number of iterations.</summary>
    public uint Iterations { get; init; }
    /// <summary>Cluster labels per data point.</summary>
    public uint[] Labels { get; init; } = [];
}

/// <summary>DBSCAN clustering result.</summary>
public class DbscanResult
{
    /// <summary>Number of clusters found.</summary>
    public uint NClusters { get; init; }
    /// <summary>Number of noise points.</summary>
    public uint NoiseCount { get; init; }
    /// <summary>Cluster labels per data point (-1 = noise).</summary>
    public int[] Labels { get; init; } = [];
}

/// <summary>Hierarchical clustering result.</summary>
public class HierarchicalResult
{
    /// <summary>Number of clusters.</summary>
    public uint NClusters { get; init; }
    /// <summary>Cluster labels per data point.</summary>
    public uint[] Labels { get; init; } = [];
    /// <summary>Merge distances in dendrogram order.</summary>
    public double[] MergeDistances { get; init; } = [];
    /// <summary>Merge sizes in dendrogram order.</summary>
    public uint[] MergeSizes { get; init; } = [];
}

/// <summary>HDBSCAN clustering result.</summary>
public class HdbscanResult
{
    /// <summary>Number of clusters found.</summary>
    public uint NClusters { get; init; }
    /// <summary>Number of noise points.</summary>
    public uint NoiseCount { get; init; }
    /// <summary>Cluster labels per data point (-1 = noise).</summary>
    public int[] Labels { get; init; } = [];
    /// <summary>Membership probabilities per data point.</summary>
    public double[] Probabilities { get; init; } = [];
}

/// <summary>Gap Statistic result for optimal K selection.</summary>
public class GapStatResult
{
    /// <summary>Best K found.</summary>
    public uint BestK { get; init; }
    /// <summary>Gap values for each K tested.</summary>
    public double[] GapValues { get; init; } = [];
    /// <summary>Standard errors for each K tested.</summary>
    public double[] StdErrors { get; init; } = [];
}

/// <summary>PCA result.</summary>
public class PcaResult
{
    /// <summary>Number of components.</summary>
    public uint NComponents { get; init; }
    /// <summary>Explained variance ratios per component.</summary>
    public double[] ExplainedVariance { get; init; } = [];
}

/// <summary>Anomaly detection result (Isolation Forest / LOF).</summary>
public class AnomalyResult
{
    /// <summary>Anomaly scores per data point.</summary>
    public double[] Scores { get; init; } = [];
    /// <summary>Anomaly flags per data point (1 = anomaly).</summary>
    public byte[] Anomalies { get; init; } = [];
    /// <summary>Number of anomalies detected.</summary>
    public uint AnomalyCount { get; init; }
    /// <summary>Anomaly threshold used.</summary>
    public double Threshold { get; init; }
}

/// <summary>Mahalanobis distance result.</summary>
public class MahalanobisResult
{
    /// <summary>Mahalanobis distances per data point.</summary>
    public double[] Distances { get; init; } = [];
    /// <summary>Outlier flags per data point (1 = outlier).</summary>
    public byte[] Anomalies { get; init; } = [];
    /// <summary>Chi-squared threshold.</summary>
    public double Threshold { get; init; }
    /// <summary>Number of outliers detected.</summary>
    public uint OutlierCount { get; init; }
}

/// <summary>Correlation matrix result.</summary>
public class CorrelationResult
{
    /// <summary>Number of variables.</summary>
    public uint NVars { get; init; }
    /// <summary>Correlation matrix (NVars x NVars).</summary>
    public double[,] Matrix { get; init; } = new double[0, 0];
    /// <summary>Number of high-correlation pairs.</summary>
    public uint NHighPairs { get; init; }
}

/// <summary>Linear regression result.</summary>
public class RegressionResult
{
    /// <summary>Y-intercept.</summary>
    public double Intercept { get; init; }
    /// <summary>Slope coefficient.</summary>
    public double Slope { get; init; }
    /// <summary>R-squared value.</summary>
    public double RSquared { get; init; }
    /// <summary>Adjusted R-squared value.</summary>
    public double AdjRSquared { get; init; }
    /// <summary>F-test p-value.</summary>
    public double FPValue { get; init; }
}

/// <summary>Cramer's V contingency result.</summary>
public class CramersVResult
{
    /// <summary>Cramer's V value.</summary>
    public double V { get; init; }
    /// <summary>Chi-squared statistic.</summary>
    public double ChiSquared { get; init; }
    /// <summary>P-value.</summary>
    public double PValue { get; init; }
}

/// <summary>Distribution normality test result.</summary>
public class DistributionResult
{
    /// <summary>Sample size.</summary>
    public uint N { get; init; }
    /// <summary>KS test statistic.</summary>
    public double KsStatistic { get; init; }
    /// <summary>KS test p-value.</summary>
    public double KsPValue { get; init; }
    /// <summary>Jarque-Bera test statistic.</summary>
    public double JbStatistic { get; init; }
    /// <summary>Jarque-Bera test p-value.</summary>
    public double JbPValue { get; init; }
    /// <summary>Shapiro-Wilk test statistic.</summary>
    public double SwStatistic { get; init; }
    /// <summary>Shapiro-Wilk test p-value.</summary>
    public double SwPValue { get; init; }
    /// <summary>Anderson-Darling test statistic.</summary>
    public double AdStatistic { get; init; }
    /// <summary>Anderson-Darling test p-value.</summary>
    public double AdPValue { get; init; }
    /// <summary>Whether the data is considered normal.</summary>
    public bool IsNormal { get; init; }
}

/// <summary>Feature importance result.</summary>
public class FeatureImportanceResult
{
    /// <summary>Importance scores per feature.</summary>
    public double[] Scores { get; init; } = [];
    /// <summary>Condition number of the correlation matrix.</summary>
    public double ConditionNumber { get; init; }
    /// <summary>Number of low-variance features.</summary>
    public uint NLowVariance { get; init; }
    /// <summary>Number of high-correlation feature pairs.</summary>
    public uint NHighCorrPairs { get; init; }
}

/// <summary>ANOVA feature.</summary>
public class AnovaFeature
{
    /// <summary>Feature index.</summary>
    public uint Index { get; init; }
    /// <summary>F-statistic.</summary>
    public double FStatistic { get; init; }
    /// <summary>P-value.</summary>
    public double PValue { get; init; }
}

/// <summary>ANOVA feature selection result.</summary>
public class AnovaSelectionResult
{
    /// <summary>Features with F-statistics.</summary>
    public AnovaFeature[] Features { get; init; } = [];
    /// <summary>Number of selected features.</summary>
    public uint NSelected { get; init; }
}

/// <summary>Mutual information feature.</summary>
public class MutualInfoFeature
{
    /// <summary>Feature index.</summary>
    public uint Index { get; init; }
    /// <summary>Mutual information value.</summary>
    public double Mi { get; init; }
}

/// <summary>Mutual information result.</summary>
public class MutualInfoResult
{
    /// <summary>Features with MI scores.</summary>
    public MutualInfoFeature[] Features { get; init; } = [];
}

/// <summary>Permutation importance feature.</summary>
public class PermImportanceFeature
{
    /// <summary>Feature index.</summary>
    public uint Index { get; init; }
    /// <summary>Importance score.</summary>
    public double Importance { get; init; }
    /// <summary>Standard deviation across repeats.</summary>
    public double StdDev { get; init; }
}

/// <summary>Permutation importance result.</summary>
public class PermImportanceResult
{
    /// <summary>Baseline model score.</summary>
    public double BaselineScore { get; init; }
    /// <summary>Features with importance scores.</summary>
    public PermImportanceFeature[] Features { get; init; } = [];
}

#endregion
