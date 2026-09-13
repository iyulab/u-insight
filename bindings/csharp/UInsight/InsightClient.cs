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
        return BuildProfileResult(ctx);
    }

    /// <summary>
    /// Profiles a column-major JSON string and returns column summaries.
    /// </summary>
    /// <remarks>
    /// Expected format: <c>{"col1": [v1, v2, ...], "col2": [...]}</c>.
    /// Values can be numbers, booleans, strings, or null.
    /// Column types are inferred automatically.
    /// </remarks>
    public ProfileResult ProfileJson(string jsonData)
    {
        var ctx = Native.insight_profile_json(jsonData);
        if (ctx == IntPtr.Zero)
            throw new InsightException(-1, "Failed to profile JSON data");
        return BuildProfileResult(ctx);
    }

    private static ProfileResult BuildProfileResult(IntPtr ctx)
    {
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

    /// <summary>
    /// Computes silhouette scores for an existing clustering assignment.
    /// </summary>
    /// <param name="data">Input matrix (rows = samples, columns = features).</param>
    /// <param name="labels">Cluster label per sample. Each label must be in <c>[0, k)</c>.</param>
    /// <param name="k">Number of distinct clusters represented in <paramref name="labels"/>.</param>
    /// <returns>Silhouette analysis containing the mean and per-sample scores.</returns>
    /// <remarks>
    /// Silhouette ranges from -1 (wrong cluster) to +1 (well-separated). O(n²) memory and time —
    /// use sparingly on large inputs. Singleton clusters and points with no other cluster present
    /// are excluded from the mean and reported as 0.0 in <see cref="SilhouetteResult.PerSample"/>.
    /// </remarks>
    public SilhouetteResult Silhouette(double[,] data, uint[] labels, uint k)
    {
        ArgumentNullException.ThrowIfNull(labels);
        var (nRows, nCols, flat) = Flatten(data);
        if (labels.Length != nRows)
        {
            throw new ArgumentException(
                $"labels length ({labels.Length}) must match data row count ({nRows})",
                nameof(labels));
        }

        var native = new NativeStructs.CSilhouetteResult();
        unsafe
        {
            fixed (double* dataPtr = flat)
            fixed (uint* labelsPtr = labels)
            {
                Native.ThrowIfFailed(
                    Native.insight_silhouette(dataPtr, nRows, nCols, labelsPtr, k, ref native));
            }
        }

        try
        {
            return new SilhouetteResult
            {
                Avg = native.Avg,
                PerSample = CopyF64Array(native.PerSample, native.NSamples)
            };
        }
        finally
        {
            Native.insight_free_f64_array(native.PerSample, native.NSamples);
        }
    }

    #endregion

    #region PCA

    /// <summary>
    /// Runs Principal Component Analysis.
    /// </summary>
    /// <param name="data">Input matrix (rows = samples, columns = features).</param>
    /// <param name="nComponents">Number of principal components to retain.</param>
    /// <param name="autoScale">When true, standardise each feature to unit variance before PCA
    /// (correlation-PCA). When false, only mean-centre the data (covariance-PCA).</param>
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
            var k = native.NComponents;
            var p = native.NFeatures;
            var n = native.NSamples;

            return new PcaResult
            {
                NComponents = k,
                NFeatures = p,
                NSamples = n,
                ExplainedVariance = CopyF64Array(native.ExplainedVariance, k),
                CumulativeVariance = CopyF64Array(native.CumulativeVariance, k),
                Loadings = Reshape2D(CopyF64Array(native.Loadings, k * p), (int)k, (int)p),
                Scores = ReshapeJagged(CopyF64Array(native.Scores, n * k), (int)n, (int)k)
            };
        }
        finally
        {
            Native.insight_free_f64_array(native.ExplainedVariance, native.NComponents);
            Native.insight_free_f64_array(native.CumulativeVariance, native.NComponents);
            Native.insight_free_f64_array(native.Loadings, native.NComponents * native.NFeatures);
            Native.insight_free_f64_array(native.Scores, native.NSamples * native.NComponents);
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
    /// Computes a correlation matrix using the chosen method.
    /// </summary>
    /// <param name="data">Row-major numeric data; rows are observations, columns are variables.</param>
    /// <param name="method">Correlation method (default: Pearson).</param>
    public CorrelationResult Correlation(
        double[,] data,
        CorrelationMethodKind method = CorrelationMethodKind.Pearson)
    {
        var (nRows, nCols, flat) = Flatten(data);
        var native = new NativeStructs.CCorrelationResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_correlation(ptr, nRows, nCols, (uint)method, ref native));
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

    #region Changepoint Detection

    /// <summary>
    /// Detect changepoints using the PELT algorithm (Killick et al., 2012).
    /// </summary>
    /// <param name="data">Univariate time series.</param>
    /// <param name="cost">0 = L2 (mean change), 1 = Normal (mean+variance).</param>
    /// <param name="penalty">Penalty per changepoint. 0.0 = BIC (automatic).</param>
    /// <param name="minSegmentLen">Minimum segment length (>= 2).</param>
    public PeltResult Pelt(double[] data, uint cost = 0, double penalty = 0.0, uint minSegmentLen = 2)
    {
        var native = new NativeStructs.CPeltResult();

        unsafe
        {
            fixed (double* ptr = data)
            {
                Native.ThrowIfFailed(
                    Native.insight_pelt(ptr, (uint)data.Length, cost, penalty, minSegmentLen, ref native));
            }
        }

        try
        {
            var changepoints = CopyU32Array(native.Changepoints, native.NChangepoints);

            return new PeltResult
            {
                Changepoints = changepoints,
                NSegments = native.NChangepoints + 1
            };
        }
        finally
        {
            Native.insight_free_pelt_result(ref native);
        }
    }

    /// <summary>
    /// Estimate the dominant period of a univariate series (AutoPeriod —
    /// Vlachos, Yu &amp; Castelli 2005: permutation-thresholded periodogram
    /// peaks refined on the autocorrelation function). Deterministic for a
    /// series. <see cref="PeriodEstimate.Period"/> is <c>null</c> — not an
    /// error — when no periodicity passes both stages.
    /// </summary>
    /// <param name="data">Univariate series, at least 8 finite values.</param>
    public PeriodEstimate EstimatePeriod(double[] data)
    {
        var native = new NativeStructs.CPeriodEstimate();
        unsafe
        {
            fixed (double* ptr = data)
            {
                Native.ThrowIfFailed(Native.insight_estimate_period(ptr, (uint)data.Length, ref native));
            }
        }

        try
        {
            var candidates = new PeriodCandidate[native.NCandidates];
            unsafe
            {
                if (native.NCandidates > 0 && native.Candidates != IntPtr.Zero)
                {
                    var raw = (NativeStructs.CPeriodCandidate*)native.Candidates;
                    for (var i = 0; i < native.NCandidates; i++)
                    {
                        candidates[i] = new PeriodCandidate
                        {
                            Period = raw[i].Period,
                            Acf = raw[i].Acf,
                            Bin = raw[i].Bin,
                            Power = raw[i].Power,
                            PowerShare = raw[i].PowerShare,
                        };
                    }
                }
            }
            return new PeriodEstimate
            {
                Period = native.Period == 0 ? null : native.Period,
                Candidates = candidates,
                N = native.N,
                AcfThreshold = native.AcfThreshold,
                PowerThreshold = native.PowerThreshold,
            };
        }
        finally
        {
            Native.insight_free_period_estimate(ref native);
        }
    }

    /// <summary>
    /// Score every point of a series for anomalies by spectral residual
    /// saliency (Ren et al. 2019) — spikes, steps and dropouts, without a
    /// trained model and without assuming a period. Options left
    /// <c>null</c> take the paper's defaults (q = 3, z = 40, τ = 3, z-score
    /// gate 1.5, 70% band, no batching).
    /// </summary>
    /// <param name="data">Univariate series, at least 12 finite values.</param>
    /// <param name="options">Scoring options, or <c>null</c> for the defaults.</param>
    public SpectralResidualResult SpectralResidual(double[] data, SpectralResidualOptions? options = null)
    {
        var native = new NativeStructs.CSpectralResidualResult();
        unsafe
        {
            fixed (double* ptr = data)
            {
                if (options is null)
                {
                    Native.ThrowIfFailed(
                        Native.insight_spectral_residual(ptr, (uint)data.Length, null, ref native));
                }
                else
                {
                    var o = new NativeStructs.CSpectralResidualOptions
                    {
                        AveragingWindow = options.AveragingWindow,
                        JudgementWindow = options.JudgementWindow,
                        Threshold = options.Threshold,
                        MinZscore = options.MinZscore,
                        Sensitivity = options.Sensitivity,
                        BatchSize = options.BatchSize ?? 0,
                    };
                    Native.ThrowIfFailed(
                        Native.insight_spectral_residual(ptr, (uint)data.Length, &o, ref native));
                }
            }
        }

        try
        {
            var points = new SrPoint[native.NPoints];
            unsafe
            {
                if (native.NPoints > 0 && native.Points != IntPtr.Zero)
                {
                    var raw = (NativeStructs.CSrPoint*)native.Points;
                    for (var i = 0; i < native.NPoints; i++)
                    {
                        points[i] = new SrPoint
                        {
                            Index = raw[i].Index,
                            Value = raw[i].Value,
                            Saliency = raw[i].Saliency,
                            Score = raw[i].Score,
                            Expected = raw[i].Expected,
                            Lower = raw[i].Lower,
                            Upper = raw[i].Upper,
                            IsAnomaly = raw[i].IsAnomaly != 0,
                        };
                    }
                }
            }
            return new SpectralResidualResult
            {
                Points = points,
                Anomalies = points.Where(p => p.IsAnomaly).Select(p => p.Index).ToArray(),
            };
        }
        finally
        {
            Native.insight_free_spectral_residual_result(ref native);
        }
    }

    /// <summary>
    /// Detect changepoints in multivariate (multi-channel) time-series data using PELT.
    /// </summary>
    /// <param name="data">
    /// Multivariate observations matrix. Rows = time-series samples, columns = signal channels —
    /// the same convention as <see cref="Pca"/>, <see cref="KMeans"/>, and every other
    /// multi-dimensional API in this library.
    /// </param>
    /// <param name="cost">0 = L2 (mean change), 1 = Normal (mean+variance).</param>
    /// <param name="penalty">Penalty per changepoint. 0.0 = BIC (automatic).</param>
    /// <param name="minSegmentLen">Minimum segment length (>= 2).</param>
    public PeltResult PeltMulti(double[,] data, uint cost = 0, double penalty = 0.0, uint minSegmentLen = 2)
    {
        var (nSamples, nChannels, flat) = Flatten(data);
        var native = new NativeStructs.CPeltResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_pelt_multi(ptr, nSamples, nChannels, cost, penalty, minSegmentLen, ref native));
            }
        }

        try
        {
            var changepoints = CopyU32Array(native.Changepoints, native.NChangepoints);

            return new PeltResult
            {
                Changepoints = changepoints,
                NSegments = native.NChangepoints + 1
            };
        }
        finally
        {
            Native.insight_free_pelt_result(ref native);
        }
    }

    #endregion

    #region Trend & Density Estimation

    /// <summary>
    /// Mann-Kendall non-parametric trend test with Sen's slope estimator.
    /// Tests H0: no monotonic trend vs H1: monotonic trend exists.
    /// </summary>
    /// <param name="data">Time-ordered observations (needs at least 4 points).</param>
    public MannKendallResult MannKendall(double[] data)
    {
        var native = new NativeStructs.CMannKendallResult();

        unsafe
        {
            fixed (double* ptr = data)
            {
                Native.ThrowIfFailed(
                    Native.insight_mann_kendall(ptr, (uint)data.Length, ref native));
            }
        }

        return new MannKendallResult
        {
            SStatistic = native.SStatistic,
            Variance = native.Variance,
            ZStatistic = native.ZStatistic,
            PValue = native.PValue,
            KendallTau = native.KendallTau,
            SenSlope = native.SenSlope,
        };
    }

    /// <summary>
    /// Gaussian kernel density estimation.
    /// </summary>
    /// <param name="data">Sample observations (needs at least 2 points).</param>
    /// <param name="method">Bandwidth selection method.</param>
    /// <param name="bandwidth">Used only when <paramref name="method"/> is <see cref="KdeBandwidthMethod.Manual"/>.</param>
    /// <param name="nPoints">Number of evaluation grid points (typical: 256-1024).</param>
    public KdeResult Kde(
        double[] data,
        KdeBandwidthMethod method = KdeBandwidthMethod.Silverman,
        double bandwidth = 0.0,
        uint nPoints = 512)
    {
        var native = new NativeStructs.CKdeResult();

        unsafe
        {
            fixed (double* ptr = data)
            {
                Native.ThrowIfFailed(
                    Native.insight_kde(ptr, (uint)data.Length, (uint)method, bandwidth, nPoints, ref native));
            }
        }

        try
        {
            return new KdeResult
            {
                X = CopyF64Array(native.X, native.NPoints),
                Density = CopyF64Array(native.Density, native.NPoints),
                Bandwidth = native.Bandwidth,
            };
        }
        finally
        {
            Native.insight_free_kde_result(ref native);
        }
    }

    #endregion

    #region SPC Variables Control Charts

    /// <summary>
    /// X-bar-R control chart (subgroup mean + range). Suitable for subgroup
    /// sizes 2 to 25.
    /// </summary>
    /// <param name="data">
    /// Subgroup observations matrix. Rows = subgroups, columns = individual
    /// measurements within each subgroup — same convention as <see cref="PeltMulti"/>.
    /// </param>
    public VariablesChartResult XBarRChart(double[,] data)
    {
        var (nSubgroups, subgroupSize, flat) = Flatten(data);
        var native = new NativeStructs.CVariablesChartResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_xbar_r_chart(ptr, nSubgroups, subgroupSize, ref native));
            }
        }

        try
        {
            return BuildVariablesChartResult(native);
        }
        finally
        {
            Native.insight_free_variables_chart_result(ref native);
        }
    }

    /// <summary>
    /// X-bar-S control chart (subgroup mean + standard deviation). Preferred
    /// over X-bar-R for larger subgroups. Suitable for subgroup sizes 2 to 25.
    /// </summary>
    /// <param name="data">
    /// Subgroup observations matrix. Rows = subgroups, columns = individual
    /// measurements within each subgroup — same convention as <see cref="PeltMulti"/>.
    /// </param>
    public VariablesChartResult XBarSChart(double[,] data)
    {
        var (nSubgroups, subgroupSize, flat) = Flatten(data);
        var native = new NativeStructs.CVariablesChartResult();

        unsafe
        {
            fixed (double* ptr = flat)
            {
                Native.ThrowIfFailed(
                    Native.insight_xbar_s_chart(ptr, nSubgroups, subgroupSize, ref native));
            }
        }

        try
        {
            return BuildVariablesChartResult(native);
        }
        finally
        {
            Native.insight_free_variables_chart_result(ref native);
        }
    }

    /// <summary>
    /// Individual-MR control chart (single observations + moving range).
    /// The MR (secondary) series has one fewer point than the I (primary)
    /// series — the first moving range is undefined.
    /// </summary>
    /// <param name="data">Individual observations (needs at least 2 points).</param>
    public VariablesChartResult IndividualMrChart(double[] data)
    {
        var native = new NativeStructs.CVariablesChartResult();

        unsafe
        {
            fixed (double* ptr = data)
            {
                Native.ThrowIfFailed(
                    Native.insight_individual_mr_chart(ptr, (uint)data.Length, ref native));
            }
        }

        try
        {
            return BuildVariablesChartResult(native);
        }
        finally
        {
            Native.insight_free_variables_chart_result(ref native);
        }
    }

    private static unsafe VariablesChartResult BuildVariablesChartResult(
        NativeStructs.CVariablesChartResult native)
    {
        return new VariablesChartResult
        {
            Primary = new ControlLimits
            {
                Ucl = native.PrimaryUcl,
                Cl = native.PrimaryCl,
                Lcl = native.PrimaryLcl,
            },
            PrimaryPoints = CopySpcPoints(native.PrimaryPoints, native.NPrimaryPoints),
            Secondary = new ControlLimits
            {
                Ucl = native.SecondaryUcl,
                Cl = native.SecondaryCl,
                Lcl = native.SecondaryLcl,
            },
            SecondaryPoints = CopySpcPoints(native.SecondaryPoints, native.NSecondaryPoints),
            SigmaHat = NanToNull(native.SigmaHat),
            InControl = native.InControl != 0,
        };
    }

    private static unsafe SpcChartPoint[] CopySpcPoints(IntPtr ptr, uint count)
    {
        var result = new SpcChartPoint[count];
        if (count > 0 && ptr != IntPtr.Zero)
        {
            var native = (NativeStructs.CSpcChartPoint*)ptr;
            for (var i = 0; i < count; i++)
            {
                result[i] = new SpcChartPoint
                {
                    Value = native[i].Value,
                    ViolationMask = (SpcViolation)native[i].ViolationMask,
                };
            }
        }
        return result;
    }

    #endregion

    #region SPC Attributes Control Charts

    /// <summary>
    /// P chart (proportion nonconforming, variable sample size).
    /// </summary>
    /// <param name="defectives">Number of defective items per subgroup.</param>
    /// <param name="sampleSizes">Total sample size per subgroup (parallel to <paramref name="defectives"/>).</param>
    public AttributeChartResult PChart(ulong[] defectives, ulong[] sampleSizes)
    {
        var native = new NativeStructs.CAttributeChartResult();
        unsafe
        {
            fixed (ulong* defPtr = defectives)
            fixed (ulong* sizePtr = sampleSizes)
            {
                Native.ThrowIfFailed(
                    Native.insight_p_chart(defPtr, sizePtr, (uint)defectives.Length, ref native));
            }
        }

        try
        {
            return BuildAttributeChartResult(native);
        }
        finally
        {
            Native.insight_free_attribute_chart_result(ref native);
        }
    }

    /// <summary>
    /// NP chart (count nonconforming, constant sample size).
    /// </summary>
    /// <param name="defectiveCounts">Defective count per subgroup.</param>
    /// <param name="sampleSize">Constant sample size (must be &gt; 0).</param>
    public AttributeChartResult NpChart(ulong[] defectiveCounts, ulong sampleSize)
    {
        var native = new NativeStructs.CAttributeChartResult();
        unsafe
        {
            fixed (ulong* ptr = defectiveCounts)
            {
                Native.ThrowIfFailed(
                    Native.insight_np_chart(ptr, (uint)defectiveCounts.Length, sampleSize, ref native));
            }
        }

        try
        {
            return BuildAttributeChartResult(native);
        }
        finally
        {
            Native.insight_free_attribute_chart_result(ref native);
        }
    }

    /// <summary>
    /// C chart (defect count, constant area of opportunity).
    /// </summary>
    /// <param name="defectCounts">Defect count per inspection unit.</param>
    public AttributeChartResult CChart(ulong[] defectCounts)
    {
        var native = new NativeStructs.CAttributeChartResult();
        unsafe
        {
            fixed (ulong* ptr = defectCounts)
            {
                Native.ThrowIfFailed(
                    Native.insight_c_chart(ptr, (uint)defectCounts.Length, ref native));
            }
        }

        try
        {
            return BuildAttributeChartResult(native);
        }
        finally
        {
            Native.insight_free_attribute_chart_result(ref native);
        }
    }

    /// <summary>
    /// U chart (defects per unit, variable area of opportunity).
    /// </summary>
    /// <param name="defects">Defect count per subgroup.</param>
    /// <param name="unitsInspected">Units inspected per subgroup (parallel to <paramref name="defects"/>).</param>
    public AttributeChartResult UChart(ulong[] defects, double[] unitsInspected)
    {
        var native = new NativeStructs.CAttributeChartResult();
        unsafe
        {
            fixed (ulong* defPtr = defects)
            fixed (double* unitPtr = unitsInspected)
            {
                Native.ThrowIfFailed(
                    Native.insight_u_chart(defPtr, unitPtr, (uint)defects.Length, ref native));
            }
        }

        try
        {
            return BuildAttributeChartResult(native);
        }
        finally
        {
            Native.insight_free_attribute_chart_result(ref native);
        }
    }

    private static AttributeChartResult BuildAttributeChartResult(
        NativeStructs.CAttributeChartResult native)
    {
        return new AttributeChartResult
        {
            Points = CopyAttributePoints(native.Points, native.NPoints),
        };
    }

    #endregion

    #region SPC Laney P'/U' + G/T Charts

    /// <summary>
    /// Laney P' chart (overdispersion-adjusted proportion nonconforming).
    /// Needs at least 3 subgroups.
    /// </summary>
    /// <param name="defectives">Number of defective items per subgroup.</param>
    /// <param name="sampleSizes">Total sample size per subgroup (parallel to <paramref name="defectives"/>).</param>
    public LaneyChartResult LaneyPChart(ulong[] defectives, ulong[] sampleSizes)
    {
        var native = new NativeStructs.CLaneyChartResult();
        unsafe
        {
            fixed (ulong* defPtr = defectives)
            fixed (ulong* sizePtr = sampleSizes)
            {
                Native.ThrowIfFailed(
                    Native.insight_laney_p_chart(defPtr, sizePtr, (uint)defectives.Length, ref native));
            }
        }

        try
        {
            return BuildLaneyChartResult(native);
        }
        finally
        {
            Native.insight_free_laney_chart_result(ref native);
        }
    }

    /// <summary>
    /// Laney U' chart (overdispersion-adjusted defect rate).
    /// Needs at least 3 subgroups.
    /// </summary>
    /// <param name="defects">Defect count per subgroup.</param>
    /// <param name="unitsInspected">Units inspected per subgroup (parallel to <paramref name="defects"/>).</param>
    public LaneyChartResult LaneyUChart(ulong[] defects, double[] unitsInspected)
    {
        var native = new NativeStructs.CLaneyChartResult();
        unsafe
        {
            fixed (ulong* defPtr = defects)
            fixed (double* unitPtr = unitsInspected)
            {
                Native.ThrowIfFailed(
                    Native.insight_laney_u_chart(defPtr, unitPtr, (uint)defects.Length, ref native));
            }
        }

        try
        {
            return BuildLaneyChartResult(native);
        }
        finally
        {
            Native.insight_free_laney_chart_result(ref native);
        }
    }

    /// <summary>
    /// G chart (geometric distribution) for rare-event monitoring — number
    /// of conforming units between consecutive defects. Needs at least 3 points.
    /// </summary>
    public RareEventChartResult GChart(double[] interEventCounts)
    {
        var native = new NativeStructs.CRareEventChartResult();
        unsafe
        {
            fixed (double* ptr = interEventCounts)
            {
                Native.ThrowIfFailed(
                    Native.insight_g_chart(ptr, (uint)interEventCounts.Length, ref native));
            }
        }

        try
        {
            return BuildRareEventChartResult(native);
        }
        finally
        {
            Native.insight_free_rare_event_chart_result(ref native);
        }
    }

    /// <summary>
    /// T chart (exponential distribution) for rare-event monitoring — time
    /// between consecutive defects. Needs at least 3 points.
    /// </summary>
    public RareEventChartResult TChart(double[] interEventTimes)
    {
        var native = new NativeStructs.CRareEventChartResult();
        unsafe
        {
            fixed (double* ptr = interEventTimes)
            {
                Native.ThrowIfFailed(
                    Native.insight_t_chart(ptr, (uint)interEventTimes.Length, ref native));
            }
        }

        try
        {
            return BuildRareEventChartResult(native);
        }
        finally
        {
            Native.insight_free_rare_event_chart_result(ref native);
        }
    }

    private static unsafe LaneyChartResult BuildLaneyChartResult(NativeStructs.CLaneyChartResult native)
    {
        return new LaneyChartResult
        {
            Bar = native.Bar,
            Phi = native.Phi,
            Points = CopyAttributePoints(native.Points, native.NPoints),
        };
    }

    private static unsafe RareEventChartResult BuildRareEventChartResult(
        NativeStructs.CRareEventChartResult native)
    {
        return new RareEventChartResult
        {
            Bar = native.Bar,
            Points = CopyAttributePoints(native.Points, native.NPoints),
        };
    }

    private static unsafe AttributeChartPoint[] CopyAttributePoints(IntPtr ptr, uint count)
    {
        var result = new AttributeChartPoint[count];
        if (count > 0 && ptr != IntPtr.Zero)
        {
            var raw = (NativeStructs.CAttributeChartPoint*)ptr;
            for (var i = 0; i < count; i++)
            {
                result[i] = new AttributeChartPoint
                {
                    Value = raw[i].Value,
                    Ucl = raw[i].Ucl,
                    Cl = raw[i].Cl,
                    Lcl = raw[i].Lcl,
                    OutOfControl = raw[i].OutOfControl != 0,
                };
            }
        }
        return result;
    }

    #endregion

    #region Process Capability

    /// <summary>
    /// Standard process capability indices (Cp, Cpk, Pp, Ppk, Cpm).
    /// </summary>
    /// <param name="data">Process observations (needs at least 2 points).</param>
    /// <param name="usl">Upper specification limit. At least one of <paramref name="usl"/>/<paramref name="lsl"/> must be set.</param>
    /// <param name="lsl">Lower specification limit.</param>
    /// <param name="target">Target value for Cpm. Without it <c>Cpm</c> is <c>null</c>; pass the midpoint of <paramref name="usl"/>/<paramref name="lsl"/> explicitly if that is the target.</param>
    /// <param name="sigmaWithin">
    /// Short-term standard deviation (e.g. from a control chart's R-bar/d2,
    /// S-bar/c4 or MR-bar/d2). When omitted -- a flat vector with no subgroup
    /// structure -- only the long-term indices are reported: <c>Cp</c>,
    /// <c>Cpk</c>, <c>Cpu</c>, <c>Cpl</c> and <c>StdDevWithin</c> are
    /// <c>null</c>. They are not filled from the overall sigma, which would
    /// make <c>Cp</c> equal <c>Pp</c> for every input.
    /// </param>
    public CapabilityIndices ProcessCapability(
        double[] data, double? usl = null, double? lsl = null,
        double? target = null, double? sigmaWithin = null)
    {
        var native = new NativeStructs.CCapabilityIndices();
        unsafe
        {
            fixed (double* ptr = data)
            {
                Native.ThrowIfFailed(Native.insight_process_capability(
                    ptr, (uint)data.Length,
                    usl ?? double.NaN, lsl ?? double.NaN,
                    target ?? double.NaN, sigmaWithin ?? double.NaN,
                    ref native));
            }
        }
        return BuildCapabilityIndices(native);
    }

    /// <summary>
    /// Process capability for non-normal data via Box-Cox transformation.
    /// </summary>
    /// <param name="data">Process observations (must all be strictly positive, needs at least 4 points).</param>
    /// <param name="usl">Upper specification limit (must be positive if set).</param>
    /// <param name="lsl">Lower specification limit (must be positive if set).</param>
    public BoxCoxCapabilityResult BoxCoxCapability(double[] data, double? usl = null, double? lsl = null)
    {
        var native = new NativeStructs.CBoxCoxCapabilityResult();
        unsafe
        {
            fixed (double* ptr = data)
            {
                Native.ThrowIfFailed(Native.insight_boxcox_capability(
                    ptr, (uint)data.Length, usl ?? double.NaN, lsl ?? double.NaN, ref native));
            }
        }
        return new BoxCoxCapabilityResult
        {
            Lambda = native.Lambda,
            Indices = BuildCapabilityIndices(native.Indices),
        };
    }

    /// <summary>
    /// Percentile-based process capability indices (ISO 22514-2), robust to non-normal data.
    /// </summary>
    /// <param name="data">Process observations (needs at least 20 points).</param>
    /// <param name="lsl">Lower specification limit. At least one of <paramref name="lsl"/>/<paramref name="usl"/> must be set.</param>
    /// <param name="usl">Upper specification limit.</param>
    public PercentileCapabilityResult PercentileCapability(double[] data, double? lsl = null, double? usl = null)
    {
        var native = new NativeStructs.CPercentileCapabilityResult();
        unsafe
        {
            fixed (double* ptr = data)
            {
                Native.ThrowIfFailed(Native.insight_percentile_capability(
                    ptr, (uint)data.Length, lsl ?? double.NaN, usl ?? double.NaN, ref native));
            }
        }
        return new PercentileCapabilityResult
        {
            CpStar = NanToNull(native.CpStar),
            CpkStar = NanToNull(native.CpkStar),
            CpuStar = NanToNull(native.CpuStar),
            CplStar = NanToNull(native.CplStar),
            Median = native.Median,
            PercentileLower = native.PercentileLower,
            PercentileUpper = native.PercentileUpper,
        };
    }

    /// <summary>Converts a sigma quality level to a parts-per-million (PPM) defect rate (Motorola 1.5-sigma shift convention).</summary>
    public double SigmaToPpm(double sigma) => Native.insight_sigma_to_ppm(sigma);

    /// <summary>
    /// Converts a parts-per-million (PPM) defect rate to a sigma quality level.
    /// Returns <c>null</c> if <paramref name="ppm"/> is outside <c>(0, 1_000_000)</c>.
    /// </summary>
    public double? PpmToSigma(double ppm) => NanToNull(Native.insight_ppm_to_sigma(ppm));

    private static CapabilityIndices BuildCapabilityIndices(NativeStructs.CCapabilityIndices native)
    {
        return new CapabilityIndices
        {
            Cp = NanToNull(native.Cp),
            Cpk = NanToNull(native.Cpk),
            Cpu = NanToNull(native.Cpu),
            Cpl = NanToNull(native.Cpl),
            Pp = NanToNull(native.Pp),
            Ppk = NanToNull(native.Ppk),
            Ppu = NanToNull(native.Ppu),
            Ppl = NanToNull(native.Ppl),
            Cpm = NanToNull(native.Cpm),
            Mean = native.Mean,
            StdDevWithin = NanToNull(native.StdDevWithin),
            StdDevOverall = native.StdDevOverall,
        };
    }

    private static double? NanToNull(double value) => double.IsNaN(value) ? null : value;

    #endregion

    #region Weibull Reliability

    /// <summary>
    /// Fits Weibull distribution parameters via Maximum Likelihood Estimation.
    /// </summary>
    /// <param name="failureTimes">Positive failure times (needs at least 2 values).</param>
    public WeibullMleResult WeibullMle(double[] failureTimes)
    {
        var native = new NativeStructs.CWeibullMleResult();
        unsafe
        {
            fixed (double* ptr = failureTimes)
            {
                Native.ThrowIfFailed(
                    Native.insight_weibull_mle(ptr, (uint)failureTimes.Length, ref native));
            }
        }
        return new WeibullMleResult
        {
            Shape = native.Shape,
            Scale = native.Scale,
            LogLikelihood = native.LogLikelihood,
            Iterations = native.Iterations,
        };
    }

    /// <summary>
    /// Fits Weibull distribution parameters via Median Rank Regression.
    /// </summary>
    /// <param name="failureTimes">Positive failure times (needs at least 2 values).</param>
    public WeibullMrrResult WeibullMrr(double[] failureTimes)
    {
        var native = new NativeStructs.CWeibullMrrResult();
        unsafe
        {
            fixed (double* ptr = failureTimes)
            {
                Native.ThrowIfFailed(
                    Native.insight_weibull_mrr(ptr, (uint)failureTimes.Length, ref native));
            }
        }
        return new WeibullMrrResult
        {
            Shape = native.Shape,
            Scale = native.Scale,
            RSquared = native.RSquared,
        };
    }

    /// <summary>Weibull reliability (survival) function R(t) = exp(-(t/eta)^beta). For t &lt;= 0, returns 1.0.</summary>
    public double WeibullReliability(double shape, double scale, double t) =>
        Native.insight_weibull_reliability(shape, scale, t);

    /// <summary>Weibull hazard (instantaneous failure) rate at time t. For t &lt;= 0, returns 0.0.</summary>
    public double WeibullHazardRate(double shape, double scale, double t) =>
        Native.insight_weibull_hazard_rate(shape, scale, t);

    /// <summary>Mean Time Between Failures (MTBF) = eta * Gamma(1 + 1/beta).</summary>
    public double WeibullMtbf(double shape, double scale) =>
        Native.insight_weibull_mtbf(shape, scale);

    /// <summary>
    /// Time at which reliability drops to level <paramref name="p"/> (solves R(t) = p for t).
    /// Returns <c>null</c> if <paramref name="p"/> is outside <c>(0, 1)</c>.
    /// </summary>
    public double? WeibullTimeToReliability(double shape, double scale, double p) =>
        NanToNull(Native.insight_weibull_time_to_reliability(shape, scale, p));

    /// <summary>
    /// B-life: time at which <paramref name="fractionFailed"/> of the population has failed
    /// (e.g. 0.10 gives the B10 life). Returns <c>null</c> if out of range <c>(0, 1)</c>.
    /// </summary>
    public double? WeibullBLife(double shape, double scale, double fractionFailed) =>
        NanToNull(Native.insight_weibull_b_life(shape, scale, fractionFailed));

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

    private static double[,] Reshape2D(double[] flat, int rows, int cols)
    {
        var result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = flat[i * cols + j];
        return result;
    }

    private static double[][] ReshapeJagged(double[] flat, int rows, int cols)
    {
        var result = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            var row = new double[cols];
            Array.Copy(flat, i * cols, row, 0, cols);
            result[i] = row;
        }
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

/// <summary>Silhouette analysis result.</summary>
public class SilhouetteResult
{
    /// <summary>
    /// Mean silhouette across samples that had a defined silhouette
    /// (singletons and points with no other cluster present are excluded).
    /// Ranges from -1 (wrong cluster) to +1 (well-separated).
    /// </summary>
    public double Avg { get; init; }

    /// <summary>
    /// Per-sample silhouette scores (length = sample count).
    /// Singleton-cluster points and points with no other cluster present report 0.0.
    /// </summary>
    public double[] PerSample { get; init; } = [];
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
    /// <summary>Number of components retained.</summary>
    public uint NComponents { get; init; }

    /// <summary>Number of original features (columns of the input matrix).</summary>
    public uint NFeatures { get; init; }

    /// <summary>Number of input samples (rows of the input matrix).</summary>
    public uint NSamples { get; init; }

    /// <summary>Explained variance ratios per component (length = NComponents).</summary>
    public double[] ExplainedVariance { get; init; } = [];

    /// <summary>Cumulative explained variance ratios per component (length = NComponents).</summary>
    public double[] CumulativeVariance { get; init; } = [];

    /// <summary>
    /// Component loadings, shape [NComponents, NFeatures].
    /// Row k contains the unit-norm loading weights of PC{k+1} on the original features —
    /// i.e. <c>Loadings[k, :]</c> is the eigenvector for component k.
    /// </summary>
    public double[,] Loadings { get; init; } = new double[0, 0];

    /// <summary>
    /// Projected scores in PC space, shape [NSamples][NComponents].
    /// Row i contains the PC-space coordinates of input sample i.
    /// </summary>
    public double[][] Scores { get; init; } = [];
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

/// <summary>
/// Correlation method kinds. Numeric values match the native
/// <c>INSIGHT_CORR_*</c> constants (0 = Pearson, 1 = Spearman, 2 = Kendall).
/// </summary>
public enum CorrelationMethodKind : uint
{
    /// <summary>Pearson product-moment correlation (assumes linear, normal-ish data).</summary>
    Pearson = 0,
    /// <summary>Spearman rank correlation (robust to monotonic non-linearity).</summary>
    Spearman = 1,
    /// <summary>Kendall tau-b rank correlation (robust to ties).</summary>
    Kendall = 2,
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

/// <summary>PELT changepoint detection result.</summary>
public class PeltResult
{
    /// <summary>Detected changepoint indices (0-based).</summary>
    public uint[] Changepoints { get; init; } = [];
    /// <summary>Number of segments (changepoints + 1).</summary>
    public uint NSegments { get; init; }
}

/// <summary>One validated period candidate of <see cref="InsightClient.EstimatePeriod"/>.</summary>
public class PeriodCandidate
{
    /// <summary>Integer period, in observations.</summary>
    public uint Period { get; init; }
    /// <summary>Autocorrelation at that lag — the strength of the periodicity.</summary>
    public double Acf { get; init; }
    /// <summary>Periodogram bin (1-based, of the padded transform) that produced it.</summary>
    public uint Bin { get; init; }
    /// <summary>Periodogram power of that bin.</summary>
    public double Power { get; init; }
    /// <summary>That bin's share of the total periodogram power.</summary>
    public double PowerShare { get; init; }
}

/// <summary>Result of <see cref="InsightClient.EstimatePeriod"/>.</summary>
public class PeriodEstimate
{
    /// <summary>The dominant period, or <c>null</c> when no periodicity passed both stages.</summary>
    public uint? Period { get; init; }
    /// <summary>Every validated candidate, strongest first.</summary>
    public PeriodCandidate[] Candidates { get; init; } = [];
    /// <summary>Number of observations.</summary>
    public uint N { get; init; }
    /// <summary>The 95% white-noise bound on the ACF, 1.96 / sqrt(n).</summary>
    public double AcfThreshold { get; init; }
    /// <summary>Periodogram power a bin had to exceed to become a candidate.</summary>
    public double PowerThreshold { get; init; }
}

/// <summary>Options of <see cref="InsightClient.SpectralResidual"/>; the defaults are Ren et al. (2019).</summary>
public class SpectralResidualOptions
{
    /// <summary>Moving-average width on the log amplitude spectrum (q, >= 1).</summary>
    public uint AveragingWindow { get; init; } = 3;
    /// <summary>Preceding saliencies a point is scored against (z, >= 1).</summary>
    public uint JudgementWindow { get; init; } = 40;
    /// <summary>Score above which a point is an anomaly (τ, > 0).</summary>
    public double Threshold { get; init; } = 3.0;
    /// <summary>Minimum z-score of the point against the window before it (>= 0; 0 disables).</summary>
    public double MinZscore { get; init; } = 1.5;
    /// <summary>Coverage in percent of the band around the expected value (0 &lt; s &lt; 100).</summary>
    public double Sensitivity { get; init; } = 70.0;
    /// <summary>Score in consecutive batches of this size (>= 12), or <c>null</c> for one batch.</summary>
    public uint? BatchSize { get; init; }
}

/// <summary>One scored point of <see cref="InsightClient.SpectralResidual"/>.</summary>
public class SrPoint
{
    /// <summary>Position in the input series.</summary>
    public uint Index { get; init; }
    /// <summary>The observed value.</summary>
    public double Value { get; init; }
    /// <summary>Spectral residual saliency (>= 0).</summary>
    public double Saliency { get; init; }
    /// <summary>Saliency relative to the preceding judgement window (>= 0).</summary>
    public double Score { get; init; }
    /// <summary>Low-frequency reconstruction of the series with anomalies removed.</summary>
    public double Expected { get; init; }
    /// <summary>Expected minus the band margin.</summary>
    public double Lower { get; init; }
    /// <summary>Expected plus the band margin.</summary>
    public double Upper { get; init; }
    /// <summary>Whether the point is an anomaly.</summary>
    public bool IsAnomaly { get; init; }
}

/// <summary>Result of <see cref="InsightClient.SpectralResidual"/>.</summary>
public class SpectralResidualResult
{
    /// <summary>One point per observation, in order.</summary>
    public SrPoint[] Points { get; init; } = [];
    /// <summary>Indices of the points flagged as anomalies.</summary>
    public uint[] Anomalies { get; init; } = [];
}

/// <summary>Mann-Kendall trend test result.</summary>
public class MannKendallResult
{
    /// <summary>Mann-Kendall S statistic: sum of sign(x_j - x_i) for all i &lt; j.</summary>
    public long SStatistic { get; init; }
    /// <summary>Variance of S (with tie correction).</summary>
    public double Variance { get; init; }
    /// <summary>Z statistic (with continuity correction).</summary>
    public double ZStatistic { get; init; }
    /// <summary>Two-tailed p-value.</summary>
    public double PValue { get; init; }
    /// <summary>Kendall's tau: S / [n(n-1)/2]. Range [-1, 1].</summary>
    public double KendallTau { get; init; }
    /// <summary>Sen's slope estimator (median of pairwise slopes).</summary>
    public double SenSlope { get; init; }
}

/// <summary>
/// Bandwidth selection method for kernel density estimation. Numeric values
/// match the native <c>INSIGHT_KDE_*</c> constants.
/// </summary>
public enum KdeBandwidthMethod : uint
{
    /// <summary>Silverman's rule of thumb (robust to outliers and multimodal distributions).</summary>
    Silverman = 0,
    /// <summary>Scott's rule (slightly smoother, assumes approximately normal data).</summary>
    Scott = 1,
    /// <summary>Manually specified bandwidth — see the <c>bandwidth</c> parameter of <see cref="InsightClient.Kde"/>.</summary>
    Manual = 2,
}

/// <summary>Kernel density estimation result.</summary>
public class KdeResult
{
    /// <summary>Evaluation points (x-axis).</summary>
    public double[] X { get; init; } = [];
    /// <summary>Density estimates at each evaluation point (y-axis).</summary>
    public double[] Density { get; init; } = [];
    /// <summary>Bandwidth actually used.</summary>
    public double Bandwidth { get; init; }
}

/// <summary>Control chart limits (upper control limit, center line, lower control limit).</summary>
public class ControlLimits
{
    /// <summary>Upper control limit.</summary>
    public double Ucl { get; init; }
    /// <summary>Center line (process mean or target).</summary>
    public double Cl { get; init; }
    /// <summary>Lower control limit.</summary>
    public double Lcl { get; init; }
}

/// <summary>
/// Nelson's eight control chart rules for detecting special-cause variation.
/// Values are bit flags — a point can violate more than one rule at once.
/// </summary>
/// <remarks>
/// Nelson, L.S. (1984). "The Shewhart Control Chart — Tests for Special
/// Causes", Journal of Quality Technology 16(4), pp. 237-239.
/// </remarks>
[Flags]
public enum SpcViolation : uint
{
    /// <summary>No violations detected.</summary>
    None = 0,
    /// <summary>Point beyond control limits (Rule 1).</summary>
    BeyondLimits = 1u << 0,
    /// <summary>9 points in a row on the same side of the center line (Rule 2).</summary>
    NineOneSide = 1u << 1,
    /// <summary>6 points in a row steadily increasing or decreasing (Rule 3).</summary>
    SixTrend = 1u << 2,
    /// <summary>14 points in a row alternating up and down (Rule 4).</summary>
    FourteenAlternating = 1u << 3,
    /// <summary>2 out of 3 points beyond 2-sigma on the same side (Rule 5).</summary>
    TwoOfThreeBeyond2Sigma = 1u << 4,
    /// <summary>4 out of 5 points beyond 1-sigma on the same side (Rule 6).</summary>
    FourOfFiveBeyond1Sigma = 1u << 5,
    /// <summary>15 points in a row within 1-sigma of the center line (Rule 7).</summary>
    FifteenWithin1Sigma = 1u << 6,
    /// <summary>8 points in a row beyond 1-sigma on either side (Rule 8).</summary>
    EightBeyond1Sigma = 1u << 7,
}

/// <summary>A single point on a control chart.</summary>
public class SpcChartPoint
{
    /// <summary>The computed statistic value (subgroup mean, range, standard deviation, individual observation, or moving range).</summary>
    public double Value { get; init; }
    /// <summary>Nelson-rule violations detected at this point (bit flags, <see cref="SpcViolation.None"/> if none).</summary>
    public SpcViolation ViolationMask { get; init; }
}

/// <summary>
/// Result of a two-series variables control chart (X-bar-R, X-bar-S, or
/// Individual-MR). The primary series is the mean/individual chart; the
/// secondary series is the variation chart (R, S, or MR).
/// </summary>
public class VariablesChartResult
{
    /// <summary>Primary (X-bar or Individual) chart control limits.</summary>
    public ControlLimits Primary { get; init; } = new();
    /// <summary>Primary chart points.</summary>
    public SpcChartPoint[] PrimaryPoints { get; init; } = [];
    /// <summary>Secondary (R, S, or MR) chart control limits.</summary>
    public ControlLimits Secondary { get; init; } = new();
    /// <summary>
    /// Secondary chart points. May contain one fewer point than <see cref="PrimaryPoints"/>
    /// for Individual-MR charts (the first moving range is undefined).
    /// </summary>
    public SpcChartPoint[] SecondaryPoints { get; init; } = [];
    /// <summary>
    /// Within-subgroup (short-term) sigma estimated from the variation chart
    /// (R-bar/d2, S-bar/c4 or MR-bar/d2). Pass it to
    /// <see cref="InsightClient.ProcessCapability"/> as <c>sigmaWithin</c> for the
    /// short-term indices. <c>null</c> when the chart could not estimate it.
    /// </summary>
    public double? SigmaHat { get; init; }
    /// <summary>True if no Nelson-rule violations were detected on either series.</summary>
    public bool InControl { get; init; }
}

/// <summary>A single point on an attributes control chart (P, NP, C, or U).</summary>
public class AttributeChartPoint
{
    /// <summary>The computed statistic (proportion, count, or rate).</summary>
    public double Value { get; init; }
    /// <summary>Upper control limit at this point (may vary per point for P and U charts).</summary>
    public double Ucl { get; init; }
    /// <summary>Center line at this point.</summary>
    public double Cl { get; init; }
    /// <summary>Lower control limit at this point.</summary>
    public double Lcl { get; init; }
    /// <summary>True if this point is beyond its control limits.</summary>
    public bool OutOfControl { get; init; }
}

/// <summary>Result of a single-series attributes control chart (P, NP, C, or U).</summary>
public class AttributeChartResult
{
    /// <summary>Chart points.</summary>
    public AttributeChartPoint[] Points { get; init; } = [];
}

/// <summary>Result of a Laney P' or U' chart (overdispersion-adjusted attributes chart).</summary>
public class LaneyChartResult
{
    /// <summary>Overall proportion defective (P') or defect rate (U').</summary>
    public double Bar { get; init; }
    /// <summary>
    /// Overdispersion/underdispersion correction factor. 1.0 means no
    /// correction was needed (equivalent to an ordinary P or U chart).
    /// </summary>
    public double Phi { get; init; }
    /// <summary>Per-subgroup chart points.</summary>
    public AttributeChartPoint[] Points { get; init; } = [];
}

/// <summary>Result of a G or T chart (rare-event monitoring).</summary>
public class RareEventChartResult
{
    /// <summary>Mean inter-event conforming count (G chart) or inter-event time (T chart).</summary>
    public double Bar { get; init; }
    /// <summary>Per-observation chart points.</summary>
    public AttributeChartPoint[] Points { get; init; } = [];
}

/// <summary>
/// Standard process capability indices. Fields are <c>null</c> when the
/// corresponding index could not be computed (e.g. Cp requires both USL and LSL).
/// </summary>
public class CapabilityIndices
{
    /// <summary>Cp = (USL - LSL) / (6 * sigma_within). Requires both limits.</summary>
    public double? Cp { get; init; }
    /// <summary>Cpk = min(Cpu, Cpl). Requires at least one limit.</summary>
    public double? Cpk { get; init; }
    /// <summary>Cpu = (USL - mean) / (3 * sigma_within). Requires USL.</summary>
    public double? Cpu { get; init; }
    /// <summary>Cpl = (mean - LSL) / (3 * sigma_within). Requires LSL.</summary>
    public double? Cpl { get; init; }
    /// <summary>Pp = (USL - LSL) / (6 * sigma_overall). Requires both limits.</summary>
    public double? Pp { get; init; }
    /// <summary>Ppk = min(Ppu, Ppl). Requires at least one limit.</summary>
    public double? Ppk { get; init; }
    /// <summary>Ppu = (USL - mean) / (3 * sigma_overall). Requires USL.</summary>
    public double? Ppu { get; init; }
    /// <summary>Ppl = (mean - LSL) / (3 * sigma_overall). Requires LSL.</summary>
    public double? Ppl { get; init; }
    /// <summary>Cpm (Taguchi index). Requires both limits and a target.</summary>
    public double? Cpm { get; init; }
    /// <summary>Sample mean of the data.</summary>
    public double Mean { get; init; }
    /// <summary>Short-term (within-group) standard deviation — the <c>sigmaWithin</c> supplied; <c>null</c> when none was.</summary>
    public double? StdDevWithin { get; init; }
    /// <summary>Long-term (overall) standard deviation.</summary>
    public double StdDevOverall { get; init; }
}

/// <summary>Result of Box-Cox-based non-normal process capability analysis.</summary>
public class BoxCoxCapabilityResult
{
    /// <summary>Estimated optimal Box-Cox transformation parameter lambda.</summary>
    public double Lambda { get; init; }
    /// <summary>Capability indices computed on the Box-Cox-transformed scale.</summary>
    public CapabilityIndices Indices { get; init; } = new();
}

/// <summary>Result of percentile-based (ISO 22514-2) process capability analysis.</summary>
public class PercentileCapabilityResult
{
    /// <summary>Cp* = (USL - LSL) / (X_99.865 - X_0.135). Requires both limits.</summary>
    public double? CpStar { get; init; }
    /// <summary>Cpk* = min(Cpu*, Cpl*). Requires at least one limit.</summary>
    public double? CpkStar { get; init; }
    /// <summary>Cpu* = (USL - median) / (X_99.865 - median). Requires USL.</summary>
    public double? CpuStar { get; init; }
    /// <summary>Cpl* = (median - LSL) / (median - X_0.135). Requires LSL.</summary>
    public double? CplStar { get; init; }
    /// <summary>Sample median.</summary>
    public double Median { get; init; }
    /// <summary>0.135th percentile value (lower natural process limit).</summary>
    public double PercentileLower { get; init; }
    /// <summary>99.865th percentile value (upper natural process limit).</summary>
    public double PercentileUpper { get; init; }
}

/// <summary>Result of Weibull Maximum Likelihood Estimation.</summary>
public class WeibullMleResult
{
    /// <summary>Shape parameter (beta).</summary>
    public double Shape { get; init; }
    /// <summary>Scale parameter (eta).</summary>
    public double Scale { get; init; }
    /// <summary>Log-likelihood at the fitted parameters.</summary>
    public double LogLikelihood { get; init; }
    /// <summary>Number of Newton-Raphson iterations used.</summary>
    public uint Iterations { get; init; }
}

/// <summary>Result of Weibull Median Rank Regression fitting.</summary>
public class WeibullMrrResult
{
    /// <summary>Shape parameter (beta).</summary>
    public double Shape { get; init; }
    /// <summary>Scale parameter (eta).</summary>
    public double Scale { get; init; }
    /// <summary>Coefficient of determination (R-squared) measuring goodness of fit.</summary>
    public double RSquared { get; init; }
}

#endregion
