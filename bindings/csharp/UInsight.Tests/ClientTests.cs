using Xunit;

namespace UInsight.Tests;

/// <summary>
/// Each client method against the real native library: the call is accepted, runs, and its
/// result reads back with the shape the engine produced. (Moved from the umbrella's
/// binding-tests/, which no build ran.)
/// </summary>
public class ClientTests : IDisposable
{
    private readonly InsightClient _client = new();

    public void Dispose() => _client.Dispose();

    [Fact]
    public void GetVersion_ReturnsNonEmpty()
    {
        var version = _client.GetVersion();
        Assert.False(string.IsNullOrEmpty(version));
    }

    [Fact]
    public void UInsightInfo_Version_ReturnsNonEmpty()
    {
        var version = UInsightInfo.Version;
        Assert.False(string.IsNullOrEmpty(version));
    }

    // ── Profiling ──

    [Fact]
    public void ProfileCsv_BasicCsv()
    {
        var csv = "name,value,active\nAlice,1.5,true\nBob,2.3,false\nCharlie,3.1,true\n";
        var result = _client.ProfileCsv(csv);

        Assert.Equal(3, result.RowCount);
        Assert.Equal(3, result.ColumnCount);
        Assert.Equal(3, result.Columns.Length);
        Assert.Equal(0u, result.Columns[0].NullCount);
    }

    // ── Clustering ──

    [Fact]
    public void KMeans_TwoClusters()
    {
        var data = new double[,]
        {
            { 0, 0 }, { 0.5, 0.5 }, { 1, 0 },
            { 10, 10 }, { 10.5, 10.5 }, { 11, 10 }
        };

        var result = _client.KMeans(data, k: 2);

        Assert.Equal(2u, result.K);
        Assert.Equal(6, result.Labels.Length);
        Assert.True(result.Wcss > 0);
        Assert.True(result.Iterations > 0);
        // First 3 and last 3 should be in different clusters
        Assert.Equal(result.Labels[0], result.Labels[1]);
        Assert.Equal(result.Labels[3], result.Labels[4]);
        Assert.NotEqual(result.Labels[0], result.Labels[3]);
    }

    [Fact]
    public void MiniBatchKMeans_TwoClusters()
    {
        var data = new double[,]
        {
            { 0, 0 }, { 0.5, 0.5 }, { 1, 0 },
            { 10, 10 }, { 10.5, 10.5 }, { 11, 10 }
        };

        var result = _client.MiniBatchKMeans(data, k: 2, batchSize: 4);

        Assert.Equal(2u, result.K);
        Assert.Equal(6, result.Labels.Length);
    }

    [Fact]
    public void Dbscan_TwoClusters()
    {
        var data = new double[,]
        {
            { 0, 0 }, { 0.5, 0.5 }, { 1, 0 },
            { 10, 10 }, { 10.5, 10.5 }, { 11, 10 }
        };

        var result = _client.Dbscan(data, epsilon: 2.0, minSamples: 2);

        Assert.Equal(2u, result.NClusters);
        Assert.Equal(0u, result.NoiseCount);
        Assert.Equal(6, result.Labels.Length);
    }

    [Fact]
    public void Hierarchical_Ward()
    {
        var data = new double[,]
        {
            { 0, 0 }, { 1, 1 },
            { 10, 10 }, { 11, 11 }
        };

        var result = _client.Hierarchical(data, Linkage.Ward, nClusters: 2);

        Assert.Equal(2u, result.NClusters);
        Assert.Equal(4, result.Labels.Length);
    }

    [Fact]
    public void Hdbscan_BasicTest()
    {
        var data = new double[,]
        {
            { 0, 0 }, { 0.1, 0.1 }, { 0.2, 0 }, { 0, 0.2 }, { 0.1, 0 },
            { 10, 10 }, { 10.1, 10.1 }, { 10.2, 10 }, { 10, 10.2 }, { 10.1, 10 }
        };

        var result = _client.Hdbscan(data, minClusterSize: 3, minSamples: 2);

        Assert.True(result.NClusters >= 1);
        Assert.Equal(10, result.Labels.Length);
        Assert.Equal(10, result.Probabilities.Length);
    }

    [Fact]
    public void GapStatistic_FindsOptimalK()
    {
        var data = new double[,]
        {
            { 0, 0 }, { 0.5, 0.5 }, { 1, 0 },
            { 10, 10 }, { 10.5, 10.5 }, { 11, 10 }
        };

        var result = _client.GapStatistic(data, kMin: 1, kMax: 4, nRefs: 5, seed: 42);

        Assert.True(result.BestK >= 1 && result.BestK <= 4);
        Assert.True(result.GapValues.Length > 0);
        Assert.Equal(result.GapValues.Length, result.StdErrors.Length);
    }

    // ── PCA ──

    [Fact]
    public void Pca_ExplainedVariance()
    {
        var data = new double[,]
        {
            { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 },
            { 2, 3, 4 }, { 5, 6, 7 }, { 8, 9, 10 }
        };

        var result = _client.Pca(data, nComponents: 2);

        Assert.Equal(2u, result.NComponents);
        Assert.Equal(2, result.ExplainedVariance.Length);
        // First component should explain most variance
        Assert.True(result.ExplainedVariance[0] > result.ExplainedVariance[1]);
    }

    // ── Anomaly Detection ──

    [Fact]
    public void IsolationForest_DetectsOutlier()
    {
        var data = new double[,]
        {
            { 1, 1 }, { 1.1, 1.1 }, { 0.9, 0.9 },
            { 1.2, 0.8 }, { 0.8, 1.2 },
            { 100, 100 } // outlier
        };

        var result = _client.IsolationForest(data, nEstimators: 100, contamination: 0.2, seed: 42);

        Assert.Equal(6, result.Scores.Length);
        Assert.Equal(6, result.Anomalies.Length);
        Assert.True(result.AnomalyCount >= 1);
    }

    [Fact]
    public void Lof_DetectsOutlier()
    {
        var data = new double[,]
        {
            { 1, 1 }, { 1.1, 1.1 }, { 0.9, 0.9 },
            { 1.2, 0.8 }, { 0.8, 1.2 },
            { 100, 100 } // outlier
        };

        var result = _client.Lof(data, k: 3, threshold: 1.5);

        Assert.Equal(6, result.Scores.Length);
        Assert.Equal(6, result.Anomalies.Length);
    }

    [Fact]
    public void Mahalanobis_DetectsOutlier()
    {
        // Need enough data points for covariance matrix
        var data = new double[,]
        {
            { 1, 1 }, { 1.1, 1.1 }, { 0.9, 0.9 },
            { 1.2, 0.8 }, { 0.8, 1.2 }, { 1.0, 1.05 },
            { 0.95, 0.95 }, { 1.15, 1.1 }, { 0.85, 1.0 },
            { 100, 100 } // outlier
        };

        var result = _client.Mahalanobis(data, chi2Quantile: 0.975);

        Assert.Equal(10, result.Distances.Length);
        Assert.Equal(10, result.Anomalies.Length);
        Assert.True(result.OutlierCount >= 1);
    }

    // ── Statistical Analysis ──

    [Fact]
    public void Correlation_Matrix()
    {
        var data = new double[,]
        {
            { 1, 2 }, { 2, 4 }, { 3, 6 },
            { 4, 8 }, { 5, 10 }
        };

        var result = _client.Correlation(data);

        Assert.Equal(2u, result.NVars);
        Assert.Equal(2, result.Matrix.GetLength(0));
        Assert.Equal(2, result.Matrix.GetLength(1));
        // Diagonal should be 1.0
        Assert.Equal(1.0, result.Matrix[0, 0], 5);
        Assert.Equal(1.0, result.Matrix[1, 1], 5);
        // Perfect positive correlation
        Assert.True(result.Matrix[0, 1] > 0.99);
    }

    [Fact]
    public void Regression_LinearFit()
    {
        var x = new double[] { 1, 2, 3, 4, 5 };
        var y = new double[] { 2, 4, 6, 8, 10 }; // y = 2x

        var result = _client.Regression(x, y);

        Assert.True(Math.Abs(result.Slope - 2.0) < 0.01);
        Assert.True(Math.Abs(result.Intercept) < 0.01);
        Assert.True(result.RSquared > 0.99);
    }

    [Fact]
    public void CramersV_ContingencyTable()
    {
        // 2x2 contingency table
        var table = new double[,]
        {
            { 10, 20 },
            { 20, 40 }
        };

        var result = _client.CramersV(table);

        Assert.True(result.V >= 0 && result.V <= 1);
        Assert.True(result.ChiSquared >= 0);
    }

    // ── Distribution ──

    [Fact]
    public void Distribution_NormalityTest()
    {
        // Approximately normal data
        var data = Enumerable.Range(0, 100)
            .Select(i => (i - 50.0) * 0.2)
            .ToArray();

        var result = _client.Distribution(data, significanceLevel: 0.05);

        Assert.Equal(100u, result.N);
        Assert.True(result.KsStatistic >= 0);
        Assert.True(result.KsPValue >= 0 && result.KsPValue <= 1);
        Assert.True(result.SwStatistic >= 0);
    }

    // ── Feature Importance ──

    [Fact]
    public void FeatureImportance_Scores()
    {
        var data = new double[,]
        {
            { 1, 100, 0.1 }, { 2, 200, 0.2 }, { 3, 300, 0.3 },
            { 4, 400, 0.4 }, { 5, 500, 0.5 }, { 6, 600, 0.6 },
            { 7, 700, 0.7 }, { 8, 800, 0.8 }, { 9, 900, 0.9 },
            { 10, 1000, 1.0 }
        };

        var result = _client.FeatureImportance(data);

        Assert.Equal(3, result.Scores.Length);
        Assert.True(result.ConditionNumber > 0);
    }

    [Fact]
    public void AnovaSelect_FeatureSelection()
    {
        var data = new double[,]
        {
            { 1.0, 10.0 }, { 1.1, 10.1 }, { 0.9, 9.9 },
            { 5.0, 10.0 }, { 5.1, 10.1 }, { 4.9, 9.9 },
            { 9.0, 10.0 }, { 9.1, 10.1 }, { 8.9, 9.9 }
        };
        var target = new uint[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 };

        var result = _client.AnovaSelect(data, target, significanceLevel: 0.05);

        Assert.True(result.Features.Length > 0);
    }

    [Fact]
    public void MutualInfo_FeatureRanking()
    {
        var data = new double[,]
        {
            { 1.0, 10.0 }, { 1.1, 10.1 }, { 0.9, 9.9 },
            { 5.0, 10.0 }, { 5.1, 10.1 }, { 4.9, 9.9 },
            { 9.0, 10.0 }, { 9.1, 10.1 }, { 8.9, 9.9 }
        };
        var target = new uint[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 };

        var result = _client.MutualInfo(data, target, nBins: 3);

        Assert.True(result.Features.Length > 0);
    }

    [Fact]
    public void PermutationImportance_Regression()
    {
        var data = new double[,]
        {
            { 1, 0.1 }, { 2, 0.2 }, { 3, 0.3 },
            { 4, 0.4 }, { 5, 0.5 }, { 6, 0.6 },
            { 7, 0.7 }, { 8, 0.8 }, { 9, 0.9 },
            { 10, 1.0 }
        };
        var target = new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 };

        var result = _client.PermutationImportance(data, target, nRepeats: 3, seed: 42);

        Assert.True(result.Features.Length > 0);
        Assert.True(result.BaselineScore != 0);
    }

    // ── Error Handling ──

    [Fact]
    public void InsightException_OnInvalidInput()
    {
        // KMeans with k=0 should fail
        var data = new double[,] { { 1, 2 }, { 3, 4 } };
        var ex = Assert.Throws<InsightException>(() =>
        {
            _client.KMeans(data, k: 0);
        });
        Assert.Equal(InsightErrorCategory.InvalidParameter, ex.Category);
        Assert.Equal("parameter_out_of_range", ex.ErrorCode);
        Assert.Equal("k", ex.Parameter);
        Assert.Null(ex.Index);
    }
}
