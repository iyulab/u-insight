using System.Text.Json;
using System.Text.Json.Nodes;
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
    public void The_code_catalog_carries_exactly_what_the_library_sent()
    {
        var raw = UInsight.Interop.NativeLibrary.GetCodesJson();
        var codes = _client.GetCodes();
        var written = JsonSerializer.SerializeToNode(codes, InsightJson.Default.Codes);
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(raw), written), $"library: {raw}\nrecord: {written}");
        Assert.Contains(codes.Refusals, r => r.Code == "insufficient_data" && r.Fields.Contains("min"));
        Assert.Empty(codes.Violations);
    }

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

    [Fact]
    public void ProfileCsvFile_ReadsTheFileAndCarriesTheEngineRefusal()
    {
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "name,value\nAlice,1.5\nBob,2.3\n");
            var result = _client.ProfileCsvFile(path);
            Assert.Equal(2, result.RowCount);
            Assert.Equal(2, result.ColumnCount);

            File.WriteAllText(path, "a,b\n1,2\n3\n");
            var e = Assert.Throws<InsightException>(() => _client.ProfileCsvFile(path));
            Assert.NotNull(e.ErrorCode);
            Assert.Contains("3", e.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ProfileCsv_UndefinedStatisticsAreNull()
    {
        // One row: the value column has a mean but no standard deviation;
        // the text column has neither.
        var result = _client.ProfileCsv("name,value\nAlice,1.5\n");

        var name = result.Columns[0];
        Assert.Null(name.Mean);
        Assert.Null(name.StdDev);
        var value = result.Columns[1];
        Assert.Equal(1.5, value.Mean);
        Assert.Equal(1.5, value.Min);
        Assert.Null(value.StdDev);
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

        var result = _client.Mahalanobis(data, quantile: 0.975);

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


    // ── Time series ──

    /// <summary>The series the engine checks its autocorrelations against (statsmodels 0.15).</summary>
    private static readonly double[] AcfSeries =
    [
        2.3, 3.1, 4.0, 3.6, 2.9, 2.2, 2.8, 3.9, 4.6, 4.1, 3.3, 2.5, 2.7, 3.8, 4.9, 4.4, 3.5, 2.6,
        3.0, 4.2, 5.1, 4.5, 3.4, 2.9,
    ];

    [Fact]
    public void Acf_matches_the_reference_values()
    {
        var r = _client.Acf(AcfSeries, maxLag: 8);
        Assert.Equal(9, r.Acf.Length);
        Assert.Equal(8, r.BartlettLimits.Length);
        Assert.Equal(1.0, r.Acf[0], 12);
        Assert.Equal(0.486403576681892, r.Acf[1], 12);
        Assert.Equal(-0.3171014941549895, r.Acf[8], 12);
        Assert.Equal(0.40007597302960923, r.BartlettLimits[0], 12);
        Assert.Equal(0.8200659919957791, r.BartlettLimits[7], 12);
        Assert.Equal(0.40007597302960923, r.ConfidenceThreshold, 12);

        // Default: min(floor(10 log10 24), 23) = 13.
        Assert.Equal(14, _client.Acf(AcfSeries).Acf.Length);
    }

    [Fact]
    public void Pacf_matches_the_reference_values()
    {
        var r = _client.Pacf(AcfSeries, maxLag: 8);
        Assert.Equal(8, r.Pacf.Length);
        Assert.Equal(0.486403576681892, r.Pacf[0], 12);
        Assert.Equal(-0.7839238375815395, r.Pacf[1], 12);
        Assert.Equal(0.03539504608148146, r.Pacf[7], 12);
        Assert.Equal(11, _client.Pacf(AcfSeries).Pacf.Length);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(24)]
    public void A_max_lag_outside_the_range_is_refused_with_its_bounds(int maxLag)
    {
        var ex = Assert.Throws<InsightException>(() => _client.Acf(AcfSeries, maxLag));
        Assert.Equal(InsightErrorCategory.InvalidParameter, ex.Category);
        Assert.Equal("parameter_out_of_range", ex.ErrorCode);
        Assert.Equal("max_lag", ex.Parameter);
        Assert.Equal(1.0, ex.Details!.Value.GetProperty("min").GetDouble());
        Assert.Equal(23.0, ex.Details!.Value.GetProperty("max").GetDouble());
        Assert.Equal(maxLag, ex.Details!.Value.GetProperty("got").GetDouble());
    }

    [Fact]
    public void A_constant_or_short_series_is_refused()
    {
        var flat = Enumerable.Repeat(3.0, 10).ToArray();
        var ex = Assert.Throws<InsightException>(() => _client.Pacf(flat));
        Assert.Equal("degenerate_data", ex.ErrorCode);
        Assert.Equal("data", ex.Details!.Value.GetProperty("parameter").GetString());
        ex = Assert.Throws<InsightException>(() => _client.Acf([1.0]));
        Assert.Equal("insufficient_data", ex.ErrorCode);
        ex = Assert.Throws<InsightException>(() => _client.Acf([1.0, double.NaN, 2.0]));
        Assert.Equal("value_not_finite", ex.ErrorCode);
        Assert.Equal(1, ex.Index);
    }

    private static double[] Ar1Series()
    {
        var x = new double[120];
        for (var t = 1; t < x.Length; t++)
        {
            var e = (t * 7919 % 101) / 50.0 - 1.0;
            x[t] = 0.7 * x[t - 1] + e;
        }
        return x;
    }

    [Fact]
    public void AdfTest_matches_the_reference_values()
    {
        // statsmodels 0.15 adfuller on the same series.
        var r = _client.AdfTest(Ar1Series(), maxLags: 8);
        Assert.Equal(-3.4360528948, r.Statistic, 8);
        Assert.Equal(5u, r.NLags);
        Assert.Equal(114u, r.NObs);
        Assert.Equal([0.01, 0.05, 0.10], r.Levels.Select(l => l.Level));
        Assert.All(r.Levels, l => Assert.Equal(r.Statistic <= l.CriticalValue, l.Rejected));

        var trend = _client.AdfTest(Ar1Series(), AdfModel.ConstantTrend, maxLags: 6);
        Assert.Equal(-3.5052470690, trend.Statistic, 8);
        var fixedLags = _client.AdfTest(Ar1Series(), lags: 3);
        Assert.Equal(-4.6529304844, fixedLags.Statistic, 8);
        Assert.Equal(3u, fixedLags.NLags);
        Assert.Equal(116u, fixedLags.NObs);
    }

    [Fact]
    public void AdfTest_refuses_both_lag_options_and_a_count_outside_the_room()
    {
        var x = Enumerable.Range(0, 20).Select(i => (double)(i * 37 % 11)).ToArray();
        var ex = Assert.Throws<InsightException>(() => _client.AdfTest(x, maxLags: 2, lags: 2));
        Assert.Equal("invalid_option", ex.ErrorCode);
        Assert.Equal("lags", ex.Parameter);
        ex = Assert.Throws<InsightException>(() => _client.AdfTest(x, maxLags: -1));
        Assert.Equal("parameter_out_of_range", ex.ErrorCode);
        Assert.Equal("max_lags", ex.Parameter);
        Assert.Equal(8.0, ex.Details!.Value.GetProperty("max").GetDouble());
        ex = Assert.Throws<InsightException>(() => _client.AdfTest(x, (AdfModel)7));
        Assert.Equal("unknown_option", ex.ErrorCode);
        Assert.Equal("model", ex.Parameter);
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
