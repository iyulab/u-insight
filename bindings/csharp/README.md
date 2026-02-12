# UInsight

.NET bindings for the u-insight statistical analysis engine.

## Features

- **CSV Profiling**: Column-level statistics, missing analysis, type inference
- **Clustering**: K-Means++, Mini-Batch K-Means, DBSCAN, Hierarchical (4 linkages), HDBSCAN
- **PCA**: Principal Component Analysis with auto-scaling
- **Anomaly Detection**: Isolation Forest, Local Outlier Factor (LOF), Mahalanobis distance
- **Statistical Analysis**: Pearson correlation, simple linear regression, Cramer's V
- **Distribution**: Normality testing (KS, Jarque-Bera, Shapiro-Wilk, Anderson-Darling)
- **Feature Importance**: Composite scores, ANOVA F-test, Mutual Information, Permutation Importance
- **Cross-Platform**: Windows, Linux, and macOS support

## Installation

```bash
dotnet add package UInsight
```

## Quick Start

### Profiling

```csharp
using UInsight;

using var client = new InsightClient();
Console.WriteLine($"Version: {client.GetVersion()}");

var profile = client.ProfileCsv("name,value\nAlice,1.5\nBob,2.3\n");
Console.WriteLine($"Rows: {profile.RowCount}, Columns: {profile.ColumnCount}");
```

### Clustering

```csharp
using UInsight;

using var client = new InsightClient();

var data = new double[,] { {0,0}, {1,1}, {10,10}, {11,11} };

// K-Means
var km = client.KMeans(data, k: 2);
Console.WriteLine($"K={km.K}, WCSS={km.Wcss:F2}");

// DBSCAN
var db = client.Dbscan(data, epsilon: 2.0, minSamples: 2);
Console.WriteLine($"Clusters: {db.NClusters}, Noise: {db.NoiseCount}");
```

### Anomaly Detection

```csharp
using UInsight;

using var client = new InsightClient();

var data = new double[,] {
    {1,1}, {2,2}, {1.5,1.5}, {2.5,2.5},
    {100,100}  // outlier
};

var result = client.IsolationForest(data, nEstimators: 100, contamination: 0.2);
Console.WriteLine($"Anomalies: {result.AnomalyCount}");
```

### Distribution Analysis

```csharp
using UInsight;

using var client = new InsightClient();

var data = Enumerable.Range(0, 100).Select(i => (i - 50.0) * 0.2).ToArray();
var dist = client.Distribution(data);
Console.WriteLine($"Normal: {dist.IsNormal}, SW p={dist.SwPValue:F4}");
```

## API Reference

### InsightClient

```csharp
public sealed class InsightClient : IDisposable
{
    // Version
    string GetVersion();

    // Profiling
    ProfileResult ProfileCsv(string csvData);

    // Clustering
    KMeansResult KMeans(double[,] data, uint k);
    KMeansResult MiniBatchKMeans(double[,] data, uint k, uint batchSize = 100, uint maxIter = 100, ulong seed = 42);
    DbscanResult Dbscan(double[,] data, double epsilon, uint minSamples);
    HierarchicalResult Hierarchical(double[,] data, uint linkage, uint nClusters);
    HdbscanResult Hdbscan(double[,] data, uint minClusterSize, uint minSamples);
    GapStatResult GapStatistic(double[,] data, uint kMin, uint kMax, uint nRefs = 10, ulong seed = 42);

    // PCA
    PcaResult Pca(double[,] data, uint nComponents, bool autoScale = true);

    // Anomaly Detection
    AnomalyResult IsolationForest(double[,] data, uint nEstimators = 100, double contamination = 0.1, ulong seed = 42);
    AnomalyResult Lof(double[,] data, uint k = 20, double threshold = 1.5);
    MahalanobisResult Mahalanobis(double[,] data, double chi2Quantile = 0.975);

    // Statistical Analysis
    CorrelationResult Correlation(double[,] data);
    RegressionResult Regression(double[] x, double[] y);
    CramersVResult CramersV(double[,] table);

    // Distribution
    DistributionResult Distribution(double[] data, double significanceLevel = 0.05);

    // Feature Importance
    FeatureImportanceResult FeatureImportance(double[,] data);
    AnovaSelectionResult AnovaSelect(double[,] data, uint[] target, double significanceLevel = 0.05);
    MutualInfoResult MutualInfo(double[,] data, uint[] target, uint nBins = 10);
    PermImportanceResult PermutationImportance(double[,] data, double[] target, uint nRepeats = 5, ulong seed = 42);
}
```

## Native Library

The native library (`u_insight.dll` / `libu_insight.so` / `libu_insight.dylib`) must be available in your application's runtime directory or system PATH.

### Building Native Library

```bash
cd <u-insight-repo>
cargo build --release
```

The built library will be in `target/release/`.

## License

MIT License - see [LICENSE](../../LICENSE) for details.
