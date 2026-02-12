using System.Runtime.InteropServices;

namespace UInsight.Interop;

/// <summary>
/// C-compatible struct definitions matching Rust FFI types.
/// </summary>
internal static class NativeStructs
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct CColumnSummary
    {
        public uint Index;
        public ulong ValidCount;
        public ulong NullCount;
        public uint DataType;
        public double Mean;
        public double StdDev;
        public double Min;
        public double Max;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CKMeansResult
    {
        public uint K;
        public double Wcss;
        public uint Iterations;
        public IntPtr Labels;
        public uint NLabels;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CDbscanResult
    {
        public uint NClusters;
        public uint NoiseCount;
        public IntPtr Labels;
        public uint NLabels;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CHierarchicalResult
    {
        public uint NClusters;
        public IntPtr Labels;
        public uint NLabels;
        public uint NMerges;
        public IntPtr MergeDistances;
        public IntPtr MergeSizes;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CHdbscanResult
    {
        public uint NClusters;
        public uint NoiseCount;
        public IntPtr Labels;
        public IntPtr Probabilities;
        public uint NLabels;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CGapStatResult
    {
        public uint BestK;
        public uint NValues;
        public IntPtr GapValues;
        public IntPtr StdErrors;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CPcaResult
    {
        public uint NComponents;
        public IntPtr ExplainedVariance;
        public uint NVariance;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CAnomalyResult
    {
        public IntPtr Scores;
        public IntPtr Anomalies;
        public uint N;
        public uint AnomalyCount;
        public double Threshold;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CMahalanobisResult
    {
        public IntPtr Distances;
        public IntPtr Anomalies;
        public uint N;
        public double Threshold;
        public uint OutlierCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CCorrelationResult
    {
        public uint NVars;
        public IntPtr Matrix;
        public uint NHighPairs;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CRegressionResult
    {
        public double Intercept;
        public double Slope;
        public double RSquared;
        public double AdjRSquared;
        public double FPValue;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CCramersVResult
    {
        public double V;
        public double ChiSquared;
        public double PValue;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CDistributionResult
    {
        public uint N;
        public double KsStatistic;
        public double KsPValue;
        public double JbStatistic;
        public double JbPValue;
        public double SwStatistic;
        public double SwPValue;
        public double AdStatistic;
        public double AdPValue;
        public int IsNormal;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CFeatureImportanceResult
    {
        public IntPtr Scores;
        public uint NScores;
        public double ConditionNumber;
        public uint NLowVariance;
        public uint NHighCorrPairs;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CAnovaFeature
    {
        public uint Index;
        public double FStatistic;
        public double PValue;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CAnovaSelectionResult
    {
        public IntPtr Features;
        public uint NFeatures;
        public uint NSelected;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CMutualInfoFeature
    {
        public uint Index;
        public double Mi;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CMutualInfoResult
    {
        public IntPtr Features;
        public uint NFeatures;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CPermImportanceFeature
    {
        public uint Index;
        public double Importance;
        public double StdDev;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CPermImportanceResult
    {
        public double BaselineScore;
        public IntPtr Features;
        public uint NFeatures;
    }
}
