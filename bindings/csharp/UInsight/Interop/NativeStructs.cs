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
        public uint NFeatures;
        public uint NSamples;
        public IntPtr ExplainedVariance;
        public IntPtr CumulativeVariance;
        public IntPtr Loadings;
        public IntPtr Scores;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CSilhouetteResult
    {
        public double Avg;
        public IntPtr PerSample;
        public uint NSamples;
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

    [StructLayout(LayoutKind.Sequential)]
    internal struct CPeltResult
    {
        public IntPtr Changepoints;
        public uint NChangepoints;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CPeriodCandidate
    {
        public uint Period;
        public double Acf;
        public uint Bin;
        public double Power;
        public double PowerShare;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CPeriodEstimate
    {
        public uint Period;
        public uint N;
        public double AcfThreshold;
        public double PowerThreshold;
        public IntPtr Candidates;
        public uint NCandidates;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CSpectralResidualOptions
    {
        public uint AveragingWindow;
        public uint JudgementWindow;
        public double Threshold;
        public double MinZscore;
        public double Sensitivity;
        public uint BatchSize;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CSrPoint
    {
        public uint Index;
        public double Value;
        public double Saliency;
        public double Score;
        public double Expected;
        public double Lower;
        public double Upper;
        public byte IsAnomaly;
        public byte NearEdge;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CSpectralResidualResult
    {
        public IntPtr Points;
        public uint NPoints;
        public uint NAnomalies;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CMannKendallResult
    {
        public long SStatistic;
        public double Variance;
        public double ZStatistic;
        public double PValue;
        public double KendallTau;
        public double SenSlope;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CKdeResult
    {
        public IntPtr X;
        public IntPtr Density;
        public uint NPoints;
        public double Bandwidth;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CSpcChartPoint
    {
        public double Value;
        public uint ViolationMask;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CVariablesChartResult
    {
        public double PrimaryUcl;
        public double PrimaryCl;
        public double PrimaryLcl;
        public IntPtr PrimaryPoints;
        public uint NPrimaryPoints;
        public double SecondaryUcl;
        public double SecondaryCl;
        public double SecondaryLcl;
        public IntPtr SecondaryPoints;
        public uint NSecondaryPoints;
        public double SigmaHat;
        public byte InControl;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CAttributeChartPoint
    {
        public double Value;
        public double Ucl;
        public double Cl;
        public double Lcl;
        public byte OutOfControl;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CAttributeChartResult
    {
        public IntPtr Points;
        public uint NPoints;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CLaneyChartResult
    {
        public double Bar;
        public double Phi;
        public IntPtr Points;
        public uint NPoints;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CRareEventChartResult
    {
        public double Bar;
        public IntPtr Points;
        public uint NPoints;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CCapabilityIndices
    {
        public double Cp;
        public double Cpk;
        public double Cpu;
        public double Cpl;
        public double Pp;
        public double Ppk;
        public double Ppu;
        public double Ppl;
        public double Cpm;
        public double Mean;
        public double StdDevWithin;
        public double StdDevOverall;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CBoxCoxCapabilityResult
    {
        public double Lambda;
        public CCapabilityIndices Indices;
        public byte LambdaAtBound;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CPercentileCapabilityResult
    {
        public double CpStar;
        public double CpkStar;
        public double CpuStar;
        public double CplStar;
        public double Median;
        public double PercentileLower;
        public double PercentileUpper;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CWeibullMleResult
    {
        public double Shape;
        public double Scale;
        public double LogLikelihood;
        public uint Iterations;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct CWeibullMrrResult
    {
        public double Shape;
        public double Scale;
        public double RSquared;
    }
}
