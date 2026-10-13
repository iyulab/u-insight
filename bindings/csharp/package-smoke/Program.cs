// Run by scripts/nuget_package_smoke.sh against the freshly packed package:
// the native library loads, answers, and refuses with the documented surface.
using UInsight;

using var insight = new InsightClient();
Console.WriteLine($"native version: {insight.GetVersion()}");

var data = new double[,] { { 0, 0 }, { 1, 1 }, { 10, 10 }, { 11, 11 } };
var clusters = insight.KMeans(data, k: 2);
if (clusters.K != 2)
    throw new Exception($"k-means returned k = {clusters.K}");

try
{
    insight.KMeans(data, k: 0);
    throw new Exception("k = 0 was accepted");
}
catch (InsightException e) when (e.ErrorCode == "parameter_out_of_range" && e.Parameter == "k")
{
}

Console.WriteLine("nuget package smoke test passed");
