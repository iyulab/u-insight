using System.Text.Json.Serialization;

namespace UInsight;

/// <summary>A refusal reason: the <c>code</c> an exception's <c>ErrorCode</c> carries and the
/// error body fields that can stand beside it.</summary>
/// <param name="Code">The stable reason, e.g. <c>insufficient_data</c>.</param>
/// <param name="Fields">The error body's fields beside it (<c>parameter</c>, <c>min</c>, …).</param>
public sealed record RefusalCode(string Code, IReadOnlyList<string> Fields);

/// <summary>A violation a checker reports: its <c>code</c>, the fields beside it, and the
/// engine operations that report it.</summary>
/// <param name="Code">The stable reason.</param>
/// <param name="Fields">The violation's fields beside <c>code</c>, in wire names.</param>
/// <param name="Checkers">The engine operations reporting it, by their engine names.</param>
public sealed record ViolationCode(string Code, IReadOnlyList<string> Fields, IReadOnlyList<string> Checkers);

/// <summary>Every code the library can answer with besides a result — the same catalog the
/// WebAssembly package's <c>codes()</c> returns.</summary>
/// <param name="Refusals">Each refusal reason.</param>
/// <param name="Violations">Each checker violation (none in this library).</param>
public sealed record Codes(IReadOnlyList<RefusalCode> Refusals, IReadOnlyList<ViolationCode> Violations);

/// <summary>Source-generated reading of the code catalog — no reflection.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    RespectNullableAnnotations = true,
    RespectRequiredConstructorParameters = true)]
[JsonSerializable(typeof(Codes))]
internal sealed partial class InsightJson : JsonSerializerContext;
