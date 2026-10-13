using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Xunit;

namespace UInsight.Tests;

/// <summary>
/// The engine's C ABI is structs, not JSON: a result is read by laying a C# struct over
/// memory the engine filled. A field the C# side declares in the wrong order, at the wrong
/// width, or not at all reads the neighbouring field's bytes -- silently, with no parse
/// error to catch it. The header cbindgen generates from the engine (kept current by CI)
/// is the layout; every struct declared here must match it field by field.
/// </summary>
public class LayoutContractTests
{
    private static readonly Type[] Declared =
        [.. typeof(InsightClient).Assembly.GetType("UInsight.Interop.NativeStructs", throwOnError: true)!
            .GetNestedTypes(BindingFlags.NonPublic | BindingFlags.Public)
            .Where(t => t.IsValueType)];

    private sealed record CField(string Name, int Size, bool Pointer);

    /// <summary>`typedef struct Name { fields } Name;` from the header, comments removed.</summary>
    private static readonly Lazy<Dictionary<string, List<CField>>> Header = new(() =>
    {
        var text = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "u_insight.h"));
        text = Regex.Replace(text, @"/\*.*?\*/", "", RegexOptions.Singleline);
        var structs = new Dictionary<string, List<CField>>();
        foreach (Match m in Regex.Matches(text, @"typedef struct (\w+) \{(.*?)\} \1;", RegexOptions.Singleline))
        {
            var fields = new List<CField>();
            foreach (var decl in m.Groups[2].Value.Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
            {
                var f = Regex.Match(decl, @"^(?:const\s+)?(?:struct\s+)?(\w+)\s*(\*?)\s*(\w+)$");
                Assert.True(f.Success, $"{m.Groups[1].Value}: cannot read the declaration '{decl}'");
                var pointer = f.Groups[2].Value == "*";
                fields.Add(new CField(f.Groups[3].Value, pointer ? IntPtr.Size : CSize(f.Groups[1].Value), pointer));
            }
            structs[m.Groups[1].Value] = fields;
        }
        return structs;
    });

    private static int CSize(string type) => type switch
    {
        "double" or "uint64_t" or "int64_t" or "size_t" or "uintptr_t" => 8,
        "float" or "uint32_t" or "int32_t" => 4,
        "uint16_t" or "int16_t" => 2,
        "bool" or "uint8_t" or "int8_t" => 1,
        _ => throw new InvalidOperationException($"no size known for the C type '{type}'"),
    };

    /// <summary>C's own rule: each field at the next multiple of its size, the whole a
    /// multiple of the largest (every field here is a scalar or a pointer).</summary>
    private static (int[] Offsets, int Size) CLayout(List<CField> fields)
    {
        var offsets = new int[fields.Count];
        int at = 0, align = 1;
        for (var i = 0; i < fields.Count; i++)
        {
            var size = fields[i].Size;
            at = (at + size - 1) / size * size;
            offsets[i] = at;
            at += size;
            align = Math.Max(align, size);
        }
        return (offsets, (at + align - 1) / align * align);
    }

    private static string Pascal(string snake) =>
        string.Concat(snake.Split('_').Select(w => w.Length == 0 ? w : char.ToUpperInvariant(w[0]) + w[1..]));

    public static TheoryData<string> Structs() => [.. Declared.Select(t => t.Name)];

    [Theory]
    [MemberData(nameof(Structs))]
    public void Each_struct_is_laid_out_as_the_header_declares_it(string name)
    {
        Assert.True(Header.Value.TryGetValue(name, out var cFields), $"{name} is not in the header");
        var type = Declared.Single(t => t.Name == name);
        var csFields = type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .OrderBy(f => Marshal.OffsetOf(type, f.Name).ToInt64())
            .ToArray();

        Assert.Equal(cFields.Select(f => Pascal(f.Name)), csFields.Select(f => f.Name));
        var (offsets, size) = CLayout(cFields);
        for (var i = 0; i < csFields.Length; i++)
        {
            Assert.True(offsets[i] == Marshal.OffsetOf(type, csFields[i].Name).ToInt32(),
                $"{name}.{csFields[i].Name}: C offset {offsets[i]}, C# offset {Marshal.OffsetOf(type, csFields[i].Name)}");
            Assert.True(!cFields[i].Pointer || csFields[i].FieldType == typeof(IntPtr),
                $"{name}.{csFields[i].Name} is a pointer in C and must be IntPtr");
        }
        Assert.Equal(size, Marshal.SizeOf(type));
    }

    [Fact]
    public void Every_struct_the_header_declares_is_declared_here()
    {
        Assert.Equal(Header.Value.Keys.Order(), Declared.Select(t => t.Name).Order());
    }
}
