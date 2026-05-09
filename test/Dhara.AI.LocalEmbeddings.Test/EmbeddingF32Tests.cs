using System.Text.Json;
using Dhara.AI.LocalEmbeddings;

namespace Dhara.AI.LocalEmbeddings.Test;

public sealed class EmbeddingF32Tests
{
    [Fact]
    public void FromModelOutput_WritesExpectedBufferLength()
    {
        var embedding = EmbeddingF32.FromModelOutput([1, 2, 3], new byte[EmbeddingF32.GetBufferByteLength(3)]);

        Assert.Equal(12, embedding.Buffer.Length);
        Assert.Equal([1, 2, 3], embedding.Values.ToArray());
    }

    [Fact]
    public void FromModelOutput_RejectsWrongBufferLength()
    {
        Assert.Throws<ArgumentException>(() =>
            EmbeddingF32.FromModelOutput([1, 2, 3], new byte[2]));
    }

    [Fact]
    public void Json_RoundTripsBase64Buffer()
    {
        var embedding = EmbeddingF32.FromModelOutput([1, 2, 3], new byte[12]);

        var json = JsonSerializer.Serialize(embedding);
        var roundTripped = JsonSerializer.Deserialize<EmbeddingF32>(json);

        Assert.Equal(embedding.Buffer.ToArray(), roundTripped.Buffer.ToArray());
        Assert.Equal(embedding.Values.ToArray(), roundTripped.Values.ToArray());
    }
}
