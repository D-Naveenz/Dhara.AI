using System.Text.Json;
using Dhara.AI.LocalEmbeddings;

namespace Dhara.AI.LocalEmbeddings.Test;

public sealed class EmbeddingI8Tests
{
    [Fact]
    public void FromModelOutput_WritesMagnitudeAndOneBytePerDimension()
    {
        var embedding = EmbeddingI8.FromModelOutput([1, 0, -1], new byte[EmbeddingI8.GetBufferByteLength(3)]);

        Assert.Equal(7, embedding.Buffer.Length);
        Assert.Equal([127, 0, -127], embedding.Values.ToArray());
        Assert.Equal(MathF.Sqrt((127 * 127) + (127 * 127)), embedding.Magnitude, precision: 4);
    }

    [Fact]
    public void FromModelOutput_RejectsWrongBufferLength()
    {
        Assert.Throws<ArgumentException>(() =>
            EmbeddingI8.FromModelOutput([1, 2, 3], new byte[2]));
    }

    [Fact]
    public void Constructor_RejectsBufferWithoutValues()
    {
        Assert.Throws<ArgumentException>(() => new EmbeddingI8(new byte[4]));
    }

    [Fact]
    public void Similarity_ReturnsOneForSameDirection()
    {
        var first = Create(1, 0, -1);
        var second = Create(1, 0, -1);

        Assert.Equal(1, MathF.Round(first.Similarity(second), 4));
    }

    [Fact]
    public void Similarity_RejectsMismatchedDimensions()
    {
        var first = Create(1, 0);
        var second = Create(1, 0, -1);

        Assert.Throws<InvalidOperationException>(() => first.Similarity(second));
    }

    [Fact]
    public void Json_RoundTripsBase64Buffer()
    {
        var embedding = Create(1, 0, -1);

        var json = JsonSerializer.Serialize(embedding);
        var roundTripped = JsonSerializer.Deserialize<EmbeddingI8>(json);

        Assert.Equal(embedding.Buffer.ToArray(), roundTripped.Buffer.ToArray());
        Assert.Equal(embedding.Values.ToArray(), roundTripped.Values.ToArray());
        Assert.Equal(embedding.Magnitude, roundTripped.Magnitude);
    }

    private static EmbeddingI8 Create(params float[] values)
        => EmbeddingI8.FromModelOutput(values, new byte[EmbeddingI8.GetBufferByteLength(values.Length)]);
}
