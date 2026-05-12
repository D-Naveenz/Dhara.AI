using System.Text.Json;
using Dhara.AI.LocalEmbeddings;

namespace Dhara.AI.LocalEmbeddings.Test;

public sealed class EmbeddingI1Tests
{
    [Fact]
    public void FromModelOutput_WritesDimensionHeaderAndPackedBits()
    {
        var embedding = EmbeddingI1.FromModelOutput(
            [1, -1, 0, -0.5f, 2, -2, 3, -3, 4],
            new byte[EmbeddingI1.GetBufferByteLength(9)]);

        Assert.Equal(9, embedding.Dimensions);
        Assert.Equal(6, embedding.Buffer.Length);
        Assert.Equal([0b10101010, 0b10000000], embedding.PackedValues.ToArray());
    }

    [Fact]
    public void GetBufferByteLength_SupportsNonMultipleOfEightDimensions()
    {
        Assert.Equal(5, EmbeddingI1.GetBufferByteLength(1));
        Assert.Equal(5, EmbeddingI1.GetBufferByteLength(8));
        Assert.Equal(6, EmbeddingI1.GetBufferByteLength(9));
    }

    [Fact]
    public void FromModelOutput_RejectsWrongBufferLength()
    {
        Assert.Throws<ArgumentException>(() =>
            EmbeddingI1.FromModelOutput([1, -1, 1], new byte[4]));
    }

    [Fact]
    public void Constructor_RejectsMismatchedDimensionHeaderAndBufferLength()
    {
        var embedding = Create(1, -1, 1, -1, 1, -1, 1, -1, 1);
        var truncated = embedding.Buffer.ToArray()[..5];

        Assert.Throws<ArgumentException>(() => new EmbeddingI1(truncated));
    }

    [Fact]
    public void Similarity_ReturnsOneForSameSigns()
    {
        var first = Create(1, -1, 1, -1, 1);
        var second = Create(2, -2, 3, -3, 4);

        Assert.Equal(1, first.Similarity(second));
    }

    [Fact]
    public void Similarity_ReturnsFractionOfMatchingSigns()
    {
        var first = Create(1, -1, 1, -1);
        var second = Create(1, 1, -1, -1);

        Assert.Equal(0.5f, first.Similarity(second));
    }

    [Fact]
    public void Similarity_RejectsMismatchedDimensions()
    {
        var first = Create(1, -1);
        var second = Create(1, -1, 1);

        Assert.Throws<InvalidOperationException>(() => first.Similarity(second));
    }

    [Fact]
    public void Json_RoundTripsBase64Buffer()
    {
        var embedding = Create(1, -1, 0, -0.5f, 2, -2, 3, -3, 4);

        var json = JsonSerializer.Serialize(embedding);
        var roundTripped = JsonSerializer.Deserialize<EmbeddingI1>(json);

        Assert.Equal(embedding.Buffer.ToArray(), roundTripped.Buffer.ToArray());
        Assert.Equal(embedding.Dimensions, roundTripped.Dimensions);
        Assert.Equal(embedding.PackedValues.ToArray(), roundTripped.PackedValues.ToArray());
    }

    private static EmbeddingI1 Create(params float[] values)
        => EmbeddingI1.FromModelOutput(values, new byte[EmbeddingI1.GetBufferByteLength(values.Length)]);
}
