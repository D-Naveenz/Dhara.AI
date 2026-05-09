using Dhara.AI.Inference;

namespace Dhara.AI.LocalEmbeddings.Test;

public sealed class EmbeddingMathTests
{
    [Fact]
    public void CosineSimilarity_ReturnsZeroForMismatchedDimensions()
    {
        var similarity = EmbeddingMath.CosineSimilarity([1, 2], [1]);

        Assert.Equal(0, similarity);
    }

    [Fact]
    public void NormalizeInPlace_ProducesUnitVector()
    {
        float[] vector = [3, 4];

        EmbeddingMath.NormalizeInPlace(vector);

        Assert.Equal(0.6f, vector[0], precision: 5);
        Assert.Equal(0.8f, vector[1], precision: 5);
    }

    [Fact]
    public void MeanPool_AveragesOnlyAttentionTokens()
    {
        float[] tokenEmbeddings =
        [
            1, 1,
            3, 5,
            100, 100
        ];

        float[] destination = [0, 0];

        EmbeddingMath.MeanPool(
            tokenEmbeddings,
            attentionMask: [1, 1, 0],
            tokenCount: 3,
            dimensions: 2,
            destination);

        Assert.Equal([2, 3], destination);
    }
}
