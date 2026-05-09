using Dhara.AI.LocalEmbeddings;
using EmbeddingF32Value = Dhara.AI.LocalEmbeddings.EmbeddingF32;

namespace Dhara.AI.LocalEmbeddings.Test;

public sealed class EmbeddingSearchTests
{
    [Fact]
    public void FindClosestWithScore_OrdersBySimilarity()
    {
        var target = Create(1, 0);
        var candidates = new[]
        {
            ("north", Create(0, 1)),
            ("east", Create(1, 0)),
            ("near-east", Create(0.9f, 0.1f))
        };

        var result = EmbeddingSearch.FindClosestWithScore(target, candidates, maxResults: 2);

        Assert.Equal(["east", "near-east"], result.Select(static item => item.Item).ToArray());
    }

    [Fact]
    public void FindClosestWithScore_AppliesThreshold()
    {
        var target = Create(1, 0);
        var candidates = new[]
        {
            ("match", Create(1, 0)),
            ("miss", Create(0, 1))
        };

        var result = EmbeddingSearch.FindClosestWithScore(target, candidates, maxResults: 10, minSimilarity: 0.5f);

        Assert.Single(result);
        Assert.Equal("match", result[0].Item);
    }

    [Fact]
    public void FindClosestWithScore_KeepsDuplicateScoresDistinct()
    {
        var target = Create(1, 0);
        var candidates = new[]
        {
            ("first", Create(1, 0)),
            ("second", Create(1, 0))
        };

        var result = EmbeddingSearch.FindClosestWithScore(target, candidates, maxResults: 10);

        Assert.Equal(["first", "second"], result.Select(static item => item.Item).ToArray());
    }

    [Fact]
    public void FindClosestWithScore_RejectsZeroMaxResults()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            EmbeddingSearch.FindClosestWithScore(Create(1, 0), Array.Empty<(string, EmbeddingF32Value)>(), maxResults: 0));
    }

    private static EmbeddingF32Value Create(params float[] values)
        => EmbeddingF32Value.FromModelOutput(values, new byte[EmbeddingF32Value.GetBufferByteLength(values.Length)]);
}
