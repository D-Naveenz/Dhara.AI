using Dhara.AI.LocalEmbeddings;
using Microsoft.Extensions.AI;

namespace Dhara.AI.LocalEmbeddings.Test;

public sealed class LocalEmbeddingGeneratorIntegrationTests
{
    [Fact]
    public async Task GenerateAsync_ProducesDefaultModelEmbeddings()
    {
        using var generator = LocalEmbeddingGenerator.Create();

        var embeddings = await generator.GenerateAsync(["cat", "database"], cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(2, embeddings.Count);
        Assert.True(embeddings[0].Vector.Length > 0);
        Assert.Equal(embeddings[0].Vector.Length, embeddings[1].Vector.Length);
    }

    [Fact]
    public async Task EmbedRangeAsync_RanksMindVaultLikeSnippets()
    {
        using var generator = LocalEmbeddingGenerator.Create();
        var query = await generator.EmbedAsync<EmbeddingF32>("semantic search for Obsidian notes", cancellationToken: TestContext.Current.CancellationToken);
        var candidates = await generator.EmbedRangeAsync<EmbeddingF32>(
        [
            "Smart Connections stores note and block embeddings for Obsidian vault search.",
            "Native AOT compiles a .NET console app into a platform-specific executable.",
            "A grocery list contains rice, tea, and milk."
        ], TestContext.Current.CancellationToken);

        var result = EmbeddingSearch.FindClosest(query, candidates, maxResults: 1);

        Assert.Equal("Smart Connections stores note and block embeddings for Obsidian vault search.", result[0]);
    }

    [Fact]
    public async Task EmbedRangeAsync_CanRankWithInt8Embeddings()
    {
        using var generator = LocalEmbeddingGenerator.Create();
        var query = await generator.EmbedAsync<EmbeddingI8>("semantic search for Obsidian notes", cancellationToken: TestContext.Current.CancellationToken);
        var candidates = await generator.EmbedRangeAsync<EmbeddingI8>(
        [
            "Smart Connections stores note and block embeddings for Obsidian vault search.",
            "Native AOT compiles a .NET console app into a platform-specific executable.",
            "A grocery list contains rice, tea, and milk."
        ], TestContext.Current.CancellationToken);

        var result = EmbeddingSearch.FindClosest(query, candidates, maxResults: 1);

        Assert.Equal("Smart Connections stores note and block embeddings for Obsidian vault search.", result[0]);
    }

    [Fact]
    public async Task EmbedRangeAsync_CanRankExactMatchesWithBinaryEmbeddings()
    {
        using var generator = LocalEmbeddingGenerator.Create();
        const string exact = "Smart Connections stores note and block embeddings for Obsidian vault search.";
        var query = await generator.EmbedAsync<EmbeddingI1>(exact, cancellationToken: TestContext.Current.CancellationToken);
        var candidates = await generator.EmbedRangeAsync<EmbeddingI1>(
        [
            "Native AOT compiles a .NET console app into a platform-specific executable.",
            exact,
            "A grocery list contains rice, tea, and milk."
        ], TestContext.Current.CancellationToken);

        var result = EmbeddingSearch.FindClosest(query, candidates, maxResults: 1);

        Assert.Equal(exact, result[0]);
    }

    [Fact]
    public async Task GenerateAsync_IsReusableAcrossConcurrentCalls()
    {
        using var generator = LocalEmbeddingGenerator.Create();
        var tasks = Enumerable.Range(0, 8)
            .Select(index => generator.GenerateVectorAsync($"mindvault query {index}", TestContext.Current.CancellationToken))
            .ToArray();

        var vectors = await Task.WhenAll(tasks);

        Assert.All(vectors, vector => Assert.True(vector.Length > 0));
        Assert.All(vectors, vector => Assert.Equal(vectors[0].Length, vector.Length));
    }

    [Fact]
    public void GetService_ReturnsMetadata()
    {
        using var generator = LocalEmbeddingGenerator.Create();

        var metadata = generator.GetService<EmbeddingGeneratorMetadata>();

        Assert.NotNull(metadata);
        Assert.Equal("Dhara.AI.LocalEmbeddings", metadata.ProviderName);
    }
}
