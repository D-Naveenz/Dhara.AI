namespace Dhara.AI.LocalEmbeddings;

/// <summary>
/// Provides ranking helpers for buffered local embeddings.
/// </summary>
public static class EmbeddingSearch
{
    /// <summary>
    /// Computes the similarity between two embeddings.
    /// </summary>
    /// <typeparam name="TEmbedding">The embedding representation.</typeparam>
    /// <param name="left">The first embedding.</param>
    /// <param name="right">The second embedding.</param>
    /// <returns>A similarity score where larger values mean more similar.</returns>
    public static float Similarity<TEmbedding>(TEmbedding left, TEmbedding right)
        where TEmbedding : IBufferedEmbedding<TEmbedding>
        => left.Similarity(right);

    /// <summary>
    /// Finds the closest items to a target embedding.
    /// </summary>
    /// <typeparam name="TItem">The item type.</typeparam>
    /// <typeparam name="TEmbedding">The embedding representation.</typeparam>
    /// <param name="target">The target embedding.</param>
    /// <param name="candidates">Candidate items paired with embeddings.</param>
    /// <param name="maxResults">The maximum number of results to return.</param>
    /// <param name="minSimilarity">The minimum accepted similarity score.</param>
    /// <returns>The closest items, ordered from most similar to least similar.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="maxResults"/> is less than one.</exception>
    public static TItem[] FindClosest<TItem, TEmbedding>(
        TEmbedding target,
        IEnumerable<(TItem Item, TEmbedding Embedding)> candidates,
        int maxResults,
        float? minSimilarity = null)
        where TEmbedding : IBufferedEmbedding<TEmbedding>
        => FindClosestWithScore(target, candidates, maxResults, minSimilarity)
            .Select(static score => score.Item)
            .ToArray();

    /// <summary>
    /// Finds the closest items to a target embedding and includes similarity scores.
    /// </summary>
    /// <typeparam name="TItem">The item type.</typeparam>
    /// <typeparam name="TEmbedding">The embedding representation.</typeparam>
    /// <param name="target">The target embedding.</param>
    /// <param name="candidates">Candidate items paired with embeddings.</param>
    /// <param name="maxResults">The maximum number of results to return.</param>
    /// <param name="minSimilarity">The minimum accepted similarity score.</param>
    /// <returns>The closest items and scores, ordered from most similar to least similar.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="maxResults"/> is less than one.</exception>
    public static SimilarityScore<TItem>[] FindClosestWithScore<TItem, TEmbedding>(
        TEmbedding target,
        IEnumerable<(TItem Item, TEmbedding Embedding)> candidates,
        int maxResults,
        float? minSimilarity = null)
        where TEmbedding : IBufferedEmbedding<TEmbedding>
    {
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxResults);

        var threshold = minSimilarity ?? float.MinValue;
        var sortedTopK = new SortedSet<SimilarityScore<TItem>>(SimilarityScore<TItem>.Comparer);
        var index = 0L;

        foreach (var candidate in candidates)
        {
            var similarity = target.Similarity(candidate.Embedding);
            if (similarity < threshold)
            {
                index++;
                continue;
            }

            var score = new SimilarityScore<TItem>(similarity, candidate.Item, index++);
            if (sortedTopK.Count < maxResults)
            {
                sortedTopK.Add(score);
                continue;
            }

            var worst = sortedTopK.Max;
            if (worst.Similarity < similarity)
            {
                sortedTopK.Remove(worst);
                sortedTopK.Add(score);
            }
        }

        return sortedTopK.ToArray();
    }
}
