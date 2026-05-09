namespace Dhara.AI.LocalEmbeddings;

/// <summary>
/// Represents an item paired with a similarity score.
/// </summary>
/// <typeparam name="TItem">The ranked item type.</typeparam>
/// <param name="similarity">The similarity score.</param>
/// <param name="item">The ranked item.</param>
public readonly struct SimilarityScore<TItem>(float similarity, TItem item)
{
    private readonly long _uniqueIndex;

    internal SimilarityScore(float similarity, TItem item, long uniqueIndex)
        : this(similarity, item)
    {
        _uniqueIndex = uniqueIndex;
    }

    /// <summary>
    /// Gets the similarity score.
    /// </summary>
    public float Similarity => similarity;

    /// <summary>
    /// Gets the ranked item.
    /// </summary>
    public TItem Item => item;

    internal static IComparer<SimilarityScore<TItem>> Comparer { get; } =
        Comparer<SimilarityScore<TItem>>.Create(static (left, right) =>
        {
            var comparison = right.Similarity.CompareTo(left.Similarity);
            return comparison == 0
                ? left._uniqueIndex.CompareTo(right._uniqueIndex)
                : comparison;
        });
}
