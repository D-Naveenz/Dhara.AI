namespace Dhara.AI.Inference;

/// <summary>
/// Provides allocation-conscious vector math used by local embedding models.
/// </summary>
public static class EmbeddingMath
{
    /// <summary>
    /// Computes the cosine similarity between two vectors.
    /// </summary>
    /// <param name="left">The first vector.</param>
    /// <param name="right">The second vector.</param>
    /// <returns>The cosine similarity, or 0 when the vectors cannot be compared.</returns>
    public static float CosineSimilarity(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        if (left.Length == 0 || left.Length != right.Length)
        {
            return 0;
        }

        var dot = 0f;
        var leftNorm = 0f;
        var rightNorm = 0f;

        for (var index = 0; index < left.Length; index++)
        {
            var leftValue = left[index];
            var rightValue = right[index];
            dot += leftValue * rightValue;
            leftNorm += leftValue * leftValue;
            rightNorm += rightValue * rightValue;
        }

        var denominator = MathF.Sqrt(leftNorm) * MathF.Sqrt(rightNorm);
        return denominator == 0 ? 0 : dot / denominator;
    }

    /// <summary>
    /// Normalizes a vector in place using its L2 norm.
    /// </summary>
    /// <param name="vector">The vector to normalize.</param>
    public static void NormalizeInPlace(Span<float> vector)
    {
        var sum = 0f;
        foreach (var value in vector)
        {
            sum += value * value;
        }

        var magnitude = MathF.Sqrt(sum);
        if (magnitude == 0)
        {
            return;
        }

        for (var index = 0; index < vector.Length; index++)
        {
            vector[index] /= magnitude;
        }
    }

    /// <summary>
    /// Mean-pools a token-level embedding tensor into a single vector.
    /// </summary>
    /// <param name="tokenEmbeddings">The flattened token embeddings for one sequence.</param>
    /// <param name="attentionMask">The attention mask for the same sequence.</param>
    /// <param name="tokenCount">The number of tokens in the sequence.</param>
    /// <param name="dimensions">The embedding dimension count.</param>
    /// <param name="destination">The vector that receives the pooled embedding.</param>
    /// <exception cref="ArgumentException">Thrown when input spans are too small for the requested shape.</exception>
    public static void MeanPool(
        ReadOnlySpan<float> tokenEmbeddings,
        ReadOnlySpan<long> attentionMask,
        int tokenCount,
        int dimensions,
        Span<float> destination)
    {
        if (tokenCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenCount), tokenCount, "Token count must be positive.");
        }

        if (dimensions <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "Dimension count must be positive.");
        }

        if (tokenEmbeddings.Length < tokenCount * dimensions)
        {
            throw new ArgumentException("The token embedding span is shorter than the supplied tensor shape.", nameof(tokenEmbeddings));
        }

        if (attentionMask.Length < tokenCount)
        {
            throw new ArgumentException("The attention mask span is shorter than the supplied token count.", nameof(attentionMask));
        }

        if (destination.Length < dimensions)
        {
            throw new ArgumentException("The destination span is shorter than the supplied dimension count.", nameof(destination));
        }

        destination[..dimensions].Clear();
        var includedTokens = 0;

        for (var tokenIndex = 0; tokenIndex < tokenCount; tokenIndex++)
        {
            if (attentionMask[tokenIndex] == 0)
            {
                continue;
            }

            includedTokens++;
            var tokenOffset = tokenIndex * dimensions;
            for (var dimension = 0; dimension < dimensions; dimension++)
            {
                destination[dimension] += tokenEmbeddings[tokenOffset + dimension];
            }
        }

        if (includedTokens == 0)
        {
            return;
        }

        for (var dimension = 0; dimension < dimensions; dimension++)
        {
            destination[dimension] /= includedTokens;
        }
    }
}
