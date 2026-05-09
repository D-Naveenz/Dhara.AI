namespace Dhara.AI.LocalEmbeddings;

/// <summary>
/// Defines a compact embedding representation that can be created from raw model output.
/// </summary>
/// <typeparam name="TEmbedding">The concrete embedding representation.</typeparam>
public interface IBufferedEmbedding<TEmbedding>
    where TEmbedding : IBufferedEmbedding<TEmbedding>
{
    /// <summary>
    /// Gets the number of bytes required to store an embedding with the specified dimension count.
    /// </summary>
    /// <param name="dimensions">The number of vector dimensions.</param>
    /// <returns>The number of bytes required by <typeparamref name="TEmbedding"/>.</returns>
    static abstract int GetBufferByteLength(int dimensions);

    /// <summary>
    /// Creates an embedding representation from raw model output.
    /// </summary>
    /// <param name="input">The raw floating-point model output.</param>
    /// <param name="buffer">The destination buffer used to persist the embedding.</param>
    /// <returns>The concrete embedding value.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="buffer"/> has the wrong length.</exception>
    static abstract TEmbedding FromModelOutput(ReadOnlySpan<float> input, Memory<byte> buffer);

    /// <summary>
    /// Computes similarity to another embedding of the same representation.
    /// </summary>
    /// <param name="other">The other embedding.</param>
    /// <returns>A similarity score where larger values mean more similar.</returns>
    float Similarity(TEmbedding other);
}
