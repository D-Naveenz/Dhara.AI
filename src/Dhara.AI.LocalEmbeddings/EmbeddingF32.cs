using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Dhara.AI.Inference;

namespace Dhara.AI.LocalEmbeddings;

/// <summary>
/// Represents an embedding value using one <see cref="float"/> for each vector dimension.
///
/// This is the raw, unquantized output from the embedding model. Each <see cref="float"/>
/// uses 4 bytes, so a 384-dimensional embedding from the default model uses
/// 1,536 bytes and a 768-dimensional sentence-transformer embedding uses 3,072 bytes.
/// </summary>
/// <remarks>
/// <see cref="EmbeddingF32"/> is the highest-fidelity representation in this package.
/// It keeps the model output exactly as 32-bit floating-point values, which makes it the
/// best choice for learning, debugging, evaluating model quality, and final result
/// rescoring. It uses more storage than <see cref="EmbeddingI8"/> and <see cref="EmbeddingI1"/>,
/// but it avoids quantization loss.
/// </remarks>
[JsonConverter(typeof(EmbeddingF32JsonConverter))]
public readonly struct EmbeddingF32 : IBufferedEmbedding<EmbeddingF32>
{
    private readonly ReadOnlyMemory<byte> _buffer;
    private readonly ReadOnlyMemory<float> _values;

    /// <summary>
    /// Initializes a new instance of the <see cref="EmbeddingF32"/> struct from persisted bytes.
    /// </summary>
    /// <param name="buffer">
    /// The byte buffer previously produced by <see cref="Buffer"/> or another compatible
    /// persistence layer. The buffer is interpreted as consecutive 32-bit floating-point values.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="buffer"/> length is not divisible by 4, because a
    /// <see cref="float"/> cannot be reconstructed from a partial byte group.
    /// </exception>
    public EmbeddingF32(ReadOnlyMemory<byte> buffer)
    {
        if (buffer.Length % sizeof(float) != 0)
        {
            throw new ArgumentException("An EmbeddingF32 buffer length must be divisible by four.", nameof(buffer));
        }

        _buffer = buffer;
        _values = MemoryMarshal.Cast<byte, float>(buffer.Span).ToArray();
    }

    /// <summary>
    /// Gets the byte representation of the embedding for storage or JSON serialization.
    ///
    /// For this representation, the buffer is simply the vector's <see cref="float"/>
    /// values laid out as bytes. For example, 384 dimensions produce a 1,536-byte buffer,
    /// while 768 dimensions produce a 3,072-byte buffer.
    /// </summary>
    public ReadOnlyMemory<byte> Buffer => _buffer;

    /// <summary>
    /// Gets the embedding vector values as 32-bit floating-point numbers.
    ///
    /// Each number is one coordinate in the model's semantic vector space. Texts with
    /// similar meaning should produce vectors that point in a similar direction. Most
    /// retrieval systems compare these directions with cosine similarity.
    /// </summary>
    public ReadOnlyMemory<float> Values => _values;

    /// <summary>
    /// Gets the byte length required to store an <see cref="EmbeddingF32"/> with the supplied dimension count.
    /// </summary>
    /// <param name="dimensions">The number of vector dimensions produced by the embedding model.</param>
    /// <returns>
    /// The number of bytes required for the embedding. This is <paramref name="dimensions"/>
    /// multiplied by 4 because each <see cref="float"/> is 4 bytes. For example,
    /// 384 dimensions require 1,536 bytes, and 768 dimensions require 3,072 bytes.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="dimensions"/> is zero or negative.</exception>
    /// <exception cref="OverflowException">Thrown when the byte length is larger than <see cref="int.MaxValue"/>.</exception>
    public static int GetBufferByteLength(int dimensions)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimensions);
        return checked(dimensions * sizeof(float));
    }

    /// <summary>
    /// Creates an <see cref="EmbeddingF32"/> from raw model output.
    /// </summary>
    /// <param name="input">
    /// The vector returned by the model. Each value is copied directly into <paramref name="buffer"/>
    /// without quantization, rounding, clipping, or dimensionality reduction.
    /// </param>
    /// <param name="buffer">
    /// The destination byte buffer. Its length must equal <see cref="GetBufferByteLength(int)"/>
    /// for <paramref name="input"/>'s dimension count.
    /// </param>
    /// <returns>
    /// An embedding that stores the model output as 32-bit floating-point values. Use this
    /// representation when quality matters more than storage size.
    /// </returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="buffer"/> has the wrong byte length.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="input"/> is empty.</exception>
    public static EmbeddingF32 FromModelOutput(ReadOnlySpan<float> input, Memory<byte> buffer)
    {
        var requiredLength = GetBufferByteLength(input.Length);
        if (buffer.Length != requiredLength)
        {
            throw new ArgumentException($"The buffer length must be {requiredLength} bytes for {input.Length} dimensions.", nameof(buffer));
        }

        MemoryMarshal.AsBytes(input).CopyTo(buffer.Span);
        return new EmbeddingF32(buffer);
    }

    /// <summary>
    /// Computes cosine similarity between this embedding and another <see cref="EmbeddingF32"/>.
    /// </summary>
    /// <param name="other">The embedding to compare with this embedding.</param>
    /// <returns>
    /// A cosine similarity score. Values closer to 1 mean the vectors point in a similar
    /// direction, values near 0 mean they are mostly unrelated, and negative values mean
    /// they point in opposite directions.
    /// </returns>
    public float Similarity(EmbeddingF32 other)
        => EmbeddingMath.CosineSimilarity(Values.Span, other.Values.Span);

    private sealed class EmbeddingF32JsonConverter : JsonConverter<EmbeddingF32>
    {
        public override EmbeddingF32 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
            => new(reader.GetBytesFromBase64());

        public override void Write(Utf8JsonWriter writer, EmbeddingF32 value, JsonSerializerOptions options)
            => writer.WriteBase64StringValue(value.Buffer.Span);
    }
}
