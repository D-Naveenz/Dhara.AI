using System.Buffers.Binary;
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Dhara.AI.LocalEmbeddings;

/// <summary>
/// Represents an embedding value using one bit for each vector dimension.
///
/// This is a binary representation of the model output. Each dimension stores only
/// whether the original value was positive or negative. The embedding also stores
/// a 4-byte dimension count, so a 384-dimensional embedding uses 52 bytes and a
/// 768-dimensional embedding uses 100 bytes.
/// </summary>
/// <remarks>
/// <see cref="EmbeddingI1"/> is the most compact representation in this package.
/// It is useful for very large local indexes, coarse filtering, or approximate search
/// where memory usage matters more than exact ranking quality. For higher-quality
/// ranking, use <see cref="EmbeddingI8"/> or <see cref="EmbeddingF32"/>, or use
/// <see cref="EmbeddingI1"/> to shortlist candidates and rescore the top results with
/// a richer representation.
/// </remarks>
[JsonConverter(typeof(EmbeddingI1JsonConverter))]
public readonly struct EmbeddingI1 : IBufferedEmbedding<EmbeddingI1>
{
    private const int DimensionByteLength = sizeof(int);
    private readonly ReadOnlyMemory<byte> _buffer;
    private readonly int _dimensions;

    /// <summary>
    /// Initializes a new instance of the <see cref="EmbeddingI1"/> struct from persisted bytes.
    /// </summary>
    /// <param name="buffer">
    /// The byte buffer previously produced by <see cref="Buffer"/> or another compatible
    /// persistence layer. The first 4 bytes store the dimension count. The remaining bytes
    /// store one sign bit for each vector dimension, packed from the highest bit to the
    /// lowest bit in each byte.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="buffer"/> is too short, contains an invalid dimension
    /// count, or does not have the exact byte length required for the stored dimensions.
    /// </exception>
    public EmbeddingI1(ReadOnlyMemory<byte> buffer)
    {
        if (buffer.Length <= DimensionByteLength)
        {
            throw new ArgumentException("An EmbeddingI1 buffer must contain a 4-byte dimension count plus at least one packed byte.", nameof(buffer));
        }

        var dimensions = BinaryPrimitives.ReadInt32LittleEndian(buffer.Span[..DimensionByteLength]);
        if (dimensions <= 0)
        {
            throw new ArgumentException("An EmbeddingI1 buffer must contain a positive dimension count.", nameof(buffer));
        }

        var expectedLength = GetBufferByteLength(dimensions);
        if (buffer.Length != expectedLength)
        {
            throw new ArgumentException($"The buffer length must be {expectedLength} bytes for {dimensions} dimensions.", nameof(buffer));
        }

        _buffer = buffer;
        _dimensions = dimensions;
    }

    /// <summary>
    /// Gets the byte representation of the embedding for storage or JSON serialization.
    ///
    /// The first 4 bytes store the dimension count. Every byte after that stores up to
    /// 8 dimensions as sign bits.
    /// </summary>
    public ReadOnlyMemory<byte> Buffer => _buffer;

    /// <summary>
    /// Gets the number of vector dimensions represented by this embedding.
    /// </summary>
    public int Dimensions => _dimensions;

    /// <summary>
    /// Gets the packed sign-bit payload, excluding the 4-byte dimension header.
    /// </summary>
    public ReadOnlyMemory<byte> PackedValues => _buffer[DimensionByteLength..];

    /// <summary>
    /// Gets the byte length required to store an <see cref="EmbeddingI1"/> with the supplied dimension count.
    /// </summary>
    /// <param name="dimensions">The number of vector dimensions produced by the embedding model.</param>
    /// <returns>
    /// The number of bytes required for the embedding. This is 4 bytes for the dimension
    /// count plus one packed byte for each group of up to 8 dimensions.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="dimensions"/> is zero or negative.</exception>
    /// <exception cref="OverflowException">Thrown when the byte length is larger than <see cref="int.MaxValue"/>.</exception>
    public static int GetBufferByteLength(int dimensions)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimensions);
        return checked(DimensionByteLength + ((dimensions + 7) / 8));
    }

    /// <summary>
    /// Creates an <see cref="EmbeddingI1"/> from raw model output.
    /// </summary>
    /// <param name="input">
    /// The vector returned by the model. Each dimension is converted into one bit:
    /// 1 for values greater than or equal to 0, and 0 for negative values.
    /// </param>
    /// <param name="buffer">
    /// The destination byte buffer. Its length must equal <see cref="GetBufferByteLength(int)"/>
    /// for <paramref name="input"/>'s dimension count.
    /// </param>
    /// <returns>An embedding that stores the model output as packed sign bits.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="buffer"/> has the wrong byte length.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="input"/> is empty.</exception>
    public static EmbeddingI1 FromModelOutput(ReadOnlySpan<float> input, Memory<byte> buffer)
    {
        var requiredLength = GetBufferByteLength(input.Length);
        if (buffer.Length != requiredLength)
        {
            throw new ArgumentException($"The buffer length must be {requiredLength} bytes for {input.Length} dimensions.", nameof(buffer));
        }

        BinaryPrimitives.WriteInt32LittleEndian(buffer.Span[..DimensionByteLength], input.Length);
        var packed = buffer.Span[DimensionByteLength..];
        packed.Clear();

        for (var index = 0; index < input.Length; index++)
        {
            if (input[index] >= 0)
            {
                packed[index / 8] |= (byte)(1 << (7 - (index % 8)));
            }
        }

        return new EmbeddingI1(buffer);
    }

    /// <summary>
    /// Computes Hamming similarity between this embedding and another <see cref="EmbeddingI1"/>.
    /// </summary>
    /// <param name="other">The embedding to compare with this embedding.</param>
    /// <returns>
    /// A score from 0 to 1. A value of 1 means every sign bit matches. A value of 0
    /// means every represented dimension has the opposite sign.
    /// </returns>
    /// <exception cref="InvalidOperationException">Thrown when the embeddings have different dimension counts.</exception>
    public float Similarity(EmbeddingI1 other)
    {
        if (Dimensions != other.Dimensions)
        {
            throw new InvalidOperationException($"Cannot compare {Dimensions} dimensions with {other.Dimensions} dimensions.");
        }

        var differences = CountDifferentBits(PackedValues.Span, other.PackedValues.Span, Dimensions);
        return 1f - (differences / (float)Dimensions);
    }

    private static int CountDifferentBits(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right, int dimensions)
    {
        var fullBytes = dimensions / 8;
        var remainingBits = dimensions % 8;
        var differences = 0;

        for (var index = 0; index < fullBytes; index++)
        {
            differences += BitOperations.PopCount((uint)(left[index] ^ right[index]));
        }

        if (remainingBits > 0)
        {
            var mask = (byte)(0xFF << (8 - remainingBits));
            differences += BitOperations.PopCount((uint)((left[fullBytes] ^ right[fullBytes]) & mask));
        }

        return differences;
    }

    private sealed class EmbeddingI1JsonConverter : JsonConverter<EmbeddingI1>
    {
        public override EmbeddingI1 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
            => new(reader.GetBytesFromBase64());

        public override void Write(Utf8JsonWriter writer, EmbeddingI1 value, JsonSerializerOptions options)
            => writer.WriteBase64StringValue(value.Buffer.Span);
    }
}
