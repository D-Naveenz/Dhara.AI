using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Dhara.AI.LocalEmbeddings;

/// <summary>
/// Represents an embedding value using one signed 8-bit integer for each vector dimension.
///
/// This is a scalar-quantized representation of the model output. Each dimension is
/// rounded from a 32-bit <see cref="float"/> into an <see cref="sbyte"/> value from
/// -127 to 127. The embedding also stores 4 extra bytes for the quantized vector
/// magnitude, so a 384-dimensional embedding uses 388 bytes and a 768-dimensional
/// embedding uses 772 bytes.
/// </summary>
/// <remarks>
/// <see cref="EmbeddingI8"/> is a practical storage format for local retrieval systems:
/// it is roughly 4 times smaller than <see cref="EmbeddingF32"/> while preserving enough
/// information for useful similarity ranking. For the highest-quality final scoring,
/// keep <see cref="EmbeddingF32"/> around or rescore top candidates with full-precision
/// vectors.
/// </remarks>
[JsonConverter(typeof(EmbeddingI8JsonConverter))]
public readonly struct EmbeddingI8 : IBufferedEmbedding<EmbeddingI8>
{
    private const int MagnitudeByteLength = sizeof(float);
    private readonly ReadOnlyMemory<byte> _buffer;
    private readonly ReadOnlyMemory<sbyte> _values;
    private readonly float _magnitude;

    /// <summary>
    /// Initializes a new instance of the <see cref="EmbeddingI8"/> struct from persisted bytes.
    /// </summary>
    /// <param name="buffer">
    /// The byte buffer previously produced by <see cref="Buffer"/> or another compatible
    /// persistence layer. The first 4 bytes store the quantized vector magnitude. The
    /// remaining bytes store one signed 8-bit integer for each vector dimension.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="buffer"/> is too short to contain the magnitude and at
    /// least one quantized vector dimension.
    /// </exception>
    public EmbeddingI8(ReadOnlyMemory<byte> buffer)
    {
        if (buffer.Length <= MagnitudeByteLength)
        {
            throw new ArgumentException("An EmbeddingI8 buffer must contain a 4-byte magnitude plus at least one value.", nameof(buffer));
        }

        _buffer = buffer;
        _magnitude = BitConverter.ToSingle(buffer.Span[..MagnitudeByteLength]);
        _values = MemoryMarshal.Cast<byte, sbyte>(buffer.Span[MagnitudeByteLength..]).ToArray();
    }

    /// <summary>
    /// Gets the byte representation of the embedding for storage or JSON serialization.
    ///
    /// The first 4 bytes are the quantized vector magnitude. Every byte after that is one
    /// vector dimension represented as a signed 8-bit integer.
    /// </summary>
    public ReadOnlyMemory<byte> Buffer => _buffer;

    /// <summary>
    /// Gets the quantized vector values.
    ///
    /// Each value is an approximation of the original floating-point coordinate, scaled
    /// into the signed byte range from -127 to 127.
    /// </summary>
    public ReadOnlyMemory<sbyte> Values => _values;

    /// <summary>
    /// Gets the magnitude of the quantized vector.
    ///
    /// The magnitude lets similarity be computed directly from the compact integer values
    /// without reconstructing the original floating-point vector.
    /// </summary>
    public float Magnitude => _magnitude;

    /// <summary>
    /// Gets the byte length required to store an <see cref="EmbeddingI8"/> with the supplied dimension count.
    /// </summary>
    /// <param name="dimensions">The number of vector dimensions produced by the embedding model.</param>
    /// <returns>
    /// The number of bytes required for the embedding. This is <paramref name="dimensions"/>
    /// plus 4 bytes for the stored magnitude.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="dimensions"/> is zero or negative.</exception>
    /// <exception cref="OverflowException">Thrown when the byte length is larger than <see cref="int.MaxValue"/>.</exception>
    public static int GetBufferByteLength(int dimensions)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimensions);
        return checked(MagnitudeByteLength + dimensions);
    }

    /// <summary>
    /// Creates an <see cref="EmbeddingI8"/> from raw model output.
    /// </summary>
    /// <param name="input">
    /// The vector returned by the model. The largest absolute value in the vector is mapped
    /// to 127 or -127, and all other values are scaled proportionally.
    /// </param>
    /// <param name="buffer">
    /// The destination byte buffer. Its length must equal <see cref="GetBufferByteLength(int)"/>
    /// for <paramref name="input"/>'s dimension count.
    /// </param>
    /// <returns>An embedding that stores the model output as signed 8-bit values.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="buffer"/> has the wrong byte length.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="input"/> is empty.</exception>
    public static EmbeddingI8 FromModelOutput(ReadOnlySpan<float> input, Memory<byte> buffer)
    {
        var requiredLength = GetBufferByteLength(input.Length);
        if (buffer.Length != requiredLength)
        {
            throw new ArgumentException($"The buffer length must be {requiredLength} bytes for {input.Length} dimensions.", nameof(buffer));
        }

        var maxMagnitude = 0f;
        foreach (var value in input)
        {
            maxMagnitude = MathF.Max(maxMagnitude, MathF.Abs(value));
        }

        var valueBytes = buffer.Span[MagnitudeByteLength..];
        var quantizedMagnitudeSquared = 0L;

        if (maxMagnitude == 0)
        {
            valueBytes.Clear();
            BitConverter.TryWriteBytes(buffer.Span[..MagnitudeByteLength], 0f);
            return new EmbeddingI8(buffer);
        }

        var scale = 127f / maxMagnitude;
        for (var index = 0; index < input.Length; index++)
        {
            var quantized = (int)MathF.Round(input[index] * scale);
            quantized = Math.Clamp(quantized, sbyte.MinValue + 1, sbyte.MaxValue);
            valueBytes[index] = unchecked((byte)(sbyte)quantized);
            quantizedMagnitudeSquared += quantized * quantized;
        }

        BitConverter.TryWriteBytes(
            buffer.Span[..MagnitudeByteLength],
            MathF.Sqrt(quantizedMagnitudeSquared));

        return new EmbeddingI8(buffer);
    }

    /// <summary>
    /// Computes cosine-style similarity between this embedding and another <see cref="EmbeddingI8"/>.
    /// </summary>
    /// <param name="other">The embedding to compare with this embedding.</param>
    /// <returns>
    /// A similarity score computed from the quantized integer vectors. Values closer to 1
    /// mean the vectors point in a similar direction.
    /// </returns>
    /// <exception cref="InvalidOperationException">Thrown when the embeddings have different dimension counts.</exception>
    public float Similarity(EmbeddingI8 other)
    {
        if (Values.Length != other.Values.Length)
        {
            throw new InvalidOperationException($"Cannot compare {Values.Length} dimensions with {other.Values.Length} dimensions.");
        }

        if (Magnitude == 0 || other.Magnitude == 0)
        {
            return 0;
        }

        var dotProduct = 0L;
        var left = Values.Span;
        var right = other.Values.Span;

        for (var index = 0; index < left.Length; index++)
        {
            dotProduct += left[index] * right[index];
        }

        return dotProduct / (Magnitude * other.Magnitude);
    }

    private sealed class EmbeddingI8JsonConverter : JsonConverter<EmbeddingI8>
    {
        public override EmbeddingI8 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
            => new(reader.GetBytesFromBase64());

        public override void Write(Utf8JsonWriter writer, EmbeddingI8 value, JsonSerializerOptions options)
            => writer.WriteBase64StringValue(value.Buffer.Span);
    }
}
