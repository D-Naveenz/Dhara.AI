using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Dhara.AI.Inference;

namespace Dhara.AI.LocalEmbeddings;

/// <summary>
/// Stores an embedding as 32-bit floating-point values.
/// </summary>
/// <remarks>
/// This is the raw, unquantized vector representation. It is larger than quantized forms, but it preserves the full model output.
/// </remarks>
[JsonConverter(typeof(EmbeddingF32JsonConverter))]
public readonly struct EmbeddingF32 : IBufferedEmbedding<EmbeddingF32>
{
    private readonly ReadOnlyMemory<byte> _buffer;
    private readonly ReadOnlyMemory<float> _values;

    /// <summary>
    /// Initializes a new instance of the <see cref="EmbeddingF32"/> struct from persisted bytes.
    /// </summary>
    /// <param name="buffer">The byte buffer previously produced by <see cref="Buffer"/>.</param>
    /// <exception cref="ArgumentException">Thrown when <paramref name="buffer"/> length is not divisible by four.</exception>
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
    /// Gets the compact persisted byte representation of the embedding.
    /// </summary>
    public ReadOnlyMemory<byte> Buffer => _buffer;

    /// <summary>
    /// Gets the embedding vector values.
    /// </summary>
    public ReadOnlyMemory<float> Values => _values;

    /// <inheritdoc />
    public static int GetBufferByteLength(int dimensions)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimensions);
        return checked(dimensions * sizeof(float));
    }

    /// <inheritdoc />
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

    /// <inheritdoc />
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
