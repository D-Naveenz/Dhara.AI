using Dhara.AI.Inference;
using Microsoft.Extensions.AI;

namespace Dhara.AI.LocalEmbeddings;

/// <summary>
/// Generates text embeddings locally with a BERT-style ONNX model.
/// </summary>
/// <remarks>
/// This type is safe for concurrent generation calls. Dispose it when the owning application shuts down to release ONNX Runtime native resources.
/// </remarks>
public sealed class LocalEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly OnnxTextEmbeddingModel _model;
    private readonly EmbeddingGeneratorMetadata _metadata;

    /// <summary>
    /// Initializes a new instance of the <see cref="LocalEmbeddingGenerator"/> class.
    /// </summary>
    /// <param name="options">The model and runtime options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="options"/> is <see langword="null"/>.</exception>
    /// <exception cref="InvalidOperationException">Thrown when default model files cannot be found.</exception>
    public LocalEmbeddingGenerator(LocalEmbeddingGeneratorOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var inferenceOptions = options.ToInferenceOptions();
        _model = new OnnxTextEmbeddingModel(inferenceOptions);
        _metadata = new EmbeddingGeneratorMetadata(
            providerName: "Dhara.AI.LocalEmbeddings",
            providerUri: null,
            defaultModelId: inferenceOptions.ModelId,
            defaultModelDimensions: null);
    }

    /// <summary>
    /// Creates a local embedding generator using default model resolution.
    /// </summary>
    /// <param name="configure">An optional callback used to customize the generator options.</param>
    /// <returns>A configured local embedding generator.</returns>
    /// <exception cref="InvalidOperationException">Thrown when default model files cannot be found.</exception>
    public static LocalEmbeddingGenerator Create(Action<LocalEmbeddingGeneratorOptions>? configure = null)
    {
        var options = new LocalEmbeddingGeneratorOptions();
        configure?.Invoke(options);
        return new LocalEmbeddingGenerator(options);
    }

    /// <inheritdoc />
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var vectors = _model.Embed(values, cancellationToken);
        var embeddings = new GeneratedEmbeddings<Embedding<float>>(vectors.Count);

        foreach (var vector in vectors)
        {
            embeddings.Add(new Embedding<float>(vector)
            {
                CreatedAt = DateTimeOffset.UtcNow,
                ModelId = _model.ModelId
            });
        }

        return Task.FromResult(embeddings);
    }

    /// <summary>
    /// Generates a single embedding vector.
    /// </summary>
    /// <param name="text">The text to embed.</param>
    /// <param name="cancellationToken">A cancellation token observed before inference begins.</param>
    /// <returns>The generated embedding vector.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the generator does not return exactly one embedding.</exception>
    public async Task<ReadOnlyMemory<float>> GenerateVectorAsync(string text, CancellationToken cancellationToken = default)
    {
        var embeddings = await GenerateAsync([text], cancellationToken: cancellationToken).ConfigureAwait(false);
        if (embeddings.Count != 1)
        {
            throw new InvalidOperationException($"Expected one embedding, but received {embeddings.Count}.");
        }

        return embeddings[0].Vector;
    }

    /// <summary>
    /// Generates a single embedding in a compact representation.
    /// </summary>
    /// <typeparam name="TEmbedding">The compact embedding representation to create.</typeparam>
    /// <param name="text">The text to embed.</param>
    /// <param name="buffer">An optional caller-owned buffer for the compact embedding bytes.</param>
    /// <param name="cancellationToken">A cancellation token observed before inference begins.</param>
    /// <returns>The compact embedding representation.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="buffer"/> has the wrong length.</exception>
    public async Task<TEmbedding> EmbedAsync<TEmbedding>(
        string text,
        Memory<byte>? buffer = null,
        CancellationToken cancellationToken = default)
        where TEmbedding : IBufferedEmbedding<TEmbedding>
    {
        var vector = await GenerateVectorAsync(text, cancellationToken).ConfigureAwait(false);
        var outputBuffer = buffer ?? new byte[TEmbedding.GetBufferByteLength(vector.Length)];
        return TEmbedding.FromModelOutput(vector.Span, outputBuffer);
    }

    /// <summary>
    /// Generates compact embeddings for a batch of strings.
    /// </summary>
    /// <typeparam name="TEmbedding">The compact embedding representation to create.</typeparam>
    /// <param name="items">The strings to embed.</param>
    /// <param name="cancellationToken">A cancellation token observed before inference begins.</param>
    /// <returns>The input strings paired with embeddings.</returns>
    public async Task<IReadOnlyList<(string Item, TEmbedding Embedding)>> EmbedRangeAsync<TEmbedding>(
        IEnumerable<string> items,
        CancellationToken cancellationToken = default)
        where TEmbedding : IBufferedEmbedding<TEmbedding>
    {
        ArgumentNullException.ThrowIfNull(items);
        var materialized = items.ToArray();
        var generated = await GenerateAsync(materialized, cancellationToken: cancellationToken).ConfigureAwait(false);
        var result = new (string Item, TEmbedding Embedding)[generated.Count];

        for (var index = 0; index < generated.Count; index++)
        {
            var vector = generated[index].Vector;
            var buffer = new byte[TEmbedding.GetBufferByteLength(vector.Length)];
            result[index] = (materialized[index], TEmbedding.FromModelOutput(vector.Span, buffer));
        }

        return result;
    }

    /// <summary>
    /// Gets a service associated with this generator.
    /// </summary>
    /// <param name="serviceType">The requested service type.</param>
    /// <param name="serviceKey">An optional service key.</param>
    /// <returns>The requested service, or <see langword="null"/> when unavailable.</returns>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        return serviceKey is not null ? null :
            serviceType == typeof(EmbeddingGeneratorMetadata) ? _metadata :
            serviceType == typeof(OnnxTextEmbeddingModel) ? _model :
            serviceType.IsInstanceOfType(this) ? this :
            null;
    }

    /// <summary>
    /// Releases the underlying ONNX Runtime session.
    /// </summary>
    public void Dispose()
        => _model.Dispose();
}
