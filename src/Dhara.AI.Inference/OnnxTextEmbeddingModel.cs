using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Dhara.AI.Inference;

/// <summary>
/// Runs a local BERT-style ONNX embedding model.
/// </summary>
/// <remarks>
/// The underlying ONNX Runtime session is thread-safe for inference. Dispose the instance when the application no longer needs it.
/// </remarks>
public sealed class OnnxTextEmbeddingModel : IDisposable
{
    private readonly OnnxTextEmbeddingOptions _options;
    private readonly BertEmbeddingTokenizer _tokenizer;
    private readonly InferenceSession _session;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="OnnxTextEmbeddingModel"/> class.
    /// </summary>
    /// <param name="options">The model, tokenizer, and ONNX session options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="options"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">Thrown when a required path is empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the model or vocabulary file does not exist.</exception>
    public OnnxTextEmbeddingModel(OnnxTextEmbeddingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options.Clone();

        if (string.IsNullOrWhiteSpace(_options.ModelPath))
        {
            throw new ArgumentException("A model file path is required.", nameof(options));
        }

        if (!File.Exists(_options.ModelPath))
        {
            throw new FileNotFoundException("The ONNX model file was not found.", _options.ModelPath);
        }

        _tokenizer = new BertEmbeddingTokenizer(
            _options.VocabularyPath,
            _options.MaxTokens,
            _options.LowerCaseBeforeTokenization);

        var sessionOptions = new SessionOptions();
        _options.ConfigureSessionOptions?.Invoke(sessionOptions);
        _session = new InferenceSession(_options.ModelPath, sessionOptions);
    }

    /// <summary>
    /// Gets the stable identifier for the model used by this instance.
    /// </summary>
    public string ModelId => _options.ModelId;

    /// <summary>
    /// Generates embeddings for a batch of text values.
    /// </summary>
    /// <param name="values">The text values to embed.</param>
    /// <param name="cancellationToken">A cancellation token observed before inference begins.</param>
    /// <returns>A vector for each input value in the same order.</returns>
    /// <exception cref="ObjectDisposedException">Thrown when the instance has been disposed.</exception>
    /// <exception cref="ArgumentException">Thrown when the batch is empty or contains invalid values.</exception>
    public IReadOnlyList<float[]> Embed(IEnumerable<string> values, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        cancellationToken.ThrowIfCancellationRequested();

        var batch = _tokenizer.Tokenize(values);
        var dimensions = new long[] { batch.BatchSize, batch.SequenceLength };
        using var inputIds = OrtValue.CreateTensorValueFromMemory(batch.InputIds, dimensions);
        using var attentionMask = OrtValue.CreateTensorValueFromMemory(batch.AttentionMask, dimensions);
        using var tokenTypeIds = OrtValue.CreateTensorValueFromMemory(batch.TokenTypeIds, dimensions);

        var inputNames = new[] { _options.InputIdsName, _options.AttentionMaskName, _options.TokenTypeIdsName };
        var inputs = new[] { inputIds, attentionMask, tokenTypeIds };
        var outputNames = _options.OutputName is null ? _session.OutputNames : [_options.OutputName];
        using var output = _session.Run(new RunOptions(), inputNames, inputs, outputNames);
        var outputValue = output[0];
        var outputTensor = outputValue.GetTensorDataAsSpan<float>();
        var shape = outputValue.GetTensorTypeAndShape().Shape;

        return shape.Length switch
        {
            2 => ReadSentenceEmbeddings(outputTensor, shape),
            3 => PoolTokenEmbeddings(outputTensor, shape, batch),
            _ => throw new InvalidOperationException($"Unsupported embedding output rank {shape.Length}. Expected rank 2 or 3.")
        };
    }

    /// <summary>
    /// Releases the ONNX Runtime session and associated native resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _session.Dispose();
        _disposed = true;
    }

    private IReadOnlyList<float[]> ReadSentenceEmbeddings(ReadOnlySpan<float> outputTensor, IReadOnlyList<long> shape)
    {
        var batchSize = checked((int)shape[0]);
        var dimensions = checked((int)shape[1]);
        var result = new float[batchSize][];

        for (var row = 0; row < batchSize; row++)
        {
            var vector = outputTensor.Slice(row * dimensions, dimensions).ToArray();
            if (_options.NormalizeEmbeddings)
            {
                EmbeddingMath.NormalizeInPlace(vector);
            }

            result[row] = vector;
        }

        return result;
    }

    private IReadOnlyList<float[]> PoolTokenEmbeddings(ReadOnlySpan<float> outputTensor, IReadOnlyList<long> shape, TokenizedTextBatch batch)
    {
        var batchSize = checked((int)shape[0]);
        var sequenceLength = checked((int)shape[1]);
        var dimensions = checked((int)shape[2]);
        var result = new float[batchSize][];

        for (var row = 0; row < batchSize; row++)
        {
            var vector = new float[dimensions];
            var tokenOffset = row * sequenceLength * dimensions;
            var maskOffset = row * batch.SequenceLength;

            if (_options.PoolingMode == EmbeddingPoolingMode.Cls)
            {
                outputTensor.Slice(tokenOffset, dimensions).CopyTo(vector);
            }
            else
            {
                EmbeddingMath.MeanPool(
                    outputTensor.Slice(tokenOffset, sequenceLength * dimensions),
                    batch.AttentionMask.AsSpan(maskOffset, batch.SequenceLength),
                    sequenceLength,
                    dimensions,
                    vector);
            }

            if (_options.NormalizeEmbeddings)
            {
                EmbeddingMath.NormalizeInPlace(vector);
            }

            result[row] = vector;
        }

        return result;
    }
}
