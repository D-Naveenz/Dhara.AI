using Dhara.AI.Inference;
using Microsoft.ML.OnnxRuntime;

namespace Dhara.AI.LocalEmbeddings;

/// <summary>
/// Configures a local embedding generator backed by a BERT-style ONNX model.
/// </summary>
public sealed class LocalEmbeddingGeneratorOptions
{
    /// <summary>
    /// Gets or sets the model name used when resolving files from the default model content directory.
    /// </summary>
    public string ModelName { get; set; } = "default";

    /// <summary>
    /// Gets or sets the path to the ONNX model file.
    /// </summary>
    /// <remarks>
    /// When this value is <see langword="null"/>, the generator resolves
    /// <c>LocalEmbeddingsModel/&lt;ModelName&gt;/model.onnx</c> under <see cref="AppContext.BaseDirectory"/>.
    /// </remarks>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the BERT vocabulary file.
    /// </summary>
    /// <remarks>
    /// When this value is <see langword="null"/>, the generator resolves
    /// <c>LocalEmbeddingsModel/&lt;ModelName&gt;/vocab.txt</c> under <see cref="AppContext.BaseDirectory"/>.
    /// </remarks>
    public string? VocabularyPath { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of tokens to send to the model, including special tokens.
    /// </summary>
    public int MaxTokens { get; set; } = 512;

    /// <summary>
    /// Gets or sets a value indicating whether tokenization is case-sensitive.
    /// </summary>
    public bool CaseSensitive { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether generated vectors are L2-normalized.
    /// </summary>
    public bool NormalizeEmbeddings { get; set; } = true;

    /// <summary>
    /// Gets or sets how token-level ONNX output is reduced to one vector per input.
    /// </summary>
    public EmbeddingPoolingMode PoolingMode { get; set; } = EmbeddingPoolingMode.Mean;

    /// <summary>
    /// Gets or sets the ONNX input name for token ids.
    /// </summary>
    public string InputIdsName { get; set; } = "input_ids";

    /// <summary>
    /// Gets or sets the ONNX input name for the attention mask.
    /// </summary>
    public string AttentionMaskName { get; set; } = "attention_mask";

    /// <summary>
    /// Gets or sets the ONNX input name for token type ids.
    /// </summary>
    public string TokenTypeIdsName { get; set; } = "token_type_ids";

    /// <summary>
    /// Gets or sets the ONNX output name to read.
    /// </summary>
    /// <remarks>
    /// When this value is <see langword="null"/>, the first model output is used.
    /// </remarks>
    public string? OutputName { get; set; }

    /// <summary>
    /// Gets or sets a callback that customizes ONNX Runtime session options before the session is created.
    /// </summary>
    public Action<SessionOptions>? ConfigureSessionOptions { get; set; }

    /// <summary>
    /// Converts this high-level options object into low-level ONNX inference options.
    /// </summary>
    /// <returns>The options consumed by <see cref="OnnxTextEmbeddingModel"/>.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the resolved model or vocabulary files do not exist.</exception>
    public OnnxTextEmbeddingOptions ToInferenceOptions()
    {
        var modelPath = ModelPath ?? ResolveModelFile(ModelName, "model.onnx");
        var vocabularyPath = VocabularyPath ?? ResolveModelFile(ModelName, "vocab.txt");

        return new OnnxTextEmbeddingOptions
        {
            ModelPath = modelPath,
            VocabularyPath = vocabularyPath,
            ModelId = ModelName,
            MaxTokens = MaxTokens,
            LowerCaseBeforeTokenization = !CaseSensitive,
            InputIdsName = InputIdsName,
            AttentionMaskName = AttentionMaskName,
            TokenTypeIdsName = TokenTypeIdsName,
            OutputName = OutputName,
            PoolingMode = PoolingMode,
            NormalizeEmbeddings = NormalizeEmbeddings,
            ConfigureSessionOptions = ConfigureSessionOptions
        };
    }

    private static string ResolveModelFile(string modelName, string fileName)
    {
        var path = Path.Combine(AppContext.BaseDirectory, "LocalEmbeddingsModel", modelName, fileName);
        if (!File.Exists(path))
        {
            throw new InvalidOperationException($"Required local embedding file was not found: {path}");
        }

        return path;
    }
}
