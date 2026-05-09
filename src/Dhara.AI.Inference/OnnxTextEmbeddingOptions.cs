using Microsoft.ML.OnnxRuntime;

namespace Dhara.AI.Inference;

/// <summary>
/// Configures an ONNX BERT-style text embedding session.
/// </summary>
public sealed class OnnxTextEmbeddingOptions
{
    /// <summary>
    /// Gets or sets the path to the ONNX model file.
    /// </summary>
    public string ModelPath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the path to the BERT vocabulary file used to tokenize text.
    /// </summary>
    public string VocabularyPath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets a stable name for the model represented by this session.
    /// </summary>
    public string ModelId { get; set; } = "default";

    /// <summary>
    /// Gets or sets the maximum number of tokens sent to the model, including special tokens.
    /// </summary>
    public int MaxTokens { get; set; } = 512;

    /// <summary>
    /// Gets or sets a value indicating whether the tokenizer lowercases text before tokenization.
    /// </summary>
    public bool LowerCaseBeforeTokenization { get; set; }

    /// <summary>
    /// Gets or sets the name of the ONNX input that receives token ids.
    /// </summary>
    public string InputIdsName { get; set; } = "input_ids";

    /// <summary>
    /// Gets or sets the name of the ONNX input that receives the attention mask.
    /// </summary>
    public string AttentionMaskName { get; set; } = "attention_mask";

    /// <summary>
    /// Gets or sets the optional name of the ONNX input that receives token type ids.
    /// </summary>
    /// <remarks>
    /// Some BERT exports require <c>token_type_ids</c> to distinguish sentence pairs.
    /// DistilBERT-style sentence-transformer exports often omit this input. When this
    /// value is <see langword="null"/>, or when the loaded ONNX model has no input with
    /// this name, token type ids are not sent to the model.
    /// </remarks>
    public string? TokenTypeIdsName { get; set; } = "token_type_ids";

    /// <summary>
    /// Gets or sets an optional output name to read from the ONNX model.
    /// </summary>
    /// <remarks>
    /// When this value is <see langword="null"/>, the first output produced by the model is used.
    /// </remarks>
    public string? OutputName { get; set; }

    /// <summary>
    /// Gets or sets how token-level output is reduced to a single vector.
    /// </summary>
    public EmbeddingPoolingMode PoolingMode { get; set; } = EmbeddingPoolingMode.Mean;

    /// <summary>
    /// Gets or sets a value indicating whether generated vectors are L2-normalized.
    /// </summary>
    public bool NormalizeEmbeddings { get; set; } = true;

    /// <summary>
    /// Gets or sets a callback that customizes the ONNX Runtime session options before the session is created.
    /// </summary>
    public Action<SessionOptions>? ConfigureSessionOptions { get; set; }

    /// <summary>
    /// Creates a detached copy of the current options.
    /// </summary>
    /// <returns>A copy of the current options.</returns>
    public OnnxTextEmbeddingOptions Clone()
        => new()
        {
            ModelPath = ModelPath,
            VocabularyPath = VocabularyPath,
            ModelId = ModelId,
            MaxTokens = MaxTokens,
            LowerCaseBeforeTokenization = LowerCaseBeforeTokenization,
            InputIdsName = InputIdsName,
            AttentionMaskName = AttentionMaskName,
            TokenTypeIdsName = TokenTypeIdsName,
            OutputName = OutputName,
            PoolingMode = PoolingMode,
            NormalizeEmbeddings = NormalizeEmbeddings,
            ConfigureSessionOptions = ConfigureSessionOptions
        };
}
