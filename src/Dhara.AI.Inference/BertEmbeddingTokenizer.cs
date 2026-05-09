using Microsoft.ML.Tokenizers;

namespace Dhara.AI.Inference;

/// <summary>
/// Converts text into BERT-style ONNX model inputs.
/// </summary>
/// <remarks>
/// Instances are immutable after construction and can be reused concurrently.
/// </remarks>
public sealed class BertEmbeddingTokenizer
{
    private readonly BertTokenizer _tokenizer;
    private readonly int _maxTokens;

    /// <summary>
    /// Initializes a new instance of the <see cref="BertEmbeddingTokenizer"/> class.
    /// </summary>
    /// <param name="vocabularyPath">The path to the BERT vocabulary file.</param>
    /// <param name="maxTokens">The maximum number of tokens per text value, including special tokens.</param>
    /// <param name="lowerCaseBeforeTokenization">Whether text is lowercased before tokenization.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="vocabularyPath"/> is empty or <paramref name="maxTokens"/> is less than 3.
    /// </exception>
    /// <exception cref="FileNotFoundException">Thrown when <paramref name="vocabularyPath"/> does not exist.</exception>
    public BertEmbeddingTokenizer(string vocabularyPath, int maxTokens, bool lowerCaseBeforeTokenization)
    {
        if (string.IsNullOrWhiteSpace(vocabularyPath))
        {
            throw new ArgumentException("A vocabulary file path is required.", nameof(vocabularyPath));
        }

        if (!File.Exists(vocabularyPath))
        {
            throw new FileNotFoundException("The vocabulary file was not found.", vocabularyPath);
        }

        if (maxTokens < 3)
        {
            throw new ArgumentOutOfRangeException(nameof(maxTokens), maxTokens, "At least three tokens are required for [CLS], text, and [SEP].");
        }

        _tokenizer = BertTokenizer.Create(vocabularyPath, new BertOptions
        {
            LowerCaseBeforeTokenization = lowerCaseBeforeTokenization,
            ApplyBasicTokenization = true,
            SplitOnSpecialTokens = true
        });
        _maxTokens = maxTokens;
    }

    /// <summary>
    /// Converts a batch of text values into padded ONNX tensor data.
    /// </summary>
    /// <param name="values">The text values to tokenize.</param>
    /// <returns>Padded token ids, attention masks, and token type ids.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="values"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">Thrown when the batch is empty or contains a <see langword="null"/> value.</exception>
    public TokenizedTextBatch Tokenize(IEnumerable<string> values)
    {
        ArgumentNullException.ThrowIfNull(values);

        var encoded = new List<IReadOnlyList<int>>();
        foreach (var value in values)
        {
            if (value is null)
            {
                throw new ArgumentException("The text batch cannot contain null values.", nameof(values));
            }

            var tokenIds = _tokenizer.EncodeToIds(value, addSpecialTokens: false, considerNormalization: true);
            var truncatedLength = Math.Min(tokenIds.Count, _maxTokens - 2);
            var truncated = tokenIds.Take(truncatedLength);
            encoded.Add(_tokenizer.BuildInputsWithSpecialTokens(truncated, additionalTokenIds: null));
        }

        if (encoded.Count == 0)
        {
            throw new ArgumentException("At least one text value is required.", nameof(values));
        }

        var sequenceLength = encoded.Max(static item => item.Count);
        var inputIds = new long[encoded.Count * sequenceLength];
        var attentionMask = new long[inputIds.Length];
        var tokenTypeIds = new long[inputIds.Length];

        for (var row = 0; row < encoded.Count; row++)
        {
            var rowOffset = row * sequenceLength;
            var tokenIds = encoded[row];
            for (var column = 0; column < tokenIds.Count; column++)
            {
                inputIds[rowOffset + column] = tokenIds[column];
                attentionMask[rowOffset + column] = 1;
            }
        }

        return new TokenizedTextBatch(inputIds, attentionMask, tokenTypeIds, encoded.Count, sequenceLength);
    }
}
