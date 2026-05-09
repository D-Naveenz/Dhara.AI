namespace Dhara.AI.Inference;

/// <summary>
/// Represents padded BERT-style model inputs for a batch of text values.
/// </summary>
/// <param name="InputIds">The flattened token id tensor values.</param>
/// <param name="AttentionMask">The flattened attention mask tensor values.</param>
/// <param name="TokenTypeIds">The flattened token type id tensor values.</param>
/// <param name="BatchSize">The number of text values in the batch.</param>
/// <param name="SequenceLength">The padded sequence length used for each text value.</param>
public sealed record TokenizedTextBatch(
    long[] InputIds,
    long[] AttentionMask,
    long[] TokenTypeIds,
    int BatchSize,
    int SequenceLength);
