namespace Dhara.AI.Inference;

/// <summary>
/// Specifies how token-level model output is reduced into one embedding vector.
/// </summary>
public enum EmbeddingPoolingMode
{
    /// <summary>
    /// Computes the average hidden state across non-padding tokens.
    /// </summary>
    Mean = 0,

    /// <summary>
    /// Uses the first token hidden state, typically the BERT classification token.
    /// </summary>
    Cls = 1
}
