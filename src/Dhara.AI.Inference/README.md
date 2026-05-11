# Dhara.AI.Inference

Low-level local inference primitives for Dhara.AI packages.

`Dhara.AI.Inference` is the ONNX and tokenizer layer behind the higher-level
embedding APIs. It is useful when you want direct control over model paths,
tokenization, ONNX Runtime session options, pooling, and normalization without
taking a dependency on the convenience APIs in `Dhara.AI.LocalEmbeddings`.

## What It Provides

- `OnnxTextEmbeddingModel` for running BERT-style text embedding models from an ONNX file.
- `OnnxTextEmbeddingOptions` for model paths, vocabulary paths, input/output names, pooling, normalization, and ONNX session customization.
- `BertEmbeddingTokenizer` for vocabulary-based tokenization into `input_ids`, `attention_mask`, and `token_type_ids`.
- `EmbeddingMath` for cosine similarity, mean pooling, and vector normalization.
- `EmbeddingPoolingMode` for mean-pooling or CLS-token embedding extraction.

## Basic Usage

```csharp
using Dhara.AI.Inference;

var options = new OnnxTextEmbeddingOptions
{
    ModelPath = "LocalEmbeddingsModel/default/model.onnx",
    VocabularyPath = "LocalEmbeddingsModel/default/vocab.txt",
    ModelId = "default",
    PoolingMode = EmbeddingPoolingMode.Mean,
    NormalizeEmbeddings = true
};

using var model = new OnnxTextEmbeddingModel(options);
var vectors = model.Embed(
[
    "local embeddings for semantic search",
    "native AOT friendly inference in .NET"
]);

var score = EmbeddingMath.CosineSimilarity(vectors[0], vectors[1]);
```

## Model Compatibility

The model is expected to accept BERT-style inputs such as `input_ids` and
`attention_mask`. If the ONNX graph exposes `token_type_ids`, the runtime sends
them automatically; if it does not, the runtime omits them. This keeps the
runtime compatible with many sentence-transformer exports, including models
that produce 384-dimensional or 768-dimensional vectors.

The runtime supports output tensors shaped as either:

- `[batch, dimensions]` when the model already returns sentence embeddings.
- `[batch, sequence, dimensions]` when the runtime should pool token embeddings.

## Native AOT

The package is marked as AOT-compatible. Native AOT is validated by publishing
an application that consumes the package, because AOT behavior is determined at
the application boundary.

## Related Package

Most application code should start with
[`Dhara.AI.LocalEmbeddings`](../Dhara.AI.LocalEmbeddings/README.md), which wraps
this runtime with `Microsoft.Extensions.AI`, model acquisition, compact storage
formats, and ranking helpers.
