# Dhara.AI.LocalEmbeddings

Local text embeddings for .NET, built on `Microsoft.Extensions.AI` and ONNX.

`Dhara.AI.LocalEmbeddings` exposes a local embedding generator through
`IEmbeddingGenerator<string, Embedding<float>>` while also providing compact
embedding formats and ranking helpers inspired by SmartComponents.

## Install

```powershell
dotnet add package Dhara.AI.LocalEmbeddings
```

## Quick Start

```csharp
using Dhara.AI.LocalEmbeddings;

using var generator = LocalEmbeddingGenerator.Create();

var query = await generator.EmbedAsync<EmbeddingF32>(
    "semantic search for Obsidian notes");

var candidates = await generator.EmbedRangeAsync<EmbeddingF32>(
[
    "Smart Connections stores note and block embeddings.",
    "Native AOT publishes a self-contained executable.",
    "ONNX Runtime can run local transformer models."
]);

var closest = EmbeddingSearch.FindClosestWithScore(query, candidates, maxResults: 2);
```

## Microsoft.Extensions.AI

The primary contract is:

```csharp
IEmbeddingGenerator<string, Embedding<float>>
```

That means the generator fits modern .NET AI patterns, dependency injection,
and code that already targets `Microsoft.Extensions.AI`.

```csharp
services.AddLocalEmbeddings(options =>
{
    options.ModelName = "default";
    options.MaxTokens = 512;
});
```

## Model Acquisition

The package includes an MSBuild target that downloads the default
SmartComponents `bge-micro-v2` ONNX model and vocabulary during build, then
copies them into the consuming app output under:

```text
LocalEmbeddingsModel/<model-name>/model.onnx
LocalEmbeddingsModel/<model-name>/vocab.txt
```

The model files are not embedded into the NuGet package. This keeps the package
small and lets applications choose their own model source.

Useful MSBuild properties:

- `LocalEmbeddingsModelName`
- `LocalEmbeddingsModelUrl`
- `LocalEmbeddingsVocabUrl`
- `LocalEmbeddingsModelCacheDir`
- `LocalEmbeddingsModelPath`
- `LocalEmbeddingsVocabPath`

## Embedding Formats

`EmbeddingF32` stores the raw model output as one 32-bit floating-point value
per dimension. A 384-dimensional vector uses 1,536 bytes, and a 768-dimensional
vector uses 3,072 bytes. Use this for highest quality, debugging, evaluation,
and final reranking.

`EmbeddingI8` stores a scalar-quantized signed 8-bit representation plus the
vector magnitude. A 384-dimensional vector uses 388 bytes, and a 768-dimensional
vector uses 772 bytes. Use this when local indexes need to be much smaller while
preserving useful ranking quality.

`EmbeddingI1` stores only the sign bit of each dimension plus a dimension
header. A 384-dimensional vector uses 52 bytes, and a 768-dimensional vector
uses 100 bytes. Use this for very large indexes, coarse filtering, or
shortlist-then-rescore retrieval.

## Model Compatibility

The default model produces 384-dimensional embeddings, but the library is not
limited to that size. It sizes buffers from the vector returned by the ONNX
model, so 768-dimensional sentence-transformer exports such as
`sentence-transformers/multi-qa-distilbert-cos-v1` can be used when exported to
a compatible ONNX shape.

Mean pooling and normalization are enabled by default because they match common
sentence-transformer semantic-search models. Both can be configured through
`LocalEmbeddingGeneratorOptions`.

## Native AOT

The package is marked as AOT-compatible. Publish a consuming app with Native AOT
to validate the complete application:

```powershell
dotnet publish samples\Dhara.AI.LocalEmbeddings.Sample -c Release -r win-x64 --self-contained true
```

## Related Package

`Dhara.AI.Inference` contains the lower-level ONNX Runtime and tokenizer layer
used by this package.
