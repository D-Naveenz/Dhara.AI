# Dhara.AI

Modern local AI building blocks for .NET.

## Local Embeddings

`Dhara.AI.LocalEmbeddings` exposes a local ONNX embedding model through
`Microsoft.Extensions.AI`:

```csharp
using Dhara.AI.LocalEmbeddings;

using var generator = LocalEmbeddingGenerator.Create();
var query = await generator.EmbedAsync<EmbeddingF32>("semantic search for Obsidian notes");
var candidates = await generator.EmbedRangeAsync<EmbeddingF32>(
[
    "Smart Connections stores note and block embeddings.",
    "Native AOT publishes a self-contained executable."
]);

var closest = EmbeddingSearch.FindClosestWithScore(query, candidates, maxResults: 1);
```

The package downloads the default `bge-micro-v2` ONNX model and vocabulary during
build. Override `LocalEmbeddingsModelUrl`, `LocalEmbeddingsVocabUrl`,
`LocalEmbeddingsModelPath`, or `LocalEmbeddingsVocabPath` to use your own files.

The runtime is not limited to the default model's 384 dimensions. It sizes
embedding buffers from the vector returned by the ONNX model, so a 768-dimensional
sentence-transformer model stores `768 * 4 = 3072` bytes in `EmbeddingF32`.
Sentence-transformer exports should use mean pooling and normalized embeddings;
this is the default behavior. DistilBERT-style exports that omit `token_type_ids`
are supported automatically when the ONNX model does not expose that input.

## Commands

```powershell
dotnet build Dhara.AI.slnx
dotnet test Dhara.AI.slnx
dotnet run --project samples\Dhara.AI.LocalEmbeddings.Sample
dotnet publish samples\Dhara.AI.LocalEmbeddings.Sample -c Release -r win-x64 --self-contained true
```
