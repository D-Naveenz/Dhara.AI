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
build. Override `DharaEmbeddingModelUrl`, `DharaEmbeddingVocabUrl`,
`DharaEmbeddingModelPath`, or `DharaEmbeddingVocabPath` to use your own files.

## Commands

```powershell
dotnet build Dhara.AI.slnx
dotnet test Dhara.AI.slnx
dotnet run --project samples\Dhara.AI.LocalEmbeddings.Sample
dotnet publish samples\Dhara.AI.LocalEmbeddings.Sample -c Release -r win-x64 --self-contained true
```
