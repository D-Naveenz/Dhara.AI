# Dhara.AI

Modern local AI building blocks for .NET, focused on local-first inference,
embeddings, retrieval, and Native AOT-friendly application shapes.

## Packages

| Package | Purpose |
| --- | --- |
| [`Dhara.AI.Inference`](src/Dhara.AI.Inference/README.md) | Low-level ONNX Runtime and tokenizer primitives for local embedding models. |
| [`Dhara.AI.LocalEmbeddings`](src/Dhara.AI.LocalEmbeddings/README.md) | `Microsoft.Extensions.AI` local embedding generator, model acquisition, compact embedding formats, and ranking helpers. |

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

Use `EmbeddingF32` when you want the raw full-precision model output. Use
`EmbeddingI8` when you want a compact scalar-quantized value for local indexes:
a 384-dimensional vector uses 388 bytes instead of 1,536 bytes, and a
768-dimensional vector uses 772 bytes instead of 3,072 bytes. This is useful
when storing many note or block embeddings for retrieval.

Use `EmbeddingI1` when you need the smallest representation and can accept
approximate Hamming-similarity ranking: a 384-dimensional vector uses 52 bytes,
and a 768-dimensional vector uses 100 bytes. This is best for very large indexes,
coarse filtering, or shortlist-then-rescore retrieval.

The package downloads the default `bge-micro-v2` ONNX model and vocabulary during
build. Override `LocalEmbeddingsModelUrl`, `LocalEmbeddingsVocabUrl`,
`LocalEmbeddingsModelPath`, or `LocalEmbeddingsVocabPath` to use your own files.

The runtime is not limited to the default model's 384 dimensions. It sizes
embedding buffers from the vector returned by the ONNX model, so a 768-dimensional
sentence-transformer model stores `768 * 4 = 3072` bytes in `EmbeddingF32`.
Sentence-transformer exports should use mean pooling and normalized embeddings;
this is the default behavior. DistilBERT-style exports that omit `token_type_ids`
are supported automatically when the ONNX model does not expose that input.

## Packaging

The two library projects are packable NuGet packages. Each package has its own
README and uses `assets/branding/dhara-logo-colored_sm.png` as the NuGet icon.
Model assets are not packed into `Dhara.AI.LocalEmbeddings`; the package ships an
MSBuild target that downloads/copies the configured model files into consuming
application outputs.

Pushes to `master`, including merged pull requests, publish both packages
through `.github/workflows/publish-nuget.yml`. The workflow requires a
repository secret named `NUGET_API_KEY`.

## Commands

```powershell
dotnet build Dhara.AI.slnx
dotnet test Dhara.AI.slnx
dotnet run --project samples\Dhara.AI.LocalEmbeddings.Sample
dotnet publish samples\Dhara.AI.LocalEmbeddings.Sample -c Release -r win-x64 --self-contained true
dotnet pack src\Dhara.AI.Inference\Dhara.AI.Inference.csproj -c Release
dotnet pack src\Dhara.AI.LocalEmbeddings\Dhara.AI.LocalEmbeddings.csproj -c Release
```
