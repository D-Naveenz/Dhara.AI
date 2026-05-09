using Dhara.AI.LocalEmbeddings;

var query = args.Length == 0
    ? "How does MindVault search my local notes?"
    : string.Join(' ', args);

var snippets = new[]
{
    "MindVault MCP reads Smart Connections cache files and ranks related Obsidian notes.",
    "Native AOT trims startup cost for command line tools and local services.",
    "A recipe for cardamom tea uses warm milk, black tea, and spices.",
    "Query embeddings let lexical search fall back to semantic similarity when the wording changes."
};

using var generator = LocalEmbeddingGenerator.Create();
var queryEmbedding = await generator.EmbedAsync<EmbeddingF32>(query);
var snippetEmbeddings = await generator.EmbedRangeAsync<EmbeddingF32>(snippets);

Console.WriteLine($"Query: {query}");
Console.WriteLine();

foreach (var result in EmbeddingSearch.FindClosestWithScore(queryEmbedding, snippetEmbeddings, maxResults: 3))
{
    Console.WriteLine($"{result.Similarity:0.000}  {result.Item}");
}
