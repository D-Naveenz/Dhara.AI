using Dhara.AI.Inference;

namespace Dhara.AI.LocalEmbeddings.Test;

public sealed class BertEmbeddingTokenizerTests
{
    [Fact]
    public void Constructor_RejectsMissingVocabulary()
    {
        var exception = Assert.Throws<FileNotFoundException>(() =>
            new BertEmbeddingTokenizer("missing-vocab.txt", maxTokens: 16, lowerCaseBeforeTokenization: true));

        Assert.Equal("missing-vocab.txt", exception.FileName);
    }

    [Fact]
    public void Tokenize_CreatesPaddedBatchWithAttentionMask()
    {
        using var vocabulary = TestVocabulary.Create();
        var tokenizer = new BertEmbeddingTokenizer(vocabulary.Path, maxTokens: 8, lowerCaseBeforeTokenization: true);

        var batch = tokenizer.Tokenize(["hello", "hello world"]);

        Assert.Equal(2, batch.BatchSize);
        Assert.True(batch.SequenceLength >= 3);
        Assert.Equal(batch.InputIds.Length, batch.AttentionMask.Length);
        Assert.Equal(batch.InputIds.Length, batch.TokenTypeIds.Length);
        Assert.Equal(1, batch.AttentionMask[0]);
        Assert.Contains(0, batch.AttentionMask);
    }

    private sealed class TestVocabulary : IDisposable
    {
        private readonly string _directory;

        private TestVocabulary(string directory, string path)
        {
            _directory = directory;
            Path = path;
        }

        public string Path { get; }

        public static TestVocabulary Create()
        {
            var directory = System.IO.Path.Combine(System.IO.Path.GetTempPath(), System.IO.Path.GetRandomFileName());
            Directory.CreateDirectory(directory);
            var path = System.IO.Path.Combine(directory, "vocab.txt");
            File.WriteAllLines(path, ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]);
            return new TestVocabulary(directory, path);
        }

        public void Dispose()
            => Directory.Delete(_directory, recursive: true);
    }
}
