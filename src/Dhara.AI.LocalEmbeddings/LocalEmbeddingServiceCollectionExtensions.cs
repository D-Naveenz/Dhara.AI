using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

namespace Dhara.AI.LocalEmbeddings;

/// <summary>
/// Provides dependency injection registration helpers for local embedding generation.
/// </summary>
public static class LocalEmbeddingServiceCollectionExtensions
{
    /// <summary>
    /// Registers <see cref="LocalEmbeddingGenerator"/> as the application's embedding generator.
    /// </summary>
    /// <param name="services">The service collection to update.</param>
    /// <param name="configure">An optional callback used to customize generator options.</param>
    /// <returns>The same <paramref name="services"/> instance for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="services"/> is <see langword="null"/>.</exception>
    public static IServiceCollection AddLocalEmbeddings(
        this IServiceCollection services,
        Action<LocalEmbeddingGeneratorOptions>? configure = null)
    {
        ArgumentNullException.ThrowIfNull(services);

        services.AddSingleton(serviceProvider =>
        {
            var options = new LocalEmbeddingGeneratorOptions();
            configure?.Invoke(options);
            return new LocalEmbeddingGenerator(options);
        });

        services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(static serviceProvider =>
            serviceProvider.GetRequiredService<LocalEmbeddingGenerator>());

        return services;
    }
}
