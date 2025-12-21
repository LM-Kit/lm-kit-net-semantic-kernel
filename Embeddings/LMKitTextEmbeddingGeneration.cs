using LMKit.Embeddings;
using LMKit.Model;
using Microsoft.Extensions.AI;

namespace LMKit.Integrations.SemanticKernel.Embeddings
{
    /// <summary>
    /// Provides a text embedding generation service using LMKit's <see cref="Embedder"/>. 
    /// This service implements the <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> interface 
    /// to generate embeddings from text data for use with Microsoft Semantic Kernel.
    /// </summary>
    public sealed class LMKitTextEmbeddingGeneration : IEmbeddingGenerator<string, Embedding<float>>
    {
        private readonly Embedder _embedder;
        private readonly EmbeddingGeneratorMetadata _metadata;

        /// <inheritdoc/>
        public EmbeddingGeneratorMetadata Metadata => _metadata;

        /// <summary>
        /// Asynchronously generates embeddings for a collection of text inputs.
        /// </summary>
        /// <param name="values">A collection of text strings for which embeddings are to be generated.</param>
        /// <param name="options">Optional embedding generation options.</param>
        /// <param name="cancellationToken">A token that can be used to cancel the embedding generation operation.</param>
        /// <returns>
        /// A task that represents the asynchronous operation. The task result contains 
        /// <see cref="GeneratedEmbeddings{TEmbedding}"/> with embedding vectors corresponding to the input texts.
        /// </returns>
        public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
            IEnumerable<string> values,
            EmbeddingGenerationOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            var inputList = values as IList<string> ?? values.ToList();
            var results = await _embedder.GetEmbeddingsAsync(inputList, cancellationToken);

            var embeddings = new GeneratedEmbeddings<Embedding<float>>();
            foreach (var result in results)
            {
                embeddings.Add(new Embedding<float>(result));
            }

            return embeddings;
        }

        /// <inheritdoc/>
        public object? GetService(Type serviceType, object? serviceKey = null)
        {
            if (serviceKey is not null)
            {
                return null;
            }

            if (serviceType == typeof(EmbeddingGeneratorMetadata))
            {
                return _metadata;
            }

            if (serviceType?.IsInstanceOfType(this) == true)
            {
                return this;
            }

            return null;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            // Embedder disposal is managed externally or by the LM model
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LMKitTextEmbeddingGeneration"/> class using the specified LMKit model.
        /// This constructor creates an internal <see cref="Embedder"/> instance based on the provided model.
        /// </summary>
        /// <param name="model">The LMKit model used to instantiate the embedder for generating text embeddings.</param>
        internal LMKitTextEmbeddingGeneration(LM model)
        {
            _embedder = new Embedder(model);
            _metadata = new EmbeddingGeneratorMetadata("LMKit", defaultModelId: model.Name);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LMKitTextEmbeddingGeneration"/> class with the provided embedder.
        /// </summary>
        /// <param name="embedder">An instance of <see cref="Embedder"/> used to generate text embeddings.</param>
        public LMKitTextEmbeddingGeneration(Embedder embedder)
        {
            _embedder = embedder;
            _metadata = new EmbeddingGeneratorMetadata("LMKit");
        }
    }
}