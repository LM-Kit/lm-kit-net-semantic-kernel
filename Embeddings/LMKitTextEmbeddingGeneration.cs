using LMKit.Embeddings;
using LMKit.Model;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Services;

namespace LMKit.SemanticKernel.Embeddings
{
#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
    /// <summary>
    /// Provides a text embedding generation service using LMKit's <see cref="Embedder"/>. 
    /// This service implements the <see cref="ITextEmbeddingGenerationService"/> interface to generate embeddings from text data for use with Microsoft Semantic Kernel.
    /// </summary>
    public sealed class LMKitTextEmbeddingGeneration : ITextEmbeddingGenerationService
#pragma warning restore SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
    {
        private Embedder _embedder;

        /// <inheritdoc/>
        IReadOnlyDictionary<string, object> IAIService.Attributes => throw new NotImplementedException();

        /// <summary>
        /// Asynchronously generates embeddings for a collection of text inputs.
        /// </summary>
        /// <param name="data">A list of text strings for which embeddings are to be generated.</param>
        /// <param name="kernel">The Semantic Kernel instance invoking this service.</param>
        /// <param name="cancellationToken">A token that can be used to cancel the embedding generation operation.</param>
        /// <returns>
        /// A task that represents the asynchronous operation. The task result contains a list of <see cref="ReadOnlyMemory{T}"/> instances,
        /// where each element represents the embedding vector corresponding to an input text.
        /// </returns>
        async Task<IList<ReadOnlyMemory<float>>> IEmbeddingGenerationService<string, float>.GenerateEmbeddingsAsync(
            IList<string> data,
            Kernel kernel,
            CancellationToken cancellationToken)
        {
            // The following code awaits the embedding generation from the embedder
            // and converts the result into the expected list format.
            return [.. await _embedder.GetEmbeddingsAsync(data, cancellationToken)];
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LMKitTextEmbeddingGeneration"/> class using the specified LMKit model.
        /// This constructor creates an internal <see cref="Embedder"/> instance based on the provided model.
        /// </summary>
        /// <param name="model">The LMKit model used to instantiate the embedder for generating text embeddings.</param>
        internal LMKitTextEmbeddingGeneration(LM model)
        {
            _embedder = new Embedder(model);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LMKitTextEmbeddingGeneration"/> class with the provided embedder.
        /// </summary>
        /// <param name="embedder">An instance of <see cref="Embedder"/> used to generate text embeddings.</param>
        public LMKitTextEmbeddingGeneration(Embedder embedder)
        {
            _embedder = embedder;
        }
    }
}