using LMKit.Integrations.SemanticKernel.ChatCompletion;
using LMKit.Integrations.SemanticKernel.Embeddings;
using LMKit.Integrations.SemanticKernel.TextGeneration;
using LMKit.Model;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.TextGeneration;

namespace LMKit.Integrations.SemanticKernel
{
    /// <summary>
    /// Provides extension methods for registering LMKit services—chat completion, text generation,
    /// and text embedding generation—with a Semantic Kernel <see cref="IKernelBuilder"/>.
    /// </summary>
    public static class KernelBuilderExtension
    {
        /// <summary>
        /// Adds the LMKit chat completion service to the kernel builder using the specified LM model.
        /// This method creates an instance of <see cref="LMKitChatCompletion"/> with the provided model and registers it.
        /// </summary>
        /// <param name="builder">The kernel builder to which the chat completion service is added.</param>
        /// <param name="model">The LMKit model used for chat completion.</param>
        /// <param name="defaultPromptExecutionSettings">
        /// An optional instance of <see cref="LMKitPromptExecutionSettings"/> that provides default settings
        /// for text generation. If not provided, a new instance will be created using the specified model.
        /// </param>
        public static void AddLMKitChatCompletion(this IKernelBuilder builder, LM model, LMKitPromptExecutionSettings defaultPromptExecutionSettings = null)
        {
            AddLMKitChatCompletion(builder, new LMKitChatCompletion(model, defaultPromptExecutionSettings));
        }

        /// <summary>
        /// Adds the specified LMKit chat completion instance to the kernel builder.
        /// The instance is registered as a singleton service implementing <see cref="IChatCompletionService"/>.
        /// </summary>
        /// <param name="builder">The kernel builder to which the chat completion service is added.</param>
        /// <param name="chatCompletion">An instance of <see cref="LMKitChatCompletion"/> to register.</param>
        public static void AddLMKitChatCompletion(this IKernelBuilder builder, LMKitChatCompletion chatCompletion)
        {
            builder.Services.AddSingleton<IChatCompletionService>(chatCompletion);
        }

        /// <summary>
        /// Adds the LMKit text generation service to the kernel builder using the specified LM model.
        /// This method creates an instance of <see cref="LMKitTextGeneration"/> with the provided model and registers it.
        /// </summary>
        /// <param name="builder">The kernel builder to which the text generation service is added.</param>
        /// <param name="model">The LMKit model used for text generation.</param>
        /// <param name="defaultPromptExecutionSettings">
        /// An optional instance of <see cref="LMKitPromptExecutionSettings"/> that provides default settings
        /// for text generation. If not provided, a new instance will be created using the specified model.
        /// </param>
        public static void AddLMKitTextGeneration(this IKernelBuilder builder, LM model, LMKitPromptExecutionSettings defaultPromptExecutionSettings = null)
        {
            AddLMKitTextGeneration(builder, new LMKitTextGeneration(model, defaultPromptExecutionSettings));
        }

        /// <summary>
        /// Adds the specified LMKit text generation instance to the kernel builder.
        /// The instance is registered as a singleton service implementing <see cref="ITextGenerationService"/>.
        /// </summary>
        /// <param name="builder">The kernel builder to which the text generation service is added.</param>
        /// <param name="textGeneration">An instance of <see cref="LMKitTextGeneration"/> to register.</param>
        public static void AddLMKitTextGeneration(this IKernelBuilder builder, LMKitTextGeneration textGeneration)
        {
            builder.Services.AddSingleton<ITextGenerationService>(textGeneration);
        }

        /// <summary>
        /// Adds the LMKit text embedding generation service to the kernel builder using the specified LM model.
        /// This method creates an instance of <see cref="LMKitTextEmbeddingGeneration"/> with the provided model and registers it.
        /// </summary>
        /// <param name="builder">The kernel builder to which the text embedding generation service is added.</param>
        /// <param name="model">The LMKit model used for text embedding generation.</param>
        public static void AddLMKitTextEmbeddingGeneration(this IKernelBuilder builder, LM model)
        {
            AddLMKitTextEmbeddingGeneration(builder, new LMKitTextEmbeddingGeneration(model));
        }

        /// <summary>
        /// Adds the specified LMKit text embedding generation instance to the kernel builder.
        /// The instance is registered as a singleton service implementing <see cref="ITextEmbeddingGenerationService"/>.
        /// </summary>
        /// <param name="builder">The kernel builder to which the text embedding generation service is added.</param>
        /// <param name="textEmbeddingGeneration">An instance of <see cref="LMKitTextEmbeddingGeneration"/> to register.</param>
        public static void AddLMKitTextEmbeddingGeneration(this IKernelBuilder builder, LMKitTextEmbeddingGeneration textEmbeddingGeneration)
        {
#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
            builder.Services.AddSingleton<ITextEmbeddingGenerationService>(textEmbeddingGeneration);
#pragma warning restore SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
        }
    }
}