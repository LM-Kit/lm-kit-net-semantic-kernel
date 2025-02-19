using LMKit.Model;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Services;
using Microsoft.SemanticKernel.TextGeneration;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace LMKit.SemanticKernel.TextGeneration
{
    /// <summary>
    /// Implements a text generation service using an LMKit model.
    /// This service supports both streaming and non-streaming text generation operations.
    /// </summary>
    public class LMKitTextGeneration : ITextGenerationService
    {
        private readonly LM _model;
        private readonly LMKitPromptExecutionSettings _defaultPromptExecutionSettings;

        /// <summary>
        /// Gets the attributes associated with the AI service.
        /// Returns an empty dictionary as attributes are not implemented.
        /// </summary>
        IReadOnlyDictionary<string, object> IAIService.Attributes => new Dictionary<string, object>();

        /// <summary>
        /// Asynchronously generates streaming text contents based on the specified prompt and execution settings.
        /// The generated text is provided as a sequence of <see cref="StreamingTextContent"/> instances.
        /// </summary>
        /// <param name="prompt">The prompt to generate text for.</param>
        /// <param name="executionSettings">The settings that control the text generation behavior.</param>
        /// <param name="kernel">The Semantic Kernel instance.</param>
        /// <param name="cancellationToken">A token that can be used to cancel the asynchronous operation.</param>
        /// <returns>An asynchronous stream of <see cref="StreamingTextContent"/> containing the generated text chunks.</returns>
        async IAsyncEnumerable<StreamingTextContent> ITextGenerationService.GetStreamingTextContentsAsync(
            string prompt,
            PromptExecutionSettings executionSettings,
            Kernel kernel,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var queue = new ConcurrentQueue<StreamingTextContent>();
            using var semaphore = new SemaphoreSlim(0);
            bool done = false;
            Exception backgroundException = null;

            // Event handler that enqueues text completion results and signals the semaphore.
            void AfterTextCompletion(object sender, LMKit.TextGeneration.Events.AfterTextCompletionEventArgs e)
            {
                queue.Enqueue(new StreamingTextContent(e.Text));
                semaphore.Release();
            }

            // Start the text generation in a background task.
            var backgroundTask = Task.Run(async () =>
            {
                try
                {
                    var promptExecutionSettings = new LMKitPromptExecutionSettings(_defaultPromptExecutionSettings, executionSettings);

                    var chat = new LMKit.TextGeneration.SingleTurnConversation(_model, promptExecutionSettings)
                    {
                        SystemPrompt = promptExecutionSettings.SystemPrompt
                    };

                    chat.AfterTextCompletion += AfterTextCompletion;

                    try
                    {
                        await chat.SubmitAsync(prompt, cancellationToken).ConfigureAwait(false);
                    }
                    finally
                    {
                        chat.AfterTextCompletion -= AfterTextCompletion;
                    }
                }
                catch (Exception ex)
                {
                    backgroundException = ex;
                }
                finally
                {
                    done = true;
                    semaphore.Release();
                }
            }, cancellationToken);

            // Yield streaming text content as it becomes available.
            while (!done || !queue.IsEmpty)
            {
                await semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);

                while (queue.TryDequeue(out var chunk))
                {
                    yield return chunk;
                }
            }

            // Propagate any exceptions that occurred in the background task.
            if (backgroundException is not null)
            {
                throw backgroundException;
            }
        }

        /// <summary>
        /// Asynchronously generates complete text content based on the specified prompt and execution settings.
        /// This method returns the complete generated text as a list of <see cref="TextContent"/>.
        /// </summary>
        /// <param name="prompt">The prompt to generate a completion for.</param>
        /// <param name="executionSettings">The settings that control the text generation behavior.</param>
        /// <param name="kernel">The Semantic Kernel instance.</param>
        /// <param name="cancellationToken">A token that can be used to cancel the asynchronous operation.</param>
        /// <returns>
        /// A task that represents the asynchronous operation.
        /// The task result contains a read-only list of <see cref="TextContent"/> with the generated text.
        /// </returns>
        async Task<IReadOnlyList<TextContent>> ITextGenerationService.GetTextContentsAsync(
            string prompt,
            PromptExecutionSettings executionSettings,
            Kernel kernel,
            CancellationToken cancellationToken)
        {
            var promptExecutionSettings = new LMKitPromptExecutionSettings(_defaultPromptExecutionSettings, executionSettings);

            var chat = new LMKit.TextGeneration.SingleTurnConversation(_model, promptExecutionSettings)
            {
                SystemPrompt = promptExecutionSettings.SystemPrompt
            };

            var result = await chat.SubmitAsync(prompt, cancellationToken).ConfigureAwait(false);

            return new List<TextContent> { new TextContent(result.Completion) };
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LMKitTextGeneration"/> class with the specified LMKit model.
        /// Optionally, a default prompt execution settings instance can be provided.
        /// </summary>
        /// <param name="model">The LMKit model used for text generation.</param>
        /// <param name="defaultPromptExecutionSettings">
        /// An optional instance of <see cref="LMKitPromptExecutionSettings"/> that provides default settings
        /// for text generation. If not provided, a new instance will be created using the specified model.
        /// </param>
        public LMKitTextGeneration(LM model, LMKitPromptExecutionSettings defaultPromptExecutionSettings = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _defaultPromptExecutionSettings = defaultPromptExecutionSettings ?? new LMKitPromptExecutionSettings(model);
        }
    }
}