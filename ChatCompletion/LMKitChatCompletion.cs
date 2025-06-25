using LMKit.Model;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Services;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace LMKit.Integrations.SemanticKernel.ChatCompletion
{
    /// <summary>
    /// Implements a chat completion service using the LMKit model.
    /// This service supports both synchronous and streaming chat completions.
    /// </summary>
    public sealed class LMKitChatCompletion : IChatCompletionService
    {
        private readonly LM _model;
        private readonly LMKitPromptExecutionSettings _defaultPromptExecutionSettings;

        /// <summary>
        /// Gets the attributes associated with the AI service.
        /// Returns an empty dictionary.
        /// </summary>
        IReadOnlyDictionary<string, object> IAIService.Attributes => new Dictionary<string, object>();

        /// <summary>
        /// Asynchronously retrieves chat message contents based on the provided chat history and execution settings.
        /// </summary>
        /// <param name="chatHistory">The chat history used to generate a chat completion.</param>
        /// <param name="executionSettings">Settings that control the execution of the prompt.</param>
        /// <param name="kernel">The Semantic Kernel instance invoking this service.</param>
        /// <param name="cancellationToken">A token that can be used to cancel the operation.</param>
        /// <returns>
        /// A task that represents the asynchronous operation.
        /// The task result contains a read-only list of <see cref="ChatMessageContent"/> representing the chat completion.
        /// </returns>
        async Task<IReadOnlyList<ChatMessageContent>> IChatCompletionService.GetChatMessageContentsAsync(
            ChatHistory chatHistory,
            PromptExecutionSettings executionSettings,
            Kernel kernel,
            CancellationToken cancellationToken)
        {
            var promptExecutionSettings = new LMKitPromptExecutionSettings(_defaultPromptExecutionSettings, executionSettings);

            using var chat = new LMKit.TextGeneration.MultiTurnConversation(
                _model,
                ToLMKitChatHistory(chatHistory, promptExecutionSettings),
                -1,
                promptExecutionSettings);

            var result = await chat.RegenerateResponseAsync(cancellationToken).ConfigureAwait(false);
            return new List<ChatMessageContent>
            {
                new(AuthorRole.Assistant, result.Completion)
            };
        }

        /// <summary>
        /// Asynchronously streams chat message contents based on the provided chat history and execution settings.
        /// </summary>
        /// <param name="chatHistory">The chat history used to generate a chat completion.</param>
        /// <param name="executionSettings">Settings that control the execution of the prompt.</param>
        /// <param name="kernel">The Semantic Kernel instance invoking this service.</param>
        /// <param name="cancellationToken">A token that can be used to cancel the operation.</param>
        /// <returns>
        /// An asynchronous stream of <see cref="StreamingChatMessageContent"/> that yields chat completion messages as they become available.
        /// </returns>
        async IAsyncEnumerable<StreamingChatMessageContent> IChatCompletionService.GetStreamingChatMessageContentsAsync(
            ChatHistory chatHistory,
            PromptExecutionSettings executionSettings,
            Kernel kernel,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var queue = new ConcurrentQueue<StreamingChatMessageContent>();
            using var semaphore = new SemaphoreSlim(0);
            bool done = false;
            Exception backgroundException = null;

            // Event handler that enqueues chat message content and signals the semaphore.
            void AfterTextCompletion(object sender, LMKit.TextGeneration.Events.AfterTextCompletionEventArgs e)
            {
                queue.Enqueue(new StreamingChatMessageContent(AuthorRole.Assistant, e.Text));
                semaphore.Release();
            }

            // Run the chat completion in a background task.
            _ = Task.Run(async () =>
            {
                try
                {
                    var promptExecutionSettings = new LMKitPromptExecutionSettings(_defaultPromptExecutionSettings, executionSettings);

                    using var chat = new LMKit.TextGeneration.MultiTurnConversation(
                        _model,
                        ToLMKitChatHistory(chatHistory, promptExecutionSettings),
                        -1,
                        promptExecutionSettings);

                    chat.AfterTextCompletion += AfterTextCompletion;
                    try
                    {
                        await chat.RegenerateResponseAsync(cancellationToken).ConfigureAwait(false);
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

            // Yield streaming chat message content as it becomes available.
            while (!done || !queue.IsEmpty)
            {
                await semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
                while (queue.TryDequeue(out var chunk))
                {
                    yield return chunk;
                }
            }

            // Propagate any exception that occurred in the background task.
            if (backgroundException is not null)
            {
                throw backgroundException;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LMKitChatCompletion"/> class with the specified LMKit model.
        /// Optionally, default prompt execution settings can be provided.
        /// </summary>
        /// <param name="model">The LMKit model used for generating chat completions.</param>
        /// <param name="defaultPromptExecutionSettings">
        /// Optional default prompt execution settings. If not provided, a new instance will be created based on the model.
        /// </param>
        public LMKitChatCompletion(LM model, LMKitPromptExecutionSettings defaultPromptExecutionSettings = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _defaultPromptExecutionSettings = defaultPromptExecutionSettings ?? new LMKitPromptExecutionSettings(model);
        }

        private LMKit.TextGeneration.Chat.ChatHistory ToLMKitChatHistory(ChatHistory chatHistory, LMKitPromptExecutionSettings promptExecutionSettings)
        {
            var lmkitChatHistory = new LMKit.TextGeneration.Chat.ChatHistory(_model);

            // Only add the system prompt if there are messages and the first message is not a system message.
            if (chatHistory.Any() && !string.IsNullOrWhiteSpace(promptExecutionSettings.SystemPrompt) && chatHistory.First().Role != AuthorRole.System)
            {
                lmkitChatHistory.AddMessage(LMKit.TextGeneration.Chat.AuthorRole.System, promptExecutionSettings.SystemPrompt);
            }

            foreach (var message in chatHistory)
            {
                if (message.Role == AuthorRole.Assistant)
                {
                    lmkitChatHistory.AddMessage(LMKit.TextGeneration.Chat.AuthorRole.Assistant, message.InnerContent.ToString());
                }
                else if (message.Role == AuthorRole.System)
                {
                    lmkitChatHistory.AddMessage(LMKit.TextGeneration.Chat.AuthorRole.System, message.InnerContent.ToString());
                }
                else if (message.Role == AuthorRole.User)
                {
                    lmkitChatHistory.AddMessage(LMKit.TextGeneration.Chat.AuthorRole.User, message.Content);
                }
                else
                {
                    throw new NotImplementedException($"Unsupported role: {message.Role.Label}");
                }
            }

            return lmkitChatHistory;
        }
    }
}