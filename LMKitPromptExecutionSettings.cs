using LMKit.Model;
using LMKit.TextGeneration;
using LMKit.TextGeneration.Sampling;
using Microsoft.SemanticKernel;
using System.Text;
using System.Text.Json;

namespace LMKit.SemanticKernel
{
    /// <summary>
    /// Provides a bridge between LMKit.NET text generation settings and Microsoft Semantic Kernel prompt execution settings.
    /// This class extends <see cref="PromptExecutionSettings"/> and implements <see cref="ITextGenerationSettings"/>,
    /// allowing for the configuration of various text generation parameters such as sampling, repetition penalties,
    /// stop sequences, grammar, logit bias, and maximum token counts.
    /// </summary>
    public class LMKitPromptExecutionSettings : PromptExecutionSettings, ITextGenerationSettings
    {
        private const string DEFAULT_SYSTEM_PROMPT = "You are a chatbot that always responds promptly and helpfully to user requests.";
        private const int DEFAULT_MAX_COMPLETION_TOKENS = 512;

        private RepetitionPenalty _repetitionPenalty;
        private List<string> _stopSequences;
        private TokenSampling _sampling;
        private Grammar _grammar;
        private LogitBias _logitBias;
        private int _maximumCompletionTokens;
        private int _resultsPerPrompt;
        private string _systemPrompt;

        /// <summary>
        /// Specifies the system prompt applied to the model before forwarding the user's requests.
        /// <para>
        /// The default value is 
        /// "You are a chatbot that always responds promptly and helpfully to user requests."
        /// </para>
        /// </summary>
        public string SystemPrompt
        {
            get => _systemPrompt;
            set => _systemPrompt = value ?? DEFAULT_SYSTEM_PROMPT;
        }

        /// <summary>
        /// Gets or sets the sampling mode used for token selection during text generation.
        /// </summary>
        public TokenSampling SamplingMode
        {
            get => _sampling;
            set => _sampling = value ?? throw new ArgumentNullException(nameof(value), "SamplingMode cannot be null.");
        }

        /// <summary>
        /// Gets the repetition penalty settings applied to reduce repeated token outputs during generation.
        /// </summary>
        public RepetitionPenalty RepetitionPenalty => _repetitionPenalty;

        /// <summary>
        /// Gets the list of stop sequences that signal the text generation to halt.
        /// </summary>
        public List<string> StopSequences => _stopSequences;

        /// <summary>
        /// Gets or sets the grammar rules applied during text generation.
        /// </summary>
        public Grammar Grammar
        {
            get => _grammar;
            set => _grammar = value;
        }

        /// <summary>
        /// Gets or sets the number of results to generate per prompt.
        /// </summary>
        public int ResultsPerPrompt
        {
            get => _resultsPerPrompt;
            set
            {
                if (value < 1)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "ResultsPerPrompt must be at least 1.");
                }
                _resultsPerPrompt = value;
            }
        }

        /// <summary>
        /// Gets the logit bias settings that influence token selection probabilities during generation.
        /// </summary>
        public LogitBias LogitBias => _logitBias;

        /// <summary>
        /// Gets or sets the maximum number of tokens to be generated in a single completion.
        /// </summary>
        public int MaximumCompletionTokens
        {
            get => _maximumCompletionTokens;
            set
            {
                if (value < 1)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "MaximumCompletionTokens must be at least 1.");
                }
                _maximumCompletionTokens = value;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LMKitPromptExecutionSettings"/> class using the specified LM model.
        /// </summary>
        /// <param name="model">The LM model to be used.</param>
        public LMKitPromptExecutionSettings(LM model)
        {
            _repetitionPenalty = new RepetitionPenalty();
            _stopSequences = new List<string>();
            _sampling = new RandomSampling();
            _logitBias = new LogitBias(model);
            _maximumCompletionTokens = DEFAULT_MAX_COMPLETION_TOKENS;
            _resultsPerPrompt = 1;
            _systemPrompt = DEFAULT_SYSTEM_PROMPT;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LMKitPromptExecutionSettings"/> class by cloning default settings
        /// and optionally overriding them with additional prompt execution settings.
        /// </summary>
        /// <param name="defaultSettings">The default settings to clone.</param>
        /// <param name="promptExecutionSettings">The prompt execution settings to override with.</param>
        internal LMKitPromptExecutionSettings(LMKitPromptExecutionSettings defaultSettings, PromptExecutionSettings promptExecutionSettings)
        {
            _repetitionPenalty = defaultSettings.RepetitionPenalty.Clone();
            _stopSequences = new List<string>(defaultSettings.StopSequences);
            _sampling = defaultSettings.SamplingMode.Clone();
            _logitBias = defaultSettings.LogitBias.Clone();
            _maximumCompletionTokens = defaultSettings.MaximumCompletionTokens;
            _resultsPerPrompt = defaultSettings.ResultsPerPrompt;
            _systemPrompt = defaultSettings.SystemPrompt;
            _grammar = defaultSettings.Grammar; 

            if (promptExecutionSettings != null)
            {
                OverrideSettings(promptExecutionSettings);
            }
        }

        /// <summary>
        /// Overrides the current settings with the properties provided in the specified prompt execution settings.
        /// </summary>
        /// <param name="promptExecutionSettings">The settings to override with.</param>
        /// <exception cref="InvalidOperationException">
        /// Thrown if an override is attempted on an unsupported sampling type or if JSON parsing fails.
        /// </exception>
        private void OverrideSettings(PromptExecutionSettings promptExecutionSettings)
        {
            try
            {
                // Serialize the provided prompt execution settings to JSON.
                var json = JsonSerializer.Serialize(promptExecutionSettings);
                ReadOnlySpan<byte> jsonSpan = Encoding.UTF8.GetBytes(json);
                Utf8JsonReader reader = new Utf8JsonReader(jsonSpan);
                JsonSerializerOptions options = new JsonSerializerOptions();

                // Read and apply each setting from the JSON.
                while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
                {
                    if (reader.TokenType == JsonTokenType.PropertyName)
                    {
                        var propertyName = reader.GetString()?.ToUpperInvariant();
                        reader.Read();

                        switch (propertyName)
                        {
                            case "MODELID":
                            case "MODEL_ID":
                                ModelId = reader.GetString();
                                break;

                            case "TEMPERATURE":
                                if (!(_sampling is RandomSampling))
                                {
                                    _sampling = new RandomSampling()
                                    {
                                        Temperature = reader.GetSingle()
                                    };
                                }
                                else
                                {
                                    ((RandomSampling)_sampling).Temperature = reader.GetSingle();
                                }
                                break;

                            case "TOPP":
                            case "TOP_P":
                                if (!(_sampling is RandomSampling))
                                {
                                    _sampling = new RandomSampling()
                                    {
                                        Temperature = 0
                                    };
                                }
                                ((RandomSampling)_sampling).TopP = reader.GetSingle();

                                break;

                            case "TOPK":
                            case "TOP_K":
                                if (!(_sampling is RandomSampling))
                                {
                                    _sampling = new RandomSampling()
                                    {
                                        Temperature = 0
                                    };
                                }
                                ((RandomSampling)_sampling).TopK = reader.GetInt32();
                                break;

                            case "FREQUENCYPENALTY":
                            case "FREQUENCY_PENALTY":
                                _repetitionPenalty.FrequencyPenalty = reader.GetSingle();
                                break;

                            case "PRESENCEPENALTY":
                            case "PRESENCE_PENALTY":
                                _repetitionPenalty.PresencePenalty = reader.GetSingle();
                                break;

                            case "REPEATPENALTY":
                            case "REPEAT_PENALTY":
                                _repetitionPenalty.RepeatPenalty = reader.GetSingle();
                                break;

                            case "MAXTOKENS":
                            case "MAX_TOKENS":
                                MaximumCompletionTokens = reader.GetInt32();
                                break;

                            case "STOPSEQUENCES":
                            case "STOP_SEQUENCES":
                                var stopSeqs = JsonSerializer.Deserialize<IList<string>>(ref reader, options) ?? Array.Empty<string>();
                                _stopSequences = new List<string>(stopSeqs);
                                break;

                            case "RESULTSPERPROMPT":
                            case "RESULTS_PER_PROMPT":
                                ResultsPerPrompt = reader.GetInt32();
                                break;

                            case "TOKENSELECTIONBIASES":
                            case "TOKEN_SELECTION_BIASES":
                                var tokenSelectionBiases = JsonSerializer.Deserialize<IDictionary<int, int>>(ref reader, options)
                                    ?? new Dictionary<int, int>();
                                foreach (var item in tokenSelectionBiases)
                                {
                                    _logitBias.BiasWeights[item.Key] = item.Value;
                                }
                                break;

                            default:
                                reader.Skip();
                                break;
                        }
                    }
                }
            }
            catch (JsonException ex)
            {
                throw new InvalidOperationException("Failed to parse prompt execution settings JSON.", ex);
            }
        }
    }
}