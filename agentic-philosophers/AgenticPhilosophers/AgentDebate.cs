using System.Collections.ObjectModel;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Agents.Chat;
using Microsoft.SemanticKernel.Agents.OpenAI;
using Microsoft.SemanticKernel.ChatCompletion;
using Azure.Identity;
using OpenAI.Files;
using OpenAI.VectorStores;
using Resources;
using dotenv.net;

public class AgentDebate
{
    private const string PlatoFileName = "Plato.pdf";

    private const string SocratesName = "Socrates";

    private const string PlatoName = "Plato";

    private const string AristotleName = "Aristotle";           

    protected const string AssistantSampleMetadataKey = "sksample";
    protected static readonly ReadOnlyDictionary<string, string> AssistantSampleMetadata =
        new(new Dictionary<string, string>
        {
            { AssistantSampleMetadataKey, bool.TrueString }
        });


    public async Task DebateAsync(string prompt)
    {
        // State the prompt that will start the conversation between the philosophers
        Console.WriteLine(prompt);

        // Load environment variables
        LoadEnvFile();
        var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") 
             ?? throw new InvalidOperationException("Environment variable 'AZURE_OPENAI_ENDPOINT' is not set.");

        var model = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") 
            ?? throw new InvalidOperationException("Environment variable 'AZURE_OPENAI_DEPLOYMENT_NAME' is not set.");            

        // Create a kernel for the Chat Completion agents
        var kernel = KernelFactory.CreateKernel(model, endpoint);        

        // Define the agent for Socrates
        var socratesAgentPrompt = await ReadYamlFile("PromptTemplates/SocratesAgent.yaml");
        var socratesPrompt = KernelFunctionYaml.ToPromptTemplateConfig(socratesAgentPrompt);
        ChatCompletionAgent socrates = new(socratesPrompt)
        {
            Kernel = kernel
        };

        // Define the agent for Aristotle
        var aristotleAgentPrompt = await ReadYamlFile("PromptTemplates/AristotleAgent.yaml");
        var aristotlePrompt = KernelFunctionYaml.ToPromptTemplateConfig(aristotleAgentPrompt);                
        ChatCompletionAgent aristotle = new (aristotlePrompt)
        {
            Kernel = kernel
        };

        // Use the tenant ID from the environment variable if it exists
        var tenantId = Environment.GetEnvironmentVariable("AZURE_TENANT_ID");
        var credentialOptions = new DefaultAzureCredentialOptions();
        if (!string.IsNullOrEmpty(tenantId))
        {
            credentialOptions.TenantId = tenantId;
        }
        var credential = new DefaultAzureCredential(credentialOptions);

        // Create a provider for the OpenAI client
        OpenAIClientProvider provider = OpenAIClientProvider.ForAzureOpenAI(credential, new Uri(endpoint));

        // Upload Plato's file
        OpenAIFileClient fileClient = provider.Client.GetOpenAIFileClient();
        await using Stream stream = EmbeddedResource.ReadStream(PlatoFileName)!;
        OpenAIFile fileInfo = await fileClient.UploadFileAsync(stream, PlatoFileName, FileUploadPurpose.Assistants);

        // Create a vector-store
        VectorStoreClient vectorStoreClient = provider.Client.GetVectorStoreClient();
        CreateVectorStoreOperation result =
            await vectorStoreClient.CreateVectorStoreAsync(waitUntilCompleted: false,
                new VectorStoreCreationOptions()
                {
                    FileIds = { fileInfo.Id },
                    Metadata = { { AssistantSampleMetadataKey, bool.TrueString } }
                });

        // Create the agent for Plato using a template
        string platoAgentPrompt = await ReadYamlFile("PromptTemplates/PlatoAgent.yaml");
        PromptTemplateConfig templateConfig = KernelFunctionYaml.ToPromptTemplateConfig(platoAgentPrompt);        
        OpenAIAssistantAgent plato = await OpenAIAssistantAgent.CreateFromTemplateAsync(
            provider,
            capabilities : new OpenAIAssistantCapabilities(model)
            {
                Metadata = AssistantSampleMetadata
            },
            kernel: new Kernel(),
            new KernelArguments(),
            templateConfig
        );

        // Create a thread associated with a vector-store for the agent conversation.
        string threadId =
            await plato.CreateThreadAsync(
                new OpenAIThreadCreationOptions
                {
                    VectorStoreId = result.VectorStoreId,
                    Metadata = AssistantSampleMetadata,
                });        

        try
        {
            // Create a termination function
            KernelFunction terminateFunction = KernelFunctionFactory.CreateFromPrompt(
                $$$"""
                    Make sure every participant gets a chance to speak. 

                    History:

                    {{$history}}
                    """
                );

            // Create a selection function
            KernelFunction selectionFunction = KernelFunctionFactory.CreateFromPrompt(
                $$$"""
                    Your job is to determine which participant takes the next turn in a conversation according to the action of the most recent participant.
                    State only the name of the participant to take the next turn.

                    Choose only from these participants:
                    - {{{SocratesName}}}
                    - {{{PlatoName}}}
                    - {{{AristotleName}}}

                    Always follow these steps when selecting the next participant:
                    1) After user input, it is {{{SocratesName}}}'s turn to respond.
                    2) After {{{SocratesName}}} replies, it's {{{PlatoName}}}'s turn based on {{{SocratesName}}}'s response.
                    3) After {{{PlatoName}}} replies, it's {{{AristotleName}}}'s turn based on {{{SocratesName}}}'s response.
                    4) After {{{AristotleName}}} replies, it's {{{SocratesName}}}'s turn to summarize the responses and end the conversation.

                    Make sure each participant has a turn.

                    History:
                    {{$history}}
                    """
            );

            AgentGroupChat chat = new(socrates, aristotle, plato)
            {
                ExecutionSettings = new()
                {
                    TerminationStrategy = new KernelFunctionTerminationStrategy(terminateFunction, kernel)
                    {
                        Agents = [socrates],
                        ResultParser = (result) => result.GetValue<string>()?.Contains("yes", StringComparison.OrdinalIgnoreCase) ?? false,
                        HistoryVariableName = "history",
                        MaximumIterations = 4
                    },
                    SelectionStrategy = new KernelFunctionSelectionStrategy(selectionFunction, kernel)
                    {
                        AgentsVariableName = "agents",
                        HistoryVariableName = "history"
                    }
                }
            };

            chat.AddChatMessage(new ChatMessageContent(AuthorRole.User, prompt));

            await foreach (var content in chat.InvokeAsync())
            {
                Console.WriteLine();
                string color = content.AuthorName switch
                {
                    "Socrates" => "\u001b[34m", // Blue
                    "Aristotle" => "\u001b[32m", // Green
                    "Plato" => "\u001b[35m", // Magenta
                    _ => "\u001b[0m" // Default color
                };
                Console.WriteLine($"{color}[{content.AuthorName ?? "*"}]: '{content.Content}'\u001b[0m");
                Console.WriteLine();
            }
        }
        finally
        {
            // Cleanup thread and vector-store
            await plato.DeleteThreadAsync(threadId);
            await plato.DeleteAsync();
            await vectorStoreClient.DeleteVectorStoreAsync(result.VectorStoreId);
            await fileClient.DeleteFileAsync(fileInfo.Id);
        }     
    }

    private void LoadEnvFile()
    {
        string[] possiblePaths = {
            "../.env",
            ".env"
        };        

        foreach (var path in possiblePaths)
        {
            if (File.Exists(path))
            {
                DotEnv.Load(options: new DotEnvOptions(
                    ignoreExceptions: true,
                    envFilePaths: new[] { path }
                ));
                return;
            }
        }     
    }

    private async Task<string> ReadYamlFile(string filename)
    {
        var filePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, filename);

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"File '{filename}' not found at '{filePath}'.");

        return await File.ReadAllTextAsync(filePath);
    }    
}

