using Microsoft.SemanticKernel;
using Azure.Identity;
using Azure.Core;
using dotenv.net;

internal static class KernelFactory
{
    public static Kernel CreateKernel(string deploymentName, string endpoint)
    {
        // Use the tenant ID from the environment variable if it exists
        var tenantId = Environment.GetEnvironmentVariable("AZURE_TENANT_ID");
        var credentialOptions = new DefaultAzureCredentialOptions();
        if (!string.IsNullOrEmpty(tenantId))
        {
            credentialOptions.TenantId = tenantId;
        }
        var credential = new DefaultAzureCredential(credentialOptions);

        return Kernel.CreateBuilder()
            .AddAzureOpenAIChatCompletion(deploymentName, endpoint, credential)
            .Build();
    }   
}