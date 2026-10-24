// RAG knowledge agent: per-environment (qa / prod) resources (C-028, C-029).
//
// Always:     user-assigned workload identity (C-047), Log Analytics + Application Insights
//             (C-052), Storage with source + data-lake zones (C-030, C-031), Key Vault
//             (C-046), App Service (C-045) with autoscale (C-225) and a prod staging slot for
//             blue-green releases (C-234), alerts/budget/dashboard (C-054, C-220, C-221).
// External:   MongoDB Atlas holds the chunks (vector + keyword search), the knowledge graph
//             and the ingestion manifest. Store its connection string in Key Vault as the
//             secret `mongodb-url` (the app reads it at startup); it is not provisioned here.
// Optional:   ingestion workers
//             (Service Bus, Event Grid, Container Apps jobs; C-032..C-035), AI services
//             (Content Safety, Document Intelligence, Language PII), Redis, Cosmos DB.
// Referenced: existing Azure OpenAI account and container registry (shared, quota-managed).
// All data-plane access uses Entra ID; the only stored secrets live in Key Vault.
// Not provisioned here (TODO, C-049/C-050): private endpoints/VNet integration and
// API Management / Front Door WAF in front of the App Service.

targetScope = 'resourceGroup'

@description('Deployment environment name: qa or prod')
@allowed([
  'qa'
  'prod'
])
param environmentName string

@description('Base name used to derive resource names, e.g. oissarag')
@maxLength(12)
param baseName string = 'oissarag'

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Container image to deploy, e.g. myregistry.azurecr.io/oissa-rag-agent:1234')
param containerImage string

@description('Linux App Service plan SKU (slots and autoscale need Standard/Premium)')
param appServicePlanSku string = environmentName == 'prod' ? 'P1v3' : 'B1'

@description('Existing Azure Container Registry the app pulls from')
param containerRegistryName string
param containerRegistryResourceGroup string = resourceGroup().name

@description('Existing Azure OpenAI account (chat + embedding deployments)')
param openAiAccountName string
param openAiResourceGroup string = resourceGroup().name
param openAiChatDeployment string = 'gpt-4o'
param openAiUtilityDeployment string = ''
param openAiEmbeddingDeployment string = 'text-embedding-3-large'
param openAiVisionDeployment string = ''
param openAiWhisperDeployment string = ''

@description('Override the Azure OpenAI URL (e.g. an APIM gateway). Blank = https://<account>.openai.azure.com')
param openAiEndpoint string = ''

// ---------------------------------------------------------------------------
// RAG mode switch and quality
// ---------------------------------------------------------------------------
@allowed([
  'semantic'
  'hybrid'
  'hybrid_graph'
  'graph'
  'agentic'
  'auto'
])
param ragMode string = 'hybrid'

@description('Comma-separated strategies this environment serves: semantic,hybrid,hybrid_graph,graph,agentic')
param ragEnabledModes string = 'hybrid'

@description('Allow per-request rag_mode (validation). Usually false in prod.')
param ragAllowModeOverride bool = environmentName != 'prod'

@description('Retry low-confidence answers with a more capable mode (I-03)')
param ragEscalation bool = false

@description('Stores written by ingestion by default: vector,graph')
param ingestionTargets string = 'vector'

@description('Tools the agentic strategy may use')
param agenticTools string = 'vector,keyword,fetch_document,graph,graph_global,calculator,date_math'

@allowed([
  'off'
  'request'
  'always'
])
param qualityJudge string = 'request'

@description('Share of answers judged in the background for quality dashboards (0-1, as a string)')
param qualityJudgeSampleRate string = environmentName == 'prod' ? '0.1' : '1.0'

// ---------------------------------------------------------------------------
// Security
// ---------------------------------------------------------------------------
@allowed([
  'api_key'
  'entra'
  'entra+api_key'
])
param authMode string = 'api_key'

@description('Entra tenant id and accepted audiences (API app client id / App ID URI) for AUTH_MODE=entra')
param entraTenantId string = ''
param entraApiAudience string = ''

@description('Entra app role required to ingest/delete content (blank = any authenticated caller)')
param ingestionRequiredRole string = ''

@description('Allowed browser origins for CORS (JSON array string), e.g. ["https://app.contoso.org"]')
param corsAllowOrigins string = '[]'

// ---------------------------------------------------------------------------
// Optional features
// ---------------------------------------------------------------------------
@description('Graph RAG is enabled (graph collections in MongoDB; nightly graph maintenance job)')
param enableGraphRag bool = contains(ragEnabledModes, 'graph') || contains(ragEnabledModes, 'agentic')

@description('MongoDB database for chunks, graph and manifest (connection string: Key Vault secret mongodb-url)')
param mongodbDatabase string = 'knowledge-${environmentName}'

@description('Event-driven ingestion: Service Bus + Event Grid + Container Apps jobs')
param enableIngestionWorkers bool = true
param enableContentSafety bool = environmentName == 'prod'
param enableDocIntelligence bool = false

@description('Images: auto | docintel | vision | docintel+vision | off (see IMAGE_PARSER)')
@allowed([
  'auto'
  'docintel'
  'vision'
  'docintel+vision'
  'off'
])
param imageParser string = 'auto'

@description('Document Intelligence model for images: prebuilt-layout or prebuilt-read (OCR only)')
param docIntelligenceImageModel string = 'prebuilt-layout'
param enableLanguagePii bool = false
param enableRedis bool = environmentName == 'prod'
param enableCosmos bool = false

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------
@description('Email addresses for alerts and budget notifications')
param alertEmails array = []
@description('Monthly budget for the resource group (0 = no budget)')
param monthlyBudget int = 0
param budgetStartDate string = '${utcNow('yyyy-MM')}-01'
param autoscaleMaxInstances int = environmentName == 'prod' ? 5 : 1

param tags object = {
  application: 'oissa-rag-agent'
  environment: environmentName
  owner: 'TODO-owner'
  'cost-center': 'TODO-cost-center'
}

var suffix = '${baseName}-${environmentName}'
var uniq = substring(uniqueString(resourceGroup().id, baseName, environmentName), 0, 6)
var appName = 'app-${suffix}-${uniq}'
var identityName = 'id-${suffix}-${uniq}'
var isProd = environmentName == 'prod'

// ---------------------------------------------------------------------------
// Workload identity shared by the web app and the jobs (C-047)
// ---------------------------------------------------------------------------
resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
  tags: tags
}

// ---------------------------------------------------------------------------
// Monitoring
// ---------------------------------------------------------------------------
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${suffix}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: isProd ? 90 : 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${suffix}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// ---------------------------------------------------------------------------
// Storage: source documents + data-lake zones (raw/parsed/clean/chunks/quarantine)
// ---------------------------------------------------------------------------
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: toLower('st${baseName}${environmentName}${uniq}')
  location: location
  tags: tags
  sku: { name: isProd ? 'Standard_ZRS' : 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
  properties: {
    isVersioningEnabled: true // C-031
    deleteRetentionPolicy: { enabled: true, days: 14 }
    containerDeleteRetentionPolicy: { enabled: true, days: 14 }
  }
}

resource documentsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'documents'
}

resource zonesContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'rag-zones'
}

// C-031: cool down intermediate outputs, expire quarantine and old versions
resource lifecycle 'Microsoft.Storage/storageAccounts/managementPolicies@2023-05-01' = {
  parent: storage
  name: 'default'
  properties: {
    policy: {
      rules: [
        {
          name: 'zones-to-cool'
          enabled: true
          type: 'Lifecycle'
          definition: {
            filters: { blobTypes: ['blockBlob'], prefixMatch: ['rag-zones/raw/', 'rag-zones/parsed/', 'rag-zones/clean/', 'rag-zones/chunks/'] }
            actions: { baseBlob: { tierToCool: { daysAfterModificationGreaterThan: 30 } } }
          }
        }
        {
          name: 'quarantine-expiry'
          enabled: true
          type: 'Lifecycle'
          definition: {
            filters: { blobTypes: ['blockBlob'], prefixMatch: ['rag-zones/quarantine/'] }
            actions: { baseBlob: { delete: { daysAfterModificationGreaterThan: 90 } } }
          }
        }
        {
          name: 'old-versions'
          enabled: true
          type: 'Lifecycle'
          definition: {
            filters: { blobTypes: ['blockBlob'] }
            actions: { version: { delete: { daysAfterCreationGreaterThan: 90 } } }
          }
        }
      ]
    }
  }
}

// ---------------------------------------------------------------------------
// Secrets
// ---------------------------------------------------------------------------
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv${baseName}${environmentName}${uniq}' // max 24 chars
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enableSoftDelete: true
    enablePurgeProtection: isProd ? true : null
  }
}

// ---------------------------------------------------------------------------
// Optional services
// ---------------------------------------------------------------------------
module aiServices 'modules/ai-services.bicep' = {
  name: 'ai-services-${environmentName}'
  params: {
    location: location
    suffix: suffix
    uniq: uniq
    tags: tags
    principalId: identity.properties.principalId
    enableContentSafety: enableContentSafety
    enableDocIntelligence: enableDocIntelligence
    enableLanguage: enableLanguagePii
  }
}

module dataServices 'modules/data-services.bicep' = {
  name: 'data-services-${environmentName}'
  params: {
    location: location
    suffix: suffix
    uniq: uniq
    tags: tags
    principalId: identity.properties.principalId
    keyVaultName: keyVault.name
    enableRedis: enableRedis
    enableCosmos: enableCosmos
    isProd: isProd
  }
}

// ---------------------------------------------------------------------------
// App settings (shared by the web app, its staging slot and the jobs)
// ---------------------------------------------------------------------------
var appSettings = [
  { name: 'APP_ENV', value: environmentName }
  { name: 'WEBSITES_PORT', value: '8000' }
  { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId } // user-assigned identity
  { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
  // Secrets (api-key, redis-url, ...) are read from this vault by the app at startup.
  { name: 'AZURE_KEY_VAULT_URL', value: keyVault.properties.vaultUri }
  // Security
  { name: 'AUTH_MODE', value: authMode }
  { name: 'ENTRA_TENANT_ID', value: entraTenantId }
  { name: 'ENTRA_API_AUDIENCE', value: entraApiAudience }
  { name: 'INGESTION_REQUIRED_ROLE', value: ingestionRequiredRole }
  { name: 'CORS_ALLOW_ORIGINS', value: corsAllowOrigins }
  { name: 'CONTENT_SAFETY_MODE', value: enableContentSafety ? 'azure' : 'heuristic' }
  { name: 'CONTENT_SAFETY_ENDPOINT', value: aiServices.outputs.contentSafetyEndpoint }
  // Service URLs
  { name: 'AZURE_OPENAI_ENDPOINT', value: empty(openAiEndpoint) ? 'https://${openAiAccountName}.openai.azure.com' : openAiEndpoint }
  { name: 'AZURE_STORAGE_ACCOUNT_URL', value: storage.properties.primaryEndpoints.blob }
  { name: 'DOC_INTELLIGENCE_ENDPOINT', value: aiServices.outputs.docIntelligenceEndpoint }
  { name: 'IMAGE_PARSER', value: imageParser }
  { name: 'DOC_INTELLIGENCE_IMAGE_MODEL', value: docIntelligenceImageModel }
  { name: 'LANGUAGE_ENDPOINT', value: aiServices.outputs.languageEndpoint }
  { name: 'PII_MODE', value: enableLanguagePii ? 'azure_language' : 'regex' }
  { name: 'COSMOS_ENDPOINT', value: dataServices.outputs.cosmosEndpoint }
  { name: 'CONVERSATION_STORE', value: enableCosmos ? 'cosmos' : 'memory' }
  { name: 'ANSWER_CACHE_ENABLED', value: string(enableRedis) }
  // Models
  { name: 'AZURE_OPENAI_CHAT_DEPLOYMENT', value: openAiChatDeployment }
  { name: 'AZURE_OPENAI_UTILITY_DEPLOYMENT', value: openAiUtilityDeployment }
  { name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT', value: openAiEmbeddingDeployment }
  { name: 'AZURE_OPENAI_VISION_DEPLOYMENT', value: openAiVisionDeployment }
  { name: 'AZURE_OPENAI_WHISPER_DEPLOYMENT', value: openAiWhisperDeployment }
  // RAG mode switch
  { name: 'RAG_MODE', value: ragMode }
  { name: 'RAG_ENABLED_MODES', value: ragEnabledModes }
  { name: 'RAG_ALLOW_MODE_OVERRIDE', value: string(ragAllowModeOverride) }
  { name: 'RAG_ALLOW_TRACE', value: string(!isProd) }
  { name: 'RAG_ESCALATION_ENABLED', value: string(ragEscalation) }
  { name: 'INGESTION_TARGETS', value: ingestionTargets }
  { name: 'AGENTIC_TOOLS', value: agenticTools }
  // Answer quality
  { name: 'QUALITY_JUDGE', value: qualityJudge }
  { name: 'QUALITY_JUDGE_SAMPLE_RATE', value: qualityJudgeSampleRate }
  // Ingestion bookkeeping
  { name: 'ZONES_STORE', value: 'blob' }
  { name: 'DATA_LAKE_ACCOUNT_URL', value: storage.properties.primaryEndpoints.blob }
  { name: 'MANIFEST_STORE', value: 'mongo' }
  // Vector store, graph store and manifest: MongoDB Atlas. MONGODB_URL comes from the
  // Key Vault secret mongodb-url (AZURE_KEY_VAULT_URL above).
  { name: 'MONGODB_DATABASE', value: mongodbDatabase }
]

// ---------------------------------------------------------------------------
// Compute: App Service (Linux container) + autoscale + prod staging slot
// ---------------------------------------------------------------------------
resource appServicePlan 'Microsoft.Web/serverfarms@2023-12-01' = {
  name: 'asp-${suffix}'
  location: location
  tags: tags
  sku: { name: appServicePlanSku }
  kind: 'linux'
  properties: {
    reserved: true
  }
}

var siteConfig = {
  linuxFxVersion: 'DOCKER|${containerImage}'
  acrUseManagedIdentityCreds: true
  acrUserManagedIdentityID: identity.properties.clientId
  alwaysOn: isProd
  healthCheckPath: '/api/v1/health'
  ftpsState: 'Disabled'
  minTlsVersion: '1.2'
  http20Enabled: true
  appSettings: appSettings
}

resource webApp 'Microsoft.Web/sites@2023-12-01' = {
  name: appName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  properties: {
    serverFarmId: appServicePlan.id
    httpsOnly: true
    keyVaultReferenceIdentity: identity.id
    siteConfig: siteConfig
  }
}

// C-234: blue-green - the pipeline deploys to "staging", smoke-tests it, then swaps.
resource stagingSlot 'Microsoft.Web/sites/slots@2023-12-01' = if (isProd) {
  parent: webApp
  name: 'staging'
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  properties: {
    serverFarmId: appServicePlan.id
    httpsOnly: true
    keyVaultReferenceIdentity: identity.id
    siteConfig: siteConfig
  }
}

// C-225: scale out on CPU
resource autoscale 'Microsoft.Insights/autoscalesettings@2022-10-01' = if (autoscaleMaxInstances > 1) {
  name: 'autoscale-${suffix}'
  location: location
  tags: tags
  properties: {
    enabled: true
    targetResourceUri: appServicePlan.id
    profiles: [
      {
        name: 'cpu'
        capacity: { minimum: '1', maximum: string(autoscaleMaxInstances), default: '1' }
        rules: [
          {
            metricTrigger: {
              metricName: 'CpuPercentage'
              metricResourceUri: appServicePlan.id
              timeGrain: 'PT1M'
              statistic: 'Average'
              timeWindow: 'PT10M'
              timeAggregation: 'Average'
              operator: 'GreaterThan'
              threshold: 70
            }
            scaleAction: { direction: 'Increase', type: 'ChangeCount', value: '1', cooldown: 'PT5M' }
          }
          {
            metricTrigger: {
              metricName: 'CpuPercentage'
              metricResourceUri: appServicePlan.id
              timeGrain: 'PT1M'
              statistic: 'Average'
              timeWindow: 'PT20M'
              timeAggregation: 'Average'
              operator: 'LessThan'
              threshold: 30
            }
            scaleAction: { direction: 'Decrease', type: 'ChangeCount', value: '1', cooldown: 'PT10M' }
          }
        ]
      }
    ]
  }
}

// ---------------------------------------------------------------------------
// Ingestion workers (Service Bus + Event Grid + Container Apps jobs)
// ---------------------------------------------------------------------------
module ingestion 'modules/ingestion.bicep' = if (enableIngestionWorkers) {
  name: 'ingestion-${environmentName}'
  params: {
    location: location
    suffix: suffix
    uniq: uniq
    tags: tags
    containerImage: containerImage
    registryServer: '${containerRegistryName}.azurecr.io'
    identityId: identity.id
    identityPrincipalId: identity.properties.principalId
    logAnalyticsWorkspaceName: logAnalytics.name
    storageAccountName: storage.name
    documentsContainer: documentsContainer.name
    env: filter(appSettings, s => s.name != 'WEBSITES_PORT')
    enableGraphMaintenance: enableGraphRag
  }
}

// ---------------------------------------------------------------------------
// RBAC for the workload identity (least privilege per service, C-047)
// ---------------------------------------------------------------------------
var roles = {
  storageBlobDataContributor: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // read sources, write zones
  keyVaultSecretsUser: '4633458b-17de-408a-b874-0445c86b69e6'
}

resource storageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, identity.id, roles.storageBlobDataContributor)
  scope: storage
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.storageBlobDataContributor)
  }
}

resource keyVaultRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, identity.id, roles.keyVaultSecretsUser)
  scope: keyVault
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.keyVaultSecretsUser)
  }
}

// Shared resources may live in other resource groups, so they go through modules.
module acrPull 'modules/acr-pull.bicep' = {
  name: 'acr-pull-${environmentName}'
  scope: resourceGroup(containerRegistryResourceGroup)
  params: {
    containerRegistryName: containerRegistryName
    principalId: identity.properties.principalId
  }
}

module openAiAccess 'modules/openai-access.bicep' = {
  name: 'openai-access-${environmentName}'
  scope: resourceGroup(openAiResourceGroup)
  params: {
    openAiAccountName: openAiAccountName
    principalId: identity.properties.principalId
  }
}

module monitoring 'modules/monitoring.bicep' = {
  name: 'monitoring-${environmentName}'
  params: {
    location: location
    suffix: suffix
    tags: tags
    appInsightsId: appInsights.id
    alertEmails: alertEmails
    monthlyBudget: monthlyBudget
    budgetStartDate: budgetStartDate
  }
}

output webAppName string = webApp.name
output webAppDefaultHostName string = webApp.properties.defaultHostName
output stagingHostName string = isProd ? '${webApp.name}-staging.azurewebsites.net' : ''
output storageAccountName string = storage.name
output keyVaultName string = keyVault.name
output keyVaultUri string = keyVault.properties.vaultUri
output appInsightsName string = appInsights.name
output identityClientId string = identity.properties.clientId
