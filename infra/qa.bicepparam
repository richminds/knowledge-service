using 'main.bicep'

// Non-secret configuration for QA. Becomes App Service settings (env vars) on deploy.
// Secrets go in the environment's Key Vault (see README "Configuration & secrets").

param environmentName = 'qa'
param baseName = 'oissarag'
// Overridden by the pipeline with the image built for this run.
param containerImage = 'oissaragagentacr.azurecr.io/oissa-rag-agent:latest'
param containerRegistryName = 'oissaragagentacr'

// --- Azure OpenAI ---------------------------------------------------------------------
// TODO: set to the shared Azure OpenAI account (and openAiResourceGroup if different).
param openAiAccountName = 'aoai-oissa-qa'
// Optional URL override (e.g. API Management gateway); blank = account URL.
param openAiEndpoint = ''
param openAiChatDeployment = 'gpt-4o'
param openAiUtilityDeployment = 'gpt-4o-mini'
param openAiEmbeddingDeployment = 'text-embedding-3-large'

// --- RAG modes: QA serves them all so they can be validated side by side --------------------
param ragMode = 'hybrid'
param ragEnabledModes = 'semantic,hybrid,hybrid_graph,graph,agentic'
param ragAllowModeOverride = true
param ragEscalation = true
param ingestionTargets = 'vector,graph'
param agenticTools = 'vector,keyword,fetch_document,graph,graph_global,calculator,date_math'

// --- Answer quality: judge every answer in the background to compare modes on dashboards ----
param qualityJudge = 'request'
param qualityJudgeSampleRate = '1.0'

// --- Browser access ----------------------------------------------------------------------
param corsAllowOrigins = '[]'

// --- Security (C-182, C-183) ---------------------------------------------------------------
// api_key: X-API-Key from Key Vault secret "api-key". entra: Entra ID bearer tokens.
param authMode = 'api_key'
// TODO: Entra tenant and the API app registration (client id or App ID URI) for AUTH_MODE=entra.
param entraTenantId = ''
param entraApiAudience = ''
// TODO: app role allowed to ingest/delete content, e.g. 'Knowledge.Ingest' (blank = any caller).
param ingestionRequiredRole = ''

// --- Optional services --------------------------------------------------------------------
param enableIngestionWorkers = true
param enableContentSafety = false
// TODO: set true to OCR scanned PDFs and images (PNG/JPEG/BMP/TIFF/HEIF) with Document
// Intelligence. imageParser 'auto' falls back to the vision model (openAiVisionDeployment) for
// pictures with little text and for GIF/WebP; 'docintel' = OCR only; 'off' = skip images.
param enableDocIntelligence = false
param imageParser = 'auto'
// TODO: 'prebuilt-read' is cheaper when images are plain scans (no tables/headings needed).
param docIntelligenceImageModel = 'prebuilt-layout'
param enableLanguagePii = false
param enableRedis = false
param enableCosmos = false

// --- Operations ---------------------------------------------------------------------------
// TODO: on-call distribution list for alerts and budget notifications.
param alertEmails = []
// TODO: monthly budget in the billing currency (0 = no budget alert).
param monthlyBudget = 0
param autoscaleMaxInstances = 1
