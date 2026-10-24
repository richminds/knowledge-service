// Optional Azure AI services (C-039, C-040, C-051), keyless (Entra ID only):
//   Content Safety (Prompt Shields + harm analysis), Document Intelligence (layout/OCR),
//   Language (PII detection). The workload identity gets "Cognitive Services User".
param location string
param suffix string
param uniq string
param tags object
param principalId string
param enableContentSafety bool = false
param enableDocIntelligence bool = false
param enableLanguage bool = false

var cognitiveServicesUser = 'a97b65f3-24c7-4388-baec-2e87135dc908'
var services = [
  { enabled: enableContentSafety, kind: 'ContentSafety', name: 'cs-${suffix}-${uniq}', sku: 'S0' }
  { enabled: enableDocIntelligence, kind: 'FormRecognizer', name: 'di-${suffix}-${uniq}', sku: 'S0' }
  { enabled: enableLanguage, kind: 'TextAnalytics', name: 'lang-${suffix}-${uniq}', sku: 'S' }
]

resource accounts 'Microsoft.CognitiveServices/accounts@2024-10-01' = [for svc in services: if (svc.enabled) {
  name: svc.name
  location: location
  tags: tags
  kind: svc.kind
  sku: { name: svc.sku }
  properties: {
    customSubDomainName: svc.name // required for Entra ID auth
    disableLocalAuth: true
    publicNetworkAccess: 'Enabled' // TODO (C-049): private endpoints
  }
}]

resource access 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for (svc, i) in services: if (svc.enabled) {
  name: guid(svc.name, principalId, cognitiveServicesUser)
  scope: accounts[i]
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesUser)
  }
}]

output contentSafetyEndpoint string = enableContentSafety ? 'https://${services[0].name}.cognitiveservices.azure.com' : ''
output docIntelligenceEndpoint string = enableDocIntelligence ? 'https://${services[1].name}.cognitiveservices.azure.com' : ''
output languageEndpoint string = enableLanguage ? 'https://${services[2].name}.cognitiveservices.azure.com' : ''
