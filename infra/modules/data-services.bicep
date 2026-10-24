// Optional data services:
//   Redis (C-044): answer cache, embedding cache, rate limits. Its connection URL is written
//                  to Key Vault as secret "redis-url" -> the app's REDIS_URL setting.
//   Cosmos DB (C-042): conversation history (C-157, C-191), serverless, Entra ID only,
//                  continuous backup (C-055).
param location string
param suffix string
param uniq string
param tags object
param principalId string
param keyVaultName string
param enableRedis bool = false
param enableCosmos bool = false
param isProd bool = false

resource redis 'Microsoft.Cache/redis@2024-03-01' = if (enableRedis) {
  name: 'redis-${suffix}-${uniq}'
  location: location
  tags: tags
  properties: {
    sku: { name: isProd ? 'Standard' : 'Basic', family: 'C', capacity: isProd ? 1 : 0 }
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
  }
}

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource redisSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = if (enableRedis) {
  parent: vault
  name: 'redis-url'
  properties: {
    value: enableRedis ? 'rediss://:${redis!.listKeys().primaryKey}@${redis!.properties.hostName}:6380/0' : ''
  }
}

resource cosmos 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' = if (enableCosmos) {
  name: 'cosmos-${suffix}-${uniq}'
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    disableLocalAuth: true
    capabilities: [{ name: 'EnableServerless' }]
    locations: [{ locationName: location, failoverPriority: 0 }]
    backupPolicy: { type: 'Continuous', continuousModeProperties: { tier: 'Continuous7Days' } }
    minimalTlsVersion: 'Tls12'
  }
}

resource cosmosDb 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2024-05-15' = if (enableCosmos) {
  parent: cosmos
  name: 'rag'
  properties: { resource: { id: 'rag' } }
}

resource conversations 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = if (enableCosmos) {
  parent: cosmosDb
  name: 'conversations'
  properties: {
    resource: {
      id: 'conversations'
      partitionKey: { paths: ['/user_id'], kind: 'Hash' }
      defaultTtl: isProd ? 7776000 : 2592000 // C-011 retention: 90 / 30 days
    }
  }
}

// Cosmos DB Built-in Data Contributor (data plane RBAC)
resource cosmosAccess 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = if (enableCosmos) {
  parent: cosmos
  name: guid(cosmos.id, principalId, 'data-contributor')
  properties: {
    principalId: principalId
    roleDefinitionId: '${cosmos.id}/sqlRoleDefinitions/00000000-0000-0000-0000-000000000002'
    scope: cosmos.id
  }
}

output cosmosEndpoint string = enableCosmos ? cosmos!.properties.documentEndpoint : ''
