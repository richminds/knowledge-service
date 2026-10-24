// Event-driven and scheduled ingestion (C-032..C-036, C-146, G-34):
//   Service Bus queue "ingestion" with dead-lettering (C-033)
//   Event Grid: blob created/deleted in the documents container -> queue (C-034)
//   Container Apps jobs on the app image (C-035):
//     ingestion-worker  scales on queue length (KEDA)     scripts/ingestion_worker.py --once
//     reconcile         nightly full sync                 scripts/reconcile.py
//     graph-maintenance nightly communities/resolution    scripts/graph_maintenance.py all
param location string
param suffix string
param uniq string
param tags object
param containerImage string
param registryServer string
param identityId string
param identityPrincipalId string
param logAnalyticsWorkspaceName string
param storageAccountName string
param documentsContainer string = 'documents'
param env array // same name/value settings as the web app
param enableGraphMaintenance bool = false

var serviceBusSender = '69a216fc-b8fb-44d8-bc22-1f3c2cd27a39'
var serviceBusReceiver = '4f6d3b9b-027b-4f4c-9142-0e5a2a2247e0'

resource bus 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' = {
  name: 'sb-${suffix}-${uniq}'
  location: location
  tags: tags
  sku: { name: 'Standard', tier: 'Standard' }
  properties: { disableLocalAuth: true, minimumTlsVersion: '1.2' }
}

resource queue 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = {
  parent: bus
  name: 'ingestion'
  properties: {
    lockDuration: 'PT5M'
    maxDeliveryCount: 5 // then dead-lettered (C-033)
    deadLetteringOnMessageExpiration: true
    defaultMessageTimeToLive: 'P7D'
  }
}

resource receiverRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(queue.id, identityPrincipalId, serviceBusReceiver)
  scope: queue
  properties: {
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', serviceBusReceiver)
  }
}

resource senderRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(queue.id, identityPrincipalId, serviceBusSender)
  scope: queue
  properties: {
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', serviceBusSender)
  }
}

// -- Event Grid: storage events -> queue (C-034) --------------------------------------------
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

resource topic 'Microsoft.EventGrid/systemTopics@2024-06-01-preview' = {
  name: 'egst-${suffix}-${uniq}'
  location: location
  tags: tags
  identity: { type: 'SystemAssigned' }
  properties: {
    source: storage.id
    topicType: 'Microsoft.Storage.StorageAccounts'
  }
}

resource topicSender 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(queue.id, topic.id, serviceBusSender)
  scope: queue
  properties: {
    principalId: topic.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', serviceBusSender)
  }
}

resource subscription 'Microsoft.EventGrid/systemTopics/eventSubscriptions@2024-06-01-preview' = {
  parent: topic
  name: 'documents-to-ingestion'
  properties: {
    deliveryWithResourceIdentity: {
      identity: { type: 'SystemAssigned' }
      destination: {
        endpointType: 'ServiceBusQueue'
        properties: { resourceId: queue.id }
      }
    }
    filter: {
      includedEventTypes: ['Microsoft.Storage.BlobCreated', 'Microsoft.Storage.BlobDeleted']
      subjectBeginsWith: '/blobServices/default/containers/${documentsContainer}/'
    }
    eventDeliverySchema: 'EventGridSchema'
  }
  dependsOn: [topicSender]
}

// -- Container Apps jobs (C-035) ------------------------------------------------------------
resource workspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' existing = {
  name: logAnalyticsWorkspaceName
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${suffix}-${uniq}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: workspace.properties.customerId
        sharedKey: workspace.listKeys().primarySharedKey
      }
    }
  }
}

var jobEnv = concat(env, [
  { name: 'SERVICEBUS_NAMESPACE', value: '${bus.name}.servicebus.windows.net' }
  { name: 'INGESTION_QUEUE_NAME', value: queue.name }
])

var jobs = [
  {
    name: 'job-ingest-${uniq}'
    trigger: 'Event'
    command: ['python', 'scripts/ingestion_worker.py', '--once']
    enabled: true
  }
  {
    name: 'job-reconcile-${uniq}'
    trigger: 'Schedule'
    cron: '0 2 * * *'
    command: ['python', 'scripts/reconcile.py']
    enabled: true
  }
  {
    name: 'job-graph-${uniq}'
    trigger: 'Schedule'
    cron: '0 3 * * *'
    command: ['python', 'scripts/graph_maintenance.py', 'all']
    enabled: enableGraphMaintenance
  }
]

resource containerJobs 'Microsoft.App/jobs@2024-10-02-preview' = [for job in jobs: if (job.enabled) {
  name: job.name
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identityId}': {} }
  }
  properties: {
    environmentId: environment.id
    configuration: {
      triggerType: job.trigger
      replicaTimeout: 3600
      replicaRetryLimit: 1
      registries: [{ server: registryServer, identity: identityId }]
      scheduleTriggerConfig: job.trigger == 'Schedule' ? { cronExpression: job.cron, parallelism: 1, replicaCompletionCount: 1 } : null
      eventTriggerConfig: job.trigger == 'Event' ? {
        parallelism: 1
        replicaCompletionCount: 1
        scale: {
          minExecutions: 0
          maxExecutions: 5 // C-147 scale-out on queue length
          pollingInterval: 30
          rules: [
            {
              name: 'servicebus-queue'
              type: 'azure-servicebus'
              metadata: {
                namespace: bus.name
                queueName: queue.name
                messageCount: '5'
              }
              identity: identityId
            }
          ]
        }
      } : null
    }
    template: {
      containers: [
        {
          name: 'worker'
          image: containerImage
          command: job.command
          env: jobEnv
          resources: { cpu: json('1.0'), memory: '2Gi' }
        }
      ]
    }
  }
}]

output serviceBusNamespace string = '${bus.name}.servicebus.windows.net'
output queueName string = queue.name
