// Alerts, budget and dashboard (C-054, C-220, C-221, I-06).
param location string
param suffix string
param tags object
param appInsightsId string
param alertEmails array
param monthlyBudget int
param budgetStartDate string // first day of a month, yyyy-MM-01

resource actionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = if (!empty(alertEmails)) {
  name: 'ag-${suffix}'
  location: 'global'
  tags: tags
  properties: {
    groupShortName: take('rag${suffix}', 12)
    enabled: true
    emailReceivers: [for (email, i) in alertEmails: { name: 'email${i}', emailAddress: email, useCommonAlertSchema: true }]
  }
}

var actions = empty(alertEmails) ? [] : [{ actionGroupId: actionGroup.id }]

// Error rate (C-221)
resource failedRequests 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'alert-${suffix}-failed-requests'
  location: 'global'
  tags: tags
  properties: {
    severity: 2
    enabled: true
    scopes: [appInsightsId]
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'failed'
          metricName: 'requests/failed'
          metricNamespace: 'microsoft.insights/components'
          operator: 'GreaterThan'
          threshold: 10
          timeAggregation: 'Count'
          criterionType: 'StaticThresholdCriterion'
        }
      ]
    }
    actions: actions
  }
}

// Latency (C-221)
resource slowRequests 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'alert-${suffix}-latency'
  location: 'global'
  tags: tags
  properties: {
    severity: 3
    enabled: true
    scopes: [appInsightsId]
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'duration'
          metricName: 'requests/duration'
          metricNamespace: 'microsoft.insights/components'
          operator: 'GreaterThan'
          threshold: 15000
          timeAggregation: 'Average'
          criterionType: 'StaticThresholdCriterion'
        }
      ]
    }
    actions: actions
  }
}

// Quality drop: share of low-confidence answers over the last hour (C-221)
resource qualityDrop 'Microsoft.Insights/scheduledQueryRules@2023-03-15-preview' = {
  name: 'alert-${suffix}-quality-drop'
  location: location
  tags: tags
  properties: {
    severity: 3
    enabled: true
    scopes: [appInsightsId]
    evaluationFrequency: 'PT30M'
    windowSize: 'PT1H'
    criteria: {
      allOf: [
        {
          query: 'customMetrics | where name == "rag.answers" | extend c = tostring(customDimensions["confidence"]) | summarize low = sumif(valueSum, c == "low"), total = sum(valueSum) | extend ratio = iff(total > 20, low / total, 0.0) | project ratio'
          timeAggregation: 'Maximum'
          metricMeasureColumn: 'ratio'
          operator: 'GreaterThan'
          threshold: json('0.3')
          failingPeriods: { numberOfEvaluationPeriods: 1, minFailingPeriodsToAlert: 1 }
        }
      ]
    }
    actions: { actionGroups: empty(alertEmails) ? [] : [actionGroup.id] }
  }
}

// Cost spikes: token usage per hour (C-221, C-218)
resource tokenSpike 'Microsoft.Insights/scheduledQueryRules@2023-03-15-preview' = {
  name: 'alert-${suffix}-token-spike'
  location: location
  tags: tags
  properties: {
    severity: 3
    enabled: true
    scopes: [appInsightsId]
    evaluationFrequency: 'PT1H'
    windowSize: 'PT1H'
    criteria: {
      allOf: [
        {
          query: 'customMetrics | where name == "rag.llm.tokens" | summarize tokens = sum(valueSum)'
          timeAggregation: 'Maximum'
          metricMeasureColumn: 'tokens'
          operator: 'GreaterThan'
          threshold: 2000000 // TODO: set from the cost budget (C-010)
          failingPeriods: { numberOfEvaluationPeriods: 1, minFailingPeriodsToAlert: 1 }
        }
      ]
    }
    actions: { actionGroups: empty(alertEmails) ? [] : [actionGroup.id] }
  }
}

// Budget for the resource group (C-054)
resource budget 'Microsoft.Consumption/budgets@2023-11-01' = if (monthlyBudget > 0 && !empty(alertEmails)) {
  name: 'budget-${suffix}'
  properties: {
    category: 'Cost'
    amount: monthlyBudget
    timeGrain: 'Monthly'
    timePeriod: { startDate: budgetStartDate }
    notifications: {
      actual80: { enabled: true, operator: 'GreaterThan', threshold: 80, contactEmails: alertEmails }
      forecast100: { enabled: true, operator: 'GreaterThan', threshold: 100, thresholdType: 'Forecasted', contactEmails: alertEmails }
    }
  }
}

// Combined dashboard (C-220, I-06): traffic split, quality, latency and cost per RAG mode
resource workbook 'Microsoft.Insights/workbooks@2023-06-01' = {
  name: guid(resourceGroup().id, 'rag-workbook', suffix)
  location: location
  tags: tags
  kind: 'shared'
  properties: {
    displayName: 'RAG agent - ${suffix}'
    category: 'workbook'
    sourceId: appInsightsId
    serializedData: string({
      version: 'Notebook/1.0'
      items: [
        { type: 1, content: { json: '## RAG agent: traffic, quality, latency and cost by mode' } }
        { type: 3, content: { version: 'KqlItem/1.0', title: 'Answers by mode and confidence', visualization: 'barchart', query: 'customMetrics | where name == "rag.answers" | summarize answers = sum(valueSum) by mode = tostring(customDimensions["rag.mode"]), confidence = tostring(customDimensions["confidence"])' } }
        { type: 3, content: { version: 'KqlItem/1.0', title: 'Judge scores by mode', visualization: 'table', query: 'customMetrics | where name == "rag.quality.score" and customDimensions["evaluator"] == "llm_judge" | summarize avg_score = sum(valueSum) / sum(valueCount) by mode = tostring(customDimensions["rag.mode"]), metric = tostring(customDimensions["quality.metric"])' } }
        { type: 3, content: { version: 'KqlItem/1.0', title: 'Latency by mode (ms)', visualization: 'timechart', query: 'customMetrics | where name == "rag.request.duration" | summarize avg_ms = sum(valueSum) / sum(valueCount) by bin(timestamp, 1h), mode = tostring(customDimensions["rag.mode"])' } }
        { type: 3, content: { version: 'KqlItem/1.0', title: 'Tokens by purpose', visualization: 'piechart', query: 'customMetrics | where name == "rag.llm.tokens" | summarize tokens = sum(valueSum) by purpose = tostring(customDimensions["llm.purpose"])' } }
        { type: 3, content: { version: 'KqlItem/1.0', title: 'Unanswered questions (content gaps, C-239)', visualization: 'table', query: 'traces | where message == "rag_answer" and tostring(customDimensions.answer_type) == "not_found" | summarize count() by collection = tostring(customDimensions.collection), bin(timestamp, 1d)' } }
      ]
    })
  }
}
