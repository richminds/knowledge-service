# Getting started with the knowledge agent

This sample document lets you smoke-test ingestion and retrieval locally.

## Submitting a new ingestion source

Call `POST /api/v1/ingestion` with a `source_uri` that is either a path under the
ingestion root (for example `sample/`) or a blob location such as
`blob://documents/policies/`. The call returns a job id; poll
`GET /api/v1/ingestion/{job_id}` until the status is `completed`.

To ingest a single file interactively, upload it to `POST /api/v1/ingestion/upload`.

## Supported file formats

Plain text, Markdown, CSV, JSON and PDF are parsed natively. Word (.docx),
PowerPoint (.pptx), Excel (.xlsx), HTML and email files are parsed with the
`unstructured` library.
