"""Database connector (C-061): rows rendered as text records.

URI: db://<name>, defined under `databases:` in config/sources.yaml:
    dsn: ${HR_DB_DSN}                      # PostgreSQL DSN (psycopg)
    query: SELECT id, question, answer, updated_at FROM faq
    id_column: id
    title_column: question
    text_template: "Q: {question}\nA: {answer}"
    updated_column: updated_at            # change marker (C-067)
    acl: ["*"]                            # or acl_column: allowed_groups (comma list)
Other engines (SQL Server, Oracle...) need their driver; add a branch in _rows().
"""

from collections.abc import AsyncIterator

from app.rag.ingestion.connectors.base import Connector
from app.rag.ingestion.connectors.sources_config import source_definition
from app.rag.ingestion.models import SourceDocument


class DatabaseConnector(Connector):
    schemes = ("db://",)

    def __init__(self, sources_path: str) -> None:
        self._sources_path = sources_path

    async def _rows(self, definition: dict) -> AsyncIterator[dict]:
        import psycopg
        from psycopg.rows import dict_row

        async with await psycopg.AsyncConnection.connect(definition["dsn"]) as conn:
            async with conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(definition["query"])
                async for row in cursor:
                    yield row

    async def iterate(self, uri: str) -> AsyncIterator[SourceDocument]:
        name = uri.removeprefix("db://").strip("/")
        definition = source_definition(self._sources_path, "databases", name)
        template = definition.get("text_template") or "{__all__}"
        async for row in self._rows(definition):
            values = {k: "" if v is None else str(v) for k, v in row.items()}
            text = (
                "\n".join(f"{k}: {v}" for k, v in values.items())
                if template == "{__all__}"
                else template.format_map(values)
            )
            row_id = values[definition.get("id_column", "id")]
            acl_column = definition.get("acl_column")
            acl = (
                [a.strip() for a in values.get(acl_column, "").split(",") if a.strip()]
                if acl_column
                else list(definition.get("acl") or [])
            )
            yield SourceDocument(
                source=f"db://{name}/{row_id}",
                name=f"{name}-{row_id}.txt",
                content=text.encode(),
                content_type="text/plain",
                acl=acl,
                etag=values.get(definition.get("updated_column", ""), ""),
                last_modified=values.get(definition.get("updated_column", "")) or None,
                metadata={
                    "title": values.get(definition.get("title_column", ""), ""),
                    "table_source": name,
                },
            )
