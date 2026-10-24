import logging

from app.core.logging import StructlogToOtelHandler, get_logger

logger = get_logger(__name__)


def configure_telemetry(connection_string: str, service_name: str, environment: str) -> None:
    """Send traces, metrics and logs to Application Insights via OpenTelemetry.

    The Azure Monitor distro auto-instruments FastAPI and outgoing HTTP calls (Azure OpenAI,
    MongoDB via pymongo shows as dependencies), exports the custom RAG metrics
    from app.core.metrics and the RAG step spans, and forwards application logs.
    No-op when the connection string is unset (local dev, tests): the OpenTelemetry API
    calls in the code then do nothing.
    """
    if not connection_string:
        return
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry.sdk.resources import Resource

    root = logging.getLogger()
    before = set(root.handlers)
    configure_azure_monitor(
        connection_string=connection_string,
        resource=Resource.create(
            {"service.name": service_name, "deployment.environment": environment}
        ),
        enable_live_metrics=True,
    )
    # Wrap the exporter's log handler so structlog fields become customDimensions.
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            root.addHandler(StructlogToOtelHandler(handler))
    logger.info("telemetry_configured", exporter="azure_monitor")
