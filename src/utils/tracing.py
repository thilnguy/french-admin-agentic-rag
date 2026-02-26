import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from src.config import settings

logger = logging.getLogger(__name__)

def setup_tracing():
    if not settings.OTEL_ENABLED:
        logger.info("OpenTelemetry Tracing is disabled.")
        return trace.get_tracer(__name__)
        
    resource = Resource.create({"service.name": settings.OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    
    try:
        exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        
        trace.set_tracer_provider(provider)
        logger.info(f"OpenTelemetry Tracing initialized -> {settings.OTEL_EXPORTER_OTLP_ENDPOINT}")
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
        
    return trace.get_tracer(settings.OTEL_SERVICE_NAME)

# Global tracer instance
tracer = setup_tracing()
