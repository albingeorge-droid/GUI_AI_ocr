from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def setup_tracing(service_name: str, phoenix_otlp_endpoint: str | None) -> None:
    """
    Sends OpenTelemetry traces to Phoenix via OTLP gRPC.
    Example endpoint: http://localhost:4317
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    if phoenix_otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=phoenix_otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
