import logging

# OTLP boilerplate
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)


def setup(level):
    logging.basicConfig(
        level=level,
        force=True,
        style="{",
        format="{relativeCreatedSecs: >9.3f}s {levelname[0]} {name:<6s} {message}",
    )

    def add_relative_seconds(record):
        record.relativeCreatedSecs = record.relativeCreated / 1000.0
        return True

    logging.getLogger().handlers[0].addFilter(add_relative_seconds)


resource = Resource.create(attributes={SERVICE_NAME: "nefelis"})


def setup_otlp():
    # Configure endpoint using OTEL_EXPORTER_OTLP_ENDPOINT=http://hostname:4318
    # Configure metrics
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(), export_interval_millis=1000
    )
    mprovider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(mprovider)
    # Configure traces
    tprovider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter())
    tprovider.add_span_processor(processor)
    trace.set_tracer_provider(tprovider)
    # Avoid annoying debugging logs
    logging.getLogger("urllib3").setLevel(logging.INFO)
