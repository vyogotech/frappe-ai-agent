"""OpenTelemetry metrics setup."""

from __future__ import annotations

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource


def create_meter(service_name: str) -> metrics.Meter:
    """Create an OTEL Meter for application metrics."""
    resource = Resource.create({"service.name": service_name})
    provider = MeterProvider(resource=resource)
    metrics.set_meter_provider(provider)
    return provider.get_meter(service_name)
