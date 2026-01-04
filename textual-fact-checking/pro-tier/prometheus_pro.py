%%writefile prometheus_pro.py
import os
from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.view import View, ExplicitBucketHistogramAggregation
from opentelemetry.semconv.resource import ResourceAttributes

# "NO SIMPLIFICATION" Buckets:
# High fidelity for Pro Tier (0.1s - 2s)
# High fidelity for Free Tier (5s - 15s)
# Standard bucket set covering both ranges with precision suitable for P50, P75, P90, P95, P99
LATENCY_BUCKETS = [
    0.05, 0.1, 0.25, 0.5, 0.75, 
    1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 
    5.0, 6.0, 7.5, 10.0, 12.5, 15.0, 20.0, 30.0, 45.0, 60.0
]

class NvyraMetrics:
    def __init__(self, service_name: str, tier: str):
        self.tier = tier
        
        # Initialize OTEL if configured
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
        
        if endpoint and headers:
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: service_name,
                "tier": tier
            })
            
            # Create Views to enforce explicit buckets for all Histograms
            # This ensures we can calculate P95, P99 accurately in Grafana
            latency_view = View(
                instrument_name="*_seconds",  # Applies to req_duration, ttft, thinking, cold_start
                aggregation=ExplicitBucketHistogramAggregation(LATENCY_BUCKETS)
            )
            
            # Simple view for counts (like extracted claims) - 1 to 50
            claims_buckets = [1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100]
            counts_view = View(
                instrument_name="*_count", # Applies to claims_extracted
                aggregation=ExplicitBucketHistogramAggregation(claims_buckets)
            )

            # Accuracy score buckets (0.0 to 1.0)
            score_buckets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
            score_view = View(
                instrument_name="*_score", # Applies to confidence_score, disinfo_score
                aggregation=ExplicitBucketHistogramAggregation(score_buckets)
            )

            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=endpoint, headers=headers)
            )
            
            provider = MeterProvider(
                resource=resource, 
                metric_readers=[reader],
                views=[latency_view, counts_view, score_view]
            )
            
            metrics.set_meter_provider(provider)
            self.meter = metrics.get_meter("nvyra.inference")
            self.enabled = True
        else:
            self.meter = metrics.get_meter("nvyra.inference.noop")
            self.enabled = False

        # --- 1. Accuracy & Business ---
        self.verdict_counter = self.meter.create_counter(
            "nvyra_verdict_total", description="Count of final verdicts"
        )
        self.confidence_score = self.meter.create_histogram(
            "nvyra_confidence_score", description="Confidence score distribution"
        )
        self.disinfo_score = self.meter.create_histogram(
            "nvyra_disinfo_score", description="Disinformation score distribution"
        )
        self.claims_extracted = self.meter.create_histogram(
            "nvyra_claims_extracted_count", description="Number of claims extracted per request"
        )
        self.citation_domains = self.meter.create_counter(
            "nvyra_citation_domain_total", description="Count of citations by domain (e.g. bbc.com)"
        )

        # --- 2. Advanced Performance ---
        self.requests_total = self.meter.create_counter(
            "nvyra_requests_total", description="Total requests processed"
        )
        self.tokens_generated = self.meter.create_counter(
            "nvyra_tokens_generated_total", description="Total LLM tokens generated"
        )
        self.req_duration = self.meter.create_histogram(
            "nvyra_req_duration_seconds", description="End-to-end request latency"
        )
        self.ttft = self.meter.create_histogram(
            "nvyra_ttft_seconds", description="Time to first token"
        )
        self.thinking_duration = self.meter.create_histogram(
            "nvyra_thinking_duration_seconds", description="Latency of reasoning/thinking phase"
        )
        self.cold_start = self.meter.create_histogram(
            "nvyra_cold_start_seconds", description="Container initialization latency"
        )

        # --- 3. Efficiency ---
        self.cache_events = self.meter.create_counter(
            "nvyra_cache_events_total", description="Cache hits and misses"
        )
        self.external_search = self.meter.create_counter(
            "nvyra_external_search_total", description="External web searches performed"
        )
        self.route_usage = self.meter.create_counter(
            "nvyra_route_usage_total", description="Routing decisions made"
        )

        # --- 4. Operational ---
        self.errors = self.meter.create_counter(
            "nvyra_errors_total", description="Error details"
        )
        self.safety_triggers = self.meter.create_counter(
            "nvyra_safety_triggers_total", description="Safety guardrail triggers"
        )
        self.gpu_util = self.meter.create_observable_gauge(
            "nvyra_gpu_utilization", callbacks=[], description="GPU Utilization %"
        )

    # --- Helper methods to simplify recording ---
    
    def record_request(self, route: str, status: str = "success"):
        if self.enabled:
            self.requests_total.add(1, {"tier": self.tier, "route": route, "status": status})

    def record_verdict(self, verdict: str):
        if self.enabled:
            self.verdict_counter.add(1, {"tier": self.tier, "verdict": verdict})

    def record_confidence(self, score: float):
        if self.enabled:
            self.confidence_score.record(score, {"tier": self.tier})

    def record_latency(self, duration: float, route: str, cache_hit: bool):
        if self.enabled:
            self.req_duration.record(duration, {"tier": self.tier, "route": route, "cache_hit": str(cache_hit).lower()})

    def record_cache(self, hit: bool):
        if self.enabled:
            result = "hit" if hit else "miss"
            self.cache_events.add(1, {"tier": self.tier, "result": result})

    def record_citation(self, url: str):
        if self.enabled:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.replace("www.", "")
                if domain:
                    self.citation_domains.add(1, {"tier": self.tier, "domain": domain})
            except:
                pass
