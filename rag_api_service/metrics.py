# rag_api_service/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry

# Create a new registry for these metrics (optional but good practice)
registry = CollectorRegistry()

# Define metrics

# Counter for total queries
# Labels allow filtering/grouping by status (success, client_error, server_error)
RAG_QUERY_TOTAL = Counter(
    'rag_query_total',
    'Total number of RAG queries processed',
    ['status'],
    registry=registry
)

# Histogram for query duration
# Measures the time taken for a complete RAG query process
RAG_QUERY_DURATION_SECONDS = Histogram(
    'rag_query_duration_seconds',
    'Histogram of RAG query duration from start to end',
    registry=registry
)

# Counter for errors specifically from calling the LLM service
LLM_CALL_ERRORS_TOTAL = Counter(
    'llm_call_errors_total',
    'Total number of errors when calling the LLM service',
    registry=registry
)

# Counter for errors specifically from calling Elasticsearch
ES_CALL_ERRORS_TOTAL = Counter(
    'es_call_errors_total',
    'Total number of errors when calling Elasticsearch',
    registry=registry
)

# Gauge to track the number of active requests
# Good for understanding concurrent load
ACTIVE_REQUESTS = Gauge(
    'rag_active_requests',
    'Number of currently active RAG query requests',
    registry=registry
)


# Function to get metrics data in Prometheus format
def get_metrics():
    """Returns metrics data formatted for Prometheus."""
    return generate_latest(registry)

# You can add more metrics as needed, e.g.:
# - Retrieval duration histogram
# - LLM call duration histogram
# - Number of documents retrieved per query histogram
# - Cache hit/miss ratio (if caching is implemented)