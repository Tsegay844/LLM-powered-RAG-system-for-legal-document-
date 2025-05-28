# rag_api_service/metrics.py
#
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry

# Create a new registry for these metrics (optional but good practice)
registry = CollectorRegistry()

# Define metrics

# Counter for total queries
# Labels allow filtering/grouping by status (success, client_error, server_error)
RAG_QUERY_TOTAL = Counter(
    'rag_query_total',
    'Total number of RAG queries processed',
    ['status'], # Label for status (success, client_error, server_error)
    registry=registry
)

# Histogram for query duration
# Measures the time taken for a complete RAG query process
# This metric will help identify performance issues in the RAG system
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

# --- NEW METRIC for User Feedback ---
# Counter for total feedback submitted
# Label for feedback type (satisfied, unsatisfied, or other types)
## This metric will help track user satisfaction and feedback trends
# This can be useful for improving the RAG system based on user input
# This metric will be incremented each time a user submits feedback
## It can be used to analyze the effectiveness of the RAG system and identify areas for improvement
RAG_FEEDBACK_TOTAL = Counter(
    'rag_feedback_total',           #  This is the metric name
    'Total number of user feedback submissions',  #  This is the description
    ['feedback_type'],
    registry=registry
)
# --- END NEW METRIC ---

# --- NEW METRICS for Component Latency ---
# Histograms to measure the duration of calls to dependencies
# These metrics will help identify bottlenecks in the RAG system
# Histogram for Elasticsearch retrieval duration
# This metric will track how long it takes to retrieve documents from Elasticsearch
# It can help identify performance issues in the retrieval process  
#
RAG_RETRIEVAL_DURATION_SECONDS = Histogram(
    'rag_retrieval_duration_seconds',
    'Histogram of duration for Elasticsearch retrieval calls',
    registry=registry
)

# Histogram for LLM service API call duration
# This metric will track how long it takes to call the LLM service API  
# It can help identify performance issues in the LLM service calls

RAG_LLM_CALL_DURATION_SECONDS = Histogram(
    'rag_llm_call_duration_seconds',
    'Histogram of duration for LLM service API calls',
    registry=registry
)
# --- END NEW METRICS ---


# Function to get metrics data in Prometheus format
def get_metrics():
    """Returns metrics data formatted for Prometheus."""
    return generate_latest(registry)



