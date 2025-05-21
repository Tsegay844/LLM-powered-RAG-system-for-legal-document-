# rag_api_service/api.py
import os
import sys
import time
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, status as http_status
from pydantic import BaseModel, ValidationError # Import ValidationError to catch it specifically
from typing import List, Dict, Any, Optional # <-- Import Optional
import traceback # Import traceback for logging exceptions

# Elasticsearch client
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError, TransportError as ESTransportError, NotFoundError as ESNotFoundError

# LangChain components for embedding the query
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document # To structure retrieved data

# Prometheus metrics
# Import metrics and the registry from metrics.py
from metrics import RAG_QUERY_TOTAL, RAG_QUERY_DURATION_SECONDS, LLM_CALL_ERRORS_TOTAL, ES_CALL_ERRORS_TOTAL, ACTIVE_REQUESTS, get_metrics


# --- Configuration Constants (Read from Environment Variables) ---
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "elasticsearch")
ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", 9200))
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "legal_docs")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm_service:8000") # Internal URL of the LLM service
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 5))

# --- FastAPI App Setup ---
app = FastAPI(
    title="Legal RAG API Service",
    description="Orchestrates document retrieval from Elasticsearch and text generation from LLM service."
)

# --- Clients Initialization ---
es_client = None
embedding_function = None
llm_service_client = None # Using requests.Session for simplicity

def initialize_clients():
    """Initializes Elasticsearch, Embedding, and LLM Service clients."""
    global es_client, embedding_function, llm_service_client

    # Elasticsearch Client
    if es_client is None:
        print(f"Initializing Elasticsearch client for {ELASTICSEARCH_HOST}:{ELASTICSEARCH_PORT}")
        try:
            es_client = Elasticsearch(
                hosts=[{"host": ELASTICSEARCH_HOST, "port": ELASTICSEARCH_PORT, "scheme": "http"}],
                timeout=30,
                sniff_on_start=False
            )
            es_client.info()
            if not es_client.indices.exists(index=ES_INDEX_NAME):
                 print(f"Warning: Elasticsearch index '{ES_INDEX_NAME}' does not exist. Run indexing.", file=sys.stderr)
            else:
                 print(f"Connected to Elasticsearch and index '{ES_INDEX_NAME}' exists.")
        except (ESConnectionError, ESTransportError) as e:
            print(f"Error connecting to Elasticsearch on startup: {e}", file=sys.stderr)
            es_client = None
            # Don't exit, allow service to start but queries will fail

    # Embedding Function (for query embedding)
    if embedding_function is None:
        print(f"Initializing embedding model for query embedding: {EMBEDDING_MODEL}")
        try:
            embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
            global VECTOR_DIMENSION
            VECTOR_DIMENSION = len(embedding_function.embed_query("test query"))
            print(f"Embedding vector dimension for query: {VECTOR_DIMENSION}")
        except Exception as e:
            print(f"Error initializing embedding model '{EMBEDDING_MODEL}': {e}", file=sys.stderr)
            embedding_function = None
            # Don't exit, allow service to start but queries will fail

    # LLM Service Client
    if llm_service_client is None:
        print(f"Initializing LLM service client for {LLM_SERVICE_URL}")
        llm_service_client = requests.Session()
        try:
             llm_health_response = llm_service_client.get(f"{LLM_SERVICE_URL}/health", timeout=5)
             llm_health_response.raise_for_status()
             print("LLM service health check successful.")
        except requests.exceptions.RequestException as e:
             print(f"Warning: LLM service health check failed at {LLM_SERVICE_URL}/health: {e}", file=sys.stderr)
             # Don't exit, allow service to start but queries will fail


# Initialize clients when the app starts
@app.on_event("startup")
async def startup_event():
    load_dotenv()
    initialize_clients()

# --- Request and Response Models ---
class QueryRequest(BaseModel):
    question: str

class DocumentSource(BaseModel):
    source: str # File path
    page: Optional[int] = None # <-- CHANGE 1: Use Optional[int] and keep default None
    text_snippet: str = None # Snippet of the text chunk (optional)

class QueryResponse(BaseModel):
    answer: str
    sources: List[DocumentSource]


# --- API Endpoints ---

@app.post("/query", response_model=QueryResponse)
async def query_rag_system(request: QueryRequest):
    """
    Processes a user query using RAG: retrieves docs from ES and generates answer using LLM service.
    """
    start_time = time.time()
    status = "server_error"

    ACTIVE_REQUESTS.inc()

    try:
        if not es_client:
             status = "server_error"
             raise HTTPException(
                 status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
                 detail="Elasticsearch client not initialized. Cannot connect to Elasticsearch."
            )
        if not embedding_function:
             status = "server_error"
             raise HTTPException(
                 status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
                 detail="Embedding function not initialized. Cannot embed query."
            )
        if not llm_service_client:
             status = "server_error"
             raise HTTPException(
                 status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
                 detail="LLM service client not initialized. Cannot connect to LLM service."
            )


        print(f"Received query: '{request.question}'")

        # --- Retrieval (Elasticsearch Vector Search) ---
        print("Performing Elasticsearch vector search...")
        try:
            query_vector = embedding_function.embed_query(request.question)

            search_body = {
                "size": RETRIEVAL_K,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                },
                "_source": ["text", "source", "page"]
            }

            search_response = es_client.search(index=ES_INDEX_NAME, body=search_body)
            retrieved_hits = search_response.get('hits', {}).get('hits', [])
            print(f"Retrieved {len(retrieved_hits)} relevant documents from Elasticsearch.")

        except ESNotFoundError:
             ES_CALL_ERRORS_TOTAL.inc()
             print(f"Elasticsearch index '{ES_INDEX_NAME}' not found.", file=sys.stderr)
             raise HTTPException(
                 status_code=http_status.HTTP_404_NOT_FOUND,
                 detail=f"Elasticsearch index '{ES_INDEX_NAME}' not found. Please run the indexer service."
            )
        except (ESConnectionError, ESTransportError) as e:
            ES_CALL_ERRORS_TOTAL.inc()
            print(f"Elasticsearch connection/transport error during search: {e}", file=sys.stderr)
            raise HTTPException(
                status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Error connecting to Elasticsearch: {e}"
            )
        except Exception as e:
            ES_CALL_ERRORS_TOTAL.inc()
            print(f"An unexpected error occurred during Elasticsearch search: {e}", file=sys.stderr)
            raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during document retrieval: {e}")


        # Prepare context for the LLM
        context_docs: List[DocumentSource] = []
        context_text = ""
        if retrieved_hits:
            context_text_chunks = [hit['_source'].get('text', '') for hit in retrieved_hits if hit['_source'].get('text')]
            context_text = "\n\n".join(context_text_chunks)

            # Extract source information for the response model
            for hit in retrieved_hits:
                 source_data = hit['_source']

                 # --- CHANGE 2: Safely get and process the 'page' value ---
                 raw_page = source_data.get('page')

                 page_value = None # Default to None
                 if raw_page is not None:
                     try:
                         # Attempt conversion to integer. Handle floating point numbers like 1.0
                         if isinstance(raw_page, (int, float)):
                             page_value = int(raw_page)
                         elif isinstance(raw_page, str):
                             # Try converting string digits
                             page_value = int(raw_page)
                         else:
                              # If type is unexpected, keep as None
                              print(f"Warning: Unexpected type for page metadata: {type(raw_page)}. Value: {raw_page}. Setting page to None.", file=sys.stderr)
                              page_value = None
                     except (ValueError, TypeError):
                         # If conversion fails (e.g., non-digit string), keep as None
                         print(f"Warning: Could not convert page metadata '{raw_page}' to integer for source {source_data.get('source')}. Setting page to None.", file=sys.stderr)
                         page_value = None # Explicitly set to None on failure

                 # --- CHANGE 3: Wrap the DocumentSource creation in a try-except for ValidationError ---
                 try:
                     context_docs.append(DocumentSource(
                         source=os.path.basename(source_data.get('source', 'N/A')),
                         page=page_value, # <-- Pass the safely processed page_value
                         text_snippet=source_data.get('text', '')[:200] + '...' if source_data.get('text') else None
                     ))
                 except ValidationError as e:
                     # Catch validation errors specifically for this DocumentSource instance
                     print(f"Pydantic validation error creating DocumentSource for hit: {e}", file=sys.stderr)
                     print(f"Problematic data: source={source_data.get('source')}, page={raw_page}, text_snippet (partial)={source_data.get('text', '')[:50]}...", file=sys.stderr)
                     # Decide how to handle: skip this source or raise error.
                     # For robustness, let's skip this source and log the issue.
                     continue # Skip adding this problematic source

                 except Exception as e:
                      # Catch any other unexpected errors during DocumentSource creation
                      print(f"An unexpected error occurred creating DocumentSource for hit: {e}", file=sys.stderr)
                      continue # Skip this source
            # --- End CHANGE 3 ---


        else:
            print("No documents retrieved from Elasticsearch.")


        # --- Generation (Call LLM Service) ---
        print("Calling LLM service for text generation...")
        llm_prompt = f"""
        You are a helpful legal document assistant. Answer the following question based *only* on the provided context documents.
        If you cannot find the answer in the context, clearly state that you cannot find the information in the documents.
        Do not make up information.

        Context Documents:
        ---
        {context_text if context_text else "No relevant documents found."}
        ---

        Question:
        {request.question}

        Answer:
        """

        try:
            llm_response = llm_service_client.post(
                f"{LLM_SERVICE_URL}/generate",
                json={"prompt": llm_prompt, "temperature": 0.0, "max_output_tokens": 512},
                timeout=120
            )
            llm_response.raise_for_status()

            llm_response_data = llm_response.json()
            generated_answer = llm_response_data.get("text", "Error: No text generated by LLM or LLM returned empty response.")
            if not generated_answer.strip():
                 generated_answer = "The LLM service returned an empty response."
            print("LLM generation successful.")

        except requests.exceptions.RequestException as e:
            LLM_CALL_ERRORS_TOTAL.inc()
            print(f"Error calling LLM service at {LLM_SERVICE_URL}: {e}", file=sys.stderr)
            llm_status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            detail_message = f"Error communicating with LLM service: {e}"
            if llm_status_code:
                 detail_message += f" (Status: {llm_status_code})"
            detail_message += ". Check LLM service logs."
            raise HTTPException(
                status_code=llm_status_code if llm_status_code in range(400, 600) else http_status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=detail_message
            )
        except Exception as e:
            LLM_CALL_ERRORS_TOTAL.inc()
            print(f"An unexpected error occurred processing LLM response: {e}", file=sys.stderr)
            traceback.print_exc()
            raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error processing LLM response: {e}")


        # --- Response ---
        status = "success"
        print("Query process completed successfully.")
        return QueryResponse(answer=generated_answer, sources=context_docs)

    except HTTPException:
        raise
    except Exception as e:
        # Catch any other unexpected errors not handled by specific blocks
        print(f"An unexpected error occurred during query processing: {e}", file=sys.stderr)
        traceback.print_exc()
        status = "server_error"
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected internal error occurred: {e}")

    finally:
        duration = time.time() - start_time
        if 'status' in locals():
             final_status_label = status
        else:
             final_status_label = "server_error"

        if 'e' in locals() and isinstance(e, HTTPException):
             if 400 <= e.status_code < 500:
                  final_status_label = "client_error"
             elif 500 <= e.status_code < 600:
                  final_status_label = "server_error"

        RAG_QUERY_TOTAL.labels(status=final_status_label).inc()
        RAG_QUERY_DURATION_SECONDS.observe(duration)
        ACTIVE_REQUESTS.dec()
        print(f"Query finished in {duration:.2f} seconds with final status: {final_status_label}")


# --- Metrics Endpoint ---
@app.get("/metrics")
def get_prometheus_metrics():
    """Endpoint to expose Prometheus metrics."""
    return Response(content=get_metrics(), media_type="text/plain; version=0.0.4; charset=utf-8")

# Optional: Basic health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check for the RAG API service and its dependencies."""
    status_detail = {"status": "ok"}
    overall_status_code = http_status.HTTP_200_OK

    # Check Elasticsearch dependency
    try:
        if es_client:
             es_client.info()
             status_detail["elasticsearch"] = "ok"
        else:
             status_detail["elasticsearch"] = "uninitialized"
             overall_status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE
    except Exception as e:
        status_detail["elasticsearch"] = f"error: {e}"
        overall_status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE

    # Check LLM Service dependency
    try:
        if llm_service_client:
            llm_health_response = llm_service_client.get(f"{LLM_SERVICE_URL}/health", timeout=5)
            llm_health_response.raise_for_status()
            status_detail["llm_service"] = llm_health_response.json()
            if status_detail["llm_service"].get("status") != "ok":
                 overall_status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE

        else:
             status_detail["llm_service"] = "uninitialized"
             overall_status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE
    except requests.exceptions.RequestException as e:
        status_detail["llm_service"] = f"error: {e}"
        overall_status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE
    except Exception as e:
         status_detail["llm_service"] = f"unexpected error: {e}"
         overall_status_code = http_status.HTTP_500_INTERNAL_SERVER_ERROR

    # Check Embedding Function initialization
    if embedding_function:
         status_detail["embedding_function"] = "initialized"
    else:
         status_detail["embedding_function"] = "uninitialized"
         overall_status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE
         status_detail["status"] = "degraded"


    if overall_status_code != http_status.HTTP_200_OK:
         status_detail["status"] = "degraded" if overall_status_code == http_status.HTTP_503_SERVICE_UNAVAILABLE else "error"
         raise HTTPException(status_code=overall_status_code, detail=status_detail)

    status_detail["status"] = "ok"
    return status_detail


# To run locally for testing (outside Docker):
# if __name__ == "__main__":
#     import uvicorn
#     load_dotenv()
#     print("Running RAG API service locally. Ensure Elasticsearch and LLM service are accessible.")
#     initialize_clients()
#     uvicorn.run(app, host="0.0.0.0", port=8000)

