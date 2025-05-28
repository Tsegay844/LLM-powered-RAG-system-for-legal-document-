Okay, congratulations on getting your project working perfectly! That's a significant achievement, especially with the microservices architecture. A professional README is essential for sharing your work and making it easy for others (or your future self) to understand and run.

Here is a structured and comprehensive README template tailored to your project. Fill in the bracketed placeholders `[...]` with your specific project details.

---

# Legal Document RAG System (Microservices)

![Project Logo or Header Image (Optional)](path/to/your/logo.png)

A Retrieval-Augmented Generation (RAG) system designed to assist with legal document analysis. This project is built using a microservices architecture orchestrated with Docker Compose, leveraging Elasticsearch for retrieval, the Google Gemini API for generation, Streamlit for the user interface, and Prometheus/Grafana for monitoring and evaluation.

## ✨ Features

*   **Microservices Architecture:** Project components are separated into distinct services for better scalability, maintainability, and fault isolation.
*   **Document Ingestion:** Process and index legal documents from various formats (PDF, TXT) into Elasticsearch.
*   **Elasticsearch Retrieval:** Utilize Elasticsearch's powerful search capabilities (including vector search) to find relevant document chunks based on a user query.
*   **Google Gemini Generation:** Leverage the Google Gemini API to generate coherent and contextually relevant answers based on the retrieved document chunks and the user's query.
*   **Streamlit User Interface:** An intuitive web interface for users to submit queries and receive answers.
*   **Persistent Storage:** Document index data in Elasticsearch and monitoring data in Prometheus/Grafana are persisted using Docker volumes.
*   **Prometheus & Grafana Monitoring:** Collect key application metrics (query volume, latency, errors, feedback) and visualize them in real-time dashboards.
*   **User Feedback:** Capture explicit user feedback (satisfied/unsatisfied) on query results to inform system evaluation and improvement.

## 🏛️ Architecture

The project follows a microservices pattern, with each service running in its own Docker container:

1.  **`elasticsearch`**: Stores and indexes the legal document chunks and their vector embeddings. Acts as the knowledge base for retrieval.
2.  **`indexer_service`**: A batch job responsible for loading documents from the `./docs` directory, splitting them, generating embeddings using Sentence Transformers, and pushing them into Elasticsearch.
3.  **`llm_service`**: A simple API wrapper around the Google Gemini API. Receives prompts and returns generated text. Decouples the main RAG logic from the specific LLM provider.
4.  **`rag_api_service`**: The core orchestration service. Receives user queries from the UI, performs vector search in Elasticsearch, formats the prompt with retrieved context, calls the `llm_service` for generation, collects application metrics, and returns the answer and sources.
5.  **`streamlit_ui`**: The user-facing web application built with Streamlit. Provides the interface for querying the system and submitting feedback. Communicates with the `rag_api_service` via HTTP.
6.  **`prometheus`**: Collects time-series metrics by scraping the `/metrics` endpoint of the `rag_api_service`. Stores historical monitoring data.
7.  **`grafana`**: Visualizes the metrics stored in Prometheus through interactive dashboards. Allows monitoring system performance and user feedback.

These services communicate over Docker's internal network. Persistent data is stored in named Docker volumes.

```
+-------------+     HTTP      +---------------+     HTTP      +-------------+
| Streamlit UI|------------->| RAG API Service|------------->| LLM Service |
|             |             | (+ Metrics)   |             | (Gemini API)|
+-------------+             +-------+-------+             +-------------+
      ^                             |
      |                             | HTTP
      |                             |
      |                             |
      |                             |
      |         +-------------------+-----------------+
      |         |                   |                 |
      |         |                   |                 | Elasticsearch Client
      |         |                   |                 | (Search/Index)
      |         |                   |                 |
+-----+-------+ |         +-------+-------+         +-------------+
| User (Browser)| |--------->| Elasticsearch |<--------|  Indexer    |
+-------------+ |           | (Vector Store)|         |  (Job)      |
                |           +-------+-------+         +-------------+
                |                   ^
                |                   | Scrapes
                |                   | Metrics
                |         +---------+---------+
                |         |                   |
                |         |    Prometheus     |
                |         |   (Metrics DB)    |
                |         +---------+---------+
                |                   ^
                |                   | Queries
                |                   | Metrics
                |         +---------+---------+
                |         |                 |
                +-------->|     Grafana     |
                          | (Dashboards)    |
                          +-----------------+
```

## 📦 Technologies Used

*   **Orchestration:** Docker, Docker Compose
*   **Backend Frameworks:** FastAPI (for API services)
*   **Frontend Framework:** Streamlit
*   **Language:** Python 3.9
*   **Vector Database / Search Engine:** Elasticsearch 7.17
*   **LLM Provider:** Google Gemini API (gemini-1.5-flash-latest)
*   **Embedding Model:** Sentence Transformers (all-MiniLM-L6-v2)
*   **Metric Collection:** Prometheus Client (Python), Prometheus
*   **Visualization:** Grafana
*   **Document Processing:** LangChain Community, pypdf, etc.

## 🛠️ Setup and Installation

1.  **Prerequisites:**
    *   [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Engine and Docker Compose) or Docker Engine and Docker Compose Plugin installed.
    *   [Git](https://git-scm.com/)

2.  **Clone the Repository:**
    ```bash
    git clone [Your GitHub Repo URL]
    cd [your-repo-name] # Navigate to the project root directory
    ```

3.  **Add Your Documents:**
    Place your legal documents (PDFs, TXTs) inside the `./docs/` directory in the project root. You can organize them in subfolders.

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory of the project and populate it with the necessary API keys and configuration.
```bash
    # In the project root
    touch .env
    ```

    Open `.env` in a text editor and add the following, replacing placeholders with your actual values:

    ```dotenv
    # .env

    # --- API Keys ---
    # Google Generative AI API Key (for llm_service)
    # Get this from https://makersuite.google.com/app/apikeys or Google Cloud Console
    GOOGLE_API_KEY=AIzaSy... # Replace with your actual Google API Key

    # --- Model Configuration ---
    # Google Gemini LLM Model (for llm_service)
    # Choose a model available with your Google API Key (e.g., gemini-pro, gemini-1.5-flash-latest)
    LLM_MODEL_NAME=gemini-1.5-flash-latest # Example: using the current recommended alias

    # Embedding Model (for indexer_service and rag_api_service)
    # Using a Sentence Transformer model that runs locally in the container
    EMBEDDING_MODEL=all-MiniLM-L6-v2 # This model is downloaded by the container

    # --- Elasticsearch Configuration ---
    # Service name in docker-compose internal network (do not change unless changing service name)
    ELASTICSEARCH_HOST=elasticsearch
    ELASTICSEARCH_PORT=9200
    ES_INDEX_NAME=legal_docs # Name of the Elasticsearch index

    # --- RAG API Configuration ---
    # Retrieval K: Number of documents to retrieve from Elasticsearch for a query
    RETRIEVAL_K=5
    # Internal URL of the LLM service (do not change unless changing service name/port)
    LLM_SERVICE_URL=http://llm_service:8000

    # --- Document Processing Configuration (used by indexer_service) ---
    CHUNK_SIZE=1000 # Characters per document chunk
    CHUNK_OVERLAP=200 # Overlap between chunks

    # --- Streamlit UI Configuration ---
    # URL of the RAG API service (accessible from the host where Streamlit runs)
    # If Docker runs on localhost, this is http://localhost:8000
    # If Docker runs on a remote machine, replace localhost with its IP/hostname
    RAG_API_URL=http://localhost:8000

    # --- Monitoring Configuration ---
    # Expose ports for UIs on the host machine
    PROMETHEUS_PORT=9090
    GRAFANA_PORT=3000
    RAG_API_PORT=8000
    STREAMLIT_PORT=8501
    ELASTICSEARCH_HOST_PORT=9200 # Optional: Port to expose Elasticsearch on the host

    # Grafana Admin User/Password (CHANGE THESE FOR PRODUCTION!)
    GRAFANA_ADMIN_USER=admin
    GRAFANA_ADMIN_PASSWORD=admin # !! Change this !!

    # Optional: Remove the 'version' attribute warning in docker-compose.yml
    # COMPOSE_IGNORE_ORPHANS=True
    ```

    **Security Note:** The `.env` file contains your API key. **Do NOT commit this file to Git.** Ensure `.env` is listed in your `.gitignore` file.

5.  **Build Docker Images:**
    Navigate to the project root directory in your terminal and build the

    
