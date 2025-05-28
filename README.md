# ⚖️ Legal Document Assistant RAG System (Microservices) ---
A Retrieval-Augmented Generation (RAG) system designed to assist with legal document analysis. This project is built using a microservices architecture orchestrated with Docker Compose, leveraging Elasticsearch for retrieval, the Google Gemini API for generation, Streamlit for the user interface, and Prometheus/Grafana for monitoring and evaluation.

The system allows users to ask natural language questions about a corpus of legal documents. It retrieves the most relevant information from the documents using Elasticsearch and synthesizes an answer using a powerful Large Language Model (LLM) accessed via a dedicated service. The system's performance and usage are monitored in real-time using Prometheus and Grafana, and user feedback is collected to evaluate answer quality.

## Features

*   **Microservices Architecture:** Project components are separated into distinct services for better scalability, maintainability, and fault isolation.
*   **Document Ingestion:** Process and index legal documents from various formats (PDF, TXT) into Elasticsearch.
*   **Elasticsearch Retrieval:** Utilize Elasticsearch's powerful search capabilities (including vector search) to find relevant document chunks based on a user query.
*   **Google Gemini Generation:** Leverage the Google Gemini API to generate coherent and contextually relevant answers based on the retrieved document chunks and the user's query.
*   **Streamlit User Interface:** An intuitive web interface for users to submit queries and receive answers.
*   **Persistent Storage:** Document index data in Elasticsearch and monitoring data in Prometheus/Grafana are persisted using Docker volumes.
*   **Prometheus & Grafana Monitoring:** Collect key application metrics (query volume, latency, errors, feedback) and visualize them in real-time dashboards.
*   **User Feedback:** Capture explicit user feedback (satisfied/unsatisfied) on query results to inform system evaluation and improvement.

## System Architecture

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

## Technologies Used

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
    git clone project
    cd https://github.com/Tsegay844/LLM-powered-RAG-system-for-legal-document-) # Navigate to the project root directory
    ```
3.  **Add Your Documents:**
    Place your legal documents (PDFs, TXTs) inside the `./docs/` directory in the project root.

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory of the project and populate it with the necessary API keys and configuration.

    Open `.env` in a text editor and add the following, replacing placeholders with your actual values:

    ```dotenv
    # .env
    # --- API Keys ---
    # Google Generative AI API Key 
    GOOGLE_API_KEY= [PUT your API Key]
    # Google Gemini LLM Model 
    LLM_MODEL_NAME=gemini-2.0-flash
    # Embedding Model (for indexer_service and rag_api_service)
    EMBEDDING_MODEL=all-MiniLM-L6-v2

    # Elasticsearch Configuration
    ELASTICSEARCH_HOST=elasticsearch
    ELASTICSEARCH_PORT=9200
    ES_INDEX_NAME=legal_docs
    
    # RAG API Configuration
    RETRIEVAL_K=5  # Number of documents to retrieve from Elasticsearch
    # Internal URL of the LLM service
    LLM_SERVICE_URL=http://llm_service:8000
    # RAG API Port (default is 8000)
    RAG_API_PORT=8000
    
    # Document Processing Configuration (used by indexer_service)
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=200
    
    # Streamlit UI Configuration
    RAG_API_URL=http://localhost:8000
    STREAMLIT_PORT=8501
    ELASTICSEARCH_HOST_PORT=9200 

    # Monitoring Configuration 
    # Prometheus Port (default is 9090)
    PROMETHEUS_PORT=9090
    # Grafana Port (default is 3000)
    GRAFANA_PORT=3000
    # Default admin user is 'admin' and password is 'admin'
    # !! Change these for production use to secure your Grafana instance !!
    GRAFANA_ADMIN_USER=admin
    GRAFANA_ADMIN_PASSWORD=admin # !! Change this !!
    
    # Optional: Remove the 'version' attribute warning in docker-compose.yml
    # COMPOSE_IGNORE_ORPHANS=True
    ```

5.  **Build Docker Images:**
    Navigate to the project root directory in your terminal and build the Docker images for all the services.
    ```bash
    docker compose build
    ```

## ▶ HOW To RUN
1.  **Start Infrastructure and Application Services:**
    This command starts all services except the `indexer_service` (which is a one-time job to index the documents).
    ```bash
    docker compose up -d
    ```
    Check their status with `docker compose ps`. Elasticsearch should show as `healthy`.
2.  **Indexing Documents:**
    The `indexer_service` is run as a separate job whenever you need to index new or updated documents. Ensure Elasticsearch is running (`docker compose ps`).
    **Run the Indexing Job:**
    ```bash
    docker compose run --rm indexer_service python indexer.py --recreate-index
    ```
    **Verify Indexing Progress (Optional):**
    While the indexer is running, you can check the number of documents indexed in Elasticsearch:
    ```bash
    curl http://localhost:9200/${ES_INDEX_NAME}/_count
    ```
    Replace `${ES_INDEX_NAME}` with the value from your `.env` (default `legal_docs`).
3.  **🌐 Access Legal Document Assistant UI**
    Once all services (`docker compose ps` should show them Up) and the indexing job is complete:
    **Access the Streamlit UI:**
    Open your web browser and go to:
    ```
    http://localhost:8501
    ```
    (Using the `STREAMLIT_PORT` from your `.env`)
    **Ask a Question:** Enter your legal question in the text area and click "Get Answer".
    **Provide Feedback:** After receiving an answer, use the "👍 Satisfied" or "👎 Unsatisfied" buttons to provide feedback.

4.  **📊 Monitoring and Evaluation**
    Access the monitoring UIs to observe system performance and user feedback.
    **Prometheus UI:**
    *   Access: `http://localhost:9090` (Using the `PROMETHEUS_PORT` from your `.env`)
    *   Use the "Graph" tab to explore raw metrics or run PromQL queries. Check "Status" -> "Targets" to ensure `rag_api_service:8000` is UP.
    **Grafana UI:**
    *   Access: `http://localhost:3000` (Using the `GRAFANA_PORT` from your `.env`)
    *   Log in with the admin user and password from your `.env`. Change the password upon first login.
    *   **Configure Prometheus Datasource:**
        *   Go to "Connections" -> "Data sources" -> "Add data source" -> "Prometheus".
        *   Set the URL to `http://prometheus:9090` (This is the internal Docker service address).
        *   Click "Save & test".
    *   **Import Dashboard:** import configured dashboard and panels from the directory `grafana/dashboard.json`

## 🛑 Stopping the Project
To stop all running services and remove their containers:
```bash
docker compose down
```


