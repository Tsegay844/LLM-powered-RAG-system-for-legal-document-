Okay, congratulations on getting your project working perfectly! That's a significant achievement, especially with the microservices architecture. A professional README is essential for sharing your work and making it easy for others (or your future self) to understand and run.

Here is a structured and comprehensive README template tailored to your project. Fill in the bracketed placeholders `[...]` with your specific project details.

---

###### ⚖️ Legal Document Assistant RAG System (Microservices)

A Retrieval-Augmented Generation (RAG) system designed to assist with legal document analysis. This project is built using a microservices architecture orchestrated with Docker Compose, leveraging Elasticsearch for retrieval, the Google Gemini API for generation, Streamlit for the user interface, and Prometheus/Grafana for monitoring and evaluation.

## Features

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


 Docker images for the services.
    ```bash
    docker compose build
    ```

## ▶️ Running the Project

1.  **Start Infrastructure and Application Services:**
    This command starts all services except the `indexer_service` (which is a job).
    ```bash
    docker compose up -d elasticsearch llm_service rag_api_service prometheus grafana streamlit_ui
    ```
    *Give the services a minute or two to start up properly, especially Elasticsearch.* You can check their status with `docker compose ps`. Elasticsearch should show as `healthy`.

## 📄 Indexing Documents

The `indexer_service` is run as a separate job whenever you need to index new or updated documents. Ensure Elasticsearch is running (`docker compose ps`).

1.  **Run the Indexing Job:**
    ```bash
    docker compose run --rm indexer_service python indexer.py --recreate-index
    ```
    *   `docker compose run --rm indexer_service`: Starts a temporary container based on the `indexer_service` image and removes it after the job finishes.
    *   `python indexer.py`: Executes the indexing script inside the container.
    *   `--recreate-index`: (Optional, Recommended for initial run) This argument tells the script to delete the existing Elasticsearch index and create a new one before indexing. Omit this flag if you want to add documents to an existing index.
    *   Monitor the terminal output for indexing progress and completion messages.

2.  **Verify Indexing Progress (Optional):**
    While the indexer is running, you can check the number of documents indexed in Elasticsearch:
    ```bash
    curl http://localhost:9200/${ES_INDEX_NAME}/_count
    ```
    Replace `${ES_INDEX_NAME}` with the value from your `.env` (default `legal_docs`).

## 🌐 Using the UI

Once all services (`docker compose ps` should show them Up) and the indexing job is complete:

1.  **Access the Streamlit UI:** Open your web browser and go to:
    ```
    http://localhost:8501
    ```
    (Using the `STREAMLIT_PORT` from your `.env`)
2.  **Ask a Question:** Enter your legal question in the text area and click "Get Answer".
3.  **Provide Feedback:** After receiving an answer, use the "👍 Satisfied" or "👎 Unsatisfied" buttons to provide feedback.

## 📊 Monitoring and Evaluation

Access the monitoring UIs to observe system performance and user feedback.

1.  **Prometheus UI:**
    *   Access: `http://localhost:9090` (Using the `PROMETHEUS_PORT` from your `.env`)
    *   Use the "Graph" tab to explore raw metrics or run PromQL queries. Check "Status" -> "Targets" to ensure `rag_api_service:8000` is UP.

2.  **Grafana UI:**
    *   Access: `http://localhost:3000` (Using the `GRAFANA_PORT` from your `.env`)
    *   Log in with the admin user and password from your `.env`. Change the password upon first login.
    *   **Configure Prometheus Datasource:**
        *   Go to "Connections" -> "Data sources" -> "Add data source" -> "Prometheus".
        *   Set the URL to `http://prometheus:9090` (This is the internal Docker service address).
        *   Click "Save & test".
    *   **Build Dashboards:** Create new dashboards and add panels using the PromQL queries provided in previous responses to visualize metrics like Query Rate, Latency (95th percentile), Error Rates (Overall, LLM, ES), Active Requests, Total Feedback, and Satisfaction Rate.

## 🛑 Stopping the Project

To stop all running services and remove their containers:

```bash
docker compose down
```

To stop services and also remove the persistent volumes (clearing Elasticsearch index, Prometheus data, Grafana data):

```bash
docker compose down --volumes
```

## 📁 Project Structure

```
legal_rag_microservices/
├── .env                      # Environment variables (API keys, config) - DO NOT COMMIT
├── .gitignore                # Files/dirs to ignore (like .env, data/)
├── docker-compose.yml        # Defines all services, networks, and volumes
├── docs/                     # Source legal documents
│   └── ... (your documents)
├── elasticsearch/            # Optional: Elasticsearch configuration
│   └── elasticsearch.yml
├── prometheus/               # Prometheus configuration
│   └── prometheus.yml        # Scrape targets
├── grafana/                  # Optional: Grafana configuration
│   └── datasources/
│       └── prometheus.yml    # Auto-configure Prometheus datasource
├── indexer_service/          # Service to load, split, embed, and push to Elasticsearch
│   ├── Dockerfile
│   ├── requirements.txt
│   └──


Okay, congratulations on getting the entire microservices RAG system working! That's a significant accomplishment.

Now, let's prepare a professional, organized, and well-explained `README.md` file for your GitHub repository. This README will serve as the main entry point for anyone wanting to understand, set up, or contribute to your project.

Here is a structured `README.md` template. Fill in the placeholders with specific details about your project where indicated.

---

```markdown
# Legal Document RAG Assistant (Microservices)

## Project Description

This project implements a Retrieval-Augmented Generation (RAG) system designed to assist with querying legal documents. Leveraging a microservices architecture powered by Docker Compose, the system provides a clear separation of concerns for document ingestion, search/retrieval, language model inference, user interface, and monitoring.

The system allows users to ask natural language questions about a corpus of legal documents. It retrieves the most relevant information from the documents using Elasticsearch and synthesizes an answer using a powerful Large Language Model (LLM) accessed via a dedicated service. The system's performance and usage are monitored in real-time using Prometheus and Grafana, and user feedback is collected to evaluate answer quality.

## Features

*   **Document Ingestion:** Loads legal documents (PDF, TXT) from a directory, splits them into manageable chunks, generates vector embeddings, and indexes them into Elasticsearch.
*   **Elasticsearch Retrieval:** Utilizes Elasticsearch for efficient storage and retrieval of document chunks based on vector similarity and potential future hybrid search.
*   **LLM Service (Google Gemini):** A dedicated microservice wrapping the Google Gemini API to handle text generation based indexer.py            # Indexing script
├── llm_service/              # Service to wrap the LLM API (Google Gemini)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── llm_api.py            # FastAPI app exposing /generate
├── rag_api_service/          # Service for retrieval, generation orchestration, and metrics
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── api.py                # FastAPI app exposing /query and /metrics
│   └── metrics.py            # Prometheus metrics definition
│   └── __init__.py           # Makes rag_api_service a Python package
└── streamlit_ui/             # Service for the user interface
    ├── Dockerfile
    ├── requirements.txt
    └── app.py                # Streamlit application
```

## 🔮 Future Improvements

*   Implement Hybrid Search in Elasticsearch (combining keyword and vector search).
*   Add more sophisticated error handling and logging.
*   Support additional document formats and potentially OCR for scanned documents.
*   Implement an MLOps pipeline (e.g., using Airflow, similar to the reference project) for automated indexing, evaluation, and model retraining.
*   Develop an offline evaluation script to calculate Hit Rate, MRR, and potentially use BERT or other models for answer quality scoring based on a ground truth dataset. Push these evaluation results as custom metrics to Prometheus.
*   Improve the LLM prompt engineering for better answer quality and control.
*   Implement caching strategies in the `rag_api_service` for frequently asked questions or recent retrieval results.
*   Enhance the Streamlit UI with more features (e.g., displaying retrieved chunks, conversational history).
*   Add user authentication and authorization.
*   Prepare for production deployment (scaling services, adding security, using managed services).

## 🙏 Acknowledgements

*   [LangChain](https://github.com/langchain-ai/langchain)
*   [Sentence Transformers](https://www.sbert.net/)
*   [ChromaDB](https://www.trychroma.com/) (used in earlier iteration)
*   [Elasticsearch](https://www.elastic.co/)
*   [Google Generative AI](https://ai.google.dev/)
*   [FastAPI](https://fastapi.tiangolo.com/)
*   [Streamlit](https://streamlit.io/)
*   [Prometheus](https://prometheus.io/)
*   [Grafana](https://grafana.com/)
*   [Docker](https://www.docker.com/)
*   [The "legal-document-assistant" GitHub project](https://github.com/lixx21/legal-document-assistant) for inspiring evaluation ideas.

## 📄 License

This project is licensed under the [Choose Your License - e.g., MIT License]. See the `LICENSE` file for details.

---

Remember to add a `LICENSE` file to your repository if you haven't already. This comprehensive README should make your project much easier to understand and use! Good luck on GitHub!
