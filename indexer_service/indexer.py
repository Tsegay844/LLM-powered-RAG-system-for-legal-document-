# indexer_service/indexer.py
# is about indexing legal documents into Elasticsearch using LangChain and SentenceTransformer embeddings.
# It loads documents from a specified directory, splits them into chunks, generates embeddings,
# and indexes them into Elasticsearch with robust error handling and retry logic.
import os
import sys
import time
import argparse 
from dotenv import load_dotenv
from typing import List, Dict, Any
from uuid import uuid4 

# LangChain components for document loading/splitting/embedding
# LangChain is a framework for building applications with language models.
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document


# This is the official Elasticsearch Python client
# It provides a high-level API for interacting with Elasticsearch clusters. 
# It supports bulk indexing, connection management, and error handling.
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionError as ESConnectionError, TransportError as ESTransportError

# Configuration Constants (Read from Environment Variables)
DOCS_DIR = os.getenv("DOCS_DIR", "/app/docs") 
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "elasticsearch") 
ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", 9200)) 
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "legal_docs") # 
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2") 
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000)) 
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200)) 


# This function connects to the Elasticsearch cluster with retry logic.
# It attempts to connect multiple times with exponential backoff,
# handling common connection errors gracefully.
def connect_elasticsearch():
    """Connects to Elasticsearch with retry logic."""
    print(f"Attempting to connect to Elasticsearch at {ELASTICSEARCH_HOST}:{ELASTICSEARCH_PORT}")
    es_client = None
    for i in range(15): # Retry up to 15 times
        try:
            es_client = Elasticsearch(
                hosts=[{"host": ELASTICSEARCH_HOST, "port": ELASTICSEARCH_PORT, "scheme": "http"}],
                timeout=30,
                sniff_on_start=False, 
                
                
            )
            
            es_client.info()
            print("Connected to Elasticsearch.")
            return es_client
        except (ESConnectionError, ESTransportError) as e:
            print(f"Elasticsearch connection failed (attempt {i+1}/15): {e}. Retrying in 10 seconds...", file=sys.stderr)
            time.sleep(10) # Wait longer between retries
        except Exception as e:
             print(f"An unexpected error occurred during ES connection (attempt {i+1}/15): {e}. Retrying in 10 seconds...", file=sys.stderr)
             time.sleep(10)

    print("Failed to connect to Elasticsearch after multiple retries. Exiting.", file=sys.stderr)
    sys.exit(1) # Exit if connection fails

# This function creates an Elasticsearch index if it does not exist.
# If the index exists and recreate is True, it deletes the existing index and creates a new one.
# It defines the mapping for the index, specifically for the vector field,
# using the 'dense_vector' type with HNSW for efficient similarity search.
def create_index_if_not_exists(es_client: Elasticsearch, index_name: str, vector_dimension: int, recreate: bool = False):
    """Creates or recreates the Elasticsearch index with mapping."""
    index_exists = es_client.indices.exists(index=index_name)

    if index_exists and recreate:
        print(f"Index '{index_name}' exists and recreate=True. Deleting existing index...")
        try:
            es_client.indices.delete(index=index_name, ignore=[400, 404]) # Ignore 400/404 errors if index doesn't exist
            print(f"Index '{index_name}' deleted.")
            time.sleep(2) # Give ES a moment
        except Exception as e:
            print(f"Error deleting index '{index_name}': {e}", file=sys.stderr)
            sys.exit(1) # Critical failure

    if not index_exists or recreate:
        print(f"Creating index '{index_name}' with vector mapping...")
        # Define the mapping for the index, specifically for the vector field
        # Using the 'dense_vector' type with HNSW for efficient similarity search
        mapping = {
            "properties": {
                "text": {"type": "text"},
                "source": {"type": "keyword"}, # Source file path (use keyword for exact matching/filtering)
                "page": {"type": "integer"}, 
                "vector": {
                    "type": "dense_vector",
                    "dims": vector_dimension,
                    "index": True,
                    "similarity": "cosine" # Or "l2_norm", "dot_product" depending on your needs/model
                }
            }
        }
        # Optional settings for the index : like number of shards/replicas for scaling
        # Aadjust these settings based on your scaling and performance needs
        settings = {
            "index": {
                # "number_of_shards": 1, # Adjust for scaling
                # "number_of_replicas": 0 # Adjust for high availability
            }
        }
        try:
            es_client.indices.create(index=index_name, settings=settings, mappings=mapping)
            print(f"Index '{index_name}' created successfully with mapping.")
        except Exception as e:
            print(f"Error creating index '{index_name}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
         print(f"Index '{index_name}' already exists and recreate=False. Skipping index creation.")


# Document Processing 
# This class handles loading and splitting documents from a specified directory.
# It supports multiple file types (PDF, TXT) and uses LangChain's text splitting capabilities to create manageable chunks for indexing.
class DocumentProcessor:
    """Handles loading and splitting documents from a directory."""
    def __init__(self, docs_directory: str, chunk_size: int, chunk_overlap: int):
        self.docs_directory = docs_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.loader_mapping = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            # ".docx": DocxLoader, # Requires python-docx
        }

    # Loads documents from the specified directory, handling different file types.
    # It uses a mapping of file extensions to LangChain loaders.
    # If a file type is unsupported, it skips the file and logs a warning.
    # Returns a list of LangChain Document objects.
    def load_documents(self) -> List[Document]:
        """Loads documents from the configured directory, handling different file types."""
        print(f"Loading documents from {self.docs_directory}...")
        if not os.path.exists(self.docs_directory):
             print(f"Error: Document directory not found: {self.docs_directory}", file=sys.stderr)
             return []

        all_documents = []
        for root, _, files in os.walk(self.docs_directory):
            for file in files:
                file_path = os.path.join(root, file)
                _, file_extension = os.path.splitext(file_path)
                loader_class = self.loader_mapping.get(file_extension.lower())

                if loader_class:
                    try:
                        print(f"Loading {file_path}...")
                        loader = loader_class(file_path)
                        docs = loader.load()
                        all_documents.extend(docs)
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}", file=sys.stderr)
                else:
                    print(f"Skipping unsupported file type: {file_path}", file=sys.stderr)

        print(f"Loaded {len(all_documents)} documents in total.")
        return all_documents


    # Splits a list of LangChain Document objects into smaller chunks using the configured text splitter.
    # This is useful for creating manageable pieces of text for indexing.
    # Each chunk will have a maximum size defined by chunk_size, with some overlap defined by chunk_overlap.
    # Returns a list of Document objects, each representing a chunk of text.   
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of LangChain Document objects into smaller chunks."""
        print("Splitting documents into chunks...")
        if not documents:
            print("No documents to split.")
            return []
        split_docs = self.text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks.")
        return split_docs

#Embedding
# This function initializes the SentenceTransformer embedding model.
# It uses the model name specified in the environment variable all-MiniLM-L6-v2.
# It returns an instance of SentenceTransformerEmbeddings().
# SentenceTransformerEmbeddings is a LangChain wrapper around the popular
# SentenceTransformers library, which provides state-of-the-art models for
# generating dense vector embeddings from text. These embeddings are used for
# semantic search, similarity, and retrieval tasks. The wrapper allows you to
# easily use SentenceTransformer models within LangChain pipelines.
def get_embedding_function():
    """Initializes and returns the SentenceTransformer embedding model."""
    print(f"Initializing local embedding model: {EMBEDDING_MODEL}")
    # Using SentenceTransformer for embeddings
    try:
        embedding = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)       
        embedding.embed_query("test")
        return embedding
    except Exception as e:
        print(f"Error initializing or loading embedding model '{EMBEDDING_MODEL}': {e}", file=sys.stderr)
        print("Ensure the model name is correct and your system has enough resources.", file=sys.stderr)
        sys.exit(1) # Exit if embedding model fails to load


# Indexing Logic
# This function generates Elasticsearch bulk indexing actions from a list of LangChain Document objects.
# It uses the embedding function to generate embeddings for each document chunk.
# Each action is a dictionary that contains the document ID, index name, and source data.
def generate_actions(documents: List[Document], embedding_function):
    """Generates Elasticsearch bulk indexing actions from documents."""
    print("Generating embeddings and indexing actions for Elasticsearch...")
    for i, doc in enumerate(documents):
        # Add a check for empty page_content before embedding
        if not doc.page_content or not doc.page_content.strip():
             print(f"Skipping empty document chunk {i+1}.", file=sys.stderr)
             continue

        try:
            # Generate embedding for the document chunk
            embedding = embedding_function.embed_query(doc.page_content)

            # Prepare the document body for Elasticsearch
            # Note: We store original text, source, page, and the vector
            es_doc = {
                "_id": str(uuid4()), # Generate a unique ID for each chunk
                "_index": ES_INDEX_NAME,
                "_source": {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),   
                    "vector": embedding
                }
            }
            yield es_doc # Yield the action for bulk indexing
        except Exception as e:
            print(f"Error processing document chunk {i+1}: {e}. Skipping chunk.", file=sys.stderr)
            # Optionally log the chunk content or identifier

# This function orchestrates the entire indexing process.
# It connects to Elasticsearch, creates the index if needed, loads and splits documents,
# generates embeddings, and indexes the documents using the bulk helper.
def run_indexing(recreate_index: bool = False):
    """Main function to orchestrate the indexing process."""
    print("--- Starting Indexing Process ---")

    # 1. Connect to Elasticsearch
    es_client = connect_elasticsearch()

    # 2. Get embedding function and determine vector dimension
    embedding_function = get_embedding_function()
    try:
         vector_dimension = len(embedding_function.embed_query("test query"))
         print(f"Determined embedding vector dimension: {vector_dimension}")
    except Exception as e:
         print(f"Could not determine embedding vector dimension: {e}. Please check embedding model.", file=sys.stderr)
         sys.exit(1)

    # 3. Create or recreate Elasticsearch index
    create_index_if_not_exists(es_client, ES_INDEX_NAME, vector_dimension, recreate=recreate_index)

    # 4. Load and split documents
    doc_processor = DocumentProcessor(DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
    documents = doc_processor.load_documents()
    if not documents:
        print(f"No documents loaded from {DOCS_DIR}. Indexing aborted.")
        return

    split_docs = doc_processor.split_documents(documents)
    if not split_docs:
         print("No document chunks created after splitting. Indexing aborted.")
         return

    # 5. Index documents in Elasticsearch using bulk helper
    print(f"Indexing {len(split_docs)} document chunks into Elasticsearch index '{ES_INDEX_NAME}'...")
    success_count = 0
    # Use bulk helper for efficient indexing
    # chunk_size determines how many documents are sent in one batch
    # max_retries and initial_backoff help handle transient errors
    try:
        for ok, item in helpers.bulk(
            es_client,
            generate_actions(split_docs, embedding_function), # Generator function for actions
            index=ES_INDEX_NAME, # Specify the index name here
            chunk_size=1000, # Number of actions per bulk API call
            request_timeout=60, # Timeout for each bulk request
            max_retries=5,
            initial_backoff=2, # Start with a 2-second backoff
            yield_ok=False # Yield back only information about failed items
        ):
            # The 'ok' variable is a boolean indicating if the item was indexed successfully
            # The 'item' variable contains the result/error details
            if ok:
                success_count += 1
            else:
                # 'item' is a dictionary with {'index': {...}} details including 'error'
                print(f"Failed to index item: {item}", file=sys.stderr)

        # If yield_ok=False, helpers.bulk returns a tuple: (success_count, errors)
        # Let's re-implement the counting logic to be more explicit
        print("Bulk indexing process completed.")
        # Note: helpers.bulk returns (success_count, errors) tuple if yield_ok=False
        # Let's just count the items yielded back as failures if yield_ok=False
        # Or set yield_ok=True and count successes/failures explicitly in the loop

    except Exception as e:
        print(f"An error occurred during bulk indexing: {e}", file=sys.stderr)
        print("Indexing process may be incomplete.", file=sys.stderr)
        # Do not exit, allow the process to finish even if some items failed

    # Re-count the documents in the index to get the final count
    try:
        es_count = es_client.count(index=ES_INDEX_NAME)['count']
        print(f"Total documents in Elasticsearch index '{ES_INDEX_NAME}' after indexing: {es_count}.")
        # Note: This count includes documents from previous runs if recreate_index was False
    except Exception as e:
        print(f"Could not get final count from Elasticsearch: {e}", file=sys.stderr)


    print("--- Indexing Process Finished ---")

# --- Main Execution ---

if __name__ == "__main__":
    # Load environment variables from the .env file (useful for local testing)
    load_dotenv()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Legal Document RAG System Indexer")
    parser.add_argument(
        "--recreate-index",
        action="store_true", # If this flag is present, recreate the index
        help="Delete the existing Elasticsearch index and recreate it before indexing."
    )

    args = parser.parse_args() # Parse command-line arguments

    # Check if document directory exists and has files before attempting index
    if not os.path.exists(DOCS_DIR) or not any(os.scandir(DOCS_DIR)):
         print(f"Warning: No documents found in '{DOCS_DIR}'. Indexing will result in an empty index.", file=sys.stderr)
         # Exit gracefully if no documents are found
         # sys.exit(0) # Or allow to proceed, resulting in an empty index

    # Run the indexing process
    run_indexing(recreate_index=args.recreate_index)