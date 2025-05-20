# rag_app/main.py
import os
import argparse
import sys
from dotenv import load_dotenv # Used for loading .env file if running locally
from typing import List, Dict, Any

# --- LangChain components for R (Retrieval) ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings # Using SentenceTransformer for local embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# --- Google Generative AI components for G (Generation) ---
# Import the core library. We will use dictionary format for content/config
import google.generativeai as genai
# We will *not* import 'types' directly here, as it seems to cause issues with the installed version
# from google.generativeai import types # REMOVED OR COMMENTED OUT
import google.api_core.exceptions # Import specific exception types


# --- Configuration Constants ---
# Paths inside the Docker container (must match docker-compose.yml volumes)
DOCS_DIR = "/app/docs"
PERSIST_DIR = "/app/chroma_db"
COLLECTION_NAME = "legal_documents_collection" # Name for the ChromaDB collection

# --- Model Names ---
# Embedding Model (local, SentenceTransformer)
# This model is downloaded by the sentence-transformers library during indexing.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # A good, fast, small embedding model

# Google Gemini LLM Model (used via Google Generative AI API)
# Ensure this specific model is available via your API key.
GOOGLE_LLM_MODEL = "gemini-2.0-flash"


# Splitting parameters for breaking documents into chunks
CHUNK_SIZE = 1000 # Maximum number of characters per chunk
CHUNK_OVERLAP = 200 # Number of characters to overlap between chunks

# Retrieval parameters
K_RETRIEVE = 5 # Number of top relevant document chunks to retrieve from the vector store


# --- Component Initializers ---

def get_embedding_function():
    """Initializes and returns the SentenceTransformer embedding model."""
    print(f"Initializing local embedding model: {EMBEDDING_MODEL_NAME}")
    # SentenceTransformer models are downloaded locally (often from Hugging Face Hub).
    # This process is handled by the library and does not require an API key here.
    # You might see Hugging Face related output during the first run as the model downloads.
    # The LangChainDeprecationWarning about HuggingFaceEmbeddings is from LangChain itself,
    # indicating an alternative way to use HF embeddings in newer packages. This warning is informational.
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# The Google Generative AI client configuration is handled directly in LegalRagSystem.__init__
# using genai.configure(api_key=api_key)


# --- Core Classes ---

class DocumentProcessor:
    """Handles loading and splitting documents from a directory."""
    def __init__(self, docs_directory: str, chunk_size: int, chunk_overlap: int):
        self.docs_directory = docs_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        # Define a mapping from file extensions to LangChain document loaders
        self.loader_mapping = {
            ".pdf": PyPDFLoader, # Requires pypdf package
            ".txt": TextLoader,  # Built-in LangChain loader
            # Add mappings for other file types if needed:
            # ".docx": DocxLoader, # Requires python-docx and potentially a newer langchain-community or specific package
        }

    def load_documents(self) -> List[Document]:
        """Loads documents from the configured directory, handling different file types."""
        print(f"Loading documents from {self.docs_directory}...")
        if not os.path.exists(self.docs_directory):
             print(f"Error: Document directory not found: {self.docs_directory}", file=sys.stderr)
             return [] # Return empty list if the directory doesn't exist

        all_documents = []
        # Walk through the directory to find files recursively
        for root, _, files in os.walk(self.docs_directory):
            for file in files:
                file_path = os.path.join(root, file)
                _, file_extension = os.path.splitext(file_path)
                loader_class = self.loader_mapping.get(file_extension.lower())

                if loader_class:
                    try:
                        print(f"Loading {file_path}...")
                        loader = loader_class(file_path)
                        # Load the documents from the file and extend the list
                        docs = loader.load()
                        all_documents.extend(docs)
                    except Exception as e:
                        # Catch and report errors for individual files
                        print(f"Error loading file {file_path}: {e}", file=sys.stderr)
                else:
                    # Report files with unsupported extensions
                    print(f"Skipping unsupported file type: {file_path}", file=sys.stderr)

        print(f"Loaded {len(all_documents)} documents in total.")
        return all_documents


    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of LangChain Document objects into smaller chunks."""
        print("Splitting documents into chunks...")
        if not documents:
            print("No documents to split.")
            return []
        # Use the configured text splitter to split the loaded documents
        split_docs = self.text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks.")
        return split_docs


class VectorStoreManager:
    """Handles interactions with the ChromaDB vector store (creation and loading)."""
    def __init__(self, persist_directory: str, collection_name: str, embedding_function):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        # Ensure the directory where the vector store data will be saved exists
        os.makedirs(self.persist_directory, exist_ok=True)

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        """Creates a new ChromaDB collection or adds documents to an existing one."""
        print(f"Creating/Updating vector store '{self.collection_name}' at {self.persist_directory}...")
        # Chroma.from_documents handles the process:
        # - Initializes/connects to ChromaDB at persist_directory.
        # - Gets or creates the specified collection.
        # - Embeds the provided documents using the embedding_function.
        # - Adds the embedded documents to the collection.
        # Note: The LangChainDeprecationWarning for Chroma is from LangChain itself,
        # suggesting the separate `langchain-chroma` package for future use.
        # For this project, using the Chroma from `langchain_community` is perfectly fine.
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        # Count documents in the collection to confirm action
        print(f"Vector store creation/update complete. Total chunks: {vector_store._collection.count()}")
        return vector_store

    def load_vector_store(self) -> Chroma:
        """Loads an existing ChromaDB vector store from the persistence directory."""
        print(f"Loading vector store '{self.collection_name}' from {self.persist_directory}...")
        # Check if the persistence directory exists and is not empty
        # This is a basic check to see if indexing has likely been run before.
        if not os.path.exists(self.persist_directory) or not any(os.scandir(self.persist_directory)):
             # Raise a specific error if the index directory is not found or empty
             raise FileNotFoundError(
                 f"Vector store directory not found or is empty: {self.persist_directory}. "
                 "Please run indexing first (`python main.py --index`)."
            )

        try:
            # Initialize ChromaDB client and get the collection
            # Chroma will automatically load data from the persist_directory if it exists.
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name
            )
            # Further check if the collection itself contains any documents
            if vector_store._collection.count() == 0:
                 # This handles cases where the directory exists but the collection wasn't successfully populated
                 raise ValueError(
                     f"Vector store collection '{self.collection_name}' found but is empty. "
                     "Please run indexing (`python main.py --index`)."
                )

            print("Vector store loaded successfully.")
            return vector_store

        except Exception as e:
            # Catch any other errors during the loading process
            raise RuntimeError(f"Failed to load vector store collection '{self.collection_name}'. Error: {e}")


class LegalRagSystem:
    """Orchestrates the RAG process using ChromaDB retrieval and Google GenAI generation."""
    def __init__(self, api_key: str):
        # Initialize core components that are needed regardless of index/query mode
        # Embedding function uses local SentenceTransformer (no API key needed for this component)
        self.embedding_function = get_embedding_function()
        self.doc_processor = DocumentProcessor(DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
        self.vector_store_manager = VectorStoreManager(PERSIST_DIR, COLLECTION_NAME, self.embedding_function)

        # Configure the Google Generative AI library with the provided API key
        # This sets up the API access globally within this process.
        # If api_key is None (e.g., during index mode run without key), configuration is skipped.
        if api_key:
            print("Configuring Google Generative AI...")
            try:
                 # genai.configure sets the API key for subsequent calls
                 genai.configure(api_key=api_key)
                 print("Google Generative AI configured.")
            except Exception as e:
                 # Report error but don't necessarily exit, as indexing might still be possible.
                 # Query method will handle API errors if configuration failed or key was bad.
                 print(f"Warning: Error configuring Google Generative AI: {e}", file=sys.stderr)


    def index_documents(self):
        """Loads, splits, embeds, and indexes documents into the vector store."""
        print("\n--- Starting Indexing Process ---")
        # This process relies on DocumentProcessor and VectorStoreManager,
        # which use local file access and the local embedding function.
        # The Google GenAI API key is NOT needed for this step.

        documents = self.doc_processor.load_documents()
        if not documents:
            print("No documents loaded. Indexing aborted.")
            return

        split_docs = self.doc_processor.split_documents(documents)
        if not split_docs:
             print("No document chunks created. Indexing aborted.")
             return

        self.vector_store_manager.create_from_documents(split_docs)
        print("--- Indexing Process Finished ---")


    def query(self, query_text: str):
        """Retrieves relevant documents and uses Google GenAI to answer the query."""
        print(f"\n--- Processing Query: '{query_text}' ---")

        try:
            # 1. Load the vector store and create a retriever instance
            # This will raise an error if the index doesn't exist or is empty.
            vector_store = self.vector_store_manager.load_vector_store()
            # Create a retriever from the vector store to find similar document chunks
            retriever = vector_store.as_retriever(search_kwargs={"k": K_RETRIEVE})
            print(f"Retriever configured to fetch top {K_RETRIEVE} chunks.")

            # 2. Retrieve relevant documents based on the user's query
            print("Retrieving relevant documents...")
            # The retriever embeds the query using the same embedding function used for indexing
            # and performs a similarity search in the vector store.
            retrieved_docs: List[Document] = retriever.invoke(query_text)
            print(f"Retrieved {len(retrieved_docs)} document chunks.")

            if not retrieved_docs:
                print("No relevant documents found in the index. Cannot answer the query based on the documents.", file=sys.stderr)
                return

            # 3. Format the retrieved context for the LLM prompt
            # Combine the content of the retrieved documents into a single string for the LLM.
            context_parts = []
            source_metadata = set() # Use a set to store unique source information for printing later
            for i, doc in enumerate(retrieved_docs):
                # Add the content of each retrieved document chunk
                context_parts.append(f"Document Chunk {i+1}:\n{doc.page_content}\n\n")

                # Collect metadata about the source documents
                source = doc.metadata.get('source', 'N/A') # Get source file path
                page = doc.metadata.get('page', 'N/A')     # Get page number (common for PDF loaders)

                if source != 'N/A':
                     # Extract just the base file name for cleaner output
                     source_base_name = os.path.basename(source)
                     # Format the source string depending on available metadata (e.g., include page number)
                     if isinstance(source, str) and 'page' in doc.metadata and page is not None:
                         source_metadata.add(f"{source_base_name} (Page: {page})")
                     elif isinstance(source, str):
                         source_metadata.add(source_base_name)
                     else: # Fallback for unexpected source metadata format
                         source_metadata.add(str(source))

            formatted_context = "".join(context_parts).strip() # Join all chunk contents

            # 4. Prepare the prompt for the Google GenAI API using dictionary format
            # Create a clear instruction prompt for the LLM, including the retrieved context and the user's question.
            rag_prompt_text = f"""
You are a helpful legal document assistant. Answer the following question based *only* on the provided context documents.
If you cannot find the answer in the context, clearly state that you cannot find the information in the documents.
Do not make up information.

Context Documents:
---
{formatted_context}
---

Question:
{query_text}

Answer:
"""
            # Prepare the 'contents' list using dictionary format as expected by the API
            # This represents the conversation history. For a single RAG query, one user turn is sufficient.
            contents = [
                {
                    "role": "user", # The role sending the content
                    "parts": [
                        {"text": rag_prompt_text}, # The actual text content
                    ],
                }
            ]

            # 5. Define generation configuration for the LLM call using dictionary format
            generation_config = {
                "temperature": 0.0, # Lower temperature encourages more deterministic/factual responses
                # "max_output_tokens": 512 # Optional: set a maximum number of tokens for the response
            }

            # 6. Get the Generative Model instance and make the API call
            print(f"Calling Google Gemini model '{GOOGLE_LLM_MODEL}'...")
            try:
                # Get the specific model instance (uses the global configuration set by genai.configure)
                model = genai.GenerativeModel(model_name=GOOGLE_LLM_MODEL)

                # Call the generate_content method to get a response
                # Pass contents and generation_config as dictionaries.
                response = model.generate_content(
                    contents=contents, # The prompt contents (list of dicts)
                    generation_config=generation_config, # Configuration settings (dict)
                    # stream=True # Uncomment if you want to handle streaming responses
                )

                # 7. Process and print the generated answer
                # Check if the response contains valid candidates and content
                if response and response.candidates:
                    # Access the text from the first candidate's first content part
                    # The structure is response.candidates[index].content.parts[index].text
                    if response.candidates[0].content and response.candidates[0].content.parts:
                         # Check if the first part is a text part and has text content
                         if response.candidates[0].content.parts[0].text:
                            answer = response.candidates[0].content.parts[0].text
                            print("\n--- Answer ---")
                            print(answer)
                         else:
                             print("\n--- Answer ---")
                             print("Received content part but no text content.", file=sys.stderr)
                             # Optional: Print the part for debugging
                             # print(f"Content Part: {response.candidates[0].content.parts[0]}", file=sys.stderr)
                    else:
                         # Handle cases where candidates or their content/parts are missing/empty
                         print("\n--- Answer ---")
                         print("No valid content or content parts received from the model response.", file=sys.stderr)
                         # Optional: Print finish reason/safety ratings
                         # print(f"Finish Reason: {response.candidates[0].finish_reason}", file=sys.stderr)
                         # print(f"Safety Ratings: {response.candidates[0].safety_ratings}", file=sys.stderr)
                else:
                    # Handle cases where the API returned no candidates
                    print("\n--- Answer ---")
                    print("No candidates received from the model response.", file=sys.stderr)
                    # Optional: Check response.prompt_feedback for issues with the prompt

                # 8. Print the sources that were used to generate the answer
                print("\n--- Sources ---")
                if source_metadata:
                    print("Information based on these documents/pages:")
                    # Print unique sources sorted alphabetically
                    for src_info in sorted(list(source_metadata)):
                        print(f"- {src_info}")
                else:
                    # Indicate if no sources were identified (usually because retrieval failed or was empty)
                    print("No specific sources identified in the retrieved context metadata.")


            # --- Specific Google API Error Handling ---
            # Catch known Google API exceptions for informative error messages
            except google.api_core.exceptions.NotFound as e:
                 # Model not found or not available for the API key
                 print(f"API Error (NotFound): The model '{GOOGLE_LLM_MODEL}' was not found or is not available. Check model name and API key permissions. Details: {e}", file=sys.stderr)
            except google.api_core.exceptions.PermissionDenied as e:
                 # API key does not have permission to use the Generative Language API or the model
                 print(f"API Error (PermissionDenied): Permission denied. Check your GOOGLE_API_KEY and ensure the Generative Language API is enabled for your Google Cloud project/account. Details: {e}", file=sys.stderr)
            except google.api_core.exceptions.DeadlineExceeded as e:
                 # The API request took too long to complete
                 print(f"API Error (DeadlineExceeded): Request timed out. The model might be slow or under heavy load. Details: {e}", file=sys.stderr)
            except google.api_core.exceptions.InvalidArgument as e:
                 # The request payload (prompt, config) was invalid
                 print(f"API Error (InvalidArgument): Invalid argument provided to the API. This could be due to prompt formatting or configuration. Details: {e}", file=sys.stderr)
                 # Optional: Print the contents sent if debugging prompt issues
                 # print(f"Problematic contents: {contents}", file=sys.stderr)
            except google.api_core.exceptions.GoogleAPIError as e:
                 # Catch any other unhandled Google API errors
                 print(f"API Error (GoogleAPIError): An unexpected Google API error occurred. Details: {e}", file=sys.stderr)
            # --- End Specific Google API Error Handling ---
            except Exception as e:
                # Catch any other unexpected errors during the query execution process (e.g., processing response)
                print(f"An unexpected error occurred during query execution: {e}", file=sys.stderr)

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            # Catch errors specifically from loading the vector store (handled by VectorStoreManager)
            print(f"Query failed: {e}", file=sys.stderr)
        except Exception as e:
            # Catch any other unexpected errors before the API call (e.g., during retrieval)
            print(f"An unexpected error occurred before calling the API: {e}", file=sys.stderr)


# --- Main Application Entry Point ---

if __name__ == "__main__":
    # Load environment variables from the .env file in the project root.
    # This happens automatically in Docker Compose, but is useful if running main.py directly.
    load_dotenv()

    # --- Argument Parsing ---
    # Set up command-line argument parsing to choose between indexing and querying.
    parser = argparse.ArgumentParser(description="Legal Document RAG System using Google GenAI")
    parser.add_argument(
        "--index",
        action="store_true", # This flag means if --index is present, args.index will be True
        help="Run the indexing process to build or update the vector store."
    )
    parser.add_argument(
        "--query",
        type=str, # This argument requires a string value (the query text)
        help="Run the query process with the specified question."
    )

    args = parser.parse_args() # Parse the command-line arguments

    # --- Environment Validation ---
    # Get the Google API key from environment variables.
    # This variable is expected to be set either in the shell or loaded from .env.
    google_api_key = os.getenv("GOOGLE_API_KEY")

    # Validate that the API key is set if the user is trying to run a query.
    # Indexing does NOT require the API key with this setup (local embeddings).
    if args.query and not google_api_key:
        print("Error: GOOGLE_API_KEY environment variable is not set.", file=sys.stderr)
        print("The Google Generative AI API key is required for query mode.", file=sys.stderr)
        print("Please set it in your .env file or your shell environment.", file=sys.stderr)
        sys.exit(1) # Exit the script with an error code

    # --- Action Execution ---
    # Instantiate the main LegalRagSystem class.
    # The API key is passed here. The __init__ method will attempt to configure genai if the key is valid.
    # If key is None (during index mode), genai configuration is skipped in __init__.
    rag_system = LegalRagSystem(api_key=google_api_key)

    # Based on the parsed arguments, perform either indexing or querying.
    if args.index:
        # If --index is specified, run the indexing process.
        print("Executing indexing process...")
        # Add a check to warn the user if the docs directory is empty before indexing.
        if not os.path.exists(DOCS_DIR) or not any(os.scandir(DOCS_DIR)):
             print(f"Warning: No documents found in '{DOCS_DIR}'. Indexing will result in an empty index.", file=sys.stderr)
        rag_system.index_documents()
        print("Indexing process finished.")

    elif args.query:
        # If --query is specified, run the querying process.
        # Check if the query string is empty or contains only whitespace.
        if not args.query or not args.query.strip():
            print("Error: --query argument requires a non-empty question.", file=sys.stderr)
            parser.print_help(sys.stderr) # Print help message to stderr
            sys.exit(1) # Exit with an error code

        print(f"Executing query process for question: '{args.query}' using model: {GOOGLE_LLM_MODEL}.")
        rag_system.query(args.query)
        print("Query process finished.")

    else:
        # If neither --index nor --query is specified, print usage instructions.
        print("No action specified.", file=sys.stderr)
        print("Please specify either --index to build the index or --query '<Your question>' to ask a question.", file=sys.stderr)
        parser.print_help(sys.stderr) # Print help message to stderr
        sys.exit(1) # Exit with an error code