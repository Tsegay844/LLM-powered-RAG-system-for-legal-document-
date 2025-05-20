# rag_app/main.py

# --- Basic Python Imports ---
import os
import argparse
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any
# --- End Basic Python Imports ---

# --- LangChain and Related Library Imports ---
# Document Loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Vector Store (ChromaDB)
from langchain_community.vectorstores import Chroma
# Embedding Models
from langchain_community.embeddings import SentenceTransformerEmbeddings # For local embeddings
# LLM Wrapper (Hugging Face Inference API)
from langchain_huggingface import HuggingFaceEndpoint # For HF Inference API connection
# Core LangChain Chain Components and Primitives
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document # Represents a document chunk with metadata
from langchain_core.vectorstores import VectorStoreRetriever # Type hint for retriever
from langchain_core.runnables import Runnable # Type hint for chains

# --- REMOVED HuggingfaceHubException IMPORT as it caused ImportError ---
# --- If needed for specific HF API errors, you might need to catch requests.exceptions.RequestException or a more general Exception ---


# --- Configuration Constants (Read from environment variables, with defaults) ---
# Paths inside the Docker container (must match docker-compose.yml volumes)
DOCS_DIR = "/app/docs"
PERSIST_DIR = "/app/chroma_db"
COLLECTION_NAME = "legal_documents_collection" # A clear name for the ChromaDB collection

# --- Models configured for Hugging Face Inference API (LLM) and Local (Embeddings) ---
# Read from environment variables, provide defaults if not set
# Using a default model and task that previously worked in your direct API test
# User wants google/flan-t5-large for text2text-generation as primary goal
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "google/flan-t5-large") # Default LLM model
HF_LLM_TASK = os.getenv("HF_LLM_TASK", "text2text-generation") # Default task for LLM

LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2") # Default embedding model


# Splitting parameters
CHUNK_SIZE = 1000 # Max characters per chunk
CHUNK_OVERLAP = 200 # Overlap between chunks

# Retrieval parameters
K_RETRIEVE = 5 # Number of relevant document chunks to retrieve


# --- Component Initializers ---

def get_embedding_function():
    """Initializes and returns the embedding model (SentenceTransformer)."""
    print(f"Initializing local embedding model: {LOCAL_EMBEDDING_MODEL}")
    try:
        return SentenceTransformerEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error initializing SentenceTransformer embedding model {LOCAL_EMBEDDING_MODEL}: {e}", file=sys.stderr)
        print("Ensure model name is correct and internet access is available for initial download.", file=sys.stderr)
        raise


def get_llm():
    """Initializes and returns the language model (Hugging Face Inference API) using HuggingFaceEndpoint."""
    # Print the actual model and task being used based on env vars/defaults
    current_hf_model = os.getenv("HF_LLM_MODEL", "google/flan-t5-large")
    current_hf_task = os.getenv("HF_LLM_TASK", "text2text-generation")
    print(f"Initializing Hugging Face LLM model: {current_hf_model} ({current_hf_task} task)")

    # HuggingFaceEndpoint automatically reads the HUGGINGFACEHUB_API_TOKEN environment variable.
    # We still check for it in the main block for a cleaner startup error message.

    try:
        # HuggingFaceEndpoint connects to the Inference API.
        # repo_id: The model ID on Hugging Face Hub.
        # task: The specific task the model performs (must match the model's capability).
        # model_kwargs: Optional parameters for the model (e.g., temperature, max_length).
        # Add model_kwargs only if the model/task supports them and you need them.
        llm = HuggingFaceEndpoint(
            repo_id=current_hf_model,
            task=current_hf_task,
            # model_kwargs={"temperature": 0.0, "max_new_tokens": 500} # Example kwargs for generation tasks
        )

        # Perform a simple invoke test to catch API connection/model issues early
        print("Performing initial LLM test invoke...")
        # --- CORRECTED TEST INVOKE INPUT ---
        # Pass a simple string to llm.invoke(), as this is what the wrapper expects
        test_prompt_input = "This is a test sentence." # Simple string input for the test invoke
        # --- End CORRECTED TEST INVOKE INPUT ---

        print(f"Test Prompt Input (simple string): '{test_prompt_input}'")

        try:
             # Invoke the LLM with the test prompt input
             test_response = llm.invoke(test_prompt_input)
             # Print only a snippet to avoid flooding output for long responses
             print(f"Initial LLM test invoke successful. Response type: {type(test_response)}. Response start: '{str(test_response)[:100]}...'")
        except Exception as test_e:
             print(f"Initial LLM test invoke failed during get_llm test invoke: {test_e}", file=sys.stderr)
             print("This indicates a problem with the LLM connection or the model/task configuration, not the test input format itself now.", file=sys.stderr)
             # Re-raise the test error
             raise test_e # Re-raise the exception here

        return llm
    except Exception as e:
        # Catch exceptions during HuggingFaceEndpoint initialization or the test invoke within get_llm
        print(f"Error initializing Hugging Face LLM model {current_hf_model} with task {current_hf_task}: {e}", file=sys.stderr)
        print("Possible issues:", file=sys.stderr)
        print("- Invalid API token (HUGGINGFACEHUB_API_TOKEN).", file=sys.stderr)
        print("- The model name (repo_id) or task is incorrect or unavailable on the Inference API.", file=sys.stderr)
        print("- Network issues preventing connection to Hugging Face.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr) # Print the specific exception details
        # Re-raise the exception so the calling function (_initialize_rag_chain or test_llm_direct) handles the failure
        raise # Re-raise the exception


def get_prompt_template():
    """Defines the prompt template for the RAG chain."""
    print("Defining prompt template...")
    # This prompt should work with most generative LLMs.
    # The RAG chain (create_stuff_documents_chain) handles inserting context and input into this template.
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful legal document assistant. Answer the following question based *only* on the provided context.
    If you cannot find the answer in the context, clearly state that you cannot find the information in the documents.
    Do not make up information.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """)
    return prompt

# --- Core Classes ---

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
        # Define specific loaders for different file types
        self.loader_mapping = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            # Add more as needed:
            # ".docx": DocxLoader, # Requires python-docx library
        }

    def load_documents(self) -> List[Document]:
        """Loads documents from the configured directory."""
        print(f"Loading documents from {self.docs_directory}...")
        if not os.path.exists(self.docs_directory):
             print(f"Error: Document directory not found: {self.docs_directory}", file=sys.stderr)
             return [] # Return empty list if directory doesn't exist

        all_documents = []
        # Walk through the directory and subdirectories
        for root, _, files in os.walk(self.docs_directory):
            for file in files:
                file_path = os.path.join(root, file)
                _, file_extension = os.path.splitext(file_path)
                # Get the appropriate loader class based on file extension
                loader_class = self.loader_mapping.get(file_extension.lower())

                if loader_class:
                    try:
                        print(f"Loading {file_path}...")
                        loader = loader_class(file_path)
                        # Extend the main document list with loaded documents from this file
                        docs = loader.load()
                        all_documents.extend(docs)
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}", file=sys.stderr)
                else:
                    print(f"Skipping unsupported file type: {file_path}", file=sys.stderr)

        print(f"Loaded {len(all_documents)} documents in total.")
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of documents into smaller chunks."""
        print("Splitting documents into chunks...")
        if not documents:
            print("No documents to split.")
            return []
        split_docs = self.text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks.")
        return split_docs


class VectorStoreManager:
    """Handles interactions with the ChromaDB vector store."""
    def __init__(self, persist_directory: str, collection_name: str, embedding_function):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function # The embedding function is needed to query the store

        # Ensure persistence directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        """Creates or updates the vector store with documents."""
        print(f"Creating/Updating vector store '{self.collection_name}' at {self.persist_directory}...")
        # Chroma.from_documents handles creating the collection if it doesn't exist.
        # By default, it ADDS documents if the collection exists, it doesn't replace.
        # For a simple project, re-running index might duplicate content.
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        print(f"Vector store creation/update complete. Total chunks: {vector_store._collection.count()}")
        return vector_store

    def load_vector_store(self) -> Chroma:
        """Loads an existing vector store."""
        print(f"Loading vector store '{self.collection_name}' from {self.persist_directory}...")
        # Check if the directory exists and contains files (basic check)
        if not os.path.exists(self.persist_directory) or not any(os.scandir(self.persist_directory)):
             raise FileNotFoundError(f"Vector store directory not found or is empty: {self.persist_directory}. Please run indexing first using '--index'.")

        try:
            # Initialize Chroma client and get the collection.
            # Chroma will automatically load data from the specified persist_directory.
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function, # Must provide embedding function even for loading
                collection_name=self.collection_name
            )
            # Check if the collection is actually populated with documents
            if vector_store._collection.count() == 0:
                 raise ValueError(f"Vector store collection '{self.collection_name}' found but is empty. Please run indexing using '--index'.")

            print("Vector store loaded successfully.")
            return vector_store

        except Exception as e:
            # Catch potential errors during ChromaDB loading (e.g., corrupted files, permissions)
            raise RuntimeError(f"Failed to load vector store collection '{self.collection_name}'. Error: {e}")


class LegalRagSystem:
    """Orchestrates the RAG process."""
    def __init__(self):
        # Initialize components that are needed regardless of index/query mode
        # Embedding function must be initialized successfully.
        try:
            self.embedding_function = get_embedding_function() # Needs internet for download on first run
        except Exception as e:
             print(f"FATAL ERROR during Embedding Function initialization: {type(e).__name__} - {e}", file=sys.stderr)
             # Exit immediately as the system cannot function without embeddings
             sys.exit(1)

        self.doc_processor = DocumentProcessor(DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
        # Pass the initialized embedding function to the VectorStoreManager
        self.vector_store_manager = VectorStoreManager(PERSIST_DIR, COLLECTION_NAME, self.embedding_function)

        # Components initialized lazily only when needed (query or direct LLM test)
        self.llm = None
        self.retriever: VectorStoreRetriever = None
        self.rag_chain: Runnable = None


    def index_documents(self):
        """Loads documents, splits them into chunks, and indexes them into the vector store."""
        print("\n--- Starting Indexing Process ---")
        documents = self.doc_processor.load_documents()
        if not documents:
            print("No documents loaded from the directory. Indexing aborted.", file=sys.stderr)
            return

        split_docs = self.doc_processor.split_documents(documents)
        if not split_docs:
             print("No document chunks created from the loaded documents. Indexing aborted.", file=sys.stderr)
             return

        # Create/Update the vector store with the processed documents
        self.vector_store_manager.create_from_documents(split_docs)
        print("--- Indexing Process Finished ---")


    def _initialize_rag_chain(self):
         """Initializes LLM, loads vector store/retriever, and creates the RAG chain."""
         # Only initialize if the chain hasn't been created yet
         if self.rag_chain:
              return

         print("\n--- Initializing RAG Chain Components ---")
         try:
             # 1. Load the vector store and create a retriever instance from it
             # load_vector_store needs the embedding_function which was initialized in __init__
             vector_store = self.vector_store_manager.load_vector_store()
             self.retriever = vector_store.as_retriever(search_kwargs={"k": K_RETRIEVE})
             print(f"Retriever configured to fetch top {K_RETRIEVE} chunks based on similarity.")

             # 2. Initialize the LLM (Hugging Face Inference API).
             # get_llm includes an internal test invoke to validate the connection early.
             # Exceptions during LLM init will be caught by the outer try...except block.
             self.llm = get_llm() # This might raise exceptions caught below

             # 3. Define the prompt template for instructing the LLM.
             prompt = get_prompt_template()

             # 4. Create the chain that combines retrieval and generation.
             # create_stuff_documents_chain works with both LLMs (like HuggingFaceEndpoint) and ChatModels.
             document_chain = create_stuff_documents_chain(self.llm, prompt)
             self.rag_chain = create_retrieval_chain(self.retriever, document_chain)
             print("RAG chain created successfully.")

         except (FileNotFoundError, ValueError, RuntimeError) as e:
             # Catch specific anticipated errors during initialization (Index/Load Issue)
             print(f"FATAL ERROR during RAG chain initialization (Index/Load Issue): {type(e).__name__} - {e}", file=sys.stderr)
             print("Please check the error message above for details on what failed (e.g., index not found, value error).", file=sys.stderr)
             sys.exit(1)
         except Exception as e: # Catch any other unexpected errors, including those from get_llm() or chain creation
              # This will now catch errors that were previously HuggingfaceHubException or others
              print(f"An unexpected FATAL ERROR occurred during RAG chain initialization (LLM/Chain Issue): {type(e).__name__} - {e}", file=sys.stderr)
              # Print detailed info for debugging unexpected errors
              print(f"Error Type: {type(e)}", file=sys.stderr)
              print(f"Error Details: {e}", file=sys.stderr)
              # Attempt to print more details if it's a requests-related error wrapped inside
              if hasattr(e, 'response') and hasattr(e.response, 'text'):
                   print(f"API Response Text: {e.response.text}", file=sys.stderr)
              sys.exit(1)


    def query(self, query_text: str):
        """Processes a user query by retrieving relevant document chunks and generating an answer using the LLM."""
        print(f"\n--- Processing Query: '{query_text}' ---")
        try:
            # Initialize the RAG chain if it hasn't been already.
            # This includes LLM initialization and its internal test invoke.
            self._initialize_rag_chain()

            # At this point, if _initialize_rag_chain didn't exit, self.rag_chain should be initialized.
            if self.rag_chain is None: # Check rag_chain instead of self.llm after _initialize_rag_chain
                 print("Error: RAG system query chain was not successfully initialized. Cannot process query.", file=sys.stderr)
                 sys.exit(1)

            # Invoke the RAG chain with the user's query.
            # The chain internally uses the retriever and the LLM.
            # The input to the RAG chain is a dictionary {'input': query_text}
            # The chain handles formatting the context and query for the LLM based on the prompt template.
            response: Dict[str, Any] = self.rag_chain.invoke({"input": query_text})

            print("\n--- Answer ---")
            # Get the 'answer' key from the response dictionary, provide default if missing.
            print(response.get("answer", "No answer was generated by the model based on the provided documents."))

            print("\n--- Sources ---")
            # Get the 'context' key, which contains the retrieved Document objects.
            context_docs: List[Document] = response.get("context", [])
            if context_docs:
                 unique_sources = set()

                 for doc in context_docs:
                     # Extract source and page from document metadata
                     # Metadata structure depends on the document loader used (e.g., PyPDFLoader adds 'source' and 'page')
                     source = doc.metadata.get('source', 'N/A')
                     page = doc.metadata.get('page', 'N/A') # For PDFLoader, 'page' is usually 0-indexed

                     if source != 'N/A':
                         # Format the source information for readability
                         # Use os.path.basename to get just the file name
                         source_base_name = os.path.basename(source)
                         if isinstance(source, str) and 'page' in doc.metadata and page is not None:
                             # Format for documents with page numbers (like PDFs)
                             unique_sources.add(f"{source_base_name} (Page: {page})")
                         elif isinstance(source, str):
                              # Format for documents without explicit page numbers (like TXT)
                             unique_sources.add(source_base_name)
                         else:
                             # Fallback for unexpected metadata format
                             unique_sources.add(str(source))

                 if unique_sources:
                     print("Information retrieved from:")
                     for src_info in sorted(list(unique_sources)):
                        print(f"- {src_info}")
                 else:
                     print("No specific sources identified with 'source' metadata in retrieved context documents.")

            else:
                 print("No context documents were returned by the retriever or chain.")

        except SystemExit as e:
             # Catch SystemExit explicitly raised during initialization errors
             print(f"Query execution halted due to a system initialization error: {e}", file=sys.stderr)
             sys.exit(1)
        except Exception as e:
            # Catch any other unexpected errors during query execution (after initialization)
            print(f"An unexpected error occurred during query execution: {type(e).__name__} - {e}", file=sys.stderr)
            # Add specific hints for common LangChain/LLM related errors
            if isinstance(e, ValueError) and "Could not inker input into prompt." in str(e):
                 print("Hint: This might be due to the combined length of the query and retrieved context exceeding the model's context window limit.", file=sys.stderr)
                 print("Try reducing CHUNK_SIZE or K_RETRIEVE.", file=sys.stderr)
            # Print more details about the error that occurred
            print(f"Error Type: {type(e)}", file=sys.stderr)
            print(f"Error Details: {e}", file=sys.stderr)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                 print(f"API Response Text: {e.response.text}", file=sys.stderr)
            sys.exit(1)


# --- Debugging Function: Test LLM Directly ---
def test_llm_direct(test_prompt: str):
    """Initializes the LLM and sends a simple prompt directly, bypassing the RAG chain."""
    print(f"\n--- Testing LLM Directly with Prompt: '{test_prompt}' ---")
    try:
        # Initialize the LLM using the standard get_llm function.
        # get_llm includes the internal test invoke already. If it succeeded,
        # we expect this direct call to also succeed.
        llm = get_llm() # This might raise exceptions caught below

        # If get_llm returned successfully, send the user-provided test prompt.
        print(f"Sending direct prompt to LLM: '{test_prompt}'")
        # --- CORRECTED DIRECT INVOKE INPUT ---
        # Pass a simple string to llm.invoke() for the direct test as well
        direct_invoke_input = test_prompt # Use the prompt string directly
        # --- End CORRECTED DIRECT INVOKE INPUT ---

        print(f"Direct Invoke Input (simple string): '{direct_invoke_input}'") # Print the actual input sent to invoke

        response = llm.invoke(direct_invoke_input) # Use the simple string input

        print("\n--- Direct LLM Response ---")
        print(response) # Print the raw response from invoke
        print("\n--- Direct LLM Test Finished ---")

    except ValueError as e:
         print(f"Error: LLM initialization failed due to configuration issue.", file=sys.stderr)
         print(f"Details: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e: # Catch any other unexpected errors during direct LLM test
        print(f"An unexpected error occurred during direct LLM test: {type(e).__name__} - {e}", file=sys.stderr)
        print(f"Error Type: {type(e)}", file=sys.stderr)
        print(f"Error Details: {e}", file=sys.stderr)
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
             print(f"API Response Text: {e.response.text}", file=sys.stderr)
        sys.exit(1)


# --- Main Application Entry Point ---
if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Legal Document RAG System using Hugging Face API for LLM and local Embeddings.")
    parser.add_argument(
        "--index",
        action="store_true",
        help="Run the indexing process to build/update the vector store from documents in ./docs."
             "\nRequires internet for embedding model download on first run."
             "\nDoes NOT require the Hugging Face API token for this step."
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run the query process with the specified question using the built vector store and the Hugging Face Inference API."
             "\nRequires the vector store to be built (--index must have run successfully)."
             "\nRequires the HUGGINGFACEHUB_API_TOKEN environment variable."
    )
    parser.add_argument(
        "--test-llm",
        type=str,
        help="Test the LLM connection directly with a simple prompt, bypassing the RAG chain."
             "\nRequires the HUGGINGFACEHUB_API_TOKEN environment variable."
             "\nUses the LLM model and task configured via env vars (HF_LLM_MODEL, HF_LLM_TASK)."
    )


    args = parser.parse_args()

    # --- Environment Validation ---
    if args.query or args.test_llm:
        print(f"{'Query' if args.query else 'LLM Test'} mode selected. Checking for HUGGINGFACEHUB_API_TOKEN...")
        if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
            print("Error: HUGGINGFACEHUB_API_TOKEN environment variable is not set.", file=sys.stderr)
            print(f"You must set this environment variable (e.g., in your .env file or shell) to use the Hugging Face API for {'querying' if args.query else 'testing the LLM'}.", file=sys.stderr)
            if args.index:
                 print(f"\nIf you intended to build the index, use '--index'. Current command: {' '.join(sys.argv)}", file=sys.stderr)
            sys.exit(1)


    # --- Action Execution ---
    # Initialize the Embedding function first. It's needed for indexing and querying.
    # If it fails, the program exits within get_embedding_function().
    try:
        embedding_function_instance = get_embedding_function()
    except Exception:
        # get_embedding_function already prints the error
        sys.exit(1)

    # Instantiate the main RAG system class.
    # This initializes components like the local embedding function processor and vector store manager.
    # The embedding function instance is passed during instantiation.
    # LLM initialization is deferred until query mode or direct test via get_llm().
    # The Embedding function is initialized here and needs internet for download the first time.
    # If embedding initialization fails, LegalRagSystem.__init__ will sys.exit(1).
    # Note: The embedding_function_instance is already passed to the VectorStoreManager in LegalRagSystem.__init__
    rag_system = LegalRagSystem()


    if args.index:
        print("Indexing mode selected.")
        if not os.path.exists(DOCS_DIR):
             print(f"Warning: The documents directory '{DOCS_DIR}' does not exist. No documents will be loaded for indexing.", file=sys.stderr)
        elif not any(os.scandir(DOCS_DIR)):
             print(f"Warning: The documents directory '{DOCS_DIR}' exists but is empty. No documents will be loaded for indexing.", file=sys.stderr)

        rag_system.index_documents()

    elif args.query:
        if not args.query.strip():
            print("Error: --query argument requires a non-empty question string.", file=sys.stderr)
            parser.print_help(sys.stderr)
            sys.exit(1)

        print(f"Attempting to query using Hugging Face model: {HF_LLM_MODEL} (Task: {HF_LLM_TASK}). Ensure HUGGINGFACEHUB_API_TOKEN is set.")
        rag_system.query(args.query)

    elif args.test_llm:
         if not args.test_llm.strip():
             print("Error: --test-llm argument requires a non-empty prompt string.", file=sys.stderr)
             parser.print_help(sys.stderr)
             sys.exit(1)

         print(f"Attempting to directly test LLM: {HF_LLM_MODEL} (Task: {HF_LLM_TASK}). Ensure HUGGINGFACEHUB_API_TOKEN is set.")
         # Pass the test prompt to the test function
         test_llm_direct(args.test_llm)

    else:
        print("No valid command specified.", file=sys.stderr)
        print("Please specify one of: --index, --query '<Your question>', or --test-llm '<Your prompt>'.", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)