# rag_app/main.py
import os
import argparse
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any

# LangChain components
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- Hugging Face & Local Embedding Imports ---
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import SentenceTransformerEmbeddings
# --- End Imports ---

# REMOVE OR COMMENT OUT OLD IMPORTS
# from langchain_community.llms import HuggingFaceHub


from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import Runnable


# --- Configuration Constants ---
DOCS_DIR = "/app/docs"
PERSIST_DIR = "/app/chroma_db"
COLLECTION_NAME = "legal_documents_collection" # A clear name for the ChromaDB collection

# --- Models configured for Hugging Face Inference API (LLM) and Local (Embeddings) ---
# You might change HF_LLM_MODEL or HF_LLM_TASK via .env or docker-compose.yml
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "google/flan-t5-large")
HF_LLM_TASK = os.getenv("HF_LLM_TASK", "text2text-generation") # Or 'text-generation' etc.

LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# Splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

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
    print(f"Initializing Hugging Face LLM model: {HF_LLM_MODEL} ({HF_LLM_TASK} task)")
    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_api_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is not set.")

    try:
        # --- IMPORTANT CHANGE HERE: Removed temperature ---
        llm = HuggingFaceEndpoint(
            repo_id=HF_LLM_MODEL,
            task=HF_LLM_TASK,
            # Removed temperature=0.0 to simplify parameters
        )
        # --- End Update ---

        # Test a simple invoke to catch errors early during initialization
        try:
             # Sending a minimal prompt to see if the API call itself works
             print("Performing initial LLM test invoke...")
             test_prompt = "Once upon a time," # Simple prompt
             test_response = llm.invoke(test_prompt)
             print(f"Initial LLM test invoke successful. Response start: '{test_response[:50]}...'")
        except Exception as test_e:
             print(f"Initial LLM test invoke failed: {test_e}", file=sys.stderr)
             print("This likely means the API connection, token, model, or task is misconfigured.", file=sys.stderr)
             raise # Re-raise the test error

        return llm
    except Exception as e:
        print(f"Error initializing Hugging Face LLM model {HF_LLM_MODEL} with task {HF_LLM_TASK}: {e}", file=sys.stderr)
        print("Possible issues: Invalid API token, model name/task incorrect, model not available on Inference API, network issues.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr) # Print the specific error from the API call
        raise


def get_prompt_template():
    """Defines the prompt template for the RAG chain."""
    print("Defining prompt template...")
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
        self.text_splitter = RecursiveCharacterTextSplitter( # Corrected typo confirmed here
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.loader_mapping = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            # ".docx": DocxLoader, # Requires python-docx library
        }

    def load_documents(self) -> List[Document]:
        """Loads documents from the configured directory."""
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

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of documents into chunks."""
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
        self.embedding_function = embedding_function

        os.makedirs(self.persist_directory, exist_ok=True)

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        """Creates or updates the vector store with documents."""
        print(f"Creating/Updating vector store '{self.collection_name}' at {self.persist_directory}...")
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
        if not os.path.exists(self.persist_directory) or not any(os.scandir(self.persist_directory)):
             raise FileNotFoundError(f"Vector store directory not found or is empty: {self.persist_directory}. Please run indexing first.")

        try:
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function, # Still needs the embedding function to query
                collection_name=self.collection_name
            )
            if vector_store._collection.count() == 0:
                 raise ValueError(f"Vector store collection '{self.collection_name}' found but is empty. Please run indexing.")

            print("Vector store loaded successfully.")
            return vector_store

        except Exception as e:
            raise RuntimeError(f"Failed to load vector store collection '{self.collection_name}'. Error: {e}")


class LegalRagSystem:
    """Orchestrates the RAG process."""
    def __init__(self):
        try:
            self.embedding_function = get_embedding_function()
        except Exception as e:
             print(f"FATAL ERROR during Embedding Function initialization: {e}", file=sys.stderr)
             sys.exit(1)

        self.doc_processor = DocumentProcessor(DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
        self.vector_store_manager = VectorStoreManager(PERSIST_DIR, COLLECTION_NAME, self.embedding_function)

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

        self.vector_store_manager.create_from_documents(split_docs)
        print("--- Indexing Process Finished ---")


    def _initialize_rag_chain(self):
         """Initializes LLM, loads vector store/retriever, and creates the RAG chain."""
         if self.rag_chain:
              return

         print("\n--- Initializing RAG Chain Components ---")
         try:
             # 1. Load the vector store and create retriever
             vector_store = self.vector_store_manager.load_vector_store()
             self.retriever = vector_store.as_retriever(search_kwargs={"k": K_RETRIEVE})
             print(f"Retriever configured to fetch top {K_RETRIEVE} chunks.")

             # 2. Initialize the LLM (Hugging Face Inference API)
             # get_llm now includes an internal test invoke
             self.llm = get_llm() # This might raise exceptions caught below

             # 3. Define the prompt template for instructing the LLM.
             prompt = get_prompt_template()

             # 4. Create the chains using LangChain's component composition.
             document_chain = create_stuff_documents_chain(self.llm, prompt)
             self.rag_chain = create_retrieval_chain(self.retriever, document_chain)
             print("RAG chain created.")

         except (FileNotFoundError, ValueError, RuntimeError) as e:
             print(f"FATAL ERROR during RAG chain initialization (Index/Load Issue): {e}", file=sys.stderr)
             sys.exit(1)
         except Exception as e: # Catch exceptions from get_llm() or chain creation
              print(f"An unexpected FATAL ERROR occurred during RAG chain initialization: {e}", file=sys.stderr)
              print(f"Error Type: {type(e)}", file=sys.stderr)
              print(f"Error Attributes: {dir(e)}", file=sys.stderr)
              sys.exit(1)

    def query(self, query_text: str):
        """Processes a user query by retrieving relevant document chunks and generating an answer using the LLM."""
        print(f"\n--- Processing Query: '{query_text}' ---")
        try:
            # _initialize_rag_chain includes LLM initialization and its internal test invoke.
            # If _initialize_rag_chain completes without SystemExit, the LLM should be ready.
            self._initialize_rag_chain()

            # This LLM check is mostly defensive; _initialize_rag_chain exiting handles failure.
            if self.llm is None:
                 print("Error: RAG system LLM component was not successfully initialized. Cannot process query.", file=sys.stderr)
                 sys.exit(1)

            # --- THIS IS THE LINE WHERE THE ERROR OCCURS ---
            response: Dict[str, Any] = self.rag_chain.invoke({"input": query_text})

            print("\n--- Answer ---")
            print(response.get("answer", "No answer was generated by the model."))

            print("\n--- Sources ---")
            context_docs: List[Document] = response.get("context", [])
            if context_docs:
                 unique_sources = set()
                 for doc in context_docs:
                     source = doc.metadata.get('source', 'N/A')
                     page = doc.metadata.get('page', 'N/A')

                     if source != 'N/A':
                         source_base_name = os.path.basename(source)
                         if isinstance(source, str) and 'page' in doc.metadata and page is not None:
                             unique_sources.add(f"{source_base_name} (Page: {page})")
                         elif isinstance(source, str):
                             unique_sources.add(source_base_name)
                         else:
                             unique_sources.add(str(source))

                 if unique_sources:
                     print("Information retrieved from:")
                     for src_info in sorted(list(unique_sources)):
                        print(f"- {src_info}")
                 else:
                     print("No specific sources identified with 'source' metadata in retrieved context.")

            else:
                 print("No context documents were returned by the retriever or chain.")

        except SystemExit as e:
             print(f"Query execution halted due to a system initialization error: {e}", file=sys.stderr)
             sys.exit(1)
        except Exception as e:
            # Enhanced error printing
            print(f"An unexpected error occurred during query execution: {e}", file=sys.stderr)
            print(f"Error Type: {type(e)}", file=sys.stderr)
            print(f"Error Attributes: {dir(e)}", file=sys.stderr)
            if isinstance(e, ValueError) and "Could not inker input into prompt." in str(e):
                 print("Hint: This might be due to the combined length of the query and retrieved context exceeding the model's limit.", file=sys.stderr)
                 print("Try reducing CHUNK_SIZE or K_RETRIEVE.", file=sys.stderr)
            sys.exit(1)

# --- Debugging Function: Test LLM Directly ---
def test_llm_direct(test_prompt: str):
    """Initializes the LLM and sends a simple prompt directly."""
    print(f"\n--- Testing LLM Directly with Prompt: '{test_prompt}' ---")
    try:
        # Get LLM directly without the RAG chain
        # get_llm includes the internal test invoke already. If it succeeded,
        # we expect this direct call to also succeed.
        llm = get_llm()

        print(f"Sending direct prompt to LLM: '{test_prompt}'")
        response = llm.invoke(test_prompt)

        print("\n--- Direct LLM Response ---")
        print(response)

    except ValueError as e:
         print(f"Error: LLM initialization failed. Ensure HUGGINGFACEHUB_API_TOKEN is set and model/task is correct.", file=sys.stderr)
         print(f"Details: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during direct LLM test: {e}", file=sys.stderr)
        print(f"Error Type: {type(e)}", file=sys.stderr)
        print(f"Error Attributes: {dir(e)}", file=sys.stderr)
        sys.exit(1)


# --- Main Application Entry Point ---
if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Legal Document RAG System using Hugging Face API for LLM and local Embeddings.")
    parser.add_argument(
        "--index",
        action="store_true",
        help="Run the indexing process to build/update the vector store from documents in ./docs."
             "Requires internet for embedding model download on first run."
             "Does NOT require the Hugging Face API token for this step."
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run the query process with the specified question using the built vector store and the Hugging Face Inference API."
             "Requires the vector store to be built (--index must have run successfully)."
             "Requires the HUGGINGFACEHUB_API_TOKEN environment variable."
    )
    # --- New Debugging Argument ---
    parser.add_argument(
        "--test-llm",
        type=str,
        help="Test the LLM connection directly with a simple prompt, bypassing the RAG chain."
             "Requires the HUGGINGFACEHUB_API_TOKEN environment variable."
    )
    # --- End New Debugging Argument ---


    args = parser.parse_args()

    # --- Environment Validation ---
    # Check for HF token if running query OR test-llm mode
    if args.query or args.test_llm:
        print(f"{'Query' if args.query else 'LLM Test'} mode selected. Checking for HUGGINGFACEHUB_API_TOKEN...")
        if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
            print("Error: HUGGINGFACEHUB_API_TOKEN environment variable is not set.", file=sys.stderr)
            print(f"You must set this environment variable (e.g., in your .env file or shell) to use the Hugging Face API for {'querying' if args.query else 'testing the LLM'}.", file=sys.stderr)
            if args.index:
                 print(f"\nIf you intended to build the index, use '--index'. Current command: {' '.join(sys.argv)}", file=sys.stderr)
            sys.exit(1)


    # --- Action Execution ---
    # Instantiate the main RAG system class.
    # This initializes components like the local embedding function.
    # LLM initialization is deferred until query mode or direct test.
    # The Embedding function is initialized here and needs internet for download the first time.
    rag_system = LegalRagSystem()

    if args.index:
        print("Indexing mode selected.")
        if not os.path.exists(DOCS_DIR):
             print(f"Warning: The documents directory '{DOCS_DIR}' does not exist. No documents will be loaded.", file=sys.stderr)
        elif not any(os.scandir(DOCS_DIR)):
             print(f"Warning: The documents directory '{DOCS_DIR}' exists but is empty. No documents will be loaded.", file=sys.stderr)

        rag_system.index_documents()

    elif args.query:
        if not args.query.strip():
            print("Error: --query argument requires a non-empty question.", file=sys.stderr)
            parser.print_help(sys.stderr)
            sys.exit(1)
        print(f"Attempting to query using Hugging Face model: {HF_LLM_MODEL} (Task: {HF_LLM_TASK}). Ensure HUGGINGFACEHUB_API_TOKEN is set.")
        # Call the query method. This will internally call _initialize_rag_chain
        # which handles loading the index and initializing the LLM via the HF API, including the test invoke.
        rag_system.query(args.query)

    # --- New Debugging Execution Path ---
    elif args.test_llm:
         if not args.test_llm.strip():
             print("Error: --test-llm argument requires a non-empty prompt string.", file=sys.stderr)
             parser.print_help(sys.stderr)
             sys.exit(1)
         # Call the new test function
         # Note: get_llm is called inside test_llm_direct, which includes the test invoke.
         test_llm_direct(args.test_llm)
    # --- End New Debugging Execution Path ---

    else:
        print("No valid command specified.", file=sys.stderr)
        print("Please specify --index, --query, or --test-llm.", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)