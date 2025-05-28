# llm_service/llm_api.py
# This file provides a FastAPI service that interacts with the Google Gemini LLM API.
import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import time # Import time for basic timing

# Google Generative AI Imports ---
import google.generativeai as genai
import google.api_core.exceptions


# These are set in docker-compose.yml and can be overridden by .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash") 

# --- FastAPI App Setup ---
app = FastAPI(
    title="LLM Inference Service (Google Gemini)",
    description="Provides an API endpoint to access a Google Gemini LLM."
)

# --- GenAI Configuration ---
# Flag to indicate if genai configuration was successful
# This will be set to True if genai.configure is called successfully
# This is used to check if the service is ready to handle requests
genai_configured = False


# Function to configure the Google Generative AI library
# This will be called on startup to set the API key
# It will also set the genai_configured flag to True if successful
def configure_genai():
    """Configures the Google Generative AI library with the API key."""
    global genai_configured
    if genai_configured:
        print("Google Generative AI is already configured.")
        return

    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set. Cannot configure Google Generative AI.", file=sys.stderr)
        genai_configured = False 
        # In a real production system, you might want the service to fail startup if this is missing
        return

    try:
        # genai.configure sets the API key for subsequent calls
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Google Generative AI configured successfully.")
        genai_configured = True # Set flag to True on success
    except Exception as e:
        print(f"Error configuring Google Generative AI: {e}", file=sys.stderr)
        genai_configured = False # Set flag to False on failure
       

# Configure GenAI when the app starts
# This will be called automatically by FastAPI on startup
@app.on_event("startup")

# This function will run when the FastAPI app starts
async def startup_event():
    # Load .env if running locally without docker-compose
    load_dotenv()
    # call the configuration function
    print("Configuring Google Generative AI...")
    configure_genai()

#  Request Body Model
# this class defines the structure of the request body for the /generate endpoint
# it includes the prompt, temperature, and max_output_tokens 
#
class PromptRequest(BaseModel):
    # The prompt text to generate from
    # This is the main input for the LLM
    prompt: str
    # Optional parameters for text generation
    # temperature controls randomness (0.0 = deterministic, 1.0 = more random)
    temperature: float = 0.0 
    # max_output_tokens limits the length of the generated text
    # If not provided, the model's default will be used
    max_output_tokens: int = None 

# --- API Endpoint ---
# This endpoint generates text based on the provided prompt
@app.post("/generate")
# This function handles POST requests to the /generate endpoint
# it expects a JSON body matching the PromptRequest model
async def generate_text(request: PromptRequest):
    """
    Generates text using the configured Google Gemini LLM based on the provided prompt.
    """
    # --- CORRECTED Check for GenAI Configuration ---
    # Check if genai was configured successfully on startup
    # Removed the incorrect check using genai.get_default_options()
    if not genai_configured:
         print("Google Generative AI not configured. API key might be missing or invalid.", file=sys.stderr)
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM service not configured. API key might be missing or invalid.")


    #
    start_time = time.time()
    try:
        print(f"Received prompt (first 150 chars): {request.prompt[:150]}...")

        # Get the specific model instance
        # This will raise an error if the model name is invalid or inaccessible with the configured API key
        # Create a GenerativeModel instance with the specified model name Gemini-2.0-flash
        # Note: The model name should be set in the environment variable LLM_MODEL_NAME
        model = genai.GenerativeModel(model_name=LLM_MODEL_NAME)
        print(f"Using model: {LLM_MODEL_NAME}")

        # Prepare generation configuration
        generation_config = {
            "temperature": request.temperature,
            # Include max_output_tokens only if provided and not None
            **( {"max_output_tokens": request.max_output_tokens} if request.max_output_tokens is not None else {} )
        }

        # Prepare the prompt contents for the API call (dictionary format)
        # The contents should be a list of dictionaries with role and parts
        # The role is typically "user" for the input prompt
        # The parts is a list of dictionaries with text
        # containing the actual prompt text
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": request.prompt},
                ],
            }
        ]

        # Call the generate_content method to get a response
        # This method sends the prompt to the LLM and returns the generated content
        response = model.generate_content(
            contents=contents,
            generation_config=generation_config,
        )

        # Log the duration of the API call
        # This will help in monitoring performance
        duration = time.time() - start_time
        print(f"Gemini API call successful in {duration:.2f} seconds.")

        # Process the response
        if response and response.candidates:
            if response.candidates[0].content and response.candidates[0].content.parts:
                 # Access the text from the first candidate's first content part
                 generated_text = response.candidates[0].content.parts[0].text
                 print("Text generation successful.")
                 return {"text": generated_text}
            else:
                 # Handle cases where candidates or their content/parts are missing/empty
                 # Check for finish reason like 'SAFETY' or 'OTHER'
                 finish_reason = response.candidates[0].finish_reason if response.candidates[0].finish_reason else "unknown"
                 safety_ratings = response.candidates[0].safety_ratings if response.candidates[0].safety_ratings else []
                 detail = f"Model returned no text content. Finish reason: {finish_reason}. Safety ratings: {safety_ratings}"
                 print(detail, file=sys.stderr)
                 # Decide whether to raise error or return a specific message
                 # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)
                 return {"text": f"LLM returned no text content. Possible reason: {finish_reason}"} # Return a message


        else:
            # Handle cases where the API returned no candidates
            detail = "LLM API returned no candidates."
            # Optional: Check response.prompt_feedback for issues with the prompt
            if response and response.prompt_feedback:
                 detail += f" Prompt feedback: {response.prompt_feedback}"
            print(detail, file=sys.stderr)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)


    # --- Specific Google API Error Handling ---
    except google.api_core.exceptions.NotFound as e:
         # Model not found or not available for the API key
         duration = time.time() - start_time
         print(f"API Error (NotFound) after {duration:.2f}s: The model '{LLM_MODEL_NAME}' was not found or is not available. Check model name and API key permissions. Details: {e}", file=sys.stderr)
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"LLM Model not found or available: {LLM_MODEL_NAME}. Details: {e}")
    except google.api_core.exceptions.PermissionDenied as e:
         # API key does not have permission to use the Generative Language API or the model
         duration = time.time() - start_time
         print(f"API Error (PermissionDenied) after {duration:.2f}s: Permission denied. Check your GOOGLE_API_KEY and ensure the Generative Language API is enabled. Details: {e}", file=sys.stderr)
         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"API Permission Denied. Check GOOGLE_API_KEY. Details: {e}")
    except google.api_core.exceptions.DeadlineExceeded as e:
         # The API request took too long to complete
         duration = time.time() - start_time
         print(f"API Error (DeadlineExceeded) after {duration:.2f}s: Request timed out. Details: {e}", file=sys.stderr)
         raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=f"LLM API request timed out. Details: {e}")
    except google.api_core.exceptions.InvalidArgument as e:
         # The request payload (prompt, config) was invalid
         duration = time.time() - start_time
         print(f"API Error (InvalidArgument) after {duration:.2f}s: Invalid argument provided to the API. Details: {e}", file=sys.stderr)
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid argument for LLM API: {e}")
    except google.api_core.exceptions.GoogleAPIError as e:
         # Catch any other unhandled Google API errors
         duration = time.time() - start_time
         print(f"API Error (GoogleAPIError) after {duration:.2f}s: An unexpected Google API error occurred. Details: {e}", file=sys.stderr)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Google API Error: {e}")
    # --- End Specific Google API Error Handling ---

    except Exception as e:
        duration = time.time() - start_time
        print(f"An unexpected error occurred during text generation after {duration:.2f}s: {e}", file=sys.stderr)
        # Log traceback for unexpected errors
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error during generation: {e}")

# Optional: Basic health check endpoint
# This endpoint checks if the LLM service is configured and ready to handle requests
@app.get("/health")
async def health_check():
    """Basic health check for the LLM service."""
    # --- CORRECTED Health Check Logic ---
    # Check if genai was configured successfully on startup using the flag
    if genai_configured:
         return {"status": "ok", "model": LLM_MODEL_NAME, "configured": True}
    else:
         # Configuration failed (likely missing API key or invalid key)
         detail_msg = "Google Generative AI library failed to configure on startup."
         if not GOOGLE_API_KEY:
              detail_msg += " GOOGLE_API_KEY is not set."
         # Return 503 Service Unavailable with details
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"status": "error", "model": LLM_MODEL_NAME, "configured": False, "detail": detail_msg})
    


# To run locally for testing (outside Docker):
