# streamlit_ui/app.py
import os
import requests
import streamlit as st
from dotenv import load_dotenv

# --- Configuration (Read from Environment Variable) ---
# This is set in docker-compose.yml and can be overridden by .env
# It should point to the RAG API service accessible from the host machine
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000") # Default for local testing

# Load environment variables from .env file (useful for local testing outside docker)
load_dotenv()

# --- Streamlit App ---
st.set_page_config(page_title="Legal Document RAG Assistant", layout="wide")

st.title("⚖️ Legal Document RAG Assistant")

st.write("Ask questions about your legal documents.")

# Input field for the user query
query = st.text_area("Enter your question here:", height=100)

# Button to submit the query
if st.button("Get Answer"):
    if query:
        # --- Call the RAG API Service ---
        api_endpoint = f"{RAG_API_URL}/query"
        payload = {"question": query}

        try:
            # Use Streamlit's spinner to show loading
            with st.spinner("Searching documents and generating answer..."):
                # Send POST request to the RAG API's /query endpoint
                response = requests.post(api_endpoint, json=payload, timeout=180) # Increased timeout

            # Check if the request was successful (HTTP status code 200-299)
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "Could not get answer from the API.")
                sources = result.get("sources", [])

                st.subheader("Answer:")
                # Use st.markdown for basic formatting if answer contains markdown
                st.markdown(answer)

                if sources:
                    st.subheader("Sources:")
                    # Display sources clearly
                    for i, source in enumerate(sources):
                        source_info = f"**{source.get('source', 'N/A')}**"
                        if source.get('page') is not None:
                             source_info += f" (Page: {source.get('page')})"
                        # Display a snippet if available
                        snippet = source.get('text_snippet')
                        if snippet:
                             source_info += f": *\"{snippet}\"*"
                        st.markdown(f"- {source_info}")
                else:
                    st.info("No specific sources found for this answer.")

            else:
                # Handle API errors (non-200 status codes)
                st.error(f"Error from RAG API: Status Code {response.status_code}")
                try:
                    # Attempt to get error details from the API response body
                    error_details = response.json().get("detail", "No details provided in response.")
                    st.error(f"Details: {error_details}")
                except:
                    # Fallback if response body is not JSON
                    st.error("Could not parse error details from API response.")

        except requests.exceptions.Timeout:
            # Handle request timeout
            st.error(f"The request to the RAG API timed out after 180 seconds. The LLM might be taking too long or the service is overloaded.")
        except requests.exceptions.ConnectionError:
             # Handle network/connection errors
             st.error(f"Could not connect to the RAG API at {RAG_API_URL}. Ensure the RAG API service is running and accessible.")
        except Exception as e:
            # Handle any other unexpected errors during the request
            st.error(f"An unexpected error occurred: {e}")
            # Optionally print traceback for debugging in the console where Streamlit is running
            # import traceback
            # traceback.print_exc()

    else:
        st.warning("Please enter a question.")

# Optional: Display RAG API URL for debugging in the sidebar
# st.sidebar.info(f"RAG API URL: {RAG_API_URL}")