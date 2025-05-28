# streamlit_ui/app.py
import os
import requests
import streamlit as st
from dotenv import load_dotenv

# --- Configuration (Read from Environment Variable) ---
# This is set in docker-compose.yml and can be overridden by .env
# It should point to the RAG API service accessible from the Streamlit container
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000") # Default for local testing

# Load environment variables from .env file
load_dotenv()

# --- Streamlit App ---
# Set the page configuration for Streamlit
# This sets the title and layout of the Streamlit app
st.set_page_config(page_title="Legal Document RAG Assistant", layout="wide")

st.title("⚖️ Legal Document RAG Assistant")

st.write("Ask questions about your legal documents.")

# Input field for the user query
query = st.text_area("Enter your question here:", height=100, key="query_input") 

# Initialize session state variables if they don't exist
if 'last_query_id' not in st.session_state:
    st.session_state.last_query_id = None
if 'answer_displayed' not in st.session_state:
    st.session_state.answer_displayed = False
if 'feedback_sent' not in st.session_state: 
    st.session_state.feedback_sent = False


# Button to submit the query
if st.button("Get Answer"):
    if query:
        # --- Call the RAG API Service ---
        # Construct the API endpoint URL for querying
        # The RAG API has a /query endpoint that accepts POST requests with a JSON body
        api_endpoint = f"{RAG_API_URL}/query"
        payload = {"question": query}

        st.session_state.answer_displayed = False # Reset feedback state
        st.session_state.last_query_id = None # Reset query ID
        st.session_state.feedback_sent = False # Reset feedback sent state


        try:
            with st.spinner("Searching documents and generating answer..."):
                # Send POST request to the RAG API's /query endpoint
                # Use a longer timeout to allow for LLM processing
                # This is important for handling large documents or complex queries
                # The timeout is set to 180 seconds to accommodate longer processing times
                response = requests.post(api_endpoint, json=payload, timeout=180) # Increased timeout

            # Check if the request was successful (HTTP status code 200)
            if response.status_code == 200:
                result = response.json()
                # --- Get query_id from the response ---
                st.session_state.last_query_id = result.get("query_id") # Store the query ID
                # --- End Get query_id ---
                answer = result.get("answer", "Could not get answer from the API.")
                sources = result.get("sources", [])

                st.subheader("Answer:")
                # Use st.markdown for basic formatting if answer contains markdown
                st.markdown(answer)

                #
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

                st.session_state.answer_displayed = True # Mark that an answer was successfully displayed
                st.session_state.feedback_sent = False # Ensure feedback can be sent for this new answer

            else:
                # Handle API errors
                st.error(f"Error from RAG API: Status Code {response.status_code}")
                st.session_state.answer_displayed = False # Ensure feedback isn't shown on error
                st.session_state.last_query_id = None
                st.session_state.feedback_sent = False
                try:
                    # Attempt to get error details from the API response body
                    error_details = response.json().get("detail", "No details provided in response.")
                    st.error(f"Details: {error_details}")
                except:
                    # Fallback if response body is not JSON
                    st.error("Could not parse error details from API response.")

        except requests.exceptions.Timeout:
            st.error(f"The request to the RAG API timed out after 180 seconds. The LLM might be taking too long or the service is overloaded.")
            st.session_state.answer_displayed = False
            st.session_state.last_query_id = None
            st.session_state.feedback_sent = False
        except requests.exceptions.ConnectionError:
             st.error(f"Could not connect to the RAG API at {RAG_API_URL}. Ensure the RAG API service is running and accessible.")
             st.session_state.answer_displayed = False
             st.session_state.last_query_id = None
             st.session_state.feedback_sent = False
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.answer_displayed = False
            st.session_state.last_query_id = None
            st.session_state.feedback_sent = False

    else:
        st.warning("Please enter a question.")
        st.session_state.answer_displayed = False
        st.session_state.last_query_id = None
        st.session_state.feedback_sent = False


# --- NEW Feedback Buttons ---
# Show feedback buttons only if an answer was successfully displayed AND feedback hasn't been sent for this answer yet
if st.session_state.answer_displayed and st.session_state.last_query_id and not st.session_state.feedback_sent:
    st.subheader("Was this answer helpful?")
    col1, col2 = st.columns(2)

    feedback_endpoint = f"{RAG_API_URL}/feedback"

    def send_feedback(feedback_type: str):
        """Helper function to send feedback to the RAG API."""
        if st.session_state.last_query_id:
            payload = {
                "query_id": st.session_state.last_query_id,
                "feedback_type": feedback_type
            }
            try:
                # Send feedback asynchronously (Streamlit doesn't do true background tasks easily)
                # We'll just make a quick request and show a message
                response = requests.post(feedback_endpoint, json=payload, timeout=5) # Short timeout for feedback
                response.raise_for_status() # Raise for bad status codes
                st.success(f"Feedback '{feedback_type}' submitted for Query ID: {st.session_state.last_query_id}!")
                # Set feedback_sent flag to True to disable buttons for this answer
                st.session_state.feedback_sent = True
                # Optionally clear the query ID if you don't want it retained after feedback
                # st.session_state.last_query_id = None
                # Rerun the script to update the UI (disable buttons)
                st.rerun() # Use st.rerun() to re-execute the script from the top

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to submit feedback: {e}")
            except Exception as e:
                 st.error(f"An unexpected error occurred sending feedback: {e}")


    with col1:
        # Add a key to the button to ensure proper state management if needed, though rerunning helps
        if st.button("👍 Satisfied", key="satisfied_button"):
            send_feedback("satisfied")

    with col2:
        # Add a key to the button
        if st.button("👎 Unsatisfied", key="unsatisfied_button"):
            send_feedback("unsatisfied")
# --- END NEW Feedback Buttons ---


# Optional: Display RAG API URL and last query ID for debugging in the sidebar
# st.sidebar.info(f"RAG API URL: {RAG_API_URL}")
# if st.session_state.last_query_id:
#      st.sidebar.info(f"Last Query ID: {st.session_state.last_query_id}")