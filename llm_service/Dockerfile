# llm_service/Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


# # This command starts the FastAPI application defined in llm_api.py
# Command to run the FastAPI application using uvicorn
# Ensure that the FastAPI app is named `app` in the llm_api module
# and that uvicorn is installed in the requirements.txt.
# This is the entry point for the Docker container.
## The application will be accessible at http://localhost:8000 when the container is running.
# --host 0.0.0.0 makes the app accessible from outside the container (on the internal network)
# --port 8000 is the internal port the app listens on
CMD ["uvicorn", "llm_api:app", "--host", "0.0.0.0", "--port", "8000"]