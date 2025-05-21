# llm_service/Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Command to run the FastAPI application using uvicorn
# --host 0.0.0.0 makes the app accessible from outside the container (on the internal network)
# --port 8000 is the internal port the app listens on
CMD ["uvicorn", "llm_api:app", "--host", "0.0.0.0", "--port", "8000"]