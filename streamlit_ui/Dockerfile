# streamlit_ui/Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Command to run the Streamlit application
# --server.port 8501 is the internal port Streamlit runs on
# --server.address 0.0.0.0 makes it accessible externally (on the internal network or host)
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]