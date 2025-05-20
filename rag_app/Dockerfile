# rag_app/Dockerfile

# Use a lightweight official Python image
FROM python:3.9-slim

# Set the working directory inside the container
# All subsequent commands will run from this directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
# --no-cache-dir saves space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the working directory
COPY . .

# Command to run when the container starts.
# This command will execute main.py with arguments passed via docker-compose run
CMD ["python", "main.py"]