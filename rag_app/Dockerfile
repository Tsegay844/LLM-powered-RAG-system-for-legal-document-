# rag_app/Dockerfile

# Start from an official Python image (using a specific version tag is good practice)
FROM python:3.9-slim

# Set the working directory inside the container
# All commands like COPY, RUN, and CMD will be relative to this directory
WORKDIR /app

# Copy the requirements file into the container's working directory
COPY requirements.txt .

# Install the Python dependencies listed in requirements.txt
# --no-cache-dir helps to keep the image size smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code from the current context (. representing rag_app/)
# into the container's working directory (/app)
COPY . .

# Define the default command to run when the container starts
# This command will execute main.py with any arguments passed via docker-compose run
CMD ["python", "main.py"]