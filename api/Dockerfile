FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

ENV PYTHONPATH=/app

# Copy the requirements file into the container to install dependencies
COPY requirements.txt .

# Install dependencies and clean up cache to keep the image small
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the specific application directories needed for inference
COPY api/ ./api
COPY src/ ./src
COPY saved_models/ ./saved_models

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI server when the container starts
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]