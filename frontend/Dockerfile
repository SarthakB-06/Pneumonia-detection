FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the frontend code into the container
COPY frontend/ ./frontend

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["python", "-m", "streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]