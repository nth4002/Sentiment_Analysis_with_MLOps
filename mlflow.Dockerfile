# mlflow.Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Install all dependencies.
RUN pip install --no-cache-dir \
    mlflow==3.1.0 \
    gunicorn==20.1.0 \
    boto3 \
    psutil \
    psycopg2-binary \
    sqlalchemy

# Copy the new wait script AND the startup script
COPY wait-for-postgres.py .
COPY start-mlflow.sh .

# Make the startup script executable
RUN chmod +x ./start-mlflow.sh

EXPOSE 5000

# Set the startup command to our script.
CMD ["./start-mlflow.sh"]