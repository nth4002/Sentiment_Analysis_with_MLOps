# wait-for-postgres.py
import os
import sys
import time
import psycopg2

print("--- Waiting for PostgreSQL to be ready... ---")

try:
    db_host = os.environ["DB_HOST"]
    db_name = os.environ["POSTGRES_DB"]
    db_user = os.environ["POSTGRES_USER"]
    db_password = os.environ["POSTGRES_PASSWORD"]
    max_retries = 30
    retry_interval = 2  # seconds
except KeyError as e:
    print(f"Error: Environment variable {e} not set.")
    sys.exit(1)


retries = 0
while retries < max_retries:
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port="5432"
        )
        conn.close()
        print("--- PostgreSQL is ready. ---")
        sys.exit(0)  # Exit with success code
    except psycopg2.OperationalError:
        print(f"Attempt {retries + 1}/{max_retries}: PostgreSQL not ready yet. Retrying in {retry_interval}s...")
        retries += 1
        time.sleep(retry_interval)

print(f"--- Could not connect to PostgreSQL after {max_retries * retry_interval} seconds. Exiting. ---")
sys.exit(1)  # Exit with failure code