# DVC & MinIO Setup Guide for Sentiment MLOps Project

This guide will help you set up [DVC](https://dvc.org/) for data versioning in your project, using either Google Drive or MinIO (self-hosted S3-compatible storage).

---

## 1. Initialize Git & DVC

```bash
git init
dvc init --no-scm -f  # Use --no-scm if you don't want to link with Git, -f to force
```

---

## 2. Configure DVC Remote Storage


### Option B: MinIO (Recommended for local/on-prem)

#### a. Install & Start MinIO

1. **Download and install MinIO:**
    ```bash
    wget https://dl.min.io/server/minio/release/linux-amd64/minio
    chmod +x minio
    sudo mv minio /usr/local/bin/
    ```

2. **Set MinIO root credentials:**
    ```bash
    export MINIO_ROOT_USER=mlops_minio
    export MINIO_ROOT_PASSWORD=mlops@2025
    ```

3. **Start MinIO server:**
    ```bash
    minio server ~/minio-data --console-address :9090
    ```
    - Access the MinIO web console at [http://localhost:9090](http://localhost:9090).

4. **Create a bucket (e.g., `mlops-sentiment-data`) via the MinIO console.**

#### b. Configure DVC to use MinIO

1. **Add MinIO as a DVC remote:**
    ```bash
    dvc remote add -d my_minio_storage s3://mlops-sentiment-data
    dvc remote modify my_minio_storage endpointurl http://localhost:9000
    dvc remote modify my_minio_storage access_key_id mlops_minio
    dvc remote modify my_minio_storage secret_access_key mlops@2025
    ```

2. **If running DVC inside Docker, use the internal endpoint:**
    ```bash
    dvc remote modify my_minio_storage endpointurl http://minio:9000
    ```

3. **If working from the host machine, set a local override:**
    ```bash
    dvc remote modify --local my_minio_storage endpointurl http://localhost:9000
    ```

---

## 3. Pushing Data to Remote

1. **Set AWS credentials (if needed):**
    ```bash
    export AWS_ACCESS_KEY_ID=mlops_minio
    export AWS_SECRET_ACCESS_KEY=mlops@2025
    ```

2. **Push your data to the remote:**
    ```bash
    dvc push
    ```

---

## 4. Useful MinIO & Docker Commands

- **List running containers:**
    ```bash
    docker ps
    ```

- **Enter a running container:**
    ```bash
    docker exec -it <container_name> /bin/sh
    ```

- **Set up MinIO client alias (inside container):**
    ```bash
    mc alias set myminio http://minio:9000 mlops_minio mlops@2025
    ```

- **Create a regular user:**
    ```bash
    mc admin user add myminio user1 password1
    mc admin policy attach myminio readwrite --user user1
    ```

- **Create a service account for DVC:**
    ```bash
    mc admin user svcacct add myminio user1 --name "DVC user for user1"
    ```

---

## 5. Troubleshooting

- **To stop MinIO:** Press `Ctrl+C` in the terminal where it's running, or use `pkill minio`.
- **To restart MinIO:** Run the `minio server ...` command again.
- **If you see a warning about default credentials:** Make sure you set `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` before starting MinIO.

---

## 6. References

- [DVC Docs](https://dvc.org/doc)
- [MinIO Docs](https://min.io/docs/minio/linux/index.html)

---
# Part 2: Connect to Kaggle via ngrok

This section explains how to use [ngrok](https://ngrok.com/) to expose your local MinIO (port **9000**) and MLflow (port **5000**) services to the internet, so you can access them from a Kaggle notebook or remote environment.

---

## 1. Download and Install ngrok

If you don't have ngrok, download it from [ngrok.com/download](https://ngrok.com/download) and follow the instructions for your OS.

**Example for Linux:**
```bash
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xvf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/
```

---

## 2. Authenticate ngrok (First Time Only)

Sign up at [ngrok.com](https://ngrok.com/) to get your auth token, then run:

```bash
ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
```

---

## 3. Start ngrok Tunnels

You need to forward both ports:

- **MinIO Console:** `9000`
- **MLflow Tracking Server:** `5000`

Open two terminal windows/tabs and run:

```bash
ngrok start --all
```

ngrok will display a public URL (e.g., `https://abcd1234.ngrok.io`) for each port.

---

### 4. Use the ngrok URLs in Kaggle

- For MinIO, use the ngrok URL (e.g., `https://abcd1234.ngrok.io`) as your S3 endpoint in your Kaggle code or DVC config.
- For MLflow, use the ngrok URL (e.g., `https://wxyz5678.ngrok.io`) as your `MLFLOW_TRACKING_URI` in your Kaggle notebook.

**Example:**
```python
os.environ['MLFLOW_TRACKING_URI'] = "https://wxyz5678.ngrok.io"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "https://abcd1234.ngrok.io"
os.environ['AWS_ACCESS_KEY_ID'] = "mlops_minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "mlops@2025"
```

---

### 5. Notes

- Keep the ngrok tunnels running as long as you need remote access.
- If you restart ngrok, the URLs will change unless you have a paid plan with reserved domains.
- Make sure your firewall allows incoming connections to the forwarded ports.

---

Now you can connect your Kaggle notebook to your local MinIO and MLflow servers using the ngrok public URLs!
