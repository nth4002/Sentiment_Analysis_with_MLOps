import time
import requests

NGROK_URL = "https://your-ngrok-url.ngrok-free.app"

while True:
    try:
        response = requests.get(NGROK_URL)
        print(f"[{time.ctime()}] Status: {response.status_code}")
    except Exception as e:
        print(f"[{time.ctime()}] Failed to ping: {e}")
    time.sleep(300)  # wait 5 minutes
