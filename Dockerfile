FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HUB_OFFLINE=1
COPY . .

# 0.0.0.0 inside the container; docker-compose.yml only publishes the port on 127.0.0.1
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.headless", "true"]
