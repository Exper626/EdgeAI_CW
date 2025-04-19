FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fastapi_connection.py .

CMD ["sh", "-c", "uvicorn fastapi_connection:app --host 0.0.0.0 --port $PORT"]
