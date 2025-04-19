FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0"]
