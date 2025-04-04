FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_TELEMETRY_ENABLED=false
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.headless=true"]
