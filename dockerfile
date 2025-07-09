FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


RUN python -m scripts.download_data msk_united ekb
RUN python -m scripts.precompute_elasticity
RUN python -m scripts.preprocess_depletion_curves
RUN python -m scripts.precompute_floor_elasticity
RUN python -m scripts.preprocess_segmentation

ENV STREAMLIT_TELEMETRY_ENABLED=false
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.headless=true"]
