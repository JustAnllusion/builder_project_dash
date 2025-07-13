FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


RUN python -u -m scripts.download_data msk_united ekb \
 && echo "[`date`] ✅ download_data completed" \
 && python -u -m scripts.precompute_elasticity \
 && echo "[`date`] ✅ precompute_elasticity completed" \
 && python -u -m scripts.preprocess_depletion_curves \
 && echo "[`date`] ✅ preprocess_depletion_curves completed" \
 && python -u -m scripts.precompute_floor_elasticity \
 && echo "[`date`] ✅ precompute_floor_elasticity completed" \
 && python -u -m scripts.preprocess_segmentation \
 && echo "[`date`] ✅ preprocess_segmentation completed"

ENV STREAMLIT_TELEMETRY_ENABLED=false
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.headless=true"]