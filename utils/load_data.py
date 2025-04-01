import os
import requests
import pandas as pd
import streamlit as st

BASE_URL = "https://storage.yandexcloud.net/scienceforbusiness/data/regions/msk/market_deals"
LOCAL_DATA_DIR = "data/regions/msk/market_deals"

@st.cache_data
def _load_file_from_local_cache(local_path: str) -> pd.DataFrame:
    return pd.read_feather(local_path)

def load_data(filename: str) -> pd.DataFrame:
    local_path = os.path.join(LOCAL_DATA_DIR, filename)
    etag_path = local_path + ".etag"
    remote_url = f"{BASE_URL}/{filename}"
    
    try:
        with st.spinner(f"Проверка данных: {filename}"):
            response = requests.head(remote_url)
            response.raise_for_status()
            remote_etag = response.headers.get("ETag", "").strip('"')
    except Exception as e:
        st.error(f"Не удалось получить ETag для {filename}: {e}")
        raise

    if os.path.exists(local_path) and os.path.exists(etag_path):
        with open(etag_path, "r") as f:
            local_etag = f.read().strip()
        if local_etag == remote_etag:
            st.toast(f"Используем локальный кэш: {filename}")
            return _load_file_from_local_cache(local_path)

    try:
        with st.spinner(f"⬇️ Загружаем из облака: {filename}"):
            response = requests.get(remote_url)
            response.raise_for_status()
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(response.content)
            with open(etag_path, "w") as f:
                f.write(remote_etag)
            st.toast(f"✅ Загружено и сохранено: {filename}")
    except Exception as e:
        st.error(f"Ошибка при скачивании {filename}: {e}")
        raise

    return _load_file_from_local_cache(local_path)
