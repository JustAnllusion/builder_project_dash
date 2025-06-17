import os
import requests
import pandas as pd
import streamlit as st

BASE_URL = "https://storage.yandexcloud.net/scienceforbusiness/data/regions"
LOCAL_DATA_DIR = "data/regions"

@st.cache_data
def _load_file_from_local_cache(local_path: str) -> pd.DataFrame:
    return pd.read_parquet(local_path)

def load_data(city: str, filename: str, subfolder: str) -> pd.DataFrame:

    remote_url = f"{BASE_URL}/{city}/{subfolder}/{filename}"
    local_path = os.path.join(LOCAL_DATA_DIR, city, subfolder, filename)
    etag_path = local_path + ".etag"

    try:
        with st.spinner(f"Проверка данных для {filename} ({city})"):
            response = requests.head(remote_url)
            response.raise_for_status()
            remote_etag = response.headers.get("ETag", "").strip('"')
    except Exception as e:
        st.error(f"Не удалось получить ETag для {filename} ({city}): {e}")
        raise

    if os.path.exists(local_path) and os.path.exists(etag_path):
        with open(etag_path, "r", encoding="utf-8") as f:
            local_etag = f.read().strip()
        if local_etag == remote_etag:
            st.toast(f"Используем локальный кэш: {filename} ({city})")
            return _load_file_from_local_cache(local_path)

    try:
        with st.spinner(f"Загружаем из облака: {filename} ({city})"):
            response = requests.get(remote_url)
            response.raise_for_status()

            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            with open(local_path, "wb") as f:
                f.write(response.content)

            with open(etag_path, "w", encoding="utf-8") as f:
                f.write(remote_etag)

            st.toast(f"Загружено и сохранено: {filename} ({city})")
    except Exception as e:
        st.error(f"Ошибка при скачивании {filename} ({city}): {e}")
        raise

    return _load_file_from_local_cache(local_path)

def load_city_data(city: str) -> dict:
    city_files = {
        "apartment_data": ("market_deals", f"{city}_geo_preprocessed_market_deals.parquet"),
        "house_data":    ("houses_info",    f"{city}_houses.parquet")
    }

    results = {}
    for key, (subfolder, filename) in city_files.items():
        df = load_data(city, filename, subfolder)
        results[key] = df

    return results