#!/usr/bin/env python3
"""
Скрипт для предварительного скачивания данных в локальную папку перед запуском приложения.
Использование:
    python download_data.py city1 city2 ...
Например:
    python download_data.py moscow spb novosibirsk
"""
import os
import sys
import requests
import logging

BASE_URL = "https://storage.yandexcloud.net/scienceforbusiness/data/regions"
LOCAL_DATA_DIR = "data/regions"
CITY_FILES = {
    "apartment_data": "{city}_prep.feather",
    "house_data": "{city}_apartment.feather",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def download_for_city(city: str):

    for key, template in CITY_FILES.items():
        filename = template.format(city=city)
        remote_url = f"{BASE_URL}/{city}/market_deals/{filename}"
        local_path = os.path.join(LOCAL_DATA_DIR, city, "market_deals", filename)
        etag_path = local_path + ".etag"

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            head_resp = requests.head(remote_url)
            head_resp.raise_for_status()
            remote_etag = head_resp.headers.get("ETag", "").strip('"')
        except Exception as e:
            logging.error(f"Не удалось получить ETag для {filename} ({city}): {e}")
            continue

        if os.path.exists(local_path) and os.path.exists(etag_path):
            with open(etag_path, 'r', encoding='utf-8') as f:
                local_etag = f.read().strip()
            if local_etag == remote_etag:
                logging.info(f"[SKIP] {filename} ({city}) — без изменений")
                continue

        try:
            logging.info(f"[DOWNLOAD] {filename} ({city})")
            get_resp = requests.get(remote_url)
            get_resp.raise_for_status()

            with open(local_path, 'wb') as f:
                f.write(get_resp.content)

            with open(etag_path, 'w', encoding='utf-8') as f:
                f.write(remote_etag)

            logging.info(f"[SAVED] {filename} ({city})")
        except Exception as e:
            logging.error(f"Ошибка при скачивании {filename} ({city}): {e}")


def main(cities):
    for city in cities:
        logging.info(f"=== Обработка города: {city} ===")
        download_for_city(city)
    logging.info("Готово.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python download_data.py city1 city2 ...")
        sys.exit(1)

    cities = sys.argv[1:]
    main(cities)
