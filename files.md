# Работа с файлами и облачным хранилищем

## 1. Общая структура

Проект использует облачное хранилище (Yandex Object Storage) для хранения входных данных.  
Для повышения производительности реализовано локальное кэширование.  
Исходные данные не хранятся в репозитории и не коммитятся в git.

## 2. Хранилище данных

- Тип хранилища: Yandex Object Storage (S3)
- Бакет: scienceforbusiness  
- Путь: data/regions/msk/market_deals/

Примеры файлов:
- msk_prep.feather
- msk_apartment.feather
- msk.feather

Пример прямой ссылки:  
https://storage.yandexcloud.net/scienceforbusiness/data/regions/msk/market_deals/msk_prep.feather

Файлы публикуются с правом доступа allUsers: READ

## 3. Загрузка файлов в приложении

Все обращения к данным производятся через функцию load_data():

```python
from utils.load_data import load_data

df = load_data("msk_prep.feather")
```
Поведение `load_data()`:

- Проверяет наличие локального файла в data/regions/msk/market_deals/
- Выполняет HEAD-запрос к Object Storage для получения ETag
- Сравнивает ETag с локальным .etag файлом
- При несовпадении — загружает актуальную версию файла
- При совпадении — использует локальную копию без повторной загрузки

## 4. Локальная структура

Входные данные:  
`data/regions/msk/market_deals/`

Кэш и промежуточные результаты:  
`data/regions/msk/market_deals/cache`

Пример сохранения результата:
`df.to_feather("data/regions/msk/market_deals/cache/depletion_curves.feather")`

## 5. Добавление новых файлов

1. Загрузить файл в Object Storage по пути data/regions/msk/market_deals/
2. Установить ACL: Object ACL → allUsers → READ
3. Проверить доступность файла по прямой ссылке
4. Использовать в коде через load_data("имя_файла.feather")

## 6. Версионирование

- Версионирование в Object Storage на текущий момент отключено
- Используется ETag для контроля актуальности
- Включение полноценного versioning возможно через консоль управления Yandex Cloud

## 7. TODO

- Перейти на приватный доступ к данным через IAM + временные signed URLs
- Вынести пути и конфигурацию загрузки в отдельный config.py или .env
- Реализовать лог действий по загрузке и кэшированию
- Интеграция с системой контроля версий данных yandex storage при необходимости
