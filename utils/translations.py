import pandas as pd
import streamlit as st

rus_columns = {
    'project': 'Проект',
    'house_id': 'ID дома',
    'rooms_0': 'Доля 0-к квартир',
    'rooms_1': 'Доля 1-к квартир',
    'rooms_2': 'Доля 2-к квартир',
    'rooms_3': 'Доля 3-к квартир',
    'rooms_4': 'Доля 4-к квартир',
    'rooms_5': 'Доля 5-к квартир',
    'rooms_6': 'Доля 6-к квартир',
    'rooms_7': 'Доля 7-к квартир',
    'rooms_8': 'Доля 8-к квартир',
    'rooms_9': 'Доля 9-к квартир',
    'rooms_10': 'Доля 10-к квартир',
    'rooms_11': 'Доля 11-к квартир',
    'rooms_number_mean': 'Среднее кол-во комнат',
    'start_year_sales': 'Год начала продаж',
    'start_sales': 'Начало продаж',
    'deals_sold': 'Доля проданных квартир',
    'ndeals': 'Кол-во квартир',
    'mean_area': 'Средняя площадь',
    'mean_price': 'Средняя цена',
    'mean_price_orig': 'Средняя цена (не дисконтированная)',
    '0_mean_area': 'Средняя площадь 0-к.',
    '1_mean_area': 'Средняя площадь 1-к.',
    '2_mean_area': 'Средняя площадь 2-к.',
    '3_mean_area': 'Средняя площадь 3-к.',
    '4_mean_area': 'Средняя площадь 4-к.',
    '5_mean_area': 'Средняя площадь 5-к.',
    '6_mean_area': 'Средняя площадь 6-к.',
    '7_mean_area': 'Средняя площадь 7-к.',
    '8_mean_area': 'Средняя площадь 8-к.',
    '9_mean_area': 'Средняя площадь 9-к.',
    '10_mean_area': 'Средняя площадь 10-к.',
    '11_mean_area': 'Средняя площадь 11-к.',
    '0_mean_price': 'Средняя цена 0-к.',
    '1_mean_price': 'Средняя цена 1-к.',
    '2_mean_price': 'Средняя цена 2-к.',
    '3_mean_price': 'Средняя цена 3-к.',
    '4_mean_price': 'Средняя цена 4-к.',
    '5_mean_price': 'Средняя цена 5-к.',
    '6_mean_price': 'Средняя цена 6-к.',
    '7_mean_price': 'Средняя цена 7-к.',
    '0_mean_selling_time': 'Среднее время продажи 0-к.',
    '1_mean_selling_time': 'Среднее время продажи 1-к.',
    '2_mean_selling_time': 'Среднее время продажи 2-к.',
    '3_mean_selling_time': 'Среднее время продажи 3-к.',
    '4_mean_selling_time': 'Среднее время продажи 4-к.',
    '5_mean_selling_time': 'Среднее время продажи 5-к.',
    '6_mean_selling_time': 'Среднее время продажи 6-к.',
    '7_mean_selling_time': 'Среднее время продажи 7-к.',
    'mean_selling_time': 'Среднее время продажи',
    'median_selling_time': 'Медианное время продажи',
    'q90_selling_time': '90-й перцентиль времени продажи',
    'stage': 'Этап',
    'mean_room_area': 'Средняя площадь комнат',
    'disctrict': 'Район',
    'developer': 'Девелопер',
    'class': 'Класс',
    'latitude': 'Широта',
    'longitude': 'Долгота',
    'pantry': 'Кладовая',
    'parking': 'Парковка',
    'floor': 'Этаж'
}


def translate_df(df: pd.DataFrame, mapping: dict = rus_columns) -> pd.DataFrame:
    return df.rename(columns=mapping)


def st_dataframe(df: pd.DataFrame, **kwargs):
    st.dataframe(translate_df(df), **kwargs)


def st_table(df: pd.DataFrame, **kwargs):
    st.table(translate_df(df), **kwargs)
