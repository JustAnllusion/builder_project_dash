import pandas as pd
import streamlit as st

rus_columns = {
    'project': 'проект',
    'house_id': 'ID дома',
    'rooms_0': 'доля 0-комнатных квартир',
    'rooms_1': 'доля 1-комнатных квартир',
    'rooms_2': 'доля 2-комнатных квартир',
    'rooms_3': 'доля 3-комнатных квартир',
    'rooms_4': 'доля 4-комнатных квартир',
    'rooms_5': 'доля 5-комнатных квартир',
    'rooms_6': 'доля 6-комнатных квартир',
    'rooms_7': 'доля 7-комнатных квартир',
    'rooms_8': 'доля 8-комнатных квартир',
    'rooms_9': 'доля 9-комнатных квартир',
    'rooms_10': 'доля 10-комнатных квартир',
    'rooms_11': 'доля 11-комнатных квартир',
    'rooms_number_mean': 'ср. кол-во комнат',
    'start_year_sales': 'год начала продаж',
    'start_sales': 'начало продаж',
    'deals_sold': 'доля проданных квартир',
    'ndeals': 'кол-во квартир',
    'mean_area': 'средняя площадь',
    'mean_price': 'средняя цена',
    '0_mean_area': 'ср. площадь 0-комн.',
    '1_mean_area': 'ср. площадь 1-комн.',
    '2_mean_area': 'ср. площадь 2-комн.',
    '3_mean_area': 'ср. площадь 3-комн.',
    '4_mean_area': 'ср. площадь 4-комн.',
    '5_mean_area': 'ср. площадь 5-комн.',
    '6_mean_area': 'ср. площадь 6-комн.',
    '7_mean_area': 'ср. площадь 7-комн.',
    '8_mean_area': 'ср. площадь 8-комн.',
    '9_mean_area': 'ср. площадь 9-комн.',
    '10_mean_area': 'ср. площадь 10-комн.',
    '11_mean_area': 'ср. площадь 11-комн.',
    '0_mean_price': 'ср. цена 0-комн.',
    '1_mean_price': 'ср. цена 1-комн.',
    '2_mean_price': 'ср. цена 2-комн.',
    '3_mean_price': 'ср. цена 3-комн.',
    '4_mean_price': 'ср. цена 4-комн.',
    '5_mean_price': 'ср. цена 5-комн.',
    '6_mean_price': 'ср. цена 6-комн.',
    '7_mean_price': 'ср. цена 7-комн.',
    '0_mean_selling_time': 'ср. время продажи 0-комн.',
    '1_mean_selling_time': 'ср. время продажи 1-комн.',
    '2_mean_selling_time': 'ср. время продажи 2-комн.',
    '3_mean_selling_time': 'ср. время продажи 3-комн.',
    '4_mean_selling_time': 'ср. время продажи 4-комн.',
    '5_mean_selling_time': 'ср. время продажи 5-комн.',
    '6_mean_selling_time': 'ср. время продажи 6-комн.',
    '7_mean_selling_time': 'ср. время продажи 7-комн.',
    'mean_selling_time': 'ср. время продажи',
    'median_selling_time': 'медианное время продажи',
    'q90_selling_time': '90-й перцентиль времени продажи',
    'stage': 'этап',
    'mean_room_area': 'ср. площадь комнат',
    'disctrict': 'район',
    'developer': 'застройщик',
    'class': 'класс',
    'latitude': 'широта',
    'longitude': 'долгота',
    'pantry': 'кладовая',
    'parking': 'парковка',
    'floor': 'этаж'
}


def translate_df(df: pd.DataFrame, mapping: dict = rus_columns) -> pd.DataFrame:
    return df.rename(columns=mapping)


def st_dataframe(df: pd.DataFrame, **kwargs):
    st.dataframe(translate_df(df), **kwargs)


def st_table(df: pd.DataFrame, **kwargs):
    st.table(translate_df(df), **kwargs)
