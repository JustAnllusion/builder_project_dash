import numpy as np
import pandas as pd


def relative_number_type(types: pd.Series, type1: str, type2: str) -> float:
    """
    Функция подсчитывает количество раз, которое встречается каждый из двух указанных типов,
    и возвращает отношение количества первого типа к количеству второго. Если второй тип не
    встречается ни разу, функция возвращает NaN, чтобы избежать деления на ноль.


    :param types: pd.Series, содержащий категориальные данные, в которых осуществляется подсчёт.
    :param type1: Название первого типа для подсчёта.
    :param type2: Название второго типа для подсчёта и вычисления относительного количества по отношению к первому типу.
    :return:
    """
    number_type1 = np.sum((types == type1))
    number_type2 = np.sum((types == type2))
    if number_type2 != 0:
        return number_type1 / number_type2

    return np.nan


def remove_price_outliers(deals: pd.DataFrame, alpha: float = 0.005) -> pd.DataFrame:
    """
    Удаляет выбросы по цене, используя квантили.

    :param deals: Данные со сделками.
    :param alpha: Процент сделок выбрасываемых при фильтрации.
    :return: Сделки без выбросов.

    """
    q_left = deals["price"].quantile(alpha / 2)
    q_right = deals["price"].quantile(1 - alpha / 2)
    quantile_mask = (q_left <= deals["price"]) & (deals["price"] <= q_right)
    deals_filtered = deals[quantile_mask]
    return deals_filtered


def get_interest_rate(deals: pd.DataFrame, target_name: str = "price", is_ml: bool = True) -> pd.Series:
    """
    Функция позволяет оценить скорость изменения параметра target_name.

    :param deals: Данные со сделками.
    :param target_name: параметр, для которого оцениваем скорость изменения.
    :param is_ml: параметр, отвечающий за способ дисконтирования. True -- ретроспективно,
    False -- по актуальным данным.
    :return: временной ряд, соответствующий скорости изменения target.
    """
    pct = (1 + deals.set_index("contract_date")[target_name]
           .resample("M")
           .mean()
           .to_period()
           .pct_change())
    pct.iloc[pct.index.shift()[-1]] = pct.iloc[-1]
    if is_ml:
        pct = pct.shift()
    return pct.fillna(1).cumprod()


def discounting(deals: pd.DataFrame, interest_rate: pd.Series, target_name: str = "price") -> pd.Series:
    """
    Функция возвращает ряд с дисконтированными ценами

    :param deals: Данные со сделками.
    :param interest_rate: временной ряд, соответствующий скорости изменения параметра target
    :param target_name: изучаемый параметр.
    :return: ряд с дисконтированными ценами.
    """
    return (deals.set_index("contract_date", append=True)[target_name]
            .resample("M", level=1)
            .transform(lambda x: x / interest_rate[x.name.to_period("M")])
            .reset_index(level=1, drop=True))


def compounding(deals: pd.DataFrame, interest_rate: pd.Series, target_name: str = "price") -> pd.Series:
    """
    Функция возвращает ряд с компаундированными ценами (процедура, обратная дисконтированию).

    :param deals: Данные со сделками.
    :param interest_rate: временной ряд, соответствующий скорости изменения параметра target
    :param target_name: изучаемый параметр.
    :return: ряд с компаундированными ценами
    """
    return (deals.set_index("contract_date", append=True)[target_name]
            .resample("M", level=1)
            .transform(lambda x: x * interest_rate[x.name.to_period("M")])
            .reset_index(level=1, drop=True))


def uniform_distribute_geo(deals: pd.DataFrame, r: float):
    """
    Равномерно распределяет геоточки вокруг геоточки проекта с радиусом r

    :param deals: Данные со сделками.
    :param r: Радиус(км)
    :return: Новый набор данных с равномерно распределенными геоточками
    """
    deals = deals.copy()
    r = 1 / 111 * r
    angle = np.random.uniform(0, 2 * np.pi, len(deals["latitude"]))
    radius = np.sqrt(np.random.uniform(0, r ** 2, len(deals["longitude"])))
    deals["latitude"] = deals["latitude"] + radius * np.sin(angle)
    deals["longitude"] = deals["longitude"] + radius * np.cos(angle)
    return deals


def clean_string_columns(deals: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует все строковые колонки в DataFrame, приводя их к нижнему регистру
    и обрезая пробелы слева и справа.

    :param deals: Данные со сделками.
    :return: Новый набор данных с преобразованными строковыми колонками.
    """
    deals = deals.copy()
    string_columns = deals.select_dtypes(include=['object']).columns

    for col in string_columns:
        deals[col] = deals[col].str.lower().str.strip()

    return deals