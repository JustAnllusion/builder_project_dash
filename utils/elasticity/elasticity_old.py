import typing as tp
import warnings
from itertools import cycle

import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as st
from bokeh.models import Div, LabelSet, TabPanel, Tabs
from bokeh.models.ranges import DataRange1d
from bokeh.palettes import magma
from bokeh.plotting import ColumnDataSource, column, figure
from bokeh.transform import factor_cmap
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
# from .utils import *
# from .variables import *
import numpy as np

warnings.filterwarnings("ignore")
DICT_ELASTICITY_ROOMS_NUMBER = {
    30: [(0, 1)],
    35: [(1, 2)],
    40: [(1, 2)],
    45: [(1, 2)],
    50: [(1, 2), (2, 3)],
    55: [(1, 2), (2, 3)],
    60: [(2, 3)],
    65: [(2, 3)],
    70: [(2, 3)],
    75: [(2, 3), (3, 4)],
    80: [(2, 3), (3, 4)],
    85: [(2, 3), (3, 4)],
    90: [(2, 3), (3, 4)],
    95: [(2, 3), (3, 4)]
}

DICT_ELASTICITY_ROOMS_NUMBER_WITHOUT_SEGMENTS = {
    30: [(0,1), (1,2), (2,3), (3,4), (4,5)]
}


FLOOR_ELASTICITY = [
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25)
]




def find_segment(x, area_min, area_max, split_parameter):
    area_min = int(area_min)
    area_max = int(area_max)
    split_parameter = int(split_parameter)
    segments = list(range(area_min, area_max + split_parameter, split_parameter))
    for i in range(len(segments) - 1):
        if segments[i] <= x < segments[i + 1]:
            return segments[i]
    return segments[-1]

def get_area_intersection_for_rooms_number(
    deals: pd.DataFrame,
    area_min: tp.Optional[int] = None,
    area_max: tp.Optional[int] = None,
    q: float = 0,
    density: bool = True,
    developers: tp.Optional[list[list[int]]] = None,
    show_all: bool = False,
    names: tp.Optional[list[str]] = None,
) -> tp.Any:
    """
    Функция возвращает гистограммы для площадей проданных объектов в разбивке по комнатам
    :param deals: все сделки
    :param area_min: минимальная площадь квартиры
    :param area_max: максимальная площадь квартиры
    :param q: уровень квантили, по которой обрезаем данные
    :param density: доля или количество сделок
    :param developers: для каких застройщиков строим графики
    :param show_all: параметр, отвечающий за построения гистограмм для всех девелоперов
    :param names: названия графиков
    :return: column с гистограммами
    """

    df = deals.copy()

    if area_min is None:
        area_min = df["area"].min()
    if area_max is None:
        area_max = df["area"].max()
    df = df[(df["area"] >= area_min) & (df["area"] <= area_max)]

    if developers is None:
        developers = [sorted(list(set(df["developer"])))]
    elif show_all:
        developers.insert(0, sorted(list(set(df["developer"]))))
    if names is None:
        names = ["Рынок"]

    res = []
    names_cycle = cycle(names)
    for dev in developers:
        df_dev = df[df["developer"].isin(dev)]

        list_rooms = sorted(list(set(df_dev["rooms_number"])))

        palette = cycle(px.colors.qualitative.Plotly)
        current_name = next(names_cycle)
        p = figure(
            width=1300,
            height=500,
            title=f"Сравнение распределения площади квартир разной комнатности, {current_name}",
            x_range=DataRange1d(only_visible=True),
            y_range=DataRange1d(only_visible=True),
        )

        for i in range(len(list_rooms)):
            current_color = next(palette)
            deals_rooms = df_dev[df_dev["rooms_number"] == list_rooms[i]]
            x = deals_rooms[
                (deals_rooms["area"] <= np.quantile(deals_rooms["area"], 1 - q))
                & (deals_rooms["area"] >= np.quantile(deals_rooms["area"], q))
            ]["area"]
            hist, edges = np.histogram(x, density=density, bins=20)
            p.quad(
                top=hist,
                bottom=0,
                left=edges[:-1],
                right=edges[1:],
                fill_color=current_color,
                line_color="white",
                legend_label=f"{list_rooms[i]} комнат",
                alpha=0.7,
            )

        p.legend.click_policy = "hide"
        res.append(p)
    return column(res)


def step_plot(x, mode="argument"):
    y = np.array(x)
    if mode == "argument":
        y = [[y[i], y[i + 1]] for i in range(len(y) - 1)] + [[y[-1]]]
        y = [y[i][j] for i in range(len(y)) for j in range(len(y[i]))]
        return np.array(y)
    elif mode == "value":
        y = [[y[0]]] + [[y[i], y[i + 1]] for i in range(len(y) - 1)]
        y = [y[i][j] for i in range(len(y)) for j in range(len(y[i]))]
        return np.array(y)
    else:
        return None


def get_room_number_distribution_by_area(
    deals: pd.DataFrame,
    area_min: tp.Optional[int] = None,
    area_max: tp.Optional[int] = None,
    split_parameter: int = 1,
) -> tp.Any:
    """
    Функция строит areaplot для распределения количества комнат в каждом сегменте площади
    :param deals: все сделки
    :param area_min: минимальная площадь
    :param area_max: максимальная площадь
    :param split_parameter: параметр сегментирования площади
    :return: areaplot с распределением количества комнат в каждом сегменте площади
    """
    if area_min is None:
        area_min = deals["area"].min()
    if area_max is None:
        area_max = deals["area"].max()

    deals_temp = deals[(deals["area"] >= area_min) & (deals["area"] <= area_max)]
    deals_temp["area"] = deals_temp["area"].apply(
        lambda t: find_segment(t, area_min, area_max, split_parameter)
    )

    rooms_set = sorted(list(set(deals_temp["rooms_number"])))
    res = []
    for i in range(len(rooms_set)):
        df = deals_temp[deals_temp["rooms_number"] == rooms_set[i]]
        res.append(df.groupby(by="area").count()["house_id"])

    index_range = np.arange(area_min, area_max + 1, split_parameter)
    res = [res[i].reindex(index_range, fill_value=0) for i in range(len(res))]

    total = sum(res[i] for i in range(len(res)))
    res = [res[i] / total for i in range(len(res))]

    x = step_plot(res[0].index, "argument")
    res = [step_plot(res[i], "value") for i in range(len(res))]

    p = figure(
        width=1400,
        height=800,
        title=f"Распределение комнатности квартир в зависимости от площади, "
        f"параметр сегментирования: {split_parameter}",
    )
    palette = cycle(px.colors.qualitative.Plotly)

    prev = np.zeros(len(res[0]))
    for i in range(len(res)):
        current_color = next(palette)
        p.varea(
            x=x,
            y1=prev,
            y2=prev + res[i],
            color=current_color,
            alpha=0.75,
            legend_label=f"{rooms_set[i]} комнат",
        )
        p.line(
            x=x,
            y=prev + res[i],
            line_color="black",
            line_width=2,
            legend_label=f"{rooms_set[i]} комнат",
        )
        prev = prev + res[i]

    prev = np.zeros(len(res[0]))
    for i in range(len(res)):
        source = ColumnDataSource(
            data=dict(
                x=(x[::2] + split_parameter / 2 - 1)[:-1],
                y=((prev[::2] + (prev + res[i])[::2]) / 2 - 0.01)[:-1],
                names=[
                    (
                        str(np.round(res[i][::2][j] * 100, 0))[:-2] + "%"
                        if res[i][::2][j] > 0.027
                        else ""
                    )
                    for j in range(len(res[i][::2]))
                ][:-1],
            )
        )
        labels = LabelSet(
            x="x", y="y", text="names", source=source, text_font_size="10pt"
        )
        p.add_layout(labels)
        prev = prev + res[i]

    p.line([index_range[0], index_range[0]], [0, 1], line_color="black", line_width=2)
    p.line([index_range[-1], index_range[-1]], [0, 1], line_color="black", line_width=2)
    p.line([index_range[0], index_range[-1]], [0, 0], line_color="black", line_width=2)
    p.line([index_range[0], index_range[-1]], [1, 1], line_color="black", line_width=2)

    for i in range(len(index_range)):
        p.line(
            [index_range[i], index_range[i]],
            [0, 1],
            line_color="black",
            line_width=0.25,
            line_dash="dashed",
        )

    p.add_layout(p.legend[0], "right")
    p.legend.click_policy = "hide"

    return p


def get_intersected_rooms_number_elasticity(
    deals: pd.DataFrame,
    area_min: tp.Optional[float] = None,
    area_max: tp.Optional[float] = None,
    split_parameter: int = 1,
    discounting_mode: str = "trend",
    segmentation: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Функция возвращает коэффициенты эластичности по количеству комнат
    как по каждому из корпусов, так и агрегированную
    :param deals: все сделки
    :param area_min: минимальное значение площади
    :param area_max: максимальное значение площади
    :param split_parameter: параметр сегментирования площади
    :param discounting_mode: параметр дисконтирования
    :param segmentation: параметр, отвечающий за сегментирование площади
    :return: словарь из датафреймов (по корпусам и агрегированный)
    """

    df = deals.copy()

    if segmentation:
        current_dict = DICT_ELASTICITY_ROOMS_NUMBER
    else:
        current_dict = DICT_ELASTICITY_ROOMS_NUMBER_WITHOUT_SEGMENTS

    if area_min is None:
        area_min = df["area"].min()
    if area_max is None:
        area_max = df["area"].max()

    # if discounting_mode == "retro":
    #     interest_rate = get_interest_rate(df, is_ml=True)
    #     df["price"] = discounting(df, interest_rate)
    # if discounting_mode == "actual":
    #     interest_rate = get_interest_rate(df, is_ml=False)
    #     df["price"] = discounting(df, interest_rate)

    df = df[(df["area"] >= area_min) & (df["area"] <= area_max)]
    df["area"] = df["area"].apply(
        lambda x: find_segment(x, area_min, area_max, split_parameter)
    )

    house_dict = dict()
    for area in tqdm(current_dict.keys()):
        deals_temp = df[df["area"] == area]
        res = []
        for elem in current_dict[area]:
            temp_res = []
            for house in list(set(deals_temp["house_id"])):
                deals_temp_house = deals_temp[deals_temp["house_id"] == house]
                if set(elem).issubset(set(deals_temp_house["rooms_number"])):
                    temp_res.append(house)
            res.append(temp_res)
        house_dict.update({area: res})

    result = pd.DataFrame(
        columns=["area", "rooms", "house_id", "number_of_deals", "coefficient"]
    )
    for area in tqdm(current_dict.keys()):
        for i in range(len(current_dict[area])):
            list_of_house = house_dict[area][i]
            rooms_number = current_dict[area][i]
            df_corp = df[
                (df["rooms_number"].isin(rooms_number))
                & (df["area"] == area)
                & (df["house_id"].isin(list_of_house))
            ]
            gr = df_corp.groupby(by="house_id")
            for group in gr.groups:
                df_corp_group = gr.get_group(group)
                res = (
                    df_corp_group[["rooms_number", "price"]]
                    .groupby(by="rooms_number")
                    .mean()["price"]
                )
                res.sort_index(ascending=True, inplace=True)
                result = result.append(
                    {
                        "area": area,
                        "rooms": rooms_number,
                        "house_id": group,
                        "number_of_deals": (
                            len(
                                df_corp_group[
                                    df_corp_group["rooms_number"] == rooms_number[0]
                                ]
                            ),
                            len(
                                df_corp_group[
                                    df_corp_group["rooms_number"] == rooms_number[1]
                                ]
                            ),
                        ),
                        "coefficient": res.iloc[1] / res.iloc[0],
                    },
                    ignore_index=True,
                )

    final_result = {
        "detailed": result,
        "agg": result[["area", "rooms", "coefficient"]]
        .groupby(by=["area", "rooms"])
        .mean(),
    }

    return final_result


def house_room_filter(x: pd.DataFrame, elem: tuple) -> bool:
    """
    Функция нужна для фильтрации номера корпуса и количества комнат для данного перехода при оценке эластичности
    по номеру этажа
    :param x: сделки, сгруппированные по house_id и rooms_number
    :param elem: конкретный переход между этажами
    :return:
    """
    return {elem[0], elem[1]}.issubset(set(x["floor"]))


def floor_elasticity_hra(x: pd.DataFrame, upper_floor: pd.DataFrame) -> float:
    """
    Функция оценивает коэффициент эластичности цены от этажа при фиксированных house_id, rooms_number, area
    для конкретного перехода между этажами
    :param x: датафрейм для нижнего этажа
    :param upper_floor: датафрейм для верхнего этажа
    :return: коэффициент эластичности цены от этажа при фиксированных house_id, rooms_number, area
    """
    lower_area = x["area"].iloc[0]
    upper_floor = upper_floor[
        (upper_floor["area"] >= lower_area - 1)
        & (upper_floor["area"] <= lower_area + 1)
    ]
    if len(upper_floor) == 0:
        return -1
    return upper_floor["price"].mean() / x["price"].mean()


def floor_elasticity_hr(x: pd.DataFrame, elem: tuple) -> float:
    """
    Функция возвращает коэффициент эластичности цены от этажа при фиксированных house_id и rooms_number
    :param x: сделки с фиксированными параметрами house_id и rooms_number
    :param elem: конкретный переход между этажами
    :return: коэффициент эластичности цены от этажа при фиксированных параметрах house_id и rooms_number
    """
    down_floor = x[x["floor"] == elem[0]]
    upper_floor = x[x["floor"] == elem[1]]

    if len(down_floor) == 0 or len(upper_floor) == 0:
        return -1

    res = pd.DataFrame(
        down_floor.groupby(by="area").apply(
            lambda x_arg: floor_elasticity_hra(x_arg, upper_floor)
        )
    )
    res = res[res[0] >= 0]
    return res[0].mean()


def get_floor_elasticity(deals: pd.DataFrame, discounting_mode: str = "trend"):

    df = deals.copy()

    # if discounting_mode == "retro":
    #     interest_rate = get_interest_rate(df, is_ml=True)
    #     df["price"] = discounting(deals, interest_rate)
    # if discounting_mode == "actual":
    #     interest_rate = get_interest_rate(df, is_ml=False)
    #     df["price"] = discounting(deals, interest_rate)
    #
    dtest = [
        df.groupby(by=["house_id", "rooms_number"]).apply(
            lambda x: house_room_filter(x, FLOOR_ELASTICITY[i])
        )
        for i in tqdm(range(len(FLOOR_ELASTICITY)))
    ]
    dtest = [pd.DataFrame(dtest[i]).reset_index() for i in range(len(dtest))]
    dtest = [dtest[i][dtest[i][0]] for i in range(len(dtest))]
    dict_house = {
        FLOOR_ELASTICITY[i]: [
            (dtest[i]["house_id"].iloc[j], dtest[i]["rooms_number"].iloc[j])
            for j in range(len(dtest[i]))
        ]
        for i in range(len(FLOOR_ELASTICITY))
    }

    result = {FLOOR_ELASTICITY[i]: 0 for i in range(len(FLOOR_ELASTICITY))}
    for i in tqdm(range(len(FLOOR_ELASTICITY))):
        df_temp = df[
            df[["house_id", "rooms_number"]]
            .apply(tuple, axis=1)
            .isin(dict_house[FLOOR_ELASTICITY[i]])
        ]
        temp_result = df_temp.groupby(by=["house_id", "rooms_number"], group_keys=False).apply(
            lambda x: floor_elasticity_hr(x, FLOOR_ELASTICITY[i])
        )
        temp_result = temp_result.reset_index()
        temp_result = temp_result[~temp_result[0].isna()]
        temp_result = temp_result[temp_result[0] > 0]
        result[FLOOR_ELASTICITY[i]] = (
            temp_result[["house_id", 0]].groupby(by="house_id").mean()[0].mean()
        )

    return result


def get_atomic_rn_elasticity(
    deals: pd.DataFrame,
    number_of_projects: int = 10,
    number_of_house: int = 5,
    discounting_mode: str = "trend",
    min_year: int = 2020,
    max_year: int = 2024,
) -> column:
    """
    :param deals: сделки
    :param number_of_projects: топ проектов для которых строим графики атомарной эластичности
    :param number_of_house: топ домов внутри каждого проекта, для которых строим график атомарной эластичности
    :param discounting_mode: режим дисконтирования
    :param min_year: минимальный год для сделок
    :param max_year: максимальный год для сделок
    :return: column с атомарными эластичностями
    """
    data = deals.copy()

    # if discounting_mode == "actual":
    #     interest_rate = get_interest_rate(data, is_ml=False)
    #     data["price"] = discounting(data, interest_rate)
    #
    # if discounting_mode == "retro":
    #     interest_rate = get_interest_rate(data, is_ml=True)
    #     data["price"] = discounting(data, interest_rate)
    #
    data = data[
        (data["contract_date"].dt.year >= min_year)
        & (data["contract_date"].dt.year <= max_year)
    ]
    data["area"] = data["area"].round(0)

    list_of_projects = list(
        data["project"].value_counts().iloc[:number_of_projects:].index
    )

    result = []
    for prj in tqdm(list_of_projects):
        deals_proj = data[data["project"] == prj]
        list_of_house = list(
            deals_proj["house_id"].value_counts().iloc[:number_of_house:].index
        )
        for house in list_of_house:
            deals_house = deals_proj[deals_proj["house_id"] == house]
            floor_set = sorted(list(set(deals_house["floor"])))
            deals_house["floor"] = deals_house["floor"].astype(str)

            index_cmap = factor_cmap(
                "floor",
                palette=magma(len(set(floor_set))),
                factors=sorted(deals_house.floor.unique()),
            )

            p = figure(
                width=1300,
                height=600,
                title=(
                    prj
                    + ", дом "
                    + str(house)
                    + f", {len(deals_house)} сделок, дисконтирование={discounting_mode}"
                ),
            )

            for i in range(len(floor_set)):
                df = deals_house.loc[(deals_house.floor == f"{floor_set[i]}")]
                df["rooms_number"] = df["rooms_number"] + st.uniform.rvs(
                    -0.1, 0.2, size=len(df)
                )
                p.scatter(
                    "rooms_number",
                    "price",
                    source=df,
                    fill_alpha=0.6,
                    fill_color=index_cmap,
                    size=10,
                    legend_group="floor",
                )

            x_all_mean = deals_house.groupby(by="rooms_number").mean().index
            y_all_mean = deals_house.groupby(by="rooms_number").mean()["price"]

            p.line(
                x_all_mean,
                y_all_mean,
                line_width=2,
                line_color="red",
                legend_label="Зависимость в среднем",
            )
            p.add_layout(p.legend[0], "right")
            p.legend.click_policy = "hide"
            p.legend.label_text_font_size = "12px"
            p.legend.ncols = 2

            result.append(p)
    return column(result)
