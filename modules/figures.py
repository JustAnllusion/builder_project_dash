import datetime
import locale
from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
import plotly.express as px
from datetime import date


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shapely
import plotly.graph_objects as go



import numpy as np




def get_points_between_lines(points, k_upper, k_lower):
    """
    Возвращает точки, лежащие между двумя прямыми y = k_upper*x и y = k_lower*x
    
    Parameters:
    points (np.ndarray): массив точек формы (n, 2)
    k_upper (float): угловой коэффициент верхней прямой
    k_lower (float): угловой коэффициент нижней прямой
    
    Returns:
    tuple: (points_above, points_between, points_below)
    """
    points_above = points[points[:, 1] > k_upper * points[:, 0]]
    points_below = points[points[:, 1] < k_lower * points[:, 0]]
    points_between = points[
        (points[:, 1] <= k_upper * points[:, 0]) & 
        (points[:, 1] >= k_lower * points[:, 0])
    ]
    return points_above, points_between, points_below



def find_c(a, b, x0, y0):
  """Находит коэффициент сдвига c уравнения прямой ax + by + c = 0
  по известным коэффициентам a, b и координате точки (x0, y0), лежащей на прямой.

  Args:
    a: Коэффициент при x.
    b: Коэффициент при y.
    x0: X-координата точки на прямой.
    y0: Y-координата точки на прямой.

  Returns:
    Значение коэффициента сдвига c.
  """

  c = -a * x0 - b * y0
  return c

def find_c_max(a, b, points):
  """Находит максимальное значение c и соответствующую точку для заданных a, b и множества точек.

  Args:
    a: Коэффициент при x.
    b: Коэффициент при y.
    points: Список точек (x, y).

  Returns:
    Кортеж (max_c, (x, y)), где max_c - максимальное значение c,
    (x, y) - соответствующая точка.
  """

  results = []
  for x, y in points:
    c = -a * x - b * y
    results.append(c)

  results.sort(reverse=True)
  return results[0]
def find_c_min(a, b, points):
  """Находит максимальное значение c и соответствующую точку для заданных a, b и множества точек.

  Args:
    a: Коэффициент при x.
    b: Коэффициент при y.
    points: Список точек (x, y).

  Returns:
    Кортеж (max_c, (x, y)), где max_c - максимальное значение c,
    (x, y) - соответствующая точка.
  """

  results = []
  for x, y in points:
    c = -a * x - b * y
    results.append(c)

  results.sort(reverse=False)
  return results[0]


def add_points_to_voronoi(fig, data, house_ids, group_configs=None):
    """
    Add scatter points to existing Voronoi visualization with colors based on groups
    Returns updated Plotly figure
    """
    segment = 0
    date_ranges = pd.date_range(start='2000-01-01', end='2024-12-31', periods=2)
    
    color_palette_markers = [
        'rgba(255, 105, 180, 1)',  # hotpink (тёплый)
        'rgba(255, 165, 0, 1)',    # orange (тёплый)
        'rgba(255, 69, 0, 1)',     # orangered (тёплый)
        'rgba(30, 144, 255, 1)',   # dodgerblue (холодный)
        'rgba(218, 165, 32, 1)',   # goldenrod (тёплый)
        'rgba(60, 179, 113, 1)',   # mediumseagreen (холодный)
        'rgba(186, 85, 211, 1)',   # mediumorchid (тёплый)
        'rgba(255, 255, 0, 1)',    # yellow (тёплый)
        'rgba(255, 69, 0, 1)',     # orangered (тёплый)
        'rgba(0, 0, 205, 1)',      # mediumblue (холодный)
        'rgba(255, 0, 0, 1)',      # red (тёплый)
        'rgba(0, 206, 209, 1)',    # darkturquoise (холодный)
        'rgba(255, 215, 0, 1)',    # gold (тёплый)
        'rgba(70, 130, 180, 1)',   # steelblue (холодный)
        'rgba(255, 140, 0, 1)',    # darkorange (тёплый)
        'rgba(0, 191, 255, 1)',    # deepskyblue (холодный)
        'rgba(220, 20, 60, 1)',    # crimson (тёплый)
    ]

    segment_data = data.copy()
    segment_data = segment_data[segment_data['house_id'].isin(house_ids)]
    
    # Создаем словарь для хранения информации о принадлежности house_id к группам
    house_to_group = {}
    
    # Если предоставлены настройки групп, используем их для определения цветов
    if group_configs:
        for group_name, config in group_configs.items():
            if "filtered_data" in config and "house_id" in config["filtered_data"].columns:
                group_house_ids = config["filtered_data"]["house_id"].dropna().unique().tolist()
                for house_id in group_house_ids:
                    if house_id in house_ids:  # Учитываем только те дома, которые в запрошенных house_ids
                        house_to_group[house_id] = group_name
    
    # Если house_id не принадлежит ни к одной группе, присваиваем ему группу "other"
    for house_id in house_ids:
        if house_id not in house_to_group:
            house_to_group[house_id] = "other"
    
    # Создаем маппинг групп на цвета
    unique_groups = list(set(house_to_group.values()))
    group_to_color = {}
    
    for i, group_name in enumerate(unique_groups):
        # Если это настроенная группа, берем цвет из ее конфигурации
        if group_configs and group_name in group_configs and "vis" in group_configs[group_name] and "color" in group_configs[group_name]["vis"]:
            group_to_color[group_name] = group_configs[group_name]["vis"]["color"]
        else:
            # Иначе используем цвет из палитры
            group_to_color[group_name] = color_palette_markers[i % len(color_palette_markers)]
    
    # Присваиваем цвета на основе группы, а не house_id
    marker_colors = [group_to_color[house_to_group[house_id]] for house_id in segment_data['house_id']]
    
    start_date = date_ranges[segment]
    end_date = date_ranges[segment + 1]
    segment_title = f"Сегмент от {start_date.strftime('%d-%m-%Y')} до {end_date.strftime('%d-%m-%Y')}"

    # Добавляем точки для каждой группы отдельно (для корректной легенды)
    for group_name in unique_groups:
        group_houses = [house_id for house_id, group in house_to_group.items() if group == group_name]
        group_data = segment_data[segment_data['house_id'].isin(group_houses)]
        
        if not group_data.empty:
            fig.add_trace(go.Scattergl(
                x=group_data["price_disc"], 
                y=group_data["total_price_discounted"],
                mode='markers',
                marker=dict(
                    color=group_to_color[group_name],
                    size=5,
                    line=dict(color='black', width=1) 
                ),
                name=group_name
            ))
    
    return fig



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
           .resample("ME")
           .mean()
           .to_period()
           .pct_change(fill_method=None))
    
    pct.iloc[-1] = pct.iloc[-1]  

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