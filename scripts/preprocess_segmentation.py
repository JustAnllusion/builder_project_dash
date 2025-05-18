import os
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import shapely
from scipy.spatial import Voronoi
import geopandas as gpd
import plotly.graph_objects as go



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


def create_voronoi_base(intersec_point, intersec_Line_up_line_upper, 
                       intersec_line_upper_line_right, intersec_line_right_line_low,
                       intersec_point_old, intersec_Line_up_line_upper_old,
                       intersec_line_upper_line_right_old, intersec_line_right_line_low_old,
                       centroids, centroids_old, scales, means):
    """
    Create base Voronoi tessellation visualization
    Returns a Plotly figure with Voronoi regions
    """

    color_palette = [
    'rgba(220, 20, 60, 0.9)',
    'rgba(232, 65, 24, 0.9)',
    'rgba(255, 127, 0, 0.9)',
    'rgba(255, 215, 0, 0.9)',
    'rgba(173, 255, 47, 0.9)',
    'rgba(124, 252, 0, 0.9)',
    'rgba(34, 139, 34, 0.9)'
    ]

    # Create the initial polygon
    polygon = shapely.wkt.loads(f'Polygon (({intersec_point[0]} {intersec_point[1]},'
                    f'{intersec_Line_up_line_upper[0]} {intersec_Line_up_line_upper[1]},'
                    f'{intersec_line_upper_line_right[0]} {intersec_line_upper_line_right[1]},'
                    f'{intersec_line_right_line_low[0]} {intersec_line_right_line_low[1]},'
                    f'{intersec_point[0]} {intersec_point[1]}))')

    # Create boundary points
    bound = polygon.buffer(30).envelope.boundary
    boundarypoints = [bound.interpolate(distance=d) for d in range(0, int(np.ceil(bound.length)), 1)]
    boundarycoords = np.array([[p.x, p.y] for p in boundarypoints])
    
    # Combine boundary and centroid coordinates
    all_coords = np.concatenate((boundarycoords, centroids))

    # Generate Voronoi tessellation
    vor = Voronoi(points=all_coords)
    vor.vertices = vor.vertices * scales + means
    lines = [shapely.geometry.LineString(vor.vertices[line]) 
             for line in vor.ridge_vertices if -1 not in line]

    # Create the original polygon
    polygon_old = shapely.wkt.loads(f'Polygon (({intersec_point_old[0]} {intersec_point_old[1]},'
                    f'{intersec_Line_up_line_upper_old[0]} {intersec_Line_up_line_upper_old[1]},'
                    f'{intersec_line_upper_line_right_old[0]} {intersec_line_upper_line_right_old[1]},'
                    f'{intersec_line_right_line_low_old[0]} {intersec_line_right_line_low_old[1]},'
                    f'{intersec_point_old[0]} {intersec_point_old[1]}))')

    # Generate Voronoi polygons
    polys = shapely.ops.polygonize(lines)
    voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys), crs="epsg:3006")
    polydf = gpd.GeoDataFrame(geometry=[polygon_old], crs="epsg:3006")
    result = gpd.overlay(df1=voronois, df2=polydf, how="intersection")

    # Create the Plotly figure
    fig = go.Figure()

    # Add the main polygon boundary
    x, y = polygon_old.exterior.xy
    fig.add_trace(go.Scatter(
        x=list(x), y=list(y),
        mode='lines',
        showlegend=False,
        line=dict(color='blue', width=2),
        name='Boundary'
    ))

    # dump = [0]*len(result.geometry)
    # dump[0]=result.geometry[1]
    # dump[1]=result.geometry[0]
    # dump[2]=result.geometry[2]
    # dump[3]=result.geometry[3]
    # dump[4]=result.geometry[4]
    # dump[5]=result.geometry[6]
    # dump[6]=result.geometry[5]

    # result.geometry = dump

    # Add Voronoi regions
    for i, poly in enumerate(result.geometry):
        x, y = poly.exterior.xy
        color_idx = i % len(color_palette)
        fig.add_trace(go.Scatter(x=list(x), y=list(y),fill='toself',mode='lines',line=dict(color='black', width=1),fillcolor=color_palette[color_idx],name=f'Region {i+1}'))

    # Добавляем прямую линию от intersec_point_old до intersec_Line_up_line_upper_old
    fig.add_trace(go.Scatter(
        x=[intersec_point_old[0], intersec_Line_up_line_upper_old[0]],
        y=[intersec_point_old[1], intersec_Line_up_line_upper_old[1]],
        mode='lines',
        showlegend=False,
        line=dict(color='red', width=1),
        name='Верхняя граница - прямая часть'
    ))
    
    # Рассчитываем направление верхней линии для продолжения
    direction1 = np.array([
        intersec_Line_up_line_upper_old[0] - intersec_point_old[0],
        intersec_Line_up_line_upper_old[1] - intersec_point_old[1]
    ])
    # Нормализуем вектор направления
    direction1 = direction1 / np.linalg.norm(direction1)
    
    # Определяем длину продолжения и количество точек для плавной кривой
    extension_length1 = 2.2e8
    num_points = 100
    
    # Создаем параметры для квадратичной функции
    # Начальная точка для квадратичной функции
    start_point = intersec_Line_up_line_upper_old
    
    # Создаем точки для квадратичной кривой
    curve_x = []
    curve_y = []
    
    for i in range(num_points):
        # Коэффициент t для параметризации
        # p = 20  # Экспонента, можно регулировать
        t = (i / (num_points - 1))
        # t = i / (num_points - 1)
        
        # Расстояние от начальной точки вдоль направления
        distance = t * extension_length1
        
        # Добавляем квадратичную компоненту с увеличением по мере удаления
        # Используем квадратичную функцию для отклонения от прямой
        # Чем дальше от начальной точки, тем больше отклонение
        quadratic_factor = 3e-7 * (distance ** 1.5 )  # Коэффициент можно настроить
        # tri_factor = 
        
        # Рассчитываем координаты точки на кривой
        # Меняем знаки для изменения направления выгиба кривой
        point_x = start_point[0] + direction1[0] * distance - direction1[1] * quadratic_factor
        point_y = start_point[1] + direction1[1] * distance + direction1[0] * quadratic_factor
        
        curve_x.append(point_x)
        curve_y.append(point_y)
    
    # Добавляем квадратичную часть кривой
    fig.add_trace(go.Scatter(
        x=curve_x,
        y=curve_y,
        mode='lines',
        line=dict(color='red', width=0.5, dash='dash'),
        showlegend=False,
        name='Верхняя граница - квадратичное продолжение'
    ))
    
    # Аналогично для второго направления: от intersec_point к intersec_line_right_line_low
    direction2 = np.array([
        intersec_line_right_line_low_old[0] - intersec_point_old[0],
        intersec_line_right_line_low_old[1] - intersec_point_old[1]
    ])
    direction2 = direction2 / np.linalg.norm(direction2)
    
    extension_length2 = 2e7
    extension_point2 = [
        intersec_point_old[0] + direction2[0] * extension_length2,
        intersec_point_old[1] + direction2[1] * extension_length2
    ]
    fig.add_trace(go.Scatter(
        x=[intersec_point_old[0], extension_point2[0]],
        y=[intersec_point_old[1], extension_point2[1]],
        mode='lines',
        line=dict(color='red', width=0.5, dash='dash'),
        showlegend=False,
        name='Нижняя граница'
    ))
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        height=800,
        title='Новая сегментация по приведенным к 2024 ценам за квадратный метр и общим ценам за квартиру',
        xaxis_title="Приведенная цена за квадратный метр",
        yaxis_title="Общая приведенная цена",
        xaxis=dict(
            # scaleanchor='y',
            # scaleratio=100,
            showgrid=True
        ),
        yaxis=dict(
            showgrid=True
        ),
        hovermode='closest'
    )
    return fig


def find_quantile_line(points, quantile, upper=True):
    """
    Находит угловой коэффициент прямой y = kx, проходящей через начало координат,
    которая отсекает заданную квантиль точек сверху или снизу.
    
    Parameters:
    points (np.ndarray): массив точек формы (n, 2), где n - количество точек
    quantile (float): квантиль в диапазоне [0, 1]
    upper (bool): если True, отсекает точки сверху, если False - снизу
    
    Returns:
    float: угловой коэффициент прямой
    """
    assert np.all(points >= 0), "Все точки должны быть в первой четверти"
    
    valid_points = points[points[:, 0] > 0]
    slopes = valid_points[:, 1] / valid_points[:, 0]
    
    # Для верхней линии используем 1-quantile, для нижней - quantile
    k = np.quantile(slopes, 1 - quantile if upper else quantile)
    
    return k

def normalize_line(a, b, c, scales, means):
    """
    Нормализует коэффициенты прямой ax + by + c = 0
    с учётом массивов scales и means.
    
    :param a: Коэффициент при x
    :param b: Коэффициент при y
    :param c: Свободный член
    :param scales: Массив [sx, sy] масштабов
    :param means: Массив [mx, my] средних значений
    :return: Новые коэффициенты (a', b', c')
    """
    sx, sy = scales
    mx, my = means

    a_new = a * sx
    b_new = b * sy
    c_new = c + a * mx + b * my
    
    return a_new, b_new, c_new


def intersection_point(a1, b1, c1, a2, b2, c2):
  """Находит точку пересечения двух прямых.

  Args:
    a1, b1, c1: Коэффициенты первого уравнения.
    a2, b2, c2: Коэффициенты второго уравнения.

  Returns:
    Кортеж (x, y) - координаты точки пересечения,
    или None, если прямые параллельны или совпадают.
  """

  # Проверка на параллельность
  if a1*b2 == a2*b1 and a1*c2 != a2*c1:
      return None  # Прямые параллельны

  # Проверка на совпадение
  if a1*b2 == a2*b1 and a1*c2 == a2*c1:
      return None  # Прямые совпадают

  # Вычисление координат точки пересечения
  x = (c1*b2 - c2*b1) / (a1*b2 - a2*b1)
  y = (a2*c1 - a1*c2) / (a1*b2 - a2*b1)
  
  return [x, y]



def make_new_clusterisation(msk_prep,house_ids,n_cl = 7):
  data = msk_prep[['total_price_discounted','contract_date','discounting_price','builder','house_id','area']].copy()

  quantile_value_sq = data['discounting_price'].quantile(0.99)
  quantile_value_total = data['total_price_discounted'].quantile(0.99)
  data = data[(data['discounting_price'] <= quantile_value_sq) & 
              (data['total_price_discounted'] <= quantile_value_total)].copy()

  data['contract_date'] = pd.to_datetime(data['contract_date'])
  date_ranges = pd.date_range(start='2000-01-01', end='2024-12-31', periods=2)
  data['time_segment'] = pd.cut(data['contract_date'], bins=date_ranges, labels=False, include_lowest=True)

  scaler = StandardScaler()
  data_cl = data[['total_price_discounted', 'discounting_price']].copy()
  data_cl_scaled = scaler.fit_transform(data_cl)

  kmeans = KMeans(n_clusters=n_cl, random_state=1,n_init=10)

  data['KMeans'] = -1

  for segment in range(1):
      segment_data = data[data['time_segment'] == segment]
      if not segment_data.empty:
          segment_scaled = scaler.transform(segment_data[['total_price_discounted', 'discounting_price']])
          kmeans.fit(segment_scaled)
          segment_labels = kmeans.predict(segment_scaled)
          data.loc[data['time_segment'] == segment, 'KMeans'] = segment_labels
          segment_data = segment_data.sort_values(by='KMeans')
          segment_data_scaled_ = scaler.transform(segment_data[['total_price_discounted', 'discounting_price']])
          segment_data['total_price_discounted_scaled'] = segment_data_scaled_[:,0]
          segment_data['discounting_price_scaled'] = segment_data_scaled_[:,1]


  means = scaler.mean_[::-1]
  scales = scaler.scale_[::-1]

  points = segment_data[['discounting_price','total_price_discounted']].to_numpy()

  quantile = 0.0001
  quantile2 = 0.00001

  k_upper = find_quantile_line(points, quantile, upper=True)
  k_lower = find_quantile_line(points, quantile2, upper=False)

  a,b,c = k_upper,-1,0
  Line_up = normalize_line(a, b, c, scales, means)
  a,b,c = k_lower,-1,0
  Line_low = normalize_line(a, b, c, scales, means)

  intersec_point =intersection_point(Line_low[0],Line_low[1],Line_low[2],Line_up[0],Line_up[1],Line_up[2])
  max_y = max(segment_data["total_price_discounted_scaled"])
  line_upper = [0,-1,max_y]
  max_x = max(segment_data["discounting_price_scaled"])
  line_right = [-1,0,max_x]


  intersec_Line_up_line_upper =intersection_point(line_upper[0],line_upper[1],line_upper[2],Line_up[0],Line_up[1],Line_up[2])
  intersec_line_upper_line_right =intersection_point(line_upper[0],line_upper[1],line_upper[2],line_right[0],line_right[1],line_right[2])
  intersec_line_right_line_low =intersection_point(line_right[0],line_right[1],line_right[2],Line_low[0],Line_low[1],Line_low[2])

  centroids = kmeans.cluster_centers_

  centroids_x = centroids[:,1].copy()
  centroids_y = centroids[:,0].copy()
  centroids[:,0] =centroids_x
  centroids[:,1] =centroids_y

  intersec_point[0]=-intersec_point[0]
  intersec_Line_up_line_upper[0]=-intersec_Line_up_line_upper[0]
  intersec_line_upper_line_right[0]=-intersec_line_upper_line_right[0]
  intersec_line_right_line_low[0]=-intersec_line_right_line_low[0]

  intersec_point_old = intersec_point*scales + means
  intersec_Line_up_line_upper_old = intersec_Line_up_line_upper*scales + means
  intersec_line_upper_line_right_old = intersec_line_upper_line_right*scales + means
  intersec_line_right_line_low_old =  intersec_line_right_line_low*scales + means 
  centroids_old = centroids*scales + means 


  fig = create_voronoi_base(intersec_point, intersec_Line_up_line_upper, 
                         intersec_line_upper_line_right, intersec_line_right_line_low,
                         intersec_point_old, intersec_Line_up_line_upper_old,
                         intersec_line_upper_line_right_old, intersec_line_right_line_low_old,
                         centroids, centroids_old, scales, means)
#   fig = add_points_to_voronoi(fig, data, house_ids, segment, date_ranges)


  return fig


if __name__ == "__main__":
    cities = ["msk", "ekb"] 

    cluster_mapping = {
        "msk": 7,
        "ekb": 3,
        "nsk": 3,
        "chb": 3
    }

    for city in cities:
        input_dir = os.path.join("data", "regions", city, "market_deals")
        output_dir = os.path.join("data", "regions", city, "segmentation")
        os.makedirs(output_dir, exist_ok=True)
        apartment_path = os.path.join(input_dir, f"{city}_prep.feather")
        # house_path = os.path.join(input_dir, f"{city}_apartment.feather")
        try:
            apartment_data = pd.read_feather(apartment_path)
            # house_data = pd.read_feather(house_path)
        except Exception as e:
            print(f"Ошибка загрузки данных для города {city}: {e}")
            continue
        

        ir = get_interest_rate(apartment_data, is_ml=False)
        apartment_data["discounting_price"] = discounting(apartment_data, ir)
        apartment_data["discounting_price"] = apartment_data["discounting_price"] * ir.iloc[-1]
        apartment_data['total_price_discounted'] = apartment_data['area'] * apartment_data['discounting_price']

        output_path = os.path.join(output_dir,f'{city}_fig_clusterisation.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(make_new_clusterisation(apartment_data,[],n_cl = cluster_mapping[city]), f)
        print(f"Кластеризация сохранена для города {city} в: {output_path}")