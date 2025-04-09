import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.utils import find_segment_for_elasticity, fit_hyperbolic_alpha

from utils.utils import (
    compute_avg_depletion_curve,
    find_segment_for_elasticity,
    fit_hyperbolic_alpha,
    load_depletion_curves,
)

def build_histogram(chart_data, hist_column, color_domain, color_range, normalize, height=400):
    bins = np.histogram_bin_edges(chart_data[hist_column].dropna(), bins=30)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    fig = go.Figure()
    for group in color_domain:
        df = chart_data[chart_data["group"] == group]
        counts, _ = np.histogram(df[hist_column].dropna(), bins=bins)
        if normalize:
            total = counts.sum()
            if total > 0:
                counts = counts / total
        color = "#FF0000" if group == "Глобальный" else color_range[color_domain.index(group)]
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=counts,
            name=group,
            marker_color=color,
            opacity=0.5,
            marker_line_width=1,
        ))
    fig.update_layout(
        barmode="overlay",
        height=height,
        xaxis_title=hist_column,
        yaxis_title="Доля" if normalize else "Количество",
    )
    return fig

def build_scatter(chart_data, x_col, y_col, color_domain, color_range, height=400):
    color_map = {group: ("#FF0000" if group == "Глобальный" else color_range[color_domain.index(group)])
                 for group in color_domain}
    fig = px.scatter(chart_data, x=x_col, y=y_col, color="group", color_discrete_map=color_map)
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(height=height, xaxis_title=x_col, yaxis_title=y_col)
    return fig

def build_depletion_chart(depletion_curves_file, selected_groups, global_filtered_data, group_configs, show_individual=False, height=400):
    depletion_curves = load_depletion_curves(depletion_curves_file)
    if depletion_curves.empty:
        return None
    combined_data = pd.DataFrame()
    individual_data = pd.DataFrame()
    for g in selected_groups:
        if g == "Глобальный":
            house_ids = global_filtered_data["house_id"].unique()
        else:
            house_ids = group_configs[g]["filtered_data"]["house_id"].unique()
        avg_df = compute_avg_depletion_curve(depletion_curves, house_ids)
        if not avg_df.empty:
            avg_df["group"] = g
            combined_data = pd.concat([combined_data, avg_df], ignore_index=True)
        if show_individual:
            indiv = depletion_curves[depletion_curves["house_id"].isin(house_ids)].copy()
            if not indiv.empty:
                indiv["group"] = g
                individual_data = pd.concat([individual_data, indiv], ignore_index=True)
    if combined_data.empty:
        return None
    fig = go.Figure()
    color_map = {group: ("#FF0000" if group == "Глобальный" else group_configs[group]["vis"]["color"])
                 for group in selected_groups}
    for group in combined_data["group"].unique():
        df = combined_data[combined_data["group"] == group]
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["pct"],
            mode="lines",
            line=dict(width=3, shape="hv", color=color_map.get(group, "#0000FF")),
            name=group,
        ))
    if show_individual and not individual_data.empty:
        max_individual = 100
        for group in individual_data["group"].unique():
            df = individual_data[individual_data["group"] == group]
            unique_house_ids = df["house_id"].unique()
            if len(unique_house_ids) > max_individual:
                unique_house_ids = np.random.choice(unique_house_ids, max_individual, replace=False)
            for hid in unique_house_ids:
                dff = df[df["house_id"] == hid]
                fig.add_trace(go.Scatter(
                    x=dff["time"],
                    y=dff["pct"],
                    mode="lines",
                    line=dict(width=1, shape="hv", color=color_map.get(group, "#0000FF")),
                    opacity=0.3,
                    showlegend=False,
                ))
    fig.update_layout(
        height=height, xaxis_title="Время (дни)", yaxis_title="Остаток продаж (%)"
    )
    return fig

# def build_elasticity_chart(selected_groups, global_filtered_data, group_configs, apartment_data, area_min, area_max, split_parameter, rooms_list):
#     if not rooms_list:
#         return None
#     categories = []
#     for r in rooms_list:
#         if r is None:
#             categories.append(("Все сделки", None))
#         elif r == 0:
#             categories.append(("Студия", 0))
#         elif r == 1:
#             categories.append(("1 комната", 1))
#         elif r == 2:
#             categories.append(("2 комнаты", 2))
#         elif r == 3:
#             categories.append(("3 комнаты", 3))
#         else:
#             categories.append((f"{r} комнат", r))
#     fig = make_subplots(
#         rows=len(categories),
#         cols=1,
#         shared_xaxes=False,
#         vertical_spacing=0.08,
#         subplot_titles=[cat[0] for cat in categories],
#         specs=[[{"secondary_y": True}] for _ in categories],
#     )
#     color_map = {}
#     for g in selected_groups:
#         color_map[g] = "#FF0000" if g == "Глобальный" else group_configs[g]["vis"]["color"]
#     for idx, (cat_name, rooms_val) in enumerate(categories, start=1):
#         row_i = idx
#         for group in selected_groups:
#             if group == "Глобальный":
#                 house_ids = global_filtered_data["house_id"].unique()
#             else:
#                 house_ids = group_configs[group]["filtered_data"]["house_id"].unique()
#             df_g = apartment_data[apartment_data["house_id"].isin(house_ids)].copy()
#             if rooms_val is not None:
#                 df_g = df_g[df_g["rooms_number"] == rooms_val]
#             df_g = df_g.dropna(subset=["area", "price"])
#             df_g = df_g[(df_g["area"] >= area_min) & (df_g["area"] <= area_max)]
#             if df_g.empty:
#                 continue
#             df_g["area_seg"] = df_g["area"].apply(lambda x: find_segment_for_elasticity(x, area_min, area_max, split_parameter))
#             list_of_curves = []
#             for hid, sub_df in df_g.groupby("house_id"):
#                 seg_mean = sub_df.groupby("area_seg")["price"].mean().sort_index()
#                 if seg_mean.empty:
#                     continue
#                 first_val = seg_mean.iloc[0]
#                 norm_curve = seg_mean / first_val if first_val != 0 else seg_mean
#                 list_of_curves.append(norm_curve)
#             if not list_of_curves:
#                 continue
#             all_idx = sorted(set().union(*(c.index for c in list_of_curves)))
#             aligned = [c.reindex(all_idx, method="ffill") for c in list_of_curves]
#             avg_curve = pd.concat(aligned, axis=1).mean(axis=1)
#             alpha = fit_hyperbolic_alpha(avg_curve)
#             hyperbolic = (avg_curve.index[0] ** alpha) / (avg_curve.index ** alpha)
#             fig.add_trace(go.Scatter(
#                 x=avg_curve.index,
#                 y=avg_curve.values,
#                 mode="lines+markers",
#                 name=f"{group} ({cat_name})",
#                 line=dict(color=color_map[group]),
#             ), row=row_i, col=1, secondary_y=False)
#             fig.add_trace(go.Scatter(
#                 x=avg_curve.index,
#                 y=hyperbolic,
#                 mode="lines",
#                 name=f"Гипербола {group} ({cat_name})",
#                 line=dict(color=color_map[group], dash="dash"),
#             ), row=row_i, col=1, secondary_y=False)
#             deals_count = df_g.groupby("area_seg").size()
#             x_vals = deals_count.index + split_parameter / 2.0
#             fig.add_trace(go.Bar(
#                 x=x_vals,
#                 y=deals_count.values,
#                 name=f"Сделок {group} ({cat_name})",
#                 marker_color=color_map[group],
#                 opacity=0.4,
#             ), row=row_i, col=1, secondary_y=True)
#     fig.update_layout(
#         height=220 * len(categories),
#         title=f"Кривая эластичности (шаг={split_parameter} кв.м)",
#         showlegend=True,
#     )
#     for i in range(len(categories)):
#         fig.update_yaxes(title_text="Норм. цена", row=i + 1, col=1, secondary_y=False)
#         fig.update_yaxes(title_text="Кол-во сделок", row=i + 1, col=1, secondary_y=True)
#     return fig
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.utils import fit_hyperbolic_alpha

def build_elasticity_chart(selected_groups, global_filtered_data, group_configs, split_parameter):
    """
    Строит единый график кривой эластичности с использованием предвычисленных данных,
    сохранённых в файле:
      data/regions/msk/market_deals/cache/elasticity_curves_precomputed.feather

    Фильтрует данные по выбранному шагу сегментации (split_parameter).
    На графике для каждой группы отображаются усреднённые кривые:
      - Нормированная кривая (линия с маркерами) - на основной оси Y
      - Гипербическая оценка (пунктир) - на основной оси Y
      - Столбцы с числом сделок - на дополнительной оси Y
    """

    try:
        precomputed = pd.read_feather(
            "data/regions/msk/market_deals/cache/elasticity_curves_precomputed.feather"
        )
    except Exception as e:
        print(f"Ошибка загрузки предвычисленных данных: {e}")
        return None

    # Фильтруем по выбранному шагу сегментации
    precomputed = precomputed[precomputed["split_parameter"] == split_parameter]
    if precomputed.empty:
        return None

    # Создаём фигуру с одной "ячейкой", но двумя осями Y
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    color_map = {}
    for g in selected_groups:
        # Цвет для каждой группы
        color_map[g] = "#FF0000" if g == "Глобальный" else group_configs[g]["vis"]["color"]

    for group in selected_groups:
        if group == "Глобальный":
            house_ids = global_filtered_data["house_id"].unique()
        else:
            house_ids = group_configs[group]["filtered_data"]["house_id"].unique()

        df_group = precomputed[precomputed["house_id"].isin(house_ids)]
        if df_group.empty:
            continue

        # Группируем по сегменту (area_seg), считаем количество сделок
        try:
            deals_count = df_group.groupby("area_seg").size()
        except Exception as e:
            print(f"Ошибка группировки по area_seg: {e}")
            continue

        # Собираем нормированные кривые для каждого дома и усредняем
        list_of_curves = []
        for hid, sub_df in df_group.groupby("house_id"):
            s = sub_df.set_index("area_seg")["norm_curve"]
            list_of_curves.append(s)

        if not list_of_curves:
            continue

        all_idx = sorted(set().union(*(c.index for c in list_of_curves)))
        aligned = [c.reindex(all_idx, method="ffill") for c in list_of_curves]
        avg_curve = pd.concat(aligned, axis=1).mean(axis=1)

        # Вычисляем гипербическую оценку
        alpha = fit_hyperbolic_alpha(avg_curve)
        hyper_eval = (all_idx[0] ** alpha) / (np.array(all_idx) ** alpha)

        # Линия нормированной кривой (primary y-axis)
        fig.add_trace(
            go.Scatter(
                x=all_idx,
                y=avg_curve.values,
                mode="lines+markers",
                name=f"{group} норм.",
                line=dict(color=color_map[group]),
            ),
            secondary_y=False
        )

        # Линия гипербической оценки (primary y-axis)
        fig.add_trace(
            go.Scatter(
                x=all_idx,
                y=hyper_eval,
                mode="lines",
                name=f"{group} гиперб. оценка",
                line=dict(color=color_map[group], dash="dash"),
            ),
            secondary_y=False
        )

        # Столбчатая диаграмма (secondary y-axis)
        x_vals = [x + split_parameter / 2.0 for x in deals_count.index]
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=deals_count.values,
                name=f"{group} сделки",
                marker_color=color_map[group],
                opacity=0.4,
            ),
            secondary_y=True
        )

    # Настройки макета
    fig.update_layout(
        height=500,
        title=f"Кривая эластичности (шаг сегментации = {split_parameter} кв.м)",
        showlegend=True,
        barmode='overlay'  # bars and lines overlay on the same x-axis
    )

    # Подписи осей
    fig.update_xaxes(title_text="Площадь (сегмент)")
    # Левая ось - нормированная цена
    fig.update_yaxes(title_text="Нормированная цена", secondary_y=False)
    # Правая ось - число сделок
    fig.update_yaxes(title_text="Число сделок", secondary_y=True)

    return fig
