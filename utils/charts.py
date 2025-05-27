import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils.utils import find_segment_for_elasticity

from utils.utils import (
    compute_avg_depletion_curve,
    find_segment_for_elasticity,
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
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')), opacity=0.6)
    fig.update_layout(height=height, xaxis_title=x_col, yaxis_title=y_col)
    return fig


# def build_depletion_chart(depletion_curves_file, selected_groups, global_filtered_data, group_configs, show_individual=False, height=400):
#     depletion_curves = load_depletion_curves(depletion_curves_file)
#     if depletion_curves.empty:
#         return None
#     combined_data = pd.DataFrame()
#     individual_data = pd.DataFrame()
#     for g in selected_groups:
#         if g == "Глобальный":
#             house_ids = global_filtered_data["house_id"].unique()
#         else:
#             house_ids = group_configs[g]["filtered_data"]["house_id"].unique()
#         avg_df = compute_avg_depletion_curve(depletion_curves, house_ids)
#         if not avg_df.empty:
#             avg_df["group"] = g
#             combined_data = pd.concat([combined_data, avg_df], ignore_index=True)
#         if show_individual:
#             indiv = depletion_curves[depletion_curves["house_id"].isin(house_ids)].copy()
#             if not indiv.empty:
#                 indiv["group"] = g
#                 individual_data = pd.concat([individual_data, indiv], ignore_index=True)
#     if combined_data.empty:
#         return None
#     fig = go.Figure()
#     color_map = {group: ("#FF0000" if group == "Глобальный" else group_configs[group]["vis"]["color"])
#                  for group in selected_groups}
#     for group in combined_data["group"].unique():
#         df = combined_data[combined_data["group"] == group]
#         fig.add_trace(go.Scatter(
#             x=df["time"],
#             y=df["pct"],
#             mode="lines",
#             line=dict(width=3, shape="hv", color=color_map.get(group, "#0000FF")),
#             name=group,
#         ))
#     if show_individual and not individual_data.empty:
#         max_individual = 100
#         for group in individual_data["group"].unique():
#             df = individual_data[individual_data["group"] == group]
#             unique_house_ids = df["house_id"].unique()
#             if len(unique_house_ids) > max_individual:
#                 unique_house_ids = np.random.choice(unique_house_ids, max_individual, replace=False)
#             for hid in unique_house_ids:
#                 dff = df[df["house_id"] == hid]
#                 fig.add_trace(go.Scatter(
#                     x=dff["time"],
#                     y=dff["pct"],
#                     mode="lines",
#                     line=dict(width=1, shape="hv", color=color_map.get(group, "#0000FF")),
#                     opacity=0.3,
#                     showlegend=False,
#                 ))
#     fig.update_layout(
#         height=height, xaxis_title="Время (дни)", yaxis_title="Остаток продаж (%)"
#     )
#     return fig
def build_depletion_chart(
        depletion_curves_file,
        selected_groups,
        global_filtered_data,
        group_configs,
        show_individual: bool = False,
        height: int = 400
):
    from utils.utils import load_depletion_curves, compute_avg_depletion_curve
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

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
    color_map = {
        group: (
            "#FF0000" if group == "Глобальный"
            else group_configs[group]["vis"]["color"]
        )
        for group in selected_groups
    }

    # Основные (средние) кривые
    for group in combined_data["group"].unique():
        df = combined_data[combined_data["group"] == group]
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["pct"],
            mode="lines",
            line=dict(width=3, shape="hv", color=color_map.get(group, "#0000FF")),
            name=group,
            legendgroup=group,
            showlegend=True,
        ))

    # Индивидуальные кривые
    if show_individual and not individual_data.empty:
        max_individual = 100
        for group in individual_data["group"].unique():
            df = individual_data[individual_data["group"] == group]
            unique_house_ids = df["house_id"].unique()
            if len(unique_house_ids) > max_individual:
                unique_house_ids = np.random.choice(
                    unique_house_ids, max_individual, replace=False
                )
            for hid in unique_house_ids:
                dff = df[df["house_id"] == hid]
                fig.add_trace(go.Scatter(
                    x=dff["time"],
                    y=dff["pct"],
                    mode="lines",
                    name=group,
                    legendgroup=group,
                    showlegend=False,
                    line=dict(width=1, shape="hv", color=color_map.get(group, "#0000FF")),
                    opacity=0.3,
                ))

    fig.update_layout(
        height=height,
        xaxis_title="Время (дни)",
        yaxis_title="Остаток продаж (%)",
    )

    return fig


def build_elasticity_chart(selected_groups, global_filtered_data, group_configs, split_parameter, min_seg=None, max_seg=None):
    city_key = st.session_state.get("city_key", "msk")
    precomputed_path = f"data/regions/{city_key}/market_deals/cache/elasticity_curves.feather"
    try:
        precomputed = pd.read_feather(precomputed_path)
    except Exception as e:
        print(f"Ошибка загрузки предвычисленных данных: {e}")
        return None

    precomputed = precomputed[precomputed["split_parameter"] == split_parameter]
    # Фильтрация по диапазону сегментов area_seg
    if min_seg is not None and max_seg is not None:
        precomputed = precomputed[
            (precomputed["area_seg"] >= min_seg) &
            (precomputed["area_seg"] <= max_seg)
        ]
    if precomputed.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    color_map = {}
    for g in selected_groups:
        color_map[g] = "#FF0000" if g == "Глобальный" else group_configs[g]["vis"]["color"]

    for group in selected_groups:
        if group == "Глобальный":
            house_ids = global_filtered_data["house_id"].unique()
        else:
            house_ids = group_configs[group]["filtered_data"]["house_id"].unique()

        df_group = precomputed[precomputed["house_id"].isin(house_ids)]
        if df_group.empty:
            continue

        try:
            deals_count = df_group.groupby("area_seg").size()
        except Exception as e:
            print(f"Ошибка группировки по area_seg: {e}")
            continue

        list_of_curves = []
        for hid, sub_df in df_group.groupby("house_id"):
            s = sub_df.set_index("area_seg")["norm_curve"]
            list_of_curves.append(s)

        if not list_of_curves:
            continue

        all_idx = sorted(set().union(*(c.index for c in list_of_curves)))
        aligned = [c.reindex(all_idx, method="ffill") for c in list_of_curves]
        avg_curve = pd.concat(aligned, axis=1).mean(axis=1)

        list_of_hyper = [
            sub_df.set_index("area_seg")["hyper_curve"]
            for _, sub_df in df_group.groupby("house_id")
        ]
        avg_hyper = pd.concat(
            [s.reindex(all_idx, method="ffill") for s in list_of_hyper],
            axis=1
        ).mean(axis=1)
        if min_seg not in avg_curve.index:
            min_seg_eff = all_idx[0]
        else:
            min_seg_eff = min_seg
        base_norm = avg_curve.loc[min_seg_eff]
        base_hyper = avg_hyper.loc[min_seg_eff]
        new_norm = avg_curve / base_norm
        new_hyper = avg_hyper / base_hyper

        fig.add_trace(
            go.Scatter(
                x=all_idx,
                y=new_norm.values,
                mode="lines+markers",
                name=f"{group} норм.",
                line=dict(color=color_map[group]),
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=all_idx,
                y=new_hyper.values,
                mode="lines",
                name=f"{group} гиперб. оценка",
                line=dict(color=color_map[group], dash="dash"),
            ),
            secondary_y=False
        )

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

    fig.update_layout(
        height=500,
        title=f"Кривая эластичности (шаг сегментации = {split_parameter} кв.м)",
        showlegend=True,
        barmode='overlay'
    )

    fig.update_xaxes(title_text="Площадь (сегмент)")
    fig.update_yaxes(title_text="Нормированная цена", secondary_y=False)
    fig.update_yaxes(title_text="Число сделок", secondary_y=True)

    return fig
