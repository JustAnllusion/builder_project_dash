import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.utils import compute_avg_depletion_curve, load_depletion_curves


def build_histogram(
    chart_data, hist_column, color_domain, color_range, normalize, height=400
):
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
        color = (
            "#FF0000"
            if group == "Глобальный"
            else color_range[color_domain.index(group)]
        )
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=counts,
                name=group,
                marker_color=color,
                opacity=0.5,
                marker_line_width=1,
            )
        )
    fig.update_layout(
        barmode="overlay",
        height=height,
        xaxis_title=hist_column,
        yaxis_title="Доля" if normalize else "Количество",
    )
    return fig


def build_scatter(chart_data, x_col, y_col, color_domain, color_range, height=400):
    color_map = {
        group: (
            "#FF0000"
            if group == "Глобальный"
            else color_range[color_domain.index(group)]
        )
        for group in color_domain
    }
    fig = px.scatter(
        chart_data, x=x_col, y=y_col, color="group", color_discrete_map=color_map
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(height=height, xaxis_title=x_col, yaxis_title=y_col)
    return fig


def build_depletion_chart(
    depletion_curves_file,
    selected_groups,
    global_filtered_data,
    group_configs,
    show_individual=False,
    height=400,
):
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
            indiv = depletion_curves[
                depletion_curves["house_id"].isin(house_ids)
            ].copy()
            if not indiv.empty:
                indiv["group"] = g
                individual_data = pd.concat([individual_data, indiv], ignore_index=True)
    if combined_data.empty:
        return None
    fig = go.Figure()
    color_map = {
        group: (
            "#FF0000" if group == "Глобальный" else group_configs[group]["vis"]["color"]
        )
        for group in selected_groups
    }
    for group in combined_data["group"].unique():
        df = combined_data[combined_data["group"] == group]
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["pct"],
                mode="lines",
                line=dict(width=3, shape="hv", color=color_map.get(group, "#0000FF")),
                name=group,
            )
        )
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
                fig.add_trace(
                    go.Scatter(
                        x=dff["time"],
                        y=dff["pct"],
                        mode="lines",
                        line=dict(
                            width=1, shape="hv", color=color_map.get(group, "#0000FF")
                        ),
                        opacity=0.3,
                        showlegend=False,
                    )
                )
    fig.update_layout(
        height=height, xaxis_title="Время (дни)", yaxis_title="Остаток продаж (%)"
    )
    return fig
