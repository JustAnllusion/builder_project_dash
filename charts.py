import altair as alt
import numpy as np
import pandas as pd
from utils import compute_avg_depletion_curve, load_depletion_curves


def build_histogram(
    chart_data, hist_column, color_domain, color_range, normalize, height=400
):
    if normalize:
        chart = (
            alt.Chart(chart_data)
            .transform_bin("bin", field=hist_column, bin=alt.Bin(maxbins=30))
            .transform_aggregate(count="count()", groupby=["bin", "group"])
            .transform_window(total="sum(count)", groupby=["group"])
            .transform_calculate(fraction="datum.count / datum.total")
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("bin:Q", title=hist_column),
                y=alt.Y("fraction:Q", title="Доля", axis=alt.Axis(format=".0%")),
                color=alt.Color(
                    "group:N",
                    scale=alt.Scale(domain=color_domain, range=color_range),
                    legend=alt.Legend(title="Группа"),
                ),
            )
            .properties(height=height)
        )
    else:
        chart = (
            alt.Chart(chart_data)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X(f"{hist_column}:Q", bin=alt.Bin(maxbins=30), title=hist_column),
                y=alt.Y("count()", title="Количество"),
                color=alt.Color(
                    "group:N",
                    scale=alt.Scale(domain=color_domain, range=color_range),
                    legend=alt.Legend(title="Группа"),
                ),
            )
            .properties(height=height)
        )
    return chart


def build_scatter(chart_data, x_col, y_col, color_domain, color_range, height=400):
    chart = (
        alt.Chart(chart_data)
        .mark_circle(size=60)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_col),
            y=alt.Y(f"{y_col}:Q", title=y_col),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title="Группа"),
            ),
            tooltip=list(chart_data.columns),
        )
        .interactive()
        .properties(height=height)
    )
    return chart


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
        house_ids = (
            global_filtered_data["house_id"].unique()
            if g == "Глобальный"
            else group_configs[g]["filtered_data"]["house_id"].unique()
        )
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
    base_chart = (
        alt.Chart(combined_data)
        .mark_line(strokeWidth=3, interpolate="step-after")
        .encode(
            x=alt.X("time:Q", title="Время (дни)"),
            y=alt.Y("pct:Q", title="Средний остаток продаж (%)"),
            color=alt.Color("group:N", legend=alt.Legend(title="Группа")),
        )
        .properties(height=height)
    )
    if show_individual and not individual_data.empty:
        indiv_chart = (
            alt.Chart(individual_data)
            .mark_line(interpolate="step-after", opacity=0.6)
            .encode(
                x=alt.X("time:Q", title="Время (дни)"),
                y=alt.Y("pct:Q", title="Остаток продаж (%)"),
                color=alt.Color("group:N", legend=None),
                detail="house_id:N",
            )
            .properties(height=height)
        )
        final_chart = (
            alt.layer(base_chart, indiv_chart)
            .resolve_scale(color="independent")
            .properties(height=height)
        )
    else:
        final_chart = base_chart
    return final_chart
