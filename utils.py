import colorsys
import io
import json

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return output.getvalue()


def convert_df_to_parquet(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    df.to_parquet(output, index=False)
    return output.getvalue()


def download_ui(df: pd.DataFrame, default_file_name: str):
    col_format, col_button = st.columns([1, 1])
    with col_format:
        format_option = st.selectbox(
            "Формат:",
            options=["CSV", "Excel", "Parquet"],
            key=f"download_format_{default_file_name}",
            label_visibility="collapsed",
        )
    if format_option == "CSV":
        file_data = convert_df_to_csv(df)
        file_name = f"{default_file_name}.csv"
        mime = "text/csv"
    elif format_option == "Excel":
        file_data = convert_df_to_excel(df)
        file_name = f"{default_file_name}.xlsx"
        mime = "application/vnd.ms-excel"
    else:
        file_data = convert_df_to_parquet(df)
        file_name = f"{default_file_name}.parquet"
        mime = "application/octet-stream"
    with col_button:
        st.download_button(
            "Скачать",
            file_data,
            file_name=file_name,
            mime=mime,
            key=f"download_button_{default_file_name}",
        )


@st.cache_data(show_spinner=False)
def load_depletion_curves(file_path="depletion_curves.feather"):
    try:
        return pd.read_feather(file_path)
    except Exception as e:
        st.error(f"Ошибка загрузки кривых выбытия: {e}")
        return pd.DataFrame()


def hex_to_rgba(hex_color, alpha=255):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return [r, g, b, alpha]


def generate_shades(hex_color, n):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    shades = []
    for i in range(n):
        new_l = l * (0.8 + 0.4 * i / (n - 1)) if n > 1 else l
        new_l = max(0, min(new_l, 1))
        new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, s)
        shade = "#{:02x}{:02x}{:02x}".format(
            int(new_r * 255), int(new_g * 255), int(new_b * 255)
        )
        shades.append(shade)
    return shades


@st.cache_data(show_spinner=False)
def load_data(file_path):
    try:
        return pd.read_feather(file_path)
    except Exception as e:
        st.error(f"Ошибка загрузки данных из {file_path}: {e}")
        return pd.DataFrame()


def get_top_categories(df, column, top_n=5):
    if column in df.columns:
        return df[column].value_counts().head(top_n).index.tolist()
    return []


def set_altair_theme():
    alt.themes.enable("opaque")
    return {
        "config": {
            "title": {
                "fontSize": 18,
                "font": "Roboto",
                "anchor": "start",
                "color": "#333",
            },
            "axis": {
                "labelFontSize": 12,
                "titleFontSize": 14,
                "labelColor": "#333",
                "titleColor": "#333",
            },
            "legend": {"labelFontSize": 12, "titleFontSize": 14},
            "view": {"continuousWidth": 400, "continuousHeight": 300},
        }
    }


alt.themes.register("custom_theme", set_altair_theme)
alt.themes.enable("custom_theme")


def numeric_filter_widget(col, data, key_prefix):
    if data.empty:
        return (0.0, 0.0)
    min_val = float(data.min())
    max_val = float(data.max())
    if min_val == max_val:
        st.write(f"Столбец «{col}» имеет константное значение: {min_val}.")
        return (min_val, max_val)
    use_quantile = st.checkbox(
        f"Фильтровать по квантилям для «{col}»", key=f"{key_prefix}_quantile_{col}"
    )
    if use_quantile:
        quantile_range = st.slider(
            f"Выберите квантильный диапазон для «{col}»",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
            key=f"{key_prefix}_quantile_range_{col}",
        )
        lower_q, upper_q = quantile_range
        lower_bound = float(data.quantile(lower_q))
        upper_bound = float(data.quantile(upper_q))
        st.info(
            f"Фильтрация: [{lower_q:.2f} → {upper_q:.2f}] (соответствует [{lower_bound:.2f} → {upper_bound:.2f}])"
        )
        return (lower_bound, upper_bound)
    else:
        slider_val = st.slider(
            f"Фильтр для «{col}»",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            key=f"{key_prefix}_{col}",
        )
        return slider_val


@st.cache_data(show_spinner=False)
def compute_avg_depletion_curve(
    depletion_curves: pd.DataFrame, house_ids: np.array
) -> pd.DataFrame:
    df = depletion_curves[depletion_curves["house_id"].isin(house_ids)]
    if df.empty:
        return pd.DataFrame()
    max_time = int(df["time"].max())
    time_index = np.arange(0, max_time + 1)
    pivot = df.pivot(index="time", columns="house_id", values="pct")
    pivot = pivot.reindex(time_index).ffill().fillna(100)
    avg_df = pd.DataFrame({"time": time_index, "pct": pivot.mean(axis=1)})
    return avg_df


def compute_smart_group_name(group):
    if not group.get("column_filters"):
        return group.get("group_name", "Группа")
    parts = []
    for col, filt in group.get("column_filters", {}).items():
        if isinstance(filt, tuple):
            parts.append(f"{col}:[{filt[0]}, {filt[1]}]")
        elif isinstance(filt, list) and filt:
            parts.append(f"{col}:{'&'.join(map(str, filt))}")
    if parts:
        return " & ".join(parts)
    return group.get("group_name", "Группа")


def safe_multiselect(label, options, default, key):
    safe_default = [x for x in default if x in options]
    if not safe_default:
        safe_default = options
    return st.multiselect(label, options=options, default=safe_default, key=key)
