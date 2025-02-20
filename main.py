import json

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from utils import (
    compute_avg_depletion_curve,
    compute_smart_group_name,
    convert_df_to_csv,
    convert_df_to_excel,
    convert_df_to_parquet,
    download_ui,
    get_top_categories,
    hex_to_rgba,
    load_data,
    load_depletion_curves,
    numeric_filter_widget,
    safe_multiselect,
)

st.set_page_config(
    page_title="Анализ данных Москвы", layout="wide", initial_sidebar_state="expanded"
)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
:root {--primary-color: #FF6F61; --secondary-color: #333; --accent-color: #0077b6; --bg-color: #f7f7f7; --card-bg: #ffffff;}
body, .stText, .stButton, .stMarkdown {font-family: 'Roboto', sans-serif; color: var(--secondary-color);}
.main-title {font-size: 42px !important; text-align: center; color: var(--secondary-color); margin-bottom: 20px; border-bottom: 3px solid var(--primary-color); padding-bottom: 10px; transition: color 0.3s ease;}
.main-title:hover {color: var(--primary-color);}
.sidebar-header {font-size: 20px; font-weight: 600; margin-bottom: 10px; color: var(--secondary-color);}
.sidebar-divider {border-top: 2px solid var(--primary-color); margin: 15px 0;}
.group-card {background: var(--card-bg); border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-top: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
.vega-tooltip {background-color: var(--secondary-color) !important; color: #fff !important; border-radius: 5px; padding: 8px; font-size: 14px;}
.download-panel {display: flex; justify-content: flex-end; align-items: center; margin-top: 5px;}
button[data-baseweb="button"] {height: 40px !important; min-height: 40px !important;}
div[data-testid="stFileUploader"] button {height: 40px !important; min-height: 40px !important;}
.section-header {font-size: 28px; text-align: left; font-weight: normal; color: var(--secondary-color); margin-bottom: 20px; border-bottom: 2px solid var(--primary-color); padding-bottom: 10px;}
</style>
""",
    unsafe_allow_html=True,
)

apartment_data = load_data("msk_prep.feather")
house_data = load_data("msk_apartment.feather")

st.sidebar.markdown(
    "<div class='sidebar-header'>Фильтрация</div>", unsafe_allow_html=True
)
with st.sidebar.expander("Глобальная фильтрация", expanded=True):
    st.write("Выберите признаки для фильтрации по всему набору данных:")
    selected_filter_columns = st.multiselect(
        "Признаки для фильтрации", options=list(house_data.columns), default=[]
    )
    global_filters = {}
    for col in selected_filter_columns:
        if pd.api.types.is_numeric_dtype(house_data[col]):
            col_data = house_data[col].dropna()
            global_filters[col] = numeric_filter_widget(
                col, col_data, key_prefix="global_filter"
            )
        else:
            all_categories = sorted(house_data[col].dropna().unique())
            popular_categories = get_top_categories(house_data, col, top_n=5)
            st.write(
                f"Популярные варианты для «{col}»: {', '.join(popular_categories)}"
            )
            selected_options = st.multiselect(
                f"Выберите значения для «{col}»",
                options=all_categories,
                default=[],
                key=f"global_filter_{col}",
            )
            global_filters[col] = selected_options

active_global_filters = {col: val for col, val in global_filters.items()}
global_filtered_data = house_data.copy()
for col, filter_val in active_global_filters.items():
    if isinstance(filter_val, tuple):
        global_filtered_data = global_filtered_data[
            (global_filtered_data[col] >= filter_val[0])
            & (global_filtered_data[col] <= filter_val[1])
        ]
    else:
        if filter_val:
            global_filtered_data = global_filtered_data[
                global_filtered_data[col].isin(filter_val)
            ]

st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-header'>Группы</div>", unsafe_allow_html=True)
if "dynamic_groups" not in st.session_state:
    st.session_state.dynamic_groups = []
if "processed_json_files" not in st.session_state:
    st.session_state.processed_json_files = []
if st.sidebar.button("Создать новую группу"):
    new_group = {
        "group_name": f"Группа {len(st.session_state.dynamic_groups) + 1}",
        "group_color": "#0000FF",
        "selected_filter_columns": [],
        "column_filters": {},
        "filtered_data": house_data.copy(),
        "is_static": False,
        "base_data": house_data.copy(),
    }
    st.session_state.dynamic_groups.append(new_group)
with st.sidebar.expander("Загрузить группы из JSON", expanded=False):
    st.info(
        'Поддерживаемые форматы JSON:\n- Формат 1: Список ID (например, ["id1", "id2"]).\n- Формат 2: Список списков ID (например, [["id1", "id2"], ["id3", "id4"]]).\n- Формат 3: Словарь, где ключи – имена групп, а значения – списки ID.'
    )
    json_file = st.file_uploader("Выберите JSON файл", type=["json"], key="json_group")
    if (
        json_file is not None
        and json_file.name not in st.session_state.processed_json_files
    ):
        try:
            json_data = json.load(json_file)
            groups_from_json = {}
            if isinstance(json_data, dict):
                groups_from_json = json_data
            elif isinstance(json_data, list):
                if all(isinstance(elem, list) for elem in json_data):
                    for idx, id_list in enumerate(json_data, start=1):
                        group_name = (
                            f"Группа {len(st.session_state.dynamic_groups) + 1}"
                        )
                        groups_from_json[group_name] = id_list
                else:
                    group_name = f"Группа {len(st.session_state.dynamic_groups) + 1}"
                    groups_from_json[group_name] = json_data
            else:
                st.error("Неверный формат JSON. Ожидается список или словарь.")
                groups_from_json = {}
            for group_name, ids in groups_from_json.items():
                group_color = st.color_picker(
                    f"Цвет для группы «{group_name}»",
                    value="#0000FF",
                    key=f"uploaded_group_color_{group_name}",
                )
                group_df = house_data[house_data["house_id"].isin(ids)].copy()
                new_group = {
                    "group_name": group_name,
                    "group_color": group_color,
                    "selected_filter_columns": [],
                    "column_filters": {},
                    "filtered_data": group_df.copy(),
                    "is_static": True,
                    "base_data": group_df.copy(),
                }
                st.session_state.dynamic_groups.insert(0, new_group)
                st.success(f"Группа «{group_name}» успешно добавлена.")
            st.session_state.processed_json_files.append(json_file.name)
        except Exception as e:
            st.error(f"Ошибка при чтении JSON: {e}")
group_configs = {}
if st.session_state.dynamic_groups:
    st.sidebar.markdown(
        "<div class='sidebar-header'>Созданные группы</div>", unsafe_allow_html=True
    )
    for idx, group in enumerate(st.session_state.dynamic_groups):
        with st.sidebar.expander(f"{group['group_name']}", expanded=False):
            if st.button("Удалить группу", key=f"del_dynamic_{idx}"):
                st.session_state.dynamic_groups.pop(idx)
                st.experimental_rerun()
            smart_placeholder = compute_smart_group_name(group)
            new_name = st.text_input(
                "Название группы",
                value=group["group_name"],
                placeholder=smart_placeholder,
                key=f"group_name_{idx}",
            )
            if new_name.strip() == "":
                group["group_name"] = smart_placeholder
            else:
                group["group_name"] = new_name
            if group.get("is_static", False):
                st.markdown("*Статическая группа (из JSON)*")
            group["group_color"] = st.color_picker(
                "Цвет группы", value=group["group_color"], key=f"group_color_{idx}"
            )
            group["selected_filter_columns"] = st.multiselect(
                "Выберите признаки для фильтрации",
                options=list(house_data.columns),
                default=group.get("selected_filter_columns", []),
                key=f"group_filter_columns_{idx}",
            )
            column_filters = {}
            for col in group["selected_filter_columns"]:
                if pd.api.types.is_numeric_dtype(house_data[col]):
                    col_data = group["base_data"][col].dropna()
                    column_filters[col] = numeric_filter_widget(
                        col, col_data, key_prefix=f"group_filter_{idx}"
                    )
                else:
                    all_categories = sorted(group["base_data"][col].dropna().unique())
                    popular_categories = get_top_categories(
                        group["base_data"], col, top_n=5
                    )
                    st.write(
                        f"Популярные варианты для «{col}»: {', '.join(popular_categories)}"
                    )
                    selected_options = st.multiselect(
                        f"Выберите значения для «{col}»",
                        options=all_categories,
                        default=[],
                        key=f"group_filter_{idx}_{col}",
                    )
                    column_filters[col] = selected_options
            group["column_filters"] = column_filters
            base_df = (
                group["base_data"].copy()
                if group.get("is_static", False)
                else house_data.copy()
            )
            filtered_df = base_df.copy()
            for col, filter_val in group["column_filters"].items():
                if isinstance(filter_val, tuple):
                    filtered_df = filtered_df[
                        (filtered_df[col] >= filter_val[0])
                        & (filtered_df[col] <= filter_val[1])
                    ]
                else:
                    if filter_val:
                        filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
            group["filtered_data"] = filtered_df.copy()
            group_configs[group["group_name"]] = {
                "column_filters": group["column_filters"],
                "filtered_data": group["filtered_data"],
                "vis": {
                    "color": group["group_color"],
                    "opacity": 200,
                    "radius": 50,
                    "show": True,
                },
            }

tab_data, tab_map, tab_analysis, tab_prediction, tab_clustering = st.tabs(
    ["Данные", "Карта объектов", "Анализ", "Предсказание", "Кластеризация"]
)

with tab_data:
    st.markdown(
        "<div class='section-header'>Табличное представление данных</div>",
        unsafe_allow_html=True,
    )
    main_table = global_filtered_data.copy()
    total_rows = len(house_data)
    max_table_rows = len(main_table)
    default_rows = (
        100 if max_table_rows >= 100 else (max_table_rows if max_table_rows > 0 else 1)
    )
    num_rows = st.number_input(
        "Количество строк для отображения",
        min_value=1,
        max_value=max_table_rows if max_table_rows > 0 else 1,
        value=default_rows,
        step=1,
    )
    percentage = (max_table_rows / total_rows * 100) if total_rows > 0 else 0
    st.write(
        f"Найдено строк: **{max_table_rows}** из **{total_rows}** ({percentage:.2f}%)"
    )
    visible_columns = [
        "project",
        "house_id",
        "developer",
        "class",
        "start_sales",
        "ndeals",
        "deals_sold",
        "mean_area",
        "mean_price",
        "mean_selling_time",
        "floor",
    ]
    available_columns = [col for col in visible_columns if col in main_table.columns]
    display_df = (
        main_table[available_columns].copy() if available_columns else main_table.copy()
    )
    for col in display_df.select_dtypes(include=["float", "int"]).columns:
        display_df[col] = display_df[col].round(2)
    if display_df.empty:
        st.info("Нет данных для отображения.")
    else:
        st.dataframe(display_df.head(num_rows), use_container_width=True)
        with st.container():
            st.markdown('<div class="download-panel">', unsafe_allow_html=True)
            download_ui(display_df, "global_data")
            st.markdown("</div>", unsafe_allow_html=True)
    if group_configs:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Группы</div>", unsafe_allow_html=True)
        for grp, config in group_configs.items():
            grp_df = config["filtered_data"]
            with st.expander(f"{grp} (строк: {len(grp_df)})", expanded=False):
                st.markdown("<div class='group-card'>", unsafe_allow_html=True)
                available_columns_grp = [
                    col for col in visible_columns if col in grp_df.columns
                ]
                display_grp_df = (
                    grp_df[available_columns_grp].head(num_rows).copy()
                    if available_columns_grp
                    else grp_df.head(num_rows).copy()
                )
                for col in display_grp_df.select_dtypes(
                    include=["float", "int"]
                ).columns:
                    display_grp_df[col] = display_grp_df[col].round(2)
                st.dataframe(display_grp_df, use_container_width=True)
                with st.container():
                    st.markdown('<div class="download-panel">', unsafe_allow_html=True)
                    download_ui(display_grp_df, grp)
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

with tab_map:
    st.markdown(
        "<div class='section-header'>Интерактивная карта объектов</div>",
        unsafe_allow_html=True,
    )
    required_cols = {"latitude", "longitude", "house_id"}
    if not required_cols.issubset(house_data.columns):
        st.warning(f"Для карты необходимы столбцы: {', '.join(required_cols)}")
    else:
        global_vis_color = "#FF0000"
        if "global_radius" not in st.session_state:
            st.session_state["global_radius"] = 50
        if "global_opacity" not in st.session_state:
            st.session_state["global_opacity"] = 150
        global_radius = st.session_state["global_radius"]
        global_opacity = st.session_state["global_opacity"]
        for grp in group_configs.keys():
            key_radius = f"grp_radius_{grp}"
            key_opacity = f"grp_opacity_{grp}"
            if key_radius not in st.session_state:
                st.session_state[key_radius] = group_configs[grp]["vis"]["radius"]
            if key_opacity not in st.session_state:
                st.session_state[key_opacity] = group_configs[grp]["vis"]["opacity"]
        cols = st.columns(len(group_configs) + 1)
        toggle_global = cols[0].checkbox(
            "Глобальный слой", value=True, key="toggle_global"
        )
        cols[0].markdown(
            f"<div style='width:20px;height:20px;background:{global_vis_color};border-radius:3px;'></div>",
            unsafe_allow_html=True,
        )
        toggles = {"Глобальный": toggle_global}
        for idx, grp in enumerate(group_configs.keys(), start=1):
            toggles[grp] = cols[idx].checkbox(grp, value=True, key=f"toggle_{grp}")
            cols[idx].markdown(
                f"<div style='width:20px;height:20px;background:{group_configs[grp]['vis']['color']};border-radius:3px;'></div>",
                unsafe_allow_html=True,
            )
        points_for_center = pd.DataFrame()
        global_points = global_filtered_data.dropna(
            subset=["latitude", "longitude", "house_id"]
        ).copy()[["latitude", "longitude", "house_id"]]
        layers = []
        if toggle_global and not global_points.empty:
            global_layer = pdk.Layer(
                "ScatterplotLayer",
                data=global_points,
                get_position=["longitude", "latitude"],
                get_radius=global_radius,
                get_fill_color=hex_to_rgba(global_vis_color, global_opacity),
                pickable=True,
            )
            layers.append(global_layer)
            points_for_center = pd.concat(
                [points_for_center, global_points], ignore_index=True
            )
        for grp, config in group_configs.items():
            if not toggles.get(grp, False):
                continue
            grp_data = (
                config["filtered_data"]
                .dropna(subset=["latitude", "longitude", "house_id"])
                .copy()[["latitude", "longitude", "house_id"]]
            )
            if not grp_data.empty:
                key_radius = f"grp_radius_{grp}"
                key_opacity = f"grp_opacity_{grp}"
                grp_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=grp_data,
                    get_position=["longitude", "latitude"],
                    get_radius=st.session_state[key_radius],
                    get_fill_color=hex_to_rgba(
                        config["vis"]["color"], st.session_state[key_opacity]
                    ),
                    pickable=True,
                )
                layers.append(grp_layer)
                points_for_center = pd.concat(
                    [points_for_center, grp_data], ignore_index=True
                )
        if not layers:
            st.warning("Нет данных для отображения на карте.")
        else:
            if not points_for_center.empty:
                center_lat = points_for_center["latitude"].mean()
                center_lon = points_for_center["longitude"].mean()
            else:
                center_lat, center_lon = 55.75, 37.62
            view_state = pdk.ViewState(
                latitude=center_lat, longitude=center_lon, zoom=10, pitch=0
            )
            tooltip = {
                "html": "<b>ID: {house_id}</b>",
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white",
                    "fontSize": "14px",
                },
            }
            deck = pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            )
            st.pydeck_chart(deck)
        if "show_map_settings" not in st.session_state:
            st.session_state.show_map_settings = False
        if st.button("⚙️ Настройки карты", key="toggle_map_settings"):
            st.session_state.show_map_settings = not st.session_state.show_map_settings
        if st.session_state.show_map_settings:
            with st.container():
                st.markdown("#### Настройки карты")
                st.markdown("**Глобальный слой:**")
                col1, col2 = st.columns(2)
                new_global_radius = col1.slider(
                    "Размер точек",
                    min_value=10,
                    max_value=200,
                    value=global_radius,
                    step=5,
                    key="global_radius_slider",
                )
                new_global_opacity = col2.slider(
                    "Яркость",
                    min_value=0,
                    max_value=255,
                    value=global_opacity,
                    step=5,
                    key="global_opacity_slider",
                )
                st.session_state["global_radius"] = new_global_radius
                st.session_state["global_opacity"] = new_global_opacity
                for grp in group_configs.keys():
                    st.markdown(f"**{grp}:**")
                    col1, col2 = st.columns(2)
                    new_grp_radius = col1.slider(
                        "Размер точек",
                        min_value=10,
                        max_value=200,
                        value=st.session_state[f"grp_radius_{grp}"],
                        step=5,
                        key=f"{grp}_radius_slider",
                    )
                    new_grp_opacity = col2.slider(
                        "Яркость",
                        min_value=0,
                        max_value=255,
                        value=st.session_state[f"grp_opacity_{grp}"],
                        step=5,
                        key=f"{grp}_opacity_slider",
                    )
                    st.session_state[f"grp_radius_{grp}"] = new_grp_radius
                    st.session_state[f"grp_opacity_{grp}"] = new_grp_opacity

with tab_analysis:
    st.markdown(
        "<div class='section-header'>Анализ данных</div>", unsafe_allow_html=True
    )
    analysis_mode_tabs = st.tabs(
        ["Индивидуальное построение", "Множественное построение"]
    )
    with analysis_mode_tabs[0]:
        if "graph_configs" not in st.session_state:
            st.session_state.graph_configs = []
        if len(st.session_state.graph_configs) == 0:
            default_config = {
                "id": 1,
                "name": "График 1",
                "chart_type": "Гистограмма",
                "selected_groups": ["Глобальный"]
                + (list(group_configs.keys()) if group_configs else []),
                "histogram": {
                    "column": "mean_price",
                    "log_transform": False,
                    "remove_outliers": False,
                    "lower_q": 0.05,
                    "upper_q": 0.95,
                    "normalize": False,
                },
                "scatter": {
                    "x": "mean_price",
                    "y": "mean_selling_time",
                    "normalize": False,
                },
                "depletion": {"show_individual": False},
            }
            st.session_state.graph_configs.append(default_config)
        for idx, config in enumerate(st.session_state.graph_configs):
            with st.container():
                colA, colB = st.columns(2)
                with colA:
                    config["name"] = st.text_input(
                        "Название графика",
                        value=config["name"],
                        key=f"chart_name_{idx}",
                    )
                with colB:
                    config["chart_type"] = st.selectbox(
                        "Тип графика",
                        options=["Гистограмма", "Скатерплот", "Кривая выбытия"],
                        index=["Гистограмма", "Скатерплот", "Кривая выбытия"].index(
                            config["chart_type"]
                        ),
                        key=f"chart_type_{idx}",
                    )
                available_groups = ["Глобальный"] + (
                    list(group_configs.keys()) if group_configs else []
                )
                safe_default = [
                    g
                    for g in config.get("selected_groups", available_groups)
                    if g in available_groups
                ]
                if not safe_default:
                    safe_default = available_groups
                config["selected_groups"] = st.multiselect(
                    "Группы для анализа",
                    options=available_groups,
                    default=safe_default,
                    key=f"chart_groups_{idx}",
                )
                with st.expander("Настройки графика", expanded=False):
                    if config["chart_type"] == "Гистограмма":
                        st.markdown("Настройки гистограммы", unsafe_allow_html=True)
                        numeric_cols = set()
                        for g in config["selected_groups"]:
                            df = (
                                global_filtered_data.copy()
                                if g == "Глобальный"
                                else group_configs[g]["filtered_data"].copy()
                            )
                            numeric_cols.update(
                                df.select_dtypes(include=["number"]).columns.tolist()
                            )
                        numeric_cols = sorted(list(numeric_cols))
                        default_col = (
                            "mean_price"
                            if "mean_price" in numeric_cols
                            else (numeric_cols[0] if numeric_cols else "")
                        )
                        config["histogram"]["column"] = st.selectbox(
                            "Признак для гистограммы",
                            options=numeric_cols,
                            index=(
                                numeric_cols.index(default_col)
                                if default_col in numeric_cols
                                else 0
                            ),
                            key=f"hist_column_{idx}",
                        )
                        config["histogram"]["log_transform"] = st.checkbox(
                            "Логарифмировать",
                            value=config["histogram"].get("log_transform", False),
                            key=f"hist_log_{idx}",
                        )
                        config["histogram"]["remove_outliers"] = st.checkbox(
                            "Удалить выбросы",
                            value=config["histogram"].get("remove_outliers", False),
                            key=f"hist_outliers_{idx}",
                        )
                        if config["histogram"].get("remove_outliers", False):
                            config["histogram"]["lower_q"] = st.slider(
                                "Нижний квантиль",
                                min_value=0.0,
                                max_value=0.3,
                                value=config["histogram"].get("lower_q", 0.05),
                                step=0.01,
                                key=f"hist_lq_{idx}",
                            )
                            config["histogram"]["upper_q"] = st.slider(
                                "Верхний квантиль",
                                min_value=0.7,
                                max_value=1.0,
                                value=config["histogram"].get("upper_q", 0.95),
                                step=0.01,
                                key=f"hist_uq_{idx}",
                            )
                        config["histogram"]["normalize"] = st.checkbox(
                            "Нормировать гистограмму",
                            value=config["histogram"].get("normalize", False),
                            key=f"hist_normalize_{idx}",
                        )
                    elif config["chart_type"] == "Скатерплот":
                        st.markdown("Настройки скатерплота", unsafe_allow_html=True)
                        numeric_cols = set()
                        for g in config["selected_groups"]:
                            df = (
                                global_filtered_data.copy()
                                if g == "Глобальный"
                                else group_configs[g]["filtered_data"].copy()
                            )
                            numeric_cols.update(
                                df.select_dtypes(include=["number"]).columns.tolist()
                            )
                        numeric_cols = sorted(list(numeric_cols))
                        default_x = (
                            "mean_price"
                            if "mean_price" in numeric_cols
                            else (numeric_cols[0] if numeric_cols else "")
                        )
                        default_y = (
                            "mean_selling_time"
                            if "mean_selling_time" in numeric_cols
                            else (numeric_cols[1] if len(numeric_cols) > 1 else "")
                        )
                        config["scatter"]["x"] = st.selectbox(
                            "Ось X",
                            options=numeric_cols,
                            index=(
                                numeric_cols.index(default_x)
                                if default_x in numeric_cols
                                else 0
                            ),
                            key=f"scatter_x_{idx}",
                        )
                        config["scatter"]["y"] = st.selectbox(
                            "Ось Y",
                            options=numeric_cols,
                            index=(
                                numeric_cols.index(default_y)
                                if default_y in numeric_cols
                                else 0
                            ),
                            key=f"scatter_y_{idx}",
                        )
                        config["scatter"]["normalize"] = st.checkbox(
                            "Нормировать (привести к квантилям)",
                            value=config["scatter"].get("normalize", False),
                            key=f"scatter_normalize_{idx}",
                        )
                    elif config["chart_type"] == "Кривая выбытия":
                        st.markdown("Настройки кривой выбытия", unsafe_allow_html=True)
                        config["depletion"]["show_individual"] = st.checkbox(
                            "Показать индивидуальные кривые",
                            value=config["depletion"].get("show_individual", False),
                            key=f"depletion_individual_{idx}",
                        )
                if st.button(
                    f"Удалить график {config['name']}", key=f"delete_chart_{idx}"
                ):
                    st.session_state.graph_configs.pop(idx)
                    st.experimental_rerun()
                selected_groups = config["selected_groups"]
                color_domain = selected_groups
                color_range = [
                    "#FF0000" if g == "Глобальный" else group_configs[g]["vis"]["color"]
                    for g in selected_groups
                ]
                if config["chart_type"] == "Гистограмма":
                    hist_column = config["histogram"]["column"]
                    log_transform = config["histogram"].get("log_transform", False)
                    remove_outliers = config["histogram"].get("remove_outliers", False)
                    lower_q = config["histogram"].get("lower_q", 0.05)
                    upper_q = config["histogram"].get("upper_q", 0.95)
                    normalize = config["histogram"].get("normalize", False)
                    chart_data = pd.DataFrame()
                    for g in config["selected_groups"]:
                        df = (
                            global_filtered_data.copy()
                            if g == "Глобальный"
                            else group_configs[g]["filtered_data"].copy()
                        )
                        if hist_column in df.columns:
                            col_data = df[hist_column].dropna()
                            if remove_outliers and not col_data.empty:
                                lower_bound = col_data.quantile(lower_q)
                                upper_bound = col_data.quantile(upper_q)
                                df = df[
                                    (df[hist_column] >= lower_bound)
                                    & (df[hist_column] <= upper_bound)
                                ]
                            if log_transform:
                                df = df[df[hist_column] > 0]
                                df[hist_column] = np.log(df[hist_column])
                            df = df[[hist_column]].copy()
                            df["group"] = g
                            chart_data = pd.concat([chart_data, df], ignore_index=True)
                    if not chart_data.empty:
                        if normalize:
                            chart = (
                                alt.Chart(chart_data)
                                .transform_bin(
                                    "bin", field=hist_column, bin=alt.Bin(maxbins=30)
                                )
                                .transform_aggregate(
                                    count="count()", groupby=["bin", "group"]
                                )
                                .transform_window(total="sum(count)", groupby=["group"])
                                .transform_calculate(
                                    fraction="datum.count / datum.total"
                                )
                                .mark_bar(opacity=0.7)
                                .encode(
                                    x=alt.X("bin:Q", title=hist_column),
                                    y=alt.Y(
                                        "fraction:Q",
                                        title="Доля",
                                        axis=alt.Axis(format=".0%"),
                                    ),
                                    color=alt.Color(
                                        "group:N",
                                        scale=alt.Scale(
                                            domain=color_domain, range=color_range
                                        ),
                                        legend=alt.Legend(title="Группа"),
                                    ),
                                )
                                .properties(height=400)
                            )
                        else:
                            chart = (
                                alt.Chart(chart_data)
                                .mark_bar(opacity=0.7)
                                .encode(
                                    x=alt.X(
                                        f"{hist_column}:Q",
                                        bin=alt.Bin(maxbins=30),
                                        title=hist_column,
                                    ),
                                    y=alt.Y("count()", title="Количество"),
                                    color=alt.Color(
                                        "group:N",
                                        scale=alt.Scale(
                                            domain=color_domain, range=color_range
                                        ),
                                        legend=alt.Legend(title="Группа"),
                                    ),
                                )
                                .properties(height=400)
                            )
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("Нет данных для построения гистограммы.")
                elif config["chart_type"] == "Скатерплот":
                    x_col = config["scatter"]["x"]
                    y_col = config["scatter"]["y"]
                    normalize_scatter = config["scatter"].get("normalize", False)
                    chart_data = pd.DataFrame()
                    for g in config["selected_groups"]:
                        df = (
                            global_filtered_data.copy()
                            if g == "Глобальный"
                            else group_configs[g]["filtered_data"].copy()
                        )
                        if x_col in df.columns and y_col in df.columns:
                            df = df[[x_col, y_col]].copy()
                            if normalize_scatter:
                                if df[x_col].nunique() > 1:
                                    df[x_col] = df[x_col].rank(pct=True)
                                if df[y_col].nunique() > 1:
                                    df[y_col] = df[y_col].rank(pct=True)
                            df["group"] = g
                            chart_data = pd.concat([chart_data, df], ignore_index=True)
                    if not chart_data.empty:
                        chart = (
                            alt.Chart(chart_data)
                            .mark_circle(size=60)
                            .encode(
                                x=alt.X(f"{x_col}:Q", title=x_col),
                                y=alt.Y(f"{y_col}:Q", title=y_col),
                                color=alt.Color(
                                    "group:N",
                                    scale=alt.Scale(
                                        domain=color_domain, range=color_range
                                    ),
                                    legend=alt.Legend(title="Группа"),
                                ),
                                tooltip=list(chart_data.columns),
                            )
                            .interactive()
                            .properties(height=400)
                        )
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("Нет данных для построения скатерплота.")
                elif config["chart_type"] == "Кривая выбытия":
                    depletion_curves = load_depletion_curves("depletion_curves.feather")
                    if depletion_curves.empty:
                        st.info("Нет данных для построения кривой выбытия.")
                    else:
                        combined_data = pd.DataFrame()
                        individual_data = pd.DataFrame()
                        for g in config["selected_groups"]:
                            house_ids = (
                                global_filtered_data["house_id"].unique()
                                if g == "Глобальный"
                                else group_configs[g]["filtered_data"][
                                    "house_id"
                                ].unique()
                            )
                            avg_df = compute_avg_depletion_curve(
                                depletion_curves, house_ids
                            )
                            if not avg_df.empty:
                                avg_df["group"] = g
                                combined_data = pd.concat(
                                    [combined_data, avg_df], ignore_index=True
                                )
                            if config["depletion"].get("show_individual", False):
                                indiv = depletion_curves[
                                    depletion_curves["house_id"].isin(house_ids)
                                ].copy()
                                if not indiv.empty:
                                    indiv["group"] = g
                                    individual_data = pd.concat(
                                        [individual_data, indiv], ignore_index=True
                                    )
                        if not combined_data.empty:
                            base_chart = (
                                alt.Chart(combined_data)
                                .mark_line(strokeWidth=3, interpolate="step-after")
                                .encode(
                                    x=alt.X("time:Q", title="Время (дни)"),
                                    y=alt.Y(
                                        "pct:Q", title="Средний остаток продаж (%)"
                                    ),
                                    color=alt.Color(
                                        "group:N",
                                        scale=alt.Scale(
                                            domain=color_domain, range=color_range
                                        ),
                                        legend=alt.Legend(title="Группа"),
                                    ),
                                )
                                .properties(height=400)
                            )
                            if (
                                config["depletion"].get("show_individual", False)
                                and not individual_data.empty
                            ):
                                indiv_chart = (
                                    alt.Chart(individual_data)
                                    .mark_line(interpolate="step-after", opacity=0.6)
                                    .encode(
                                        x=alt.X("time:Q", title="Время (дни)"),
                                        y=alt.Y("pct:Q", title="Остаток продаж (%)"),
                                        color=alt.Color(
                                            "group:N",
                                            scale=alt.Scale(
                                                domain=color_domain, range=color_range
                                            ),
                                            legend=None,
                                        ),
                                        detail="house_id:N",
                                    )
                                    .properties(height=400)
                                )
                                final_chart = (
                                    alt.layer(base_chart, indiv_chart)
                                    .resolve_scale(color="independent")
                                    .properties(height=400)
                                )
                            else:
                                final_chart = base_chart
                            st.altair_chart(final_chart, use_container_width=True)
                        else:
                            st.info("Нет данных для построения кривой выбытия.")
            st.markdown(
                "<hr style='border-top: 1px solid #ccc;'>", unsafe_allow_html=True
            )
            if st.button("Добавить график", key="add_chart_bottom"):
                default_selected_groups = ["Глобальный"] + (
                    list(group_configs.keys()) if group_configs else []
                )
                new_config = {
                    "id": len(st.session_state.graph_configs) + 1,
                    "name": f"График {len(st.session_state.graph_configs) + 1}",
                    "chart_type": "Гистограмма",
                    "selected_groups": default_selected_groups,
                    "histogram": {
                        "column": "mean_price",
                        "log_transform": False,
                        "remove_outliers": False,
                        "lower_q": 0.05,
                        "upper_q": 0.95,
                        "normalize": False,
                    },
                    "scatter": {
                        "x": "mean_price",
                        "y": "mean_selling_time",
                        "normalize": False,
                    },
                    "depletion": {"show_individual": False},
                }
                st.session_state.graph_configs.append(new_config)
                st.experimental_rerun()
    with analysis_mode_tabs[1]:
        if "multi_graph_configs" not in st.session_state:
            st.session_state.multi_graph_configs = []
        if len(st.session_state.multi_graph_configs) == 0:
            default_config = {
                "selected_groups": ["Глобальный"]
                + (list(group_configs.keys()) if group_configs else []),
                "selected_chart_types": ["Гистограмма"],
                "params": {
                    "Гистограмма": {
                        "column": "mean_price",
                        "log_transform": False,
                        "remove_outliers": False,
                        "lower_q": 0.05,
                        "upper_q": 0.95,
                        "normalize": False,
                    },
                    "Скатерплот": {
                        "x": "mean_price",
                        "y": "mean_selling_time",
                        "normalize": False,
                    },
                    "Кривая выбытия": {"show_individual": False},
                },
            }
            st.session_state.multi_graph_configs.append(default_config)
        for i, cfg in enumerate(st.session_state.multi_graph_configs):
            st.markdown(f"### Группа графиков {i+1}")
            available_groups = ["Глобальный"] + (
                list(group_configs.keys()) if group_configs else []
            )
            cfg["selected_groups"] = safe_multiselect(
                "Выберите группы для анализа",
                options=available_groups,
                default=cfg.get("selected_groups", available_groups),
                key=f"multi_selected_groups_{i}",
            )
            cfg["selected_chart_types"] = st.multiselect(
                "Выберите типы графиков для построения",
                options=["Гистограмма", "Скатерплот", "Кривая выбытия"],
                default=cfg.get("selected_chart_types", ["Гистограмма"]),
                key=f"multi_selected_chart_types_{i}",
            )
            if "Гистограмма" in cfg["selected_chart_types"]:
                with st.expander("Настройки гистограммы", expanded=False):
                    numeric_cols = set()
                    for g in cfg["selected_groups"]:
                        df = (
                            global_filtered_data
                            if g == "Глобальный"
                            else group_configs[g]["filtered_data"]
                        )
                        numeric_cols.update(
                            df.select_dtypes(include=["number"]).columns.tolist()
                        )
                    numeric_cols = sorted(list(numeric_cols))
                    default_col = (
                        "mean_price"
                        if "mean_price" in numeric_cols
                        else (numeric_cols[0] if numeric_cols else "")
                    )
                    cfg.setdefault("params", {}).setdefault("Гистограмма", {})
                    cfg["params"]["Гистограмма"]["column"] = st.selectbox(
                        "Признак для гистограммы",
                        options=numeric_cols,
                        index=(
                            numeric_cols.index(default_col)
                            if default_col in numeric_cols
                            else 0
                        ),
                        key=f"multi_hist_column_{i}",
                    )
                    cfg["params"]["Гистограмма"]["log_transform"] = st.checkbox(
                        "Логарифмировать гистограмму",
                        value=cfg["params"]["Гистограмма"].get("log_transform", False),
                        key=f"multi_hist_log_{i}",
                    )
                    cfg["params"]["Гистограмма"]["remove_outliers"] = st.checkbox(
                        "Удалить выбросы в гистограмме",
                        value=cfg["params"]["Гистограмма"].get(
                            "remove_outliers", False
                        ),
                        key=f"multi_hist_outliers_{i}",
                    )
                    cfg["params"]["Гистограмма"]["lower_q"] = st.slider(
                        "Нижний квантиль (гистограмма)",
                        min_value=0.0,
                        max_value=0.3,
                        value=cfg["params"]["Гистограмма"].get("lower_q", 0.05),
                        step=0.01,
                        key=f"multi_hist_lq_{i}",
                    )
                    cfg["params"]["Гистограмма"]["upper_q"] = st.slider(
                        "Верхний квантиль (гистограмма)",
                        min_value=0.7,
                        max_value=1.0,
                        value=cfg["params"]["Гистограмма"].get("upper_q", 0.95),
                        step=0.01,
                        key=f"multi_hist_uq_{i}",
                    )
                    cfg["params"]["Гистограмма"]["normalize"] = st.checkbox(
                        "Нормировать гистограмму",
                        value=cfg["params"]["Гистограмма"].get("normalize", False),
                        key=f"multi_hist_norm_{i}",
                    )
            if "Скатерплот" in cfg["selected_chart_types"]:
                with st.expander("Настройки скатерплота", expanded=False):
                    numeric_cols = set()
                    for g in cfg["selected_groups"]:
                        df = (
                            global_filtered_data
                            if g == "Глобальный"
                            else group_configs[g]["filtered_data"]
                        )
                        numeric_cols.update(
                            df.select_dtypes(include=["number"]).columns.tolist()
                        )
                    numeric_cols = sorted(list(numeric_cols))
                    default_x = (
                        "mean_price"
                        if "mean_price" in numeric_cols
                        else (numeric_cols[0] if numeric_cols else "")
                    )
                    default_y = (
                        "mean_selling_time"
                        if "mean_selling_time" in numeric_cols
                        else (numeric_cols[1] if len(numeric_cols) > 1 else "")
                    )
                    cfg.setdefault("params", {}).setdefault("Скатерплот", {})
                    cfg["params"]["Скатерплот"]["x"] = st.selectbox(
                        "Ось X (скатерплот)",
                        options=numeric_cols,
                        index=(
                            numeric_cols.index(default_x)
                            if default_x in numeric_cols
                            else 0
                        ),
                        key=f"multi_scatter_x_{i}",
                    )
                    cfg["params"]["Скатерплот"]["y"] = st.selectbox(
                        "Ось Y (скатерплот)",
                        options=numeric_cols,
                        index=(
                            numeric_cols.index(default_y)
                            if default_y in numeric_cols
                            else 0
                        ),
                        key=f"multi_scatter_y_{i}",
                    )
                    cfg["params"]["Скатерплот"]["normalize"] = st.checkbox(
                        "Нормировать скатерплот (привести к квантилям)",
                        value=cfg["params"]["Скатерплот"].get("normalize", False),
                        key=f"multi_scatter_norm_{i}",
                    )
            if "Кривая выбытия" in cfg["selected_chart_types"]:
                with st.expander("Настройки кривой выбытия", expanded=False):
                    cfg.setdefault("params", {}).setdefault("Кривая выбытия", {})
                    cfg["params"]["Кривая выбытия"]["show_individual"] = st.checkbox(
                        "Показать индивидуальные кривые",
                        value=cfg["params"]["Кривая выбытия"].get(
                            "show_individual", False
                        ),
                        key=f"multi_depletion_indiv_{i}",
                    )
            for ct in cfg["selected_chart_types"]:
                if ct == "Гистограмма":
                    hist_column = cfg["params"]["Гистограмма"]["column"]
                    log_transform = cfg["params"]["Гистограмма"]["log_transform"]
                    remove_outliers = cfg["params"]["Гистограмма"]["remove_outliers"]
                    lower_q = cfg["params"]["Гистограмма"]["lower_q"]
                    upper_q = cfg["params"]["Гистограмма"]["upper_q"]
                    normalize = cfg["params"]["Гистограмма"]["normalize"]
                    chart_data = pd.DataFrame()
                    for g in cfg["selected_groups"]:
                        df = (
                            global_filtered_data.copy()
                            if g == "Глобальный"
                            else group_configs[g]["filtered_data"].copy()
                        )
                        if hist_column in df.columns:
                            col_data = df[hist_column].dropna()
                            if remove_outliers and not col_data.empty:
                                lower_bound = col_data.quantile(lower_q)
                                upper_bound = col_data.quantile(upper_q)
                                df = df[
                                    (df[hist_column] >= lower_bound)
                                    & (df[hist_column] <= upper_bound)
                                ]
                            if log_transform:
                                df = df[df[hist_column] > 0]
                                df[hist_column] = np.log(df[hist_column])
                            df = df[[hist_column]].copy()
                            df["group"] = g
                            chart_data = pd.concat([chart_data, df], ignore_index=True)
                    if not chart_data.empty:
                        color_domain = cfg["selected_groups"]
                        color_range = [
                            (
                                "#FF0000"
                                if g == "Глобальный"
                                else group_configs[g]["vis"]["color"]
                            )
                            for g in cfg["selected_groups"]
                        ]
                        if normalize:
                            chart = (
                                alt.Chart(chart_data)
                                .transform_bin(
                                    "bin", field=hist_column, bin=alt.Bin(maxbins=30)
                                )
                                .transform_aggregate(
                                    count="count()", groupby=["bin", "group"]
                                )
                                .transform_window(total="sum(count)", groupby=["group"])
                                .transform_calculate(
                                    fraction="datum.count / datum.total"
                                )
                                .mark_bar(opacity=0.7)
                                .encode(
                                    x=alt.X("bin:Q", title=hist_column),
                                    y=alt.Y(
                                        "fraction:Q",
                                        title="Доля",
                                        axis=alt.Axis(format=".0%"),
                                    ),
                                    color=alt.Color(
                                        "group:N",
                                        scale=alt.Scale(
                                            domain=color_domain, range=color_range
                                        ),
                                        legend=alt.Legend(title="Группа"),
                                    ),
                                )
                                .properties(height=400)
                            )
                        else:
                            chart = (
                                alt.Chart(chart_data)
                                .mark_bar(opacity=0.7)
                                .encode(
                                    x=alt.X(
                                        f"{hist_column}:Q",
                                        bin=alt.Bin(maxbins=30),
                                        title=hist_column,
                                    ),
                                    y=alt.Y("count()", title="Количество"),
                                    color=alt.Color(
                                        "group:N",
                                        scale=alt.Scale(
                                            domain=color_domain, range=color_range
                                        ),
                                        legend=alt.Legend(title="Группа"),
                                    ),
                                )
                                .properties(height=400)
                            )
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("Нет данных для построения гистограммы.")
                elif ct == "Скатерплот":
                    x_col = cfg["params"]["Скатерплот"]["x"]
                    y_col = cfg["params"]["Скатерплот"]["y"]
                    normalize_scatter = cfg["params"]["Скатерплот"]["normalize"]
                    chart_data = pd.DataFrame()
                    for g in cfg["selected_groups"]:
                        df = (
                            global_filtered_data.copy()
                            if g == "Глобальный"
                            else group_configs[g]["filtered_data"].copy()
                        )
                        if x_col in df.columns and y_col in df.columns:
                            df = df[[x_col, y_col]].copy()
                            if normalize_scatter:
                                if df[x_col].nunique() > 1:
                                    df[x_col] = df[x_col].rank(pct=True)
                                if df[y_col].nunique() > 1:
                                    df[y_col] = df[y_col].rank(pct=True)
                            df["group"] = g
                            chart_data = pd.concat([chart_data, df], ignore_index=True)
                    if not chart_data.empty:
                        color_domain = cfg["selected_groups"]
                        color_range = [
                            (
                                "#FF0000"
                                if g == "Глобальный"
                                else group_configs[g]["vis"]["color"]
                            )
                            for g in cfg["selected_groups"]
                        ]
                        chart = (
                            alt.Chart(chart_data)
                            .mark_circle(size=60)
                            .encode(
                                x=alt.X(f"{x_col}:Q", title=x_col),
                                y=alt.Y(f"{y_col}:Q", title=y_col),
                                color=alt.Color(
                                    "group:N",
                                    scale=alt.Scale(
                                        domain=color_domain, range=color_range
                                    ),
                                    legend=alt.Legend(title="Группа"),
                                ),
                                tooltip=list(chart_data.columns),
                            )
                            .interactive()
                            .properties(height=400)
                        )
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("Нет данных для построения скатерплота.")
                elif ct == "Кривая выбытия":
                    depletion_curves = load_depletion_curves("depletion_curves.feather")
                    if depletion_curves.empty:
                        st.info("Нет данных для построения кривой выбытия.")
                    else:
                        combined_data = pd.DataFrame()
                        individual_data = pd.DataFrame()
                        for g in cfg["selected_groups"]:
                            house_ids = (
                                global_filtered_data["house_id"].unique()
                                if g == "Глобальный"
                                else group_configs[g]["filtered_data"][
                                    "house_id"
                                ].unique()
                            )
                            avg_df = compute_avg_depletion_curve(
                                depletion_curves, house_ids
                            )
                            if not avg_df.empty:
                                avg_df["group"] = g
                                combined_data = pd.concat(
                                    [combined_data, avg_df], ignore_index=True
                                )
                            if cfg["params"]["Кривая выбытия"].get(
                                "show_individual", False
                            ):
                                indiv = depletion_curves[
                                    depletion_curves["house_id"].isin(house_ids)
                                ].copy()
                                if not indiv.empty:
                                    indiv["group"] = g
                                    individual_data = pd.concat(
                                        [individual_data, indiv], ignore_index=True
                                    )
                        if not combined_data.empty:
                            color_domain = cfg["selected_groups"]
                            color_range = [
                                (
                                    "#FF0000"
                                    if g == "Глобальный"
                                    else group_configs[g]["vis"]["color"]
                                )
                                for g in cfg["selected_groups"]
                            ]
                            base_chart = (
                                alt.Chart(combined_data)
                                .mark_line(strokeWidth=3, interpolate="step-after")
                                .encode(
                                    x=alt.X("time:Q", title="Время (дни)"),
                                    y=alt.Y(
                                        "pct:Q", title="Средний остаток продаж (%)"
                                    ),
                                    color=alt.Color(
                                        "group:N",
                                        scale=alt.Scale(
                                            domain=color_domain, range=color_range
                                        ),
                                        legend=alt.Legend(title="Группа"),
                                    ),
                                )
                                .properties(height=400)
                            )
                            if (
                                cfg["params"]["Кривая выбытия"].get(
                                    "show_individual", False
                                )
                                and not individual_data.empty
                            ):
                                indiv_chart = (
                                    alt.Chart(individual_data)
                                    .mark_line(interpolate="step-after", opacity=0.6)
                                    .encode(
                                        x=alt.X("time:Q", title="Время (дни)"),
                                        y=alt.Y("pct:Q", title="Остаток продаж (%)"),
                                        color=alt.Color(
                                            "group:N",
                                            scale=alt.Scale(
                                                domain=color_domain, range=color_range
                                            ),
                                            legend=None,
                                        ),
                                        detail="house_id:N",
                                    )
                                    .properties(height=400)
                                )
                                final_chart = (
                                    alt.layer(base_chart, indiv_chart)
                                    .resolve_scale(color="independent")
                                    .properties(height=400)
                                )
                            else:
                                final_chart = base_chart
                            st.altair_chart(final_chart, use_container_width=True)
                        else:
                            st.info("Нет данных для построения кривой выбытия.")
            st.markdown(
                "<hr style='border-top: 1px solid #ccc;'>", unsafe_allow_html=True
            )
            if st.button("Удалить группу", key=f"delete_multi_group_{i}"):
                st.session_state.multi_graph_configs.pop(i)
                st.experimental_rerun()
        if st.button("Добавить новую группу графиков", key="add_multi_group"):
            default_config = {
                "selected_groups": ["Глобальный"]
                + (list(group_configs.keys()) if group_configs else []),
                "selected_chart_types": ["Гистограмма"],
                "params": {
                    "Гистограмма": {
                        "column": "mean_price",
                        "log_transform": False,
                        "remove_outliers": False,
                        "lower_q": 0.05,
                        "upper_q": 0.95,
                        "normalize": False,
                    },
                    "Скатерплот": {
                        "x": "mean_price",
                        "y": "mean_selling_time",
                        "normalize": False,
                    },
                    "Кривая выбытия": {"show_individual": False},
                },
            }
            st.session_state.multi_graph_configs.append(default_config)
            st.experimental_rerun()

with tab_prediction:
    st.markdown(
        "<div class='section-header'>Предсказание</div>", unsafe_allow_html=True
    )
    st.info("Раздел 'Предсказание' находится в разработке.")

with tab_clustering:
    st.markdown(
        "<div class='section-header'>Кластеризация</div>", unsafe_allow_html=True
    )
    st.info("Здесь будет кластеризация")
