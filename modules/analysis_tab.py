# analysis_tab.py


import numpy as np
import pandas as pd
import streamlit as st

from utils.charts import (build_depletion_chart, build_elasticity_chart,
                          build_histogram, build_scatter)


def render_analysis_tab(
    global_filtered_data: pd.DataFrame,
    house_data: pd.DataFrame,
    group_configs: dict,
    apartment_data: pd.DataFrame,
):
    """
    Основная вкладка "Анализ" со множеством типов графиков:
      - гистограмма
      - скатерплот
      - кривая выбытия
      - кривая эластичности (с учетом нового выбора комнат)
    """
    st.markdown(
        "<div class='section-header'>Анализ данных</div>", unsafe_allow_html=True
    )
    analysis_mode_tabs = st.tabs(
        ["Индивидуальное построение", "Множественное построение"]
    )

    with analysis_mode_tabs[0]:
        if "chart_counter" not in st.session_state:
            st.session_state.chart_counter = 0
        if "graph_configs" not in st.session_state:
            st.session_state.graph_configs = []

        if len(st.session_state.graph_configs) == 0:
            st.session_state.chart_counter += 1
            default_config = {
                "id": st.session_state.chart_counter,
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
                "elasticity": {
                    "area_min": 0,
                    "area_max": 100,
                    "split": 1,
                    "rooms_list": [None, 0, 1, 2, 3],
                },
            }
            st.session_state.graph_configs.append(default_config)

        for config in st.session_state.graph_configs:
            with st.container():
                colA, colB = st.columns(2)
                with colA:
                    config["name"] = st.text_input(
                        "Название графика",
                        value=config["name"],
                        key=f"chart_name_{config['id']}",
                    )
                with colB:
                    chart_options = [
                        "Гистограмма",
                        "Скатерплот",
                        "Кривая выбытия",
                        "Кривая эластичности",
                    ]
                    if config["chart_type"] not in chart_options:
                        config["chart_type"] = "Гистограмма"
                    config["chart_type"] = st.selectbox(
                        "Тип графика",
                        options=chart_options,
                        index=chart_options.index(config["chart_type"]),
                        key=f"chart_type_{config['id']}",
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
                    key=f"chart_groups_{config['id']}",
                )

                with st.expander("Настройки графика", expanded=False):
                    if config["chart_type"] == "Гистограмма":
                        st.markdown("Настройки гистограммы", unsafe_allow_html=True)
                        numeric_cols = set()
                        for g in config["selected_groups"]:
                            df_g = (
                                global_filtered_data
                                if g == "Глобальный"
                                else group_configs[g]["filtered_data"]
                            )
                            numeric_cols.update(
                                df_g.select_dtypes(include=["number"]).columns.tolist()
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
                            key=f"hist_column_{config['id']}",
                        )
                        config["histogram"]["log_transform"] = st.checkbox(
                            "Логарифмировать",
                            value=config["histogram"].get("log_transform", False),
                            key=f"hist_log_{config['id']}",
                        )
                        config["histogram"]["remove_outliers"] = st.checkbox(
                            "Удалить выбросы",
                            value=config["histogram"].get("remove_outliers", False),
                            key=f"hist_outliers_{config['id']}",
                        )
                        if config["histogram"].get("remove_outliers", False):
                            config["histogram"]["lower_q"] = st.slider(
                                "Нижний квантиль",
                                min_value=0.0,
                                max_value=0.3,
                                value=config["histogram"].get("lower_q", 0.05),
                                step=0.01,
                                key=f"hist_lq_{config['id']}",
                            )
                            config["histogram"]["upper_q"] = st.slider(
                                "Верхний квантиль",
                                min_value=0.7,
                                max_value=1.0,
                                value=config["histogram"].get("upper_q", 0.95),
                                step=0.01,
                                key=f"hist_uq_{config['id']}",
                            )
                        config["histogram"]["normalize"] = st.checkbox(
                            "Нормировать гистограмму",
                            value=config["histogram"].get("normalize", False),
                            key=f"hist_normalize_{config['id']}",
                        )

                    elif config["chart_type"] == "Скатерплот":
                        st.markdown("Настройки скатерплота", unsafe_allow_html=True)
                        numeric_cols = set()
                        for g in config["selected_groups"]:
                            df_g = (
                                global_filtered_data
                                if g == "Глобальный"
                                else group_configs[g]["filtered_data"]
                            )
                            numeric_cols.update(
                                df_g.select_dtypes(include=["number"]).columns.tolist()
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
                            key=f"scatter_x_{config['id']}",
                        )
                        config["scatter"]["y"] = st.selectbox(
                            "Ось Y",
                            options=numeric_cols,
                            index=(
                                numeric_cols.index(default_y)
                                if default_y in numeric_cols
                                else 0
                            ),
                            key=f"scatter_y_{config['id']}",
                        )
                        config["scatter"]["normalize"] = st.checkbox(
                            "Нормировать (привести к квантилям)",
                            value=config["scatter"].get("normalize", False),
                            key=f"scatter_normalize_{config['id']}",
                        )

                    elif config["chart_type"] == "Кривая выбытия":
                        st.markdown("Настройки кривой выбытия", unsafe_allow_html=True)
                        config["depletion"]["show_individual"] = st.checkbox(
                            "Показать индивидуальные кривые",
                            value=config["depletion"].get("show_individual", False),
                            key=f"depletion_individual_{config['id']}",
                        )

                    elif config["chart_type"] == "Кривая эластичности":
                        st.markdown(
                            "Настройки кривой эластичности", unsafe_allow_html=True
                        )
                        if "elasticity" not in config:
                            config["elasticity"] = {
                                "area_min": 0,
                                "area_max": 100,
                                "split": 1,
                                "rooms_list": [None, 0, 1, 2, 3],
                            }

                        config["elasticity"]["area_min"] = st.number_input(
                            "Минимальная площадь (area_min)",
                            min_value=0.0,
                            value=float(config["elasticity"].get("area_min", 0)),
                            step=1.0,
                            key=f"elasticity_area_min_{config['id']}",
                        )
                        config["elasticity"]["area_max"] = st.number_input(
                            "Максимальная площадь (area_max)",
                            min_value=0.0,
                            value=float(config["elasticity"].get("area_max", 100)),
                            step=1.0,
                            key=f"elasticity_area_max_{config['id']}",
                        )
                        config["elasticity"]["split"] = st.slider(
                            "Параметр сегментации (кв.м.)",
                            min_value=1,
                            max_value=10,
                            value=int(config["elasticity"].get("split", 1)),
                            step=1,
                            key=f"elasticity_split_{config['id']}",
                        )

                        rooms_options = {
                            "Все сделки (None)": None,
                            "Студия (0)": 0,
                            "1 комната": 1,
                            "2 комнаты": 2,
                            "3 комнаты": 3,
                            "4 комнаты": 4,
                        }
                        current_rooms_list = config["elasticity"].get(
                            "rooms_list", [None, 0, 1, 2, 3]
                        )

                        rev_rooms_options = {v: k for k, v in rooms_options.items()}
                        current_rooms_str = [
                            rev_rooms_options.get(r, "Все сделки (None)")
                            for r in current_rooms_list
                            if r in rev_rooms_options
                        ]

                        chosen_rooms_str = st.multiselect(
                            "Выберите категории комнат",
                            options=list(rooms_options.keys()),
                            default=current_rooms_str,
                            key=f"elasticity_rooms_{config['id']}",
                        )
                        chosen_rooms_values = [
                            rooms_options[s] for s in chosen_rooms_str
                        ]
                        config["elasticity"]["rooms_list"] = chosen_rooms_values

                if st.button(
                    f"Удалить график {config['name']}",
                    key=f"delete_chart_{config['id']}",
                ):
                    st.session_state.graph_configs = [
                        cfg
                        for cfg in st.session_state.graph_configs
                        if cfg["id"] != config["id"]
                    ]
                    st.experimental_rerun()

                unique_key = f"chart_{config['id']}_{config['chart_type']}"

                if config["chart_type"] == "Кривая эластичности":
                    fig = build_elasticity_chart(
                        selected_groups=config["selected_groups"],
                        global_filtered_data=global_filtered_data,
                        group_configs=group_configs,
                        apartment_data=apartment_data,
                        area_min=config["elasticity"]["area_min"],
                        area_max=config["elasticity"]["area_max"],
                        split_parameter=config["elasticity"]["split"],
                        rooms_list=config["elasticity"]["rooms_list"],
                    )
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True, key=unique_key)
                    else:
                        st.info("Нет данных для построения кривой эластичности.")

                elif config["chart_type"] == "Гистограмма":
                    hist_column = config["histogram"]["column"]
                    log_transform = config["histogram"]["log_transform"]
                    remove_outliers = config["histogram"]["remove_outliers"]
                    lower_q = config["histogram"]["lower_q"]
                    upper_q = config["histogram"]["upper_q"]
                    normalize = config["histogram"]["normalize"]
                    chart_data = pd.DataFrame()

                    for g in config["selected_groups"]:
                        df_g = (
                            global_filtered_data.copy()
                            if g == "Глобальный"
                            else group_configs[g]["filtered_data"].copy()
                        )
                        if hist_column in df_g.columns:
                            col_data = df_g[hist_column].dropna()
                            if remove_outliers and not col_data.empty:
                                lb = col_data.quantile(lower_q)
                                ub = col_data.quantile(upper_q)
                                df_g = df_g[
                                    (df_g[hist_column] >= lb)
                                    & (df_g[hist_column] <= ub)
                                ]
                            if log_transform:
                                df_g = df_g[df_g[hist_column] > 0]
                                df_g[hist_column] = np.log(df_g[hist_column])
                            df_g = df_g[[hist_column]].copy()
                            df_g["group"] = g
                            chart_data = pd.concat(
                                [chart_data, df_g], ignore_index=True
                            )

                    if not chart_data.empty:
                        color_list = [
                            (
                                "#FF0000"
                                if g == "Глобальный"
                                else group_configs[g]["vis"]["color"]
                            )
                            for g in config["selected_groups"]
                        ]
                        chart = build_histogram(
                            chart_data,
                            hist_column,
                            config["selected_groups"],
                            color_list,
                            normalize,
                            height=400,
                        )
                        st.plotly_chart(chart, use_container_width=True, key=unique_key)
                    else:
                        st.info("Нет данных для построения гистограммы.")

                elif config["chart_type"] == "Скатерплот":
                    x_col = config["scatter"]["x"]
                    y_col = config["scatter"]["y"]
                    normalize_scatter = config["scatter"]["normalize"]
                    chart_data = pd.DataFrame()
                    for g in config["selected_groups"]:
                        df_g = (
                            global_filtered_data.copy()
                            if g == "Глобальный"
                            else group_configs[g]["filtered_data"].copy()
                        )
                        if x_col in df_g.columns and y_col in df_g.columns:
                            df_g = df_g[[x_col, y_col]].copy()
                            if normalize_scatter:
                                if df_g[x_col].nunique() > 1:
                                    df_g[x_col] = df_g[x_col].rank(pct=True)
                                if df_g[y_col].nunique() > 1:
                                    df_g[y_col] = df_g[y_col].rank(pct=True)
                            df_g["group"] = g
                            chart_data = pd.concat(
                                [chart_data, df_g], ignore_index=True
                            )
                    if not chart_data.empty:
                        color_list = [
                            (
                                "#FF0000"
                                if g == "Глобальный"
                                else group_configs[g]["vis"]["color"]
                            )
                            for g in config["selected_groups"]
                        ]
                        chart = build_scatter(
                            chart_data,
                            x_col,
                            y_col,
                            config["selected_groups"],
                            color_list,
                            height=400,
                        )
                        st.plotly_chart(chart, use_container_width=True, key=unique_key)
                    else:
                        st.info("Нет данных для построения скатерплота.")

                elif config["chart_type"] == "Кривая выбытия":
                    chart = build_depletion_chart(
                        "depletion_curves.feather",
                        config["selected_groups"],
                        global_filtered_data,
                        group_configs,
                        show_individual=config["depletion"].get(
                            "show_individual", False
                        ),
                        height=400,
                    )
                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True, key=unique_key)
                    else:
                        st.info("Нет данных для построения кривой выбытия.")

                st.markdown(
                    "<hr style='border-top: 1px solid #ccc;'>", unsafe_allow_html=True
                )

        if st.button("Добавить график", key="add_chart_bottom"):
            st.session_state.chart_counter += 1
            default_selected_groups = ["Глобальный"] + (
                list(group_configs.keys()) if group_configs else []
            )
            new_config = {
                "id": st.session_state.chart_counter,
                "name": f"График {st.session_state.chart_counter}",
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
                "elasticity": {
                    "area_min": 0,
                    "area_max": 100,
                    "split": 1,
                    "rooms_list": [None, 0, 1, 2, 3],
                },
            }
            st.session_state.graph_configs.append(new_config)
            st.experimental_rerun()
