import streamlit as st
import pandas as pd
import numpy as np

from utils.charts import build_histogram, build_scatter, build_depletion_chart
from components.widgets import safe_multiselect
from utils.elasticity import get_area_elasticity_by_house
import plotly.graph_objects as go

def render_analysis_tab(global_filtered_data: pd.DataFrame, 
                        house_data: pd.DataFrame, 
                        group_configs: dict,
                        apartment_data: pd.DataFrame):
    st.markdown("<div class='section-header'>Анализ данных</div>", unsafe_allow_html=True)
    analysis_mode_tabs = st.tabs(["Индивидуальное построение", "Множественное построение"])
    
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
                "selected_groups": ["Глобальный"] + (list(group_configs.keys()) if group_configs else []),
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
                # Настройки для кривой эластичности
                "elasticity": {
                    "area_min": None,
                    "area_max": None,
                    "discounting_mode": "trend",
                }
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
                    chart_options = ["Гистограмма", "Скатерплот", "Кривая выбытия", "Кривая эластичности"]
                    if config["chart_type"] not in chart_options:
                        config["chart_type"] = "Гистограмма"
                    config["chart_type"] = st.selectbox(
                        "Тип графика",
                        options=chart_options,
                        index=chart_options.index(config["chart_type"]),
                        key=f"chart_type_{config['id']}",
                    )
                available_groups = ["Глобальный"] + (list(group_configs.keys()) if group_configs else [])
                safe_default = [g for g in config.get("selected_groups", available_groups) if g in available_groups]
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
                            df = global_filtered_data.copy() if g == "Глобальный" else group_configs[g]["filtered_data"].copy()
                            numeric_cols.update(df.select_dtypes(include=["number"]).columns.tolist())
                        numeric_cols = sorted(list(numeric_cols))
                        default_col = "mean_price" if "mean_price" in numeric_cols else (numeric_cols[0] if numeric_cols else "")
                        config["histogram"]["column"] = st.selectbox(
                            "Признак для гистограммы",
                            options=numeric_cols,
                            index=(numeric_cols.index(default_col) if default_col in numeric_cols else 0),
                            key=f"hist_column_{config['id']}",
                        )
                        config["histogram"]["log_transform"] = st.checkbox("Логарифмировать", value=config["histogram"].get("log_transform", False), key=f"hist_log_{config['id']}")
                        config["histogram"]["remove_outliers"] = st.checkbox("Удалить выбросы", value=config["histogram"].get("remove_outliers", False), key=f"hist_outliers_{config['id']}")
                        if config["histogram"].get("remove_outliers", False):
                            config["histogram"]["lower_q"] = st.slider("Нижний квантиль", min_value=0.0, max_value=0.3, value=config["histogram"].get("lower_q", 0.05), step=0.01, key=f"hist_lq_{config['id']}")
                            config["histogram"]["upper_q"] = st.slider("Верхний квантиль", min_value=0.7, max_value=1.0, value=config["histogram"].get("upper_q", 0.95), step=0.01, key=f"hist_uq_{config['id']}")
                        config["histogram"]["normalize"] = st.checkbox("Нормировать гистограмму", value=config["histogram"].get("normalize", False), key=f"hist_normalize_{config['id']}")
                    elif config["chart_type"] == "Скатерплот":
                        st.markdown("Настройки скатерплота", unsafe_allow_html=True)
                        numeric_cols = set()
                        for g in config["selected_groups"]:
                            df = global_filtered_data.copy() if g == "Глобальный" else group_configs[g]["filtered_data"].copy()
                            numeric_cols.update(df.select_dtypes(include=["number"]).columns.tolist())
                        numeric_cols = sorted(list(numeric_cols))
                        default_x = "mean_price" if "mean_price" in numeric_cols else (numeric_cols[0] if numeric_cols else "")
                        default_y = "mean_selling_time" if "mean_selling_time" in numeric_cols else (numeric_cols[1] if len(numeric_cols) > 1 else "")
                        config["scatter"]["x"] = st.selectbox("Ось X", options=numeric_cols, index=(numeric_cols.index(default_x) if default_x in numeric_cols else 0), key=f"scatter_x_{config['id']}")
                        config["scatter"]["y"] = st.selectbox("Ось Y", options=numeric_cols, index=(numeric_cols.index(default_y) if default_y in numeric_cols else 0), key=f"scatter_y_{config['id']}")
                        config["scatter"]["normalize"] = st.checkbox("Нормировать (привести к квантилям)", value=config["scatter"].get("normalize", False), key=f"scatter_normalize_{config['id']}")
                    elif config["chart_type"] == "Кривая выбытия":
                        st.markdown("Настройки кривой выбытия", unsafe_allow_html=True)
                        config["depletion"]["show_individual"] = st.checkbox("Показать индивидуальные кривые", value=config["depletion"].get("show_individual", False), key=f"depletion_individual_{config['id']}")
                    elif config["chart_type"] == "Кривая эластичности":
                        st.markdown("Настройки кривой эластичности", unsafe_allow_html=True)
                        if "elasticity" not in config:
                            config["elasticity"] = {
                                "area_min": None,
                                "area_max": None,
                                "discounting_mode": "trend",
                            }
                        config["elasticity"]["area_min"] = st.number_input(
                            "Минимальная площадь (area_min)",
                            min_value=0.0,
                            value=float(config["elasticity"].get("area_min") or 0),
                            step=1.0,
                            key=f"elasticity_area_min_{config['id']}",
                        )
                        config["elasticity"]["area_max"] = st.number_input(
                            "Максимальная площадь (area_max)",
                            min_value=0.0,
                            value=float(config["elasticity"].get("area_max") or 100),
                            step=1.0,
                            key=f"elasticity_area_max_{config['id']}",
                        )
                        config["elasticity"]["discounting_mode"] = st.selectbox(
                            "Режим дисконтирования",
                            options=["trend", "actual", "retro"],
                            index=["trend", "actual", "retro"].index(config["elasticity"].get("discounting_mode", "trend")),
                            key=f"elasticity_discounting_mode_{config['id']}",
                        )
                
                if st.button(f"Удалить график {config['name']}", key=f"delete_chart_{config['id']}"):
                    st.session_state.graph_configs = [cfg for cfg in st.session_state.graph_configs if cfg["id"] != config["id"]]
                    st.experimental_rerun()
                
                unique_key = f"chart_{config['id']}_{config['chart_type']}"
                if config["chart_type"] == "Кривая эластичности":
                    area_min = config["elasticity"]["area_min"] or None
                    area_max = config["elasticity"]["area_max"] or None
                    discounting_mode = config["elasticity"]["discounting_mode"]
                    
                    combined_fig = go.Figure()
                    for g in config["selected_groups"]:
                        if g == "Глобальный":
                            group_house_ids = global_filtered_data["house_id"].unique()
                            df_for_elasticity = apartment_data[apartment_data["house_id"].isin(group_house_ids)].copy()
                        else:
                            group_house_ids = group_configs[g]["filtered_data"]["house_id"].unique()
                            df_for_elasticity = apartment_data[apartment_data["house_id"].isin(group_house_ids)].copy()
                        
                        if "area" not in df_for_elasticity.columns:
                            st.error(f"В данных для группы {g} отсутствует столбец 'area'.")
                            continue
                        
                        fig_elastic = get_area_elasticity_by_house(
                            deals=df_for_elasticity,
                            area_min=area_min,
                            area_max=area_max,
                            discounting_mode=discounting_mode,
                        )
                        for trace in fig_elastic.data:
                            trace.name = g
                            trace.line.color = "#FF0000" if g == "Глобальный" else group_configs[g]["vis"]["color"]
                            combined_fig.add_trace(trace)
                    
                    combined_fig.update_layout(height=400, title="Кривая эластичности")
                    st.plotly_chart(combined_fig, use_container_width=True, key=unique_key)
                
                elif config["chart_type"] == "Гистограмма":
                    hist_column = config["histogram"]["column"]
                    log_transform = config["histogram"]["log_transform"]
                    remove_outliers = config["histogram"]["remove_outliers"]
                    lower_q = config["histogram"]["lower_q"]
                    upper_q = config["histogram"]["upper_q"]
                    normalize = config["histogram"]["normalize"]
                    chart_data = pd.DataFrame()
                    for g in config["selected_groups"]:
                        df = global_filtered_data.copy() if g == "Глобальный" else group_configs[g]["filtered_data"].copy()
                        if hist_column in df.columns:
                            col_data = df[hist_column].dropna()
                            if remove_outliers and not col_data.empty:
                                lower_bound = col_data.quantile(lower_q)
                                upper_bound = col_data.quantile(upper_q)
                                df = df[(df[hist_column] >= lower_bound) & (df[hist_column] <= upper_bound)]
                            if log_transform:
                                df = df[df[hist_column] > 0]
                                df[hist_column] = np.log(df[hist_column])
                            df = df[[hist_column]].copy()
                            df["group"] = g
                            chart_data = pd.concat([chart_data, df], ignore_index=True)
                    if not chart_data.empty:
                        chart = build_histogram(chart_data, hist_column, config["selected_groups"], 
                                                ["#FF0000" if g == "Глобальный" else group_configs[g]["vis"]["color"] for g in config["selected_groups"]], normalize, height=400)
                        st.plotly_chart(chart, use_container_width=True, key=unique_key)
                    else:
                        st.info("Нет данных для построения гистограммы.")
                elif config["chart_type"] == "Скатерплот":
                    x_col = config["scatter"]["x"]
                    y_col = config["scatter"]["y"]
                    normalize_scatter = config["scatter"]["normalize"]
                    chart_data = pd.DataFrame()
                    for g in config["selected_groups"]:
                        df = global_filtered_data.copy() if g == "Глобальный" else group_configs[g]["filtered_data"].copy()
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
                        chart = build_scatter(chart_data, x_col, y_col, config["selected_groups"],
                                                ["#FF0000" if g == "Глобальный" else group_configs[g]["vis"]["color"] for g in config["selected_groups"]], height=400)
                        st.plotly_chart(chart, use_container_width=True, key=unique_key)
                    else:
                        st.info("Нет данных для построения скатерплота.")
                elif config["chart_type"] == "Кривая выбытия":
                    chart = build_depletion_chart("depletion_curves.feather", config["selected_groups"], global_filtered_data, group_configs, show_individual=config["depletion"].get("show_individual", False), height=400)
                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True, key=unique_key)
                    else:
                        st.info("Нет данных для построения кривой выбытия.")
                
                st.markdown("<hr style='border-top: 1px solid #ccc;'>", unsafe_allow_html=True)
        
        if st.button("Добавить график", key="add_chart_bottom"):
            st.session_state.chart_counter += 1
            default_selected_groups = ["Глобальный"] + (list(group_configs.keys()) if group_configs else [])
            new_config = {
                "id": st.session_state.chart_counter,
                "name": f"График {st.session_state.chart_counter}",
                "chart_type": "Гистограмма",
                "selected_groups": default_selected_groups,
                "histogram": {"column": "mean_price", "log_transform": False, "remove_outliers": False, "lower_q": 0.05, "upper_q": 0.95, "normalize": False},
                "scatter": {"x": "mean_price", "y": "mean_selling_time", "normalize": False},
                "depletion": {"show_individual": False},
                "elasticity": {"area_min": None, "area_max": None, "discounting_mode": "trend"},
            }
            st.session_state.graph_configs.append(new_config)
            st.experimental_rerun()
    

    with analysis_mode_tabs[1]:
        if "multi_graph_configs" not in st.session_state:
            st.session_state.multi_graph_configs = []
        if len(st.session_state.multi_graph_configs) == 0:
            default_config = {
                "selected_groups": ["Глобальный"] + (list(group_configs.keys()) if group_configs else []),
                "selected_chart_types": ["Гистограмма"],
                "params": {
                    "Гистограмма": {"column": "mean_price", "log_transform": False, "remove_outliers": False, "lower_q": 0.05, "upper_q": 0.95, "normalize": False},
                    "Скатерплот": {"x": "mean_price", "y": "mean_selling_time", "normalize": False},
                    "Кривая выбытия": {"show_individual": False},
                },
            }
            st.session_state.multi_graph_configs.append(default_config)
        for i, cfg in enumerate(st.session_state.multi_graph_configs):
            st.markdown(f"### Группа графиков {i+1}")
            available_groups = ["Глобальный"] + (list(group_configs.keys()) if group_configs else [])
            cfg["selected_groups"] = safe_multiselect("Выберите группы для анализа", options=available_groups, default=cfg.get("selected_groups", available_groups), key=f"multi_selected_groups_{i}")
            cfg["selected_chart_types"] = st.multiselect("Выберите типы графиков для построения", options=["Гистограмма", "Скатерплот", "Кривая выбытия"], default=cfg.get("selected_chart_types", ["Гистограмма"]), key=f"multi_selected_chart_types_{i}")
            if "Гистограмма" in cfg["selected_chart_types"]:
                with st.expander("Настройки гистограммы", expanded=False):
                    numeric_cols = set()
                    for g in cfg["selected_groups"]:
                        df = global_filtered_data if g == "Глобальный" else group_configs[g]["filtered_data"]
                        numeric_cols.update(df.select_dtypes(include=["number"]).columns.tolist())
                    numeric_cols = sorted(list(numeric_cols))
                    default_col = "mean_price" if "mean_price" in numeric_cols else (numeric_cols[0] if numeric_cols else "")
                    cfg.setdefault("params", {}).setdefault("Гистограмма", {})
                    cfg["params"]["Гистограмма"]["column"] = st.selectbox("Признак для гистограммы", options=numeric_cols, index=(numeric_cols.index(default_col) if default_col in numeric_cols else 0), key=f"multi_hist_column_{i}")
                    cfg["params"]["Гистограмма"]["log_transform"] = st.checkbox("Логарифмировать гистограмму", value=cfg["params"]["Гистограмма"].get("log_transform", False), key=f"multi_hist_log_{i}")
                    cfg["params"]["Гистограмма"]["remove_outliers"] = st.checkbox("Удалить выбросы в гистограмме", value=cfg["params"]["Гистограмма"].get("remove_outliers", False), key=f"multi_hist_outliers_{i}")
                    cfg["params"]["Гистограмма"]["lower_q"] = st.slider("Нижний квантиль (гистограмма)", min_value=0.0, max_value=0.3, value=cfg["params"]["Гистограмма"].get("lower_q", 0.05), step=0.01, key=f"multi_hist_lq_{i}")
                    cfg["params"]["Гистограмма"]["upper_q"] = st.slider("Верхний квантиль (гистограмма)", min_value=0.7, max_value=1.0, value=cfg["params"]["Гистограмма"].get("upper_q", 0.95), step=0.01, key=f"multi_hist_uq_{i}")
                    cfg["params"]["Гистограмма"]["normalize"] = st.checkbox("Нормировать гистограмму", value=cfg["params"]["Гистограмма"].get("normalize", False), key=f"multi_hist_norm_{i}")
            if "Скатерплот" in cfg["selected_chart_types"]:
                with st.expander("Настройки скатерплота", expanded=False):
                    numeric_cols = set()
                    for g in cfg["selected_groups"]:
                        df = global_filtered_data if g == "Глобальный" else group_configs[g]["filtered_data"]
                        numeric_cols.update(df.select_dtypes(include=["number"]).columns.tolist())
                    numeric_cols = sorted(list(numeric_cols))
                    default_x = "mean_price" if "mean_price" in numeric_cols else (numeric_cols[0] if numeric_cols else "")
                    default_y = "mean_selling_time" if "mean_selling_time" in numeric_cols else (numeric_cols[1] if len(numeric_cols) > 1 else "")
                    cfg.setdefault("params", {}).setdefault("Скатерплот", {})
                    cfg["params"]["Скатерплот"]["x"] = st.selectbox("Ось X (скатерплот)", options=numeric_cols, index=(numeric_cols.index(default_x) if default_x in numeric_cols else 0), key=f"multi_scatter_x_{i}")
                    cfg["params"]["Скатерплот"]["y"] = st.selectbox("Ось Y (скатерплот)", options=numeric_cols, index=(numeric_cols.index(default_y) if default_y in numeric_cols else 0), key=f"multi_scatter_y_{i}")
                    cfg["params"]["Скатерплот"]["normalize"] = st.checkbox("Нормировать скатерплот (привести к квантилям)", value=cfg["params"]["Скатерплот"].get("normalize", False), key=f"multi_scatter_norm_{i}")
            if "Кривая выбытия" in cfg["selected_chart_types"]:
                with st.expander("Настройки кривой выбытия", expanded=False):
                    cfg.setdefault("params", {}).setdefault("Кривая выбытия", {})
                    cfg["params"]["Кривая выбытия"]["show_individual"] = st.checkbox("Показать индивидуальные кривые", value=cfg["params"]["Кривая выбытия"].get("show_individual", False), key=f"multi_depletion_indiv_{i}")
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
                        df = global_filtered_data.copy() if g == "Глобальный" else group_configs[g]["filtered_data"].copy()
                        if hist_column in df.columns:
                            col_data = df[hist_column].dropna()
                            if remove_outliers and not col_data.empty:
                                lower_bound = col_data.quantile(lower_q)
                                upper_bound = col_data.quantile(upper_q)
                                df = df[(df[hist_column] >= lower_bound) & (df[hist_column] <= upper_bound)]
                            if log_transform:
                                df = df[df[hist_column] > 0]
                                df[hist_column] = np.log(df[hist_column])
                            df = df[[hist_column]].copy()
                            df["group"] = g
                            chart_data = pd.concat([chart_data, df], ignore_index=True)
                    if not chart_data.empty:
                        color_domain = cfg["selected_groups"]
                        color_range = ["#FF0000" if g == "Глобальный" else group_configs[g]["vis"]["color"] for g in cfg["selected_groups"]]
                        chart = build_histogram(chart_data, hist_column, color_domain, color_range, normalize, height=400)
                        unique_key = f"multi_chart_{i}_Гистограмма"
                        st.plotly_chart(chart, use_container_width=True, key=unique_key)
                    else:
                        st.info("Нет данных для построения гистограммы.")
                elif ct == "Скатерплот":
                    x_col = cfg["params"]["Скатерплот"]["x"]
                    y_col = cfg["params"]["Скатерплот"]["y"]
                    normalize_scatter = cfg["params"]["Скатерплот"]["normalize"]
                    chart_data = pd.DataFrame()
                    for g in cfg["selected_groups"]:
                        df = global_filtered_data.copy() if g == "Глобальный" else group_configs[g]["filtered_data"].copy()
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
                        color_range = ["#FF0000" if g == "Глобальный" else group_configs[g]["vis"]["color"] for g in cfg["selected_groups"]]
                        chart = build_scatter(chart_data, x_col, y_col, color_domain, color_range, height=400)
                        unique_key = f"multi_chart_{i}_Скатерплот"
                        st.plotly_chart(chart, use_container_width=True, key=unique_key)
                    else:
                        st.info("Нет данных для построения скатерплота.")
                elif ct == "Кривая выбытия":
                    chart = build_depletion_chart("depletion_curves.feather", cfg["selected_groups"], global_filtered_data, group_configs, show_individual=cfg["params"]["Кривая выбытия"].get("show_individual", False), height=400)
                    if chart is not None:
                        unique_key = f"multi_chart_{i}_Кривая_выбытия"
                        st.plotly_chart(chart, use_container_width=True, key=unique_key)
                    else:
                        st.info("Нет данных для построения кривой выбытия.")
            st.markdown("<hr style='border-top: 1px solid #ccc;'>", unsafe_allow_html=True)
            if st.button("Удалить группу", key=f"delete_multi_group_{i}"):
                st.session_state.multi_graph_configs.pop(i)
                st.experimental_rerun()
        if st.button("Добавить новую группу графиков", key="add_multi_group"):
            default_config = {
                "selected_groups": ["Глобальный"] + (list(group_configs.keys()) if group_configs else []),
                "selected_chart_types": ["Гистограмма"],
                "params": {
                    "Гистограмма": {"column": "mean_price", "log_transform": False, "remove_outliers": False, "lower_q": 0.05, "upper_q": 0.95, "normalize": False},
                    "Скатерплот": {"x": "mean_price", "y": "mean_selling_time", "normalize": False},
                    "Кривая выбытия": {"show_individual": False},
                },
            }
            st.session_state.multi_graph_configs.append(default_config)
            st.experimental_rerun()