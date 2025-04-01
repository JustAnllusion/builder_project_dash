import json
import pandas as pd
import streamlit as st

from components.widgets import numeric_filter_widget, categorical_filter_widget
from utils.utils import compute_smart_group_name, get_top_categories, apply_filters


def render_sidebar(house_data: pd.DataFrame):
    if "dynamic_groups" not in st.session_state:
        st.session_state.dynamic_groups = []
    if "processed_json_files" not in st.session_state:
        st.session_state.processed_json_files = []

    st.sidebar.markdown(
        "<div class='sidebar-header'>Фильтрация</div>", unsafe_allow_html=True
    )

    with st.sidebar.expander("Глобальная фильтрация", expanded=True):
        st.write("Выберите признаки для фильтрации по всему набору данных:")
        selected_filter_columns = st.multiselect(
            "Признаки для фильтрации",
            options=list(house_data.columns),
            default=[],
            key="global_filter_cols",
        )
        global_filters = {}
        for col in selected_filter_columns:
            if pd.api.types.is_numeric_dtype(house_data[col]):
                col_data = house_data[col].dropna()
                global_filters[col] = numeric_filter_widget(
                    col, col_data, key_prefix="global_filter"
                )
            else:
                col_data = house_data[col].dropna()
                global_filters[col] = categorical_filter_widget(
                    col, col_data, key_prefix="global_filter"
                )

    global_filtered_data = apply_filters(house_data, global_filters)
    st.session_state["global_filters"] = global_filters

    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div class='sidebar-header'>Группы</div>", unsafe_allow_html=True
    )

    if st.sidebar.button("Создать новую группу", key="create_group"):
        new_group = {
            "group_name": f"Группа {len(st.session_state.dynamic_groups) + 1}",
            "group_color": "#0000FF",
            "selected_filter_columns": [],
            "column_filters": {},
            "filtered_data": house_data.copy(),
            "is_static": False,
            "base_data": house_data.copy(),
            "vis": {"color": "#0000FF", "opacity": 200, "radius": 50, "show": True},
        }
        st.session_state.dynamic_groups.append(new_group)
        st.rerun()

    with st.sidebar.expander("Загрузить группы из JSON", expanded=False):
        st.info(
            "Поддерживаемые форматы JSON:\n"
            '- Формат 1: Список ID (например, `["id1", "id2"]`).\n'
            '- Формат 2: Список списков ID (например, `[["id1", "id2"], ["id3", "id4"]]`).\n'
            "- Формат 3: Словарь, где ключи – имена групп, а значения – списки ID."
        )
        json_file = st.file_uploader(
            "Выберите JSON файл", type=["json"], key="json_group"
        )
        if (
            json_file is not None
            and json_file.name not in st.session_state.processed_json_files
        ):
            try:
                json_data = json.load(json_file)
                if isinstance(json_data, dict):
                    groups_from_json = json_data
                elif isinstance(json_data, list):
                    groups_from_json = {}
                    if all(isinstance(elem, list) for elem in json_data):
                        for idx, id_list in enumerate(json_data, start=1):
                            group_name = (
                                f"Группа {len(st.session_state.dynamic_groups) + 1}"
                            )
                            groups_from_json[group_name] = id_list
                    else:
                        group_name = (
                            f"Группа {len(st.session_state.dynamic_groups) + 1}"
                        )
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
                        "vis": {
                            "color": group_color,
                            "opacity": 200,
                            "radius": 50,
                            "show": True,
                        },
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
        for idx, group in enumerate(st.session_state.dynamic_groups.copy()):
            with st.sidebar.expander(f"{group['group_name']}", expanded=False):
                if st.button("Удалить группу", key=f"del_group_{idx}"):
                    st.session_state.dynamic_groups.remove(group)
                    st.rerun()

                smart_placeholder = compute_smart_group_name(group)
                new_name = st.text_input(
                    "Название группы",
                    value=group["group_name"],
                    placeholder=smart_placeholder,
                    key=f"group_name_{idx}",
                )
                group["group_name"] = new_name.strip() or smart_placeholder

                if group.get("is_static", False):
                    st.markdown("*Статическая группа (импортирована из JSON)*")

                group["group_color"] = st.color_picker(
                    "Цвет группы", value=group["group_color"], key=f"group_color_{idx}"
                )
                group["selected_filter_columns"] = st.multiselect(
                    "Признаки для фильтрации",
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
                        col_data = group["base_data"][col].dropna()
                        column_filters[col] = categorical_filter_widget(
                            col, col_data, key_prefix=f"group_filter_{idx}"
                        )

                group["column_filters"] = column_filters
                filtered_df = apply_filters(group["base_data"], column_filters)
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
    return st.session_state.get("global_filtered_data", house_data), group_configs
