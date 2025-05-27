import pandas as pd
import streamlit as st
from utils.utils import download_ui
from utils.translations import rus_columns, st_dataframe


def render_data_tab(
    global_filtered_data: pd.DataFrame, house_data: pd.DataFrame, group_configs: dict
):
    st.markdown(
        "<div class='section-header'> Табличное представление данных</div>",
        unsafe_allow_html=True,
    )

    main_table = global_filtered_data.copy()
    total_rows = len(house_data)
    max_table_rows = len(main_table)
    default_rows = 100 if max_table_rows >= 100 else (max_table_rows if max_table_rows > 0 else 1)

    num_rows = st.number_input(
        "Количество строк для отображения",
        min_value=1,
        max_value=max_table_rows if max_table_rows > 0 else 1,
        value=default_rows,
        step=1,
        key="data_tab_num_rows",
    )

    percentage = (max_table_rows / total_rows * 100) if total_rows > 0 else 0

    st.markdown(f"""
    <div class="data-info-box">
        Найдено строк: <strong>{max_table_rows}</strong> из <strong>{total_rows}</strong> ({percentage:.2f}%)
    </div>
    """, unsafe_allow_html=True)

    visible_columns = [
        "project", "house_id", "developer", "class", "start_sales", "ndeals",
        "deals_sold", "mean_area", "mean_price", "mean_selling_time", "floor"
    ]
    available_columns = [col for col in visible_columns if col in main_table.columns]
    # Показ русских меток, но возвращаем английские ключи
    label_map = {rus_columns.get(col, col): col for col in available_columns}
    default_labels = [rus_columns.get(col, col) for col in available_columns]
    selected_labels = st.multiselect(
        "Выберите колонки для отображения",
        options=list(label_map.keys()),
        default=default_labels,
        key="data_tab_column_selector"
    )
    selected_columns = [label_map[label] for label in selected_labels]

    display_df = main_table[selected_columns].copy()

    for col in display_df.select_dtypes(include=["float", "int"]).columns:
        display_df[col] = display_df[col].round(2)

    if display_df.empty:
        st.info("Нет данных для отображения.")
    else:
        st_dataframe(display_df.head(num_rows), use_container_width=True, height=500)
        with st.container():
            st.markdown('<div class="download-panel">', unsafe_allow_html=True)
            download_ui(display_df, "global_data")
            st.markdown("</div>", unsafe_allow_html=True)

    if group_configs:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Группы</div>", unsafe_allow_html=True)

        for grp, config in group_configs.items():
            grp_df = config["filtered_data"]
            # Show group header and row count like the global table
            st.markdown(f"<div class='section-header'>{grp}</div>", unsafe_allow_html=True)
            rows_grp = len(grp_df)
            percentage_grp = (rows_grp / total_rows * 100) if total_rows > 0 else 0
            st.markdown(f"""
            <div class="data-info-box">
                Найдено строк: <strong>{rows_grp}</strong> из <strong>{total_rows}</strong> ({percentage_grp:.2f}%)
            </div>
            """, unsafe_allow_html=True)
            color = config['vis']['color']
            with st.expander(f"{grp}", expanded=False):
                available_columns_grp = [col for col in selected_columns if col in grp_df.columns]
                display_grp_df = grp_df[available_columns_grp].head(num_rows).copy()

                for col in display_grp_df.select_dtypes(include=["float", "int"]).columns:
                    display_grp_df[col] = display_grp_df[col].round(2)

                st_dataframe(display_grp_df, use_container_width=True, height=500)

                with st.container():
                    st.markdown('<div class="download-panel">', unsafe_allow_html=True)
                    download_ui(display_grp_df, grp)
                    st.markdown("</div>", unsafe_allow_html=True)
