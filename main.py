import streamlit as st

from components.sidebar import render_sidebar
from utils import load_data

st.set_page_config(
    page_title="Анализ данных Москвы", layout="wide", initial_sidebar_state="expanded"
)

with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


apartment_data = load_data("msk_prep.feather")
house_data = load_data("msk_apartment.feather")

global_filtered_data, group_configs = render_sidebar(house_data)

tab_data, tab_map, tab_analysis, tab_prediction, tab_clustering = st.tabs(
    ["Данные", "Карта объектов", "Анализ", "Предсказание", "Кластеризация"]
)

with tab_data:
    from modules.data_tab import render_data_tab

    render_data_tab(global_filtered_data, house_data, group_configs)

with tab_map:
    from modules.map_tab import render_map_tab

    render_map_tab(global_filtered_data, house_data, group_configs)

with tab_analysis:
    from modules.analysis_tab import render_analysis_tab

    render_analysis_tab(global_filtered_data, house_data, group_configs, apartment_data)

with tab_prediction:
    st.info("Раздел 'Предсказание' находится в разработке.")

with tab_clustering:
    st.info("Здесь будет кластеризация.")
