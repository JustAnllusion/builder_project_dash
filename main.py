import streamlit as st
from modules.data_tab import render_data_tab
from modules.map_tab import render_map_tab
from modules.analysis_tab import render_analysis_tab
from modules.prediction_tab import render_prediction_tab
from modules.clustering_tab import render_clustering_tab
from components.sidebar import render_sidebar
from utils.session_data import load_user_data

st.set_page_config(
    page_title="Анализ данных Москвы", layout="wide", initial_sidebar_state="expanded"
)

with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "data_loaded" not in st.session_state:
    st.session_state.apartment_data = load_user_data(
        "apartment_data", "msk_prep.feather"
    )
    st.session_state.house_data = load_user_data("house_data", "msk_apartment.feather")
    st.session_state.data_loaded = True

st.session_state.global_filtered_data, st.session_state.group_configs = render_sidebar(
    st.session_state.house_data
)

tabs = ["Данные", "Карта объектов", "Анализ", "Предсказание", "Кластеризация"]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = tabs[0]


def update_tab():
    st.session_state.active_tab = st.session_state["tab_choice"]


active_tab = st.radio(
    "tabs",
    options=tabs,
    horizontal=True,
    index=tabs.index(st.session_state.active_tab),
    key="tab_choice",
    on_change=update_tab,
    label_visibility="hidden"
)

active_tab = st.session_state.get("tab_choice", tabs[0])

if active_tab == "Данные":
    render_data_tab(
        st.session_state.global_filtered_data,
        st.session_state.house_data,
        st.session_state.group_configs,
    )
elif active_tab == "Карта объектов":
    render_map_tab(
        st.session_state.global_filtered_data,
        st.session_state.house_data,
        st.session_state.group_configs,
    )
elif active_tab == "Анализ":
    render_analysis_tab(
        st.session_state.global_filtered_data,
        st.session_state.house_data,
        st.session_state.group_configs,
        st.session_state.apartment_data,
    )
elif active_tab == "Предсказание":
    render_prediction_tab()
elif active_tab == "Кластеризация":
    render_clustering_tab()
