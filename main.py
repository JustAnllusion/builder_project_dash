import streamlit as st
st.set_page_config(
    page_title="Анализ данных городов",
    layout="wide",
    initial_sidebar_state="expanded"
)

from auth import login, logout

st.markdown("""
<style>
[data-testid="stSidebar"] > div:first-child {
    display: flex;
    flex-direction: column;
    height: 100vh;
}
div[data-testid="stVerticalBlock"] > .stButton {
    margin-top: auto;
}
</style>
""", unsafe_allow_html=True)

name, authentication_status, username = login()

if authentication_status:
    from modules.data_tab import render_data_tab
    from modules.map_tab import render_map_tab
    from modules.analysis_tab import render_analysis_tab
    from modules.prediction_tab import render_prediction_tab
    from modules.clustering_tab import render_clustering_tab
    from components.sidebar import render_sidebar
    from utils.load_data import load_city_data 

    with open("assets/styles.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    house_data = st.session_state.get("house_data", None)
    house_data, group_configs, selected_city = render_sidebar(house_data)
    st.title(f"Анализ данных для {selected_city}")

    city_mapping = {
        "Москва": "msk",
        "Екатеринбург": "ekb",
        "Новосибирск": "nsk",
        "Челябинск": "chb"
    }

    if selected_city not in city_mapping or city_mapping[selected_city] not in ["msk", "ekb"]:
        st.info("Данные для выбранного города пока не доступны, используются данные Москвы.")
        city_key = "msk"
        selected_city = "Москва"
    else:
        city_key = city_mapping[selected_city]

    if ("data_loaded" not in st.session_state) or ("active_city" not in st.session_state) or (st.session_state.active_city != selected_city):
        with st.spinner("Загружаем данные для выбранного города..."):
            city_data = load_city_data(city_key)
            st.session_state.house_data = city_data["house_data"]
            st.session_state.apartment_data = city_data["apartment_data"]
            st.session_state.active_city = selected_city
            st.session_state.data_loaded = True
        house_data = st.session_state.house_data
        apartment_data = st.session_state.apartment_data

    tabs = ["Данные", "Карта объектов", "Анализ", "Кластеризация"]

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = tabs[0]

    active_tab = st.radio(
        "tabs",
        options=tabs,
        horizontal=True,
        key="active_tab",
        label_visibility="hidden"
    )

    if active_tab == "Данные":
        render_data_tab(house_data, st.session_state.house_data, group_configs)
    elif active_tab == "Карта объектов":
        render_map_tab(house_data, st.session_state.house_data, group_configs)
    elif active_tab == "Анализ":
        render_analysis_tab(house_data, st.session_state.house_data, group_configs)
    elif active_tab == "Предсказание":
        render_prediction_tab()
    elif active_tab == "Кластеризация":
        render_clustering_tab(st.session_state.apartment_data,group_configs,city_key)

else:
    st.warning("Пожалуйста, введите имя пользователя и пароль")