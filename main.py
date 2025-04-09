import streamlit as st
st.set_page_config(
    page_title="Анализ данных Москвы",
    layout="wide",
    initial_sidebar_state="expanded"
)

from auth import login, logout

# CSS для сайдбара: делаем его флекс-контейнером на всю высоту,
# чтобы, если потребуется, элементы (например, кнопки) можно было «утопить» вниз.
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

# Выполняем авторизацию (если пользователь не залогинен – показывается форма логина)
name, authentication_status, username = login()

if authentication_status:
    # (Опционально) можно добавить кнопку "Выйти" где-нибудь в приложении, например, в боковой панели.
    # if st.button("Выйти"):
    #     logout()

    # Импорт модулей приложения (убедитесь, что данные модули существуют, либо замените на заглушки)
    from modules.data_tab import render_data_tab
    from modules.map_tab import render_map_tab
    from modules.analysis_tab import render_analysis_tab
    from modules.prediction_tab import render_prediction_tab
    from modules.clustering_tab import render_clustering_tab
    from components.sidebar import render_sidebar
    from utils.session_data import load_user_data

    # Подключаем дополнительные стили из файла assets/styles.css
    with open("assets/styles.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Загружаем данные, если они ещё не загружены
    if "data_loaded" not in st.session_state:
        st.session_state.apartment_data = load_user_data("apartment_data", "msk_prep.feather")
        st.session_state.house_data = load_user_data("house_data", "msk_apartment.feather")
        st.session_state.data_loaded = True

    # Вызываем сайдбар для выбора города и группировки
    house_data, group_configs, selected_city = render_sidebar(st.session_state.house_data)
    st.title(f"Анализ данных для {selected_city}")

    # Определяем список вкладок
    tabs = ["Данные", "Карта объектов", "Анализ", "Предсказание", "Кластеризация"]

    # Если активная вкладка ещё не установлена, инициализируем её
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = tabs[0]

    # Используем st.radio с ключом "active_tab" — состояние сохраняется автоматически
    active_tab = st.radio(
        "tabs",
        options=tabs,
        horizontal=True,
        key="active_tab",
        label_visibility="hidden"
    )

    # Рендер содержимого в зависимости от выбранной вкладки
    if active_tab == "Данные":
        render_data_tab(house_data, st.session_state.house_data, group_configs)
    elif active_tab == "Карта объектов":
        render_map_tab(house_data, st.session_state.house_data, group_configs)
    elif active_tab == "Анализ":
        render_analysis_tab(house_data, st.session_state.house_data, group_configs)
    elif active_tab == "Предсказание":
        render_prediction_tab()
    elif active_tab == "Кластеризация":
        render_clustering_tab()

else:
    st.warning("Пожалуйста, введите имя пользователя и пароль")
