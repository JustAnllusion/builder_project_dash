import streamlit as st
import numpy as np


from utils.utils import get_top_categories


def get_unique_widget_key(prefix="widget"):
    if "unique_widget_key_counter" not in st.session_state:
        st.session_state.unique_widget_key_counter = 0
    st.session_state.unique_widget_key_counter += 1
    return f"{prefix}_{st.session_state.unique_widget_key_counter}"


def numeric_filter_widget(col, data, key_prefix):
    # Обработка пустых данных
    if data.empty:
        return (0.0, 0.0)

    # Границы исходных значений
    min_val = float(data.min())
    max_val = float(data.max())

    # Выбор метода фильтрации
    use_quantile = st.checkbox(
        f"Фильтровать по квантилям для «{col}»",
        key=f"{key_prefix}_{col}_quant"
    )

    if use_quantile:
        # Поля ввода для квантилей (0–1)
        col_qmin, col_qmax = st.columns(2)
        with col_qmin:
            q_low = st.number_input(
                "",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                key=f"{key_prefix}_{col}_qlow"
            )
            st.caption("Нижний квантиль")
        with col_qmax:
            q_high = st.number_input(
                "",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.01,
                key=f"{key_prefix}_{col}_qhigh"
            )
            st.caption("Верхний квантиль")
        low = float(data.quantile(q_low))
        high = float(data.quantile(q_high))
    else:
        # Поля ввода для точной установки значений
        col_min, col_max = st.columns(2)
        with col_min:
            low = st.number_input(
                "",
                min_value=min_val,
                max_value=max_val,
                value=min_val,
                key=f"{key_prefix}_{col}_min"
            )
            st.caption(f"Мин. для «{col}»")
        with col_max:
            high = st.number_input(
                "",
                min_value=min_val,
                max_value=max_val,
                value=max_val,
                key=f"{key_prefix}_{col}_max"
            )
            st.caption(f"Макс. для «{col}»")

    # Разделитель между фильтрами
    return (low, high)


def categorical_filter_widget(col, data, key_prefix="filter"):
    all_categories = sorted(data.dropna().unique())
    popular_categories = get_top_categories(data, col, top_n=5)
    st.write(f"Популярные варианты для «{col}»: {', '.join(popular_categories)}")

    return st.multiselect(
        f"Выберите значения для «{col}»",
        options=all_categories,
        default=[],
        key=f"{key_prefix}_{col}",
    )


def safe_multiselect(label, options, default, key):
    safe_default = [x for x in default if x in options]
    if not safe_default:
        safe_default = options
    return st.multiselect(label, options=options, default=safe_default, key=key)
