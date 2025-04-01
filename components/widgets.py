import streamlit as st


from utils.utils import get_top_categories


def get_unique_widget_key(prefix="widget"):
    if "unique_widget_key_counter" not in st.session_state:
        st.session_state.unique_widget_key_counter = 0
    st.session_state.unique_widget_key_counter += 1
    return f"{prefix}_{st.session_state.unique_widget_key_counter}"


def numeric_filter_widget(col, data, key_prefix):
    if data.empty:
        return (0.0, 0.0)

    min_val = float(data.min())
    max_val = float(data.max())
    if min_val == max_val:
        st.write(f"Столбец «{col}» имеет константное значение: {min_val}.")
        return (min_val, max_val)

    use_quantile = st.checkbox(
        f"Фильтровать по квантилям для «{col}»", key=f"{key_prefix}_quantile_{col}"
    )

    if use_quantile:
        quantile_range = st.slider(
            f"Выберите квантильный диапазон для «{col}»",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
            key=f"{key_prefix}_quantile_range_{col}",
        )
        lower_q, upper_q = quantile_range
        lower_bound = float(data.quantile(lower_q))
        upper_bound = float(data.quantile(upper_q))
        st.info(
            f"Фильтрация: [{lower_q:.2f} → {upper_q:.2f}] (соответствует [{lower_bound:.2f} → {upper_bound:.2f}])"
        )
        return (lower_bound, upper_bound)

    else:
        return st.slider(
            f"Фильтр для «{col}»",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            key=f"{key_prefix}_{col}",
        )


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
