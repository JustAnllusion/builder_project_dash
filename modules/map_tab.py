import pandas as pd
import pydeck as pdk
import streamlit as st
from utils.utils import hex_to_rgba


def render_map_tab(
    global_filtered_data: pd.DataFrame, house_data: pd.DataFrame, group_configs: dict
):
    st.markdown(
        "<div class='section-header'>Интерактивная карта объектов</div>",
        unsafe_allow_html=True,
    )
    required_cols = {"latitude", "longitude", "house_id"}
    if not required_cols.issubset(house_data.columns):
        st.warning(f"Для карты необходимы столбцы: {', '.join(required_cols)}")
        return

    global_radius = 150
    global_opacity = 200
    global_vis_color = "#FF0000"

    for grp in group_configs.keys():
        key_radius = f"grp_radius_{grp}"
        key_opacity = f"grp_opacity_{grp}"
        if key_radius not in st.session_state:
            st.session_state[key_radius] = global_radius
        if key_opacity not in st.session_state:
            st.session_state[key_opacity] = global_opacity

    cols = st.columns(len(group_configs) + 1)
    toggle_global = cols[0].checkbox("Глобальный слой", value=True, key="toggle_global")
    cols[0].markdown(
        f"<div style='width:20px;height:20px;background:{global_vis_color};border-radius:3px;'></div>",
        unsafe_allow_html=True,
    )
    toggles = {"Глобальный": toggle_global}
    for idx, grp in enumerate(group_configs.keys(), start=1):
        toggles[grp] = cols[idx].checkbox(grp, value=True, key=f"toggle_{grp}")
        cols[idx].markdown(
            f"<div style='width:20px;height:20px;background:{group_configs[grp]['vis']['color']};border-radius:3px;'></div>",
            unsafe_allow_html=True,
        )

    layers = []
    points_for_center = pd.DataFrame()
    global_points = global_filtered_data.dropna(
        subset=["latitude", "longitude", "house_id"]
    ).copy()[["latitude", "longitude", "house_id", "project", "developer"]]
    if toggles.get("Глобальный", False) and not global_points.empty:
        global_layer = pdk.Layer(
            "ScatterplotLayer",
            data=global_points,
            get_position=["longitude", "latitude"],
            get_radius=global_radius,
            get_fill_color=hex_to_rgba(global_vis_color, global_opacity),
            pickable=True,
        )
        layers.append(global_layer)
        points_for_center = pd.concat(
            [points_for_center, global_points], ignore_index=True
        )

    for grp, config in group_configs.items():
        if not toggles.get(grp, False):
            continue
        grp_data = (
            config["filtered_data"]
            .dropna(subset=["latitude", "longitude", "house_id"])
            .copy()[["latitude", "longitude", "house_id", "project", "developer"]]
        )
        if not grp_data.empty:
            key_radius = f"grp_radius_{grp}"
            key_opacity = f"grp_opacity_{grp}"
            grp_layer = pdk.Layer(
                "ScatterplotLayer",
                data=grp_data,
                get_position=["longitude", "latitude"],
                get_radius=st.session_state.get(key_radius, global_radius),
                get_fill_color=hex_to_rgba(
                    config["vis"]["color"],
                    st.session_state.get(key_opacity, global_opacity),
                ),
                pickable=True,
            )
            layers.append(grp_layer)
            points_for_center = pd.concat(
                [points_for_center, grp_data], ignore_index=True
            )

    if not layers:
        st.warning("Нет данных для отображения на карте.")
        return

    if not points_for_center.empty:
        center_lat = points_for_center["latitude"].mean()
        center_lon = points_for_center["longitude"].mean()
    else:
        center_lat, center_lon = 55.75, 37.62

    view_state = pdk.ViewState(
        latitude=center_lat, longitude=center_lon, zoom=10, pitch=0
    )
    tooltip = {
        "html": "<b>ID: {house_id}</b><br/><i>Проект: {project}</i><br/><i>Застройщик: {developer}</i>",
        "style": {
            "backgroundColor": "rgba(0, 0, 0, 0.75)",
            "color": "white",
            "fontSize": "14px",
            "padding": "5px",
        },
    }
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )
    st.pydeck_chart(deck)
