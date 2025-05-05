import streamlit as st
import pickle
import modules.figures as figures
import pandas as pd

def render_clustering_tab(apartment_data: pd.DataFrame,group_configs: dict):
    house_data_cluster = apartment_data.copy()  #(
    ir = figures.get_interest_rate(house_data_cluster, is_ml=False)
    house_data_cluster["discounting_price"] = figures.discounting(house_data_cluster, ir)
    house_data_cluster["discounting_price"] = house_data_cluster["discounting_price"] * ir[-1]
    house_data_cluster['total_price_discounted'] = house_data_cluster['area'] * house_data_cluster['discounting_price']

    with open('fig_clusterisation.pkl', 'wb') as f:
         pickle.dump(figures.make_new_clusterisation(house_data_cluster,[]),f)

    with open('fig_clusterisation.pkl', 'rb') as f:
        base_fig = pickle.load(f)
    
    if "voronoi_fig" not in st.session_state:
        st.session_state.voronoi_fig = base_fig
    
    st.markdown(
        "<div class='section-header'>Сегментация</div>", unsafe_allow_html=True
    )
    required_cols = {"house_id", "total_price_discounted", "discounting_price", "start_sales"}
    if not required_cols.issubset(house_data_cluster.columns):
        
        st.warning(f"Для корректной работы необходимы столбцы: {', '.join(required_cols)}")
    else:
        prefix = "new_tab_"  
    
        if not group_configs:
            st.warning("Нет настроенных групп для отображения.")
            st.plotly_chart(base_fig, use_container_width=True)
        else:
            for grp in group_configs.keys():
                key_radius = f"{prefix}grp_radius_{grp}"
                key_opacity = f"{prefix}grp_opacity_{grp}"
                if key_radius not in st.session_state:
                    st.session_state[key_radius] = group_configs[grp]["vis"]["radius"]
                if key_opacity not in st.session_state:
                    st.session_state[key_opacity] = group_configs[grp]["vis"]["opacity"]
            
            num_columns = max(1, len(group_configs))
            cols = st.columns(num_columns)
            
            toggles = {}
            for idx, grp in enumerate(group_configs.keys()):
                toggles[grp] = cols[idx].checkbox(grp, value=True, key=f"{prefix}toggle_{grp}")
                cols[idx].markdown(
                    f"<div style='width:20px;height:20px;background:{group_configs[grp]['vis']['color']};border-radius:3px;'></div>",
                    unsafe_allow_html=True,
                )
            
            selected_houses_data = []
            
            for grp, config in group_configs.items():
                if not toggles.get(grp, False):
                    continue
                    
                if "house_id" in config["filtered_data"].columns:
                    group_data = config["filtered_data"][["house_id", "start_sales"]].dropna(subset=["house_id"])
                    selected_houses_data.append(group_data)
            
            if selected_houses_data:
                all_selected_houses = pd.concat(selected_houses_data, ignore_index=True)
                
                if "start_sales" in all_selected_houses.columns:
                    all_selected_houses = all_selected_houses.sort_values(by="start_sales", ascending=False)
                
                unique_house_ids = all_selected_houses["house_id"].tolist()
                total_houses = len(unique_house_ids)
                
                min_houses_threshold = 50
                if total_houses > min_houses_threshold:
                    unique_house_ids = unique_house_ids[:min_houses_threshold]
                    st.warning(f"Выбрано слишком много домов ({total_houses}). Будут отображаться топ {min_houses_threshold} самых новых домов.")
                
                btn_cols = st.columns(2)
                
                if btn_cols[0].button("Обновить отображение точек"):
                    st.session_state.voronoi_fig = figures.add_points_to_voronoi(
                        base_fig, 
                        house_data_cluster, 
                        unique_house_ids, 
                        group_configs
                    )
                
                if btn_cols[1].button("Убрать отображение точек"):
                    st.session_state.voronoi_fig = base_fig
                
                st.plotly_chart(st.session_state.voronoi_fig, use_container_width=True)
            else:
                st.plotly_chart(st.session_state.base_fig, use_container_width=True)
                st.warning("Нет выбранных домов для отображения.")