import os
import pandas as pd
import numpy as np
from utils.utils import find_segment_for_elasticity, fit_hyperbolic_alpha

def precompute_elasticity_curves_all_steps(data_path, output_path):
    area_min = 0
    area_max = 100
    try:
        apartment_data = pd.read_feather(data_path)
    except Exception as e:
        print(f"Ошибка загрузки исходных данных: {e}")
        return
    results = []
    for house_id, group in apartment_data.groupby("house_id"):
        group = group.dropna(subset=["area", "price"])
        group = group[(group["area"] >= area_min) & (group["area"] <= area_max)]
        if group.empty:
            continue
        for split_parameter in range(1, 6):
            temp_df = group.copy()
            temp_df["area_seg"] = temp_df["area"].apply(
                lambda x: find_segment_for_elasticity(x, area_min, area_max, split_parameter)
            )
            seg_group = temp_df.groupby("area_seg")
            seg_mean = seg_group["price"].mean().sort_index()
            if seg_mean.empty:
                continue
            first_val = seg_mean.iloc[0]
            if first_val == 0:
                norm_curve = seg_mean.copy()
            else:
                norm_curve = seg_mean / first_val
            alpha = fit_hyperbolic_alpha(norm_curve)
            hyper_curve = pd.Series(
                (seg_mean.index[0] ** alpha) / (seg_mean.index ** alpha),
                index=seg_mean.index
            )
            size_series = seg_group.size()
            for area_seg in seg_mean.index:
                norm_val = norm_curve.loc[area_seg]
                hyper_val = hyper_curve.loc[area_seg]
                deals_count = size_series.loc[area_seg]
                results.append({
                    "house_id": house_id,
                    "split_parameter": split_parameter,
                    "area_seg": area_seg,
                    "norm_curve": norm_val,
                    "hyper_curve": hyper_val,
                    "deals_count": deals_count,
                })
    precomputed_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    precomputed_df.to_feather(output_path)
    print(f"Предвычисленные данные для кривых эластичности (1..5) сохранены в {output_path}")

if __name__ == '__main__':
    cities = ["msk", "ekb"] 
    for city in cities:
        data_path = os.path.join("data", "regions", city, "market_deals", f"{city}_prep.feather")
        output_path = os.path.join("data", "regions", city, "market_deals", "cache", "elasticity_curves_precomputed.feather")
        print(f"Обработка города: {city}")
        precompute_elasticity_curves_all_steps(data_path, output_path)