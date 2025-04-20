import os
import pandas as pd

def precompute_depletion_curves(apartment_data, house_data, convert_to_days=True, align_time_zero=True):
    curves = []
    apartment_data["contract_date"] = pd.to_datetime(apartment_data["contract_date"], errors="coerce")
    sales_by_house = apartment_data.groupby("house_id")
    for _, house in house_data.iterrows():
        house_id = house["house_id"]
        ndeals = house.get("ndeals", None)
        if ndeals is None or ndeals <= 0:
            continue
        try:
            house_sales = sales_by_house.get_group(house_id).copy()
        except KeyError:
            house_sales = pd.DataFrame()
        if house_sales.empty:
            if convert_to_days:
                df_curve = pd.DataFrame({"time": [0], "pct": [100], "house_id": [house_id]})
            else:
                continue
        else:
            house_sales = house_sales.dropna(subset=["contract_date"]).sort_values("contract_date")
            if house_sales.empty:
                continue
            if convert_to_days:
                if align_time_zero:
                    base_time = house_sales["contract_date"].min()
                    house_sales["time"] = (house_sales["contract_date"] - base_time).dt.days
                else:
                    global_ref = apartment_data["contract_date"].min()
                    house_sales["time"] = (house_sales["contract_date"] - global_ref).dt.days
            else:
                house_sales["time"] = house_sales["contract_date"]
            grouped = house_sales.groupby("time").size().reset_index(name="sales_count").sort_values("time")
            start_time = 0 if (convert_to_days and align_time_zero) else grouped["time"].min()
            times = []
            percentages = []
            if not grouped.empty and grouped["time"].iloc[0] != start_time:
                times.append(start_time)
                percentages.append(100)
            cumulative = 0
            for _, row in grouped.iterrows():
                t = row["time"]
                cumulative += row["sales_count"]
                pct = 100 * (ndeals - cumulative) / ndeals
                times.append(t)
                percentages.append(pct)
            df_curve = pd.DataFrame({"time": times, "pct": percentages, "house_id": [house_id] * len(times)})
        curves.append(df_curve)
    return pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()

if __name__ == "__main__":
    cities = ["msk", "ekb"] 
    for city in cities:
        input_dir = os.path.join("data", "regions", city, "market_deals")
        output_dir = os.path.join("data", "regions", city, "market_deals", "cache")
        os.makedirs(output_dir, exist_ok=True)
        apartment_path = os.path.join(input_dir, f"{city}_prep.feather")
        house_path = os.path.join(input_dir, f"{city}_apartment.feather")
        try:
            apartment_data = pd.read_feather(apartment_path)
            house_data = pd.read_feather(house_path)
        except Exception as e:
            print(f"Ошибка загрузки данных для города {city}: {e}")
            continue
        depletion_curves = precompute_depletion_curves(apartment_data, house_data, convert_to_days=True, align_time_zero=True)
        output_path = os.path.join(output_dir, "depletion_curves.feather")
        depletion_curves.to_feather(output_path)
        print(f"Кривые выбытия сохранены для города {city} в: {output_path}")