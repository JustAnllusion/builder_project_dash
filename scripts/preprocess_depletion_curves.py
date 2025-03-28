import pandas as pd


def precompute_depletion_curves(
    apartment_data, house_data, convert_to_days=True, align_time_zero=True
):
    curves = []

    apartment_data["contract_date"] = pd.to_datetime(
        apartment_data["contract_date"], errors="coerce"
    )
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
                df_curve = pd.DataFrame(
                    {"time": [0], "pct": [100], "house_id": [house_id]}
                )
            else:
                continue
        else:
            house_sales = house_sales.dropna(subset=["contract_date"]).sort_values(
                "contract_date"
            )
            if house_sales.empty:
                continue

            if convert_to_days:
                if align_time_zero:
                    base_time = house_sales["contract_date"].min()
                    house_sales["time"] = (
                        house_sales["contract_date"] - base_time
                    ).dt.days
                else:
                    global_ref = apartment_data["contract_date"].min()
                    house_sales["time"] = (
                        house_sales["contract_date"] - global_ref
                    ).dt.days
            else:
                house_sales["time"] = house_sales["contract_date"]

            grouped = (
                house_sales.groupby("time")
                .size()
                .reset_index(name="sales_count")
                .sort_values("time")
            )
            if convert_to_days and align_time_zero:
                start_time = 0
            else:
                start_time = grouped["time"].min()

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

            df_curve = pd.DataFrame(
                {"time": times, "pct": percentages, "house_id": [house_id] * len(times)}
            )
        curves.append(df_curve)

    if curves:
        curves_df = pd.concat(curves, ignore_index=True)
    else:
        curves_df = pd.DataFrame()

    return curves_df


if __name__ == "__main__":
    apartment_data = pd.read_feather("msk_prep.feather")
    house_data = pd.read_feather("msk_apartment.feather")

    depletion_curves = precompute_depletion_curves(
        apartment_data, house_data, convert_to_days=True, align_time_zero=True
    )

    depletion_curves.to_feather("depletion_curves.feather")
    print(
        "Кривые выбытия успешно рассчитаны и сохранены в файле 'depletion_curves.feather'."
    )
