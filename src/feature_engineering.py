import numpy as np

def create_features(df):
    df = df.copy()

    df["Floor_No"] = np.where(
        df["Floor_No"] > df["Total_Floors"],
        df["Total_Floors"],
        df["Floor_No"]
    )

    df["Age_of_Property"] = 2025 - df["Year_Built"]

    df["Price_per_SqFt"] = (
        df["Price_in_Lakhs"] * 100000
    ) / (df["Size_in_SqFt"] + 1)

    df["Infra_Score"] = df["Nearby_Schools"] + df["Nearby_Hospitals"]

    df["Quality_Score"] = (
        (df["BHK"] >= 3).astype(int)
        + (df["Age_of_Property"] < 20).astype(int)
    )

    df["Good_Investment"] = (
        (df["Price_per_SqFt"] < df["Price_per_SqFt"].median()).astype(int)
        + (df["Infra_Score"] >= df["Infra_Score"].median()).astype(int)
        + (df["Quality_Score"] >= 1).astype(int)
    ) >= 2

    df["Good_Investment"] = df["Good_Investment"].astype(int)

    df["Future_Price"] = df["Price_in_Lakhs"] * (1.05 + np.random.normal(0, 0.02, len(df)))

    return df