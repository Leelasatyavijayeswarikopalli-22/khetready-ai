import numpy as np
import pandas as pd

OPERATIONS = ["sowing","tillage","fertilization","spraying","harvesting"]
CROPS = ["rice","wheat","maize","pulses","vegetables"]
STAGES = ["pre_sowing","vegetative","flowering","maturity","harvest"]

def add_environment_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["district","date"])
    g = df.groupby("district", group_keys=False)

    for w in [1,3,7,14]:
        df[f"rainfall_{w}d"] = g["rainfall_mm"].transform(
            lambda s: s.rolling(w, min_periods=w).sum()
        )

    df["rainfall_3d_max"] = g["rainfall_mm"].transform(
        lambda s: s.rolling(3, min_periods=3).max()
    )
    df["rainfall_7d_avg"] = g["rainfall_mm"].transform(
        lambda s: s.rolling(7, min_periods=7).mean()
    )
    df["soil_moisture_change_1d"] = g["soil_moisture"].diff()
    df["soil_moisture_change_3d"] = g["soil_moisture"].diff(3)
    df["soil_moisture_3d_avg"] = g["soil_moisture"].transform(
        lambda s: s.rolling(3, min_periods=3).mean()
    )
    df["soil_moisture_7d_avg"] = g["soil_moisture"].transform(
        lambda s: s.rolling(7, min_periods=7).mean()
    )

    df["rain_soil_interaction"] = df["rainfall_3d"] * df["soil_moisture"]
    df["rain_to_soil_ratio"] = df["rainfall_3d"] / (df["soil_moisture"] + 1e-6)
    df["moisture_rain_change"] = df["soil_moisture_change_1d"] * df["rainfall_mm"]
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["monsoon_flag"] = df["month"].isin([6,7,8,9]).astype(int)
    df["season"] = np.select(
        [df["month"].isin([6,7,8,9]), df["month"].isin([10,11,12,1])],
        ["monsoon","post_monsoon_winter"],
        default="summer_pre_monsoon"
    )
    return df.dropna().reset_index(drop=True)

def add_operation_context(df, seed=42):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["operation_type"] = rng.choice(OPERATIONS, len(df))
    df["crop"] = rng.choice(CROPS, len(df))
    df["crop_stage"] = rng.choice(STAGES, len(df))
    df["field_area_acres"] = rng.uniform(1,20,len(df)).round(2)
    duration = {"sowing":5,"tillage":6,"fertilization":3,"spraying":2,"harvesting":8}
    df["operation_duration_hours"] = (
        df["operation_type"].map(duration) + rng.normal(0,0.7,len(df))
    ).clip(lower=1)
    return df

def build_features(df):
    return add_operation_context(add_environment_features(df))
