import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"


# ============================================================
# LOAD DATA
# ============================================================

sensor = pd.read_excel(
    RAW_DIR / "sensordata2021_2022_2023.xlsx"
)

summary = pd.read_excel(
    RAW_DIR / "Summary_sensors.xlsx"
)

precip = pd.read_csv(
    RAW_DIR / "precipitation_212223.csv"
)

irrigation = pd.read_csv(
    RAW_DIR / "irrigation_212223.csv"
)

eto = pd.read_csv(
    RAW_DIR / "ETo_212223.csv"
)

soil = pd.read_csv(
    RAW_DIR / "soilsampmeas_212223.csv"
)


# ============================================================
# BASIC INFORMATION
# ============================================================

datasets = {
    "SENSOR": sensor,
    "SUMMARY": summary,
    "PRECIPITATION": precip,
    "IRRIGATION": irrigation,
    "ETO": eto,
    "SOIL": soil
}


print("\n" + "=" * 70)
print("BASIC DATASET INFORMATION")
print("=" * 70)

for name, df in datasets.items():

    print(f"\n{name}")
    print("-" * 50)

    print("Shape:", df.shape)

    print("Columns:")
    print(list(df.columns))

    print("Duplicate rows:", df.duplicated().sum())

    print("Missing values:")
    print(df.isna().sum().to_string())


# ============================================================
# DATE CONVERSION
# ============================================================

sensor["Datetime"] = pd.to_datetime(sensor["Datetime"])
sensor["Date"] = sensor["Datetime"].dt.normalize()

precip["Date"] = pd.to_datetime(
    precip["Date"],
    unit="D",
    origin="1899-12-30"
)

irrigation["Date"] = pd.to_datetime(
    irrigation["Date"],
    unit="D",
    origin="1899-12-30"
)

eto["Date"] = pd.to_datetime(
    eto["Date"],
    unit="D",
    origin="1899-12-30"
)

soil["Date"] = pd.to_datetime(
    soil["Date"],
    unit="D",
    origin="1899-12-30"
)

summary["planting_date"] = pd.to_datetime(
    summary["planting_date"],
    unit="D",
    origin="1899-12-30"
)


# ============================================================
# DATE RANGES
# ============================================================

print("\n" + "=" * 70)
print("DATE RANGES")
print("=" * 70)

date_ranges = {
    "Sensor": sensor["Date"],
    "Precipitation": precip["Date"],
    "Irrigation": irrigation["Date"],
    "ETo": eto["Date"],
    "Soil": soil["Date"]
}

for name, dates in date_ranges.items():

    print(
        f"{name:15s}: "
        f"{dates.min().date()}  ->  {dates.max().date()}"
    )


# ============================================================
# SENSOR IDS
# ============================================================

sensor_ids = set(sensor["Sensor"].astype(str))

precip_long = precip.melt(
    id_vars=["year", "Date"],
    var_name="Sensor",
    value_name="precipitation"
)

irrigation_long = irrigation.melt(
    id_vars=["year", "Date"],
    var_name="Sensor",
    value_name="irrigation"
)

eto_long = eto.melt(
    id_vars=["year", "Date"],
    var_name="Sensor",
    value_name="ETo"
)

precip_ids = set(precip_long["Sensor"].astype(str))
irrigation_ids = set(irrigation_long["Sensor"].astype(str))
eto_ids = set(eto_long["Sensor"].astype(str))
summary_ids = set(summary["sensor"].astype(str))
soil_ids = set(soil["sensor"].astype(str))


# ============================================================
# SENSOR INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SENSOR INFORMATION")
print("=" * 70)

print("\nSensor data:")
print("Number of sensors:", len(sensor_ids))
print(sorted(sensor_ids))

print("\nSummary sensors:")
print("Number of sensors:", len(summary_ids))
print(sorted(summary_ids))

print("\nSoil sample sensors:")
print("Number of sensors:", len(soil_ids))
print(sorted(soil_ids))


# ============================================================
# SENSOR COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("SENSOR COVERAGE")
print("=" * 70)

sensor_coverage = (
    sensor.groupby("Sensor")["Date"]
    .agg(
        first_date="min",
        last_date="max",
        observations="count",
        unique_days="nunique"
    )
    .sort_index()
)

print(sensor_coverage.to_string())


# ============================================================
# SENSOR ID DIFFERENCES
# ============================================================

print("\n" + "=" * 70)
print("SENSOR ID DIFFERENCES")
print("=" * 70)

print("\nSensor -> Precipitation:")
print(sensor_ids - precip_ids)

print("\nSensor -> Irrigation:")
print(sensor_ids - irrigation_ids)

print("\nSensor -> ETo:")
print(sensor_ids - eto_ids)

print("\nSensor -> Summary:")
print(sensor_ids - summary_ids)

print("\nSensor -> Soil:")
print(sensor_ids - soil_ids)

print("\nPrecipitation -> Sensor:")
print(precip_ids - sensor_ids)

print("\nIrrigation -> Sensor:")
print(irrigation_ids - sensor_ids)

print("\nETo -> Sensor:")
print(eto_ids - sensor_ids)


# ============================================================
# LONG DATASET SHAPES
# ============================================================

print("\n" + "=" * 70)
print("LONG DATASET SHAPES")
print("=" * 70)

print("Precipitation:", precip_long.shape)
print("Irrigation:", irrigation_long.shape)
print("ETo:", eto_long.shape)


# ============================================================
# LONG DATASET PREVIEWS
# ============================================================

print("\n" + "=" * 70)
print("PRECIPITATION LONG")
print("=" * 70)

print(precip_long.head())


print("\n" + "=" * 70)
print("IRRIGATION LONG")
print("=" * 70)

print(irrigation_long.head())


print("\n" + "=" * 70)
print("ETO LONG")
print("=" * 70)

print(eto_long.head())


# ============================================================
# SOIL SAMPLE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("SOIL SAMPLE CHECK")
print("=" * 70)

soil_duplicates = (
    soil.groupby(["Date", "sensor"])
    .size()
    .sort_values(ascending=False)
)

print(
    "Maximum soil measurements for same Date + Sensor:",
    soil_duplicates.max()
)

print("\nDate + Sensor combinations having multiple measurements:")

print(
    soil_duplicates[
        soil_duplicates > 1
    ].head(20)
)


print("\nInspection completed.")
# ============================================================
# ADDITIONAL DUPLICATE CHECKS
# ============================================================

print("\n" + "=" * 70)
print("ADDITIONAL DUPLICATE CHECKS")
print("=" * 70)

# Exact duplicate sensor rows
print(
    "\nExact duplicate sensor rows:",
    sensor.duplicated().sum()
)

# Same Sensor + Datetime
datetime_duplicates = (
    sensor.groupby(["Sensor", "Datetime"])
    .size()
    .sort_values(ascending=False)
)

print(
    "\nMaximum observations for the same Sensor + Datetime:",
    datetime_duplicates.max()
)

print("\nSensor + Datetime combinations with >1 observation:")

print(
    datetime_duplicates[
        datetime_duplicates > 1
    ].head(20)
)


# ============================================================
# SOIL SAMPLE VALUES
# ============================================================

print("\n" + "=" * 70)
print("SOIL SAMPLE EXAMPLES")
print("=" * 70)

example_soil = soil[
    (soil["sensor"] == "C535CA") &
    (soil["Date"] == pd.Timestamp("2021-06-18"))
]

print(example_soil.to_string(index=False))