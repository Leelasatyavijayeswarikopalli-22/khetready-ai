import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD RAW DATA
# ============================================================

print("=" * 70)
print("LOADING RAW DATA")
print("=" * 70)

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


print("Sensor:", sensor.shape)
print("Summary:", summary.shape)
print("Precipitation:", precip.shape)
print("Irrigation:", irrigation.shape)
print("ETo:", eto.shape)
print("Soil:", soil.shape)


# ============================================================
# 1. DATE CLEANING
# ============================================================

print("\n" + "=" * 70)
print("DATE CLEANING")
print("=" * 70)


# Sensor datetime
sensor["Datetime"] = pd.to_datetime(
    sensor["Datetime"],
    errors="coerce"
)

sensor["Date"] = sensor["Datetime"].dt.normalize()


# Excel serial dates
precip["Date"] = pd.to_datetime(
    precip["Date"],
    unit="D",
    origin="1899-12-30",
    errors="coerce"
)

irrigation["Date"] = pd.to_datetime(
    irrigation["Date"],
    unit="D",
    origin="1899-12-30",
    errors="coerce"
)

eto["Date"] = pd.to_datetime(
    eto["Date"],
    unit="D",
    origin="1899-12-30",
    errors="coerce"
)

soil["Date"] = pd.to_datetime(
    soil["Date"],
    unit="D",
    origin="1899-12-30",
    errors="coerce"
)

summary["planting_date"] = pd.to_datetime(
    summary["planting_date"],
    unit="D",
    origin="1899-12-30",
    errors="coerce"
)


# Check invalid dates
print("Invalid sensor dates:", sensor["Date"].isna().sum())
print("Invalid precipitation dates:", precip["Date"].isna().sum())
print("Invalid irrigation dates:", irrigation["Date"].isna().sum())
print("Invalid ETo dates:", eto["Date"].isna().sum())
print("Invalid soil dates:", soil["Date"].isna().sum())
print("Invalid planting dates:", summary["planting_date"].isna().sum())


# ============================================================
# 2. SENSOR DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("SENSOR DATA CLEANING")
print("=" * 70)


before_duplicates = len(sensor)

# Remove exact duplicate rows
sensor = sensor.drop_duplicates().copy()

after_duplicates = len(sensor)

print(
    "Exact duplicate rows removed:",
    before_duplicates - after_duplicates
)

print("Sensor rows after cleaning:", len(sensor))


# ============================================================
# 3. DAILY SENSOR AGGREGATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING DAILY SENSOR DATA")
print("=" * 70)


daily_sensor = (
    sensor
    .groupby(["Date", "Sensor"])
    .agg(
        {
            "Adc0 (mV)": "mean",
            "Adc1 (mV)": "mean",
            "Adc2 (mV)": "mean",
            "vwc0 (m3/m3)": "mean",
            "vwc1 (m3/m3)": "mean",
            "vwc2 (m3/m3)": "mean",
            "pluvio": lambda x: x.sum(min_count=1),
            "temp": "mean",
            "Datetime": "count"
        }
    )
    .reset_index()
)


daily_sensor = daily_sensor.rename(
    columns={
        "Datetime": "sensor_observation_count"
    }
)


print("Daily sensor shape:", daily_sensor.shape)

print("\nDaily sensor preview:")
print(daily_sensor.head())


# ============================================================
# 4. CONVERT ENVIRONMENTAL DATA TO LONG FORMAT
# ============================================================

print("\n" + "=" * 70)
print("CONVERTING ENVIRONMENTAL DATA TO LONG FORMAT")
print("=" * 70)


# ------------------------------------------------------------
# Precipitation
# ------------------------------------------------------------

precip_long = precip.melt(
    id_vars=["year", "Date"],
    var_name="Sensor",
    value_name="precipitation"
)

# Remove unnecessary year column
precip_long = precip_long[
    ["Date", "Sensor", "precipitation"]
].copy()


# ------------------------------------------------------------
# Irrigation
# ------------------------------------------------------------

irrigation_long = irrigation.melt(
    id_vars=["year", "Date"],
    var_name="Sensor",
    value_name="irrigation"
)

irrigation_long = irrigation_long[
    ["Date", "Sensor", "irrigation"]
].copy()


# ------------------------------------------------------------
# ETo
# ------------------------------------------------------------

eto_long = eto.melt(
    id_vars=["year", "Date"],
    var_name="Sensor",
    value_name="ETo"
)

eto_long = eto_long[
    ["Date", "Sensor", "ETo"]
].copy()


# Convert environmental values to numeric
precip_long["precipitation"] = pd.to_numeric(
    precip_long["precipitation"],
    errors="coerce"
)

irrigation_long["irrigation"] = pd.to_numeric(
    irrigation_long["irrigation"],
    errors="coerce"
)

eto_long["ETo"] = pd.to_numeric(
    eto_long["ETo"],
    errors="coerce"
)


print("Precipitation long:", precip_long.shape)
print("Irrigation long:", irrigation_long.shape)
print("ETo long:", eto_long.shape)


# ============================================================
# 5. CHECK ENVIRONMENTAL DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING ENVIRONMENTAL DUPLICATES")
print("=" * 70)


def check_duplicate_keys(df, name):

    duplicates = df.duplicated(
        subset=["Date", "Sensor"]
    ).sum()

    print(f"{name} duplicate Date + Sensor rows:", duplicates)

    if duplicates > 0:

        duplicate_rows = (
            df[df.duplicated(
                subset=["Date", "Sensor"],
                keep=False
            )]
            .sort_values(["Date", "Sensor"])
        )

        print(duplicate_rows.head(20))

        raise ValueError(
            f"{name} contains duplicate Date + Sensor combinations."
        )


check_duplicate_keys(
    precip_long,
    "Precipitation"
)

check_duplicate_keys(
    irrigation_long,
    "Irrigation"
)

check_duplicate_keys(
    eto_long,
    "ETo"
)


# ============================================================
# 6. AGGREGATE SOIL SAMPLES
# ============================================================

print("\n" + "=" * 70)
print("AGGREGATING SOIL SAMPLES")
print("=" * 70)


soil_columns = [
    "0_30(grav%)",
    "30_60(grav%)",
    "0_5(grav%)"
]

for col in soil_columns:

    soil[col] = pd.to_numeric(
        soil[col],
        errors="coerce"
    )


soil_daily = (
    soil
    .groupby(["Date", "sensor"])
    .agg(
        {
            "0_30(grav%)": ["mean", "std"],
            "30_60(grav%)": ["mean", "std"],
            "0_5(grav%)": ["mean", "std"],
        }
    )
    .reset_index()
)


# Flatten multi-level column names
soil_daily.columns = [
    "Date",
    "Sensor",
    "soil_0_30_mean",
    "soil_0_30_std",
    "soil_30_60_mean",
    "soil_30_60_std",
    "soil_0_5_mean",
    "soil_0_5_std"
]


# Number of physical samples taken that day
soil_count = (
    soil
    .groupby(["Date", "sensor"])
    .size()
    .reset_index(name="soil_sample_count")
)

soil_count = soil_count.rename(
    columns={"sensor": "Sensor"}
)


soil_daily = soil_daily.merge(
    soil_count,
    on=["Date", "Sensor"],
    how="left",
    validate="one_to_one"
)


print("Aggregated soil shape:", soil_daily.shape)

print("\nSoil preview:")
print(soil_daily.head())


# ============================================================
# 7. MERGE ENVIRONMENTAL DATA INTO DAILY SENSOR DATA
# ============================================================

print("\n" + "=" * 70)
print("MERGING ENVIRONMENTAL DATA")
print("=" * 70)


master = daily_sensor.copy()


# ------------------------------------------------------------
# Precipitation
# ------------------------------------------------------------

master = master.merge(
    precip_long,
    on=["Date", "Sensor"],
    how="left",
    validate="one_to_one"
)

print("After precipitation:", master.shape)


# ------------------------------------------------------------
# Irrigation
# ------------------------------------------------------------

master = master.merge(
    irrigation_long,
    on=["Date", "Sensor"],
    how="left",
    validate="one_to_one"
)

print("After irrigation:", master.shape)


# ------------------------------------------------------------
# ETo
# ------------------------------------------------------------

master = master.merge(
    eto_long,
    on=["Date", "Sensor"],
    how="left",
    validate="one_to_one"
)

print("After ETo:", master.shape)


# ------------------------------------------------------------
# Soil
# ------------------------------------------------------------

master = master.merge(
    soil_daily,
    on=["Date", "Sensor"],
    how="left",
    validate="one_to_one"
)

print("After soil:", master.shape)


# ============================================================
# 8. MERGE SENSOR METADATA
# ============================================================

print("\n" + "=" * 70)
print("MERGING SENSOR METADATA")
print("=" * 70)


# Rename sensor column so it matches master
summary = summary.rename(
    columns={
        "sensor": "Sensor"
    }
)


# Make sure year has the same type in both datasets
summary["year"] = pd.to_numeric(
    summary["year"],
    errors="coerce"
).astype("Int64")


# Master currently has Date.
# Create year from Date so metadata can be matched
# to the correct sensor-year.
master["year"] = master["Date"].dt.year.astype("Int64")


# ------------------------------------------------------------
# Check whether Sensor + year uniquely identifies metadata
# ------------------------------------------------------------

metadata_duplicates = summary.duplicated(
    subset=["Sensor", "year"]
).sum()

print(
    "Duplicate Sensor + year metadata rows:",
    metadata_duplicates
)


if metadata_duplicates > 0:

    duplicate_metadata = (
        summary[
            summary.duplicated(
                subset=["Sensor", "year"],
                keep=False
            )
        ]
        .sort_values(["Sensor", "year"])
    )

    duplicate_keys = (
        duplicate_metadata
        .groupby(["Sensor", "year"])
        .size()
        .reset_index(name="count")
    )

    print("\n" + "=" * 70)
    print("DUPLICATE SENSOR + YEAR METADATA")
    print("=" * 70)

    print(duplicate_metadata.to_string(index=False))

    print("\n" + "=" * 70)
    print("DUPLICATE KEYS")
    print("=" * 70)

    print(duplicate_keys.to_string(index=False))

    # Save them so we don't lose the information
    duplicate_metadata.to_csv(
        PROCESSED_DIR / "duplicate_sensor_metadata.csv",
        index=False
    )

    duplicate_keys.to_csv(
        PROCESSED_DIR / "duplicate_sensor_year_keys.csv",
        index=False
    )

    raise ValueError(
        "STOP: duplicate metadata saved in "
        "data/processed/duplicate_sensor_metadata.csv"
    )


# ------------------------------------------------------------
# Merge metadata
# ------------------------------------------------------------

master = master.merge(
    summary,
    on=["Sensor", "year"],
    how="left",
    validate="many_to_one"
)


print("After metadata:", master.shape)


# ============================================================
# 9. SORT MASTER DATASET
# ============================================================

master = master.sort_values(
    ["Sensor", "Date"]
).reset_index(drop=True)


# ============================================================
# 10. BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("MASTER DATASET VALIDATION")
print("=" * 70)


print("Final shape:", master.shape)

print(
    "Unique sensors:",
    master["Sensor"].nunique()
)

print(
    "Date range:",
    master["Date"].min(),
    "->",
    master["Date"].max()
)

print(
    "Duplicate Date + Sensor rows:",
    master.duplicated(
        subset=["Date", "Sensor"]
    ).sum()
)


# ============================================================
# 11. MISSING VALUE REPORT
# ============================================================

print("\nMissing values:")

missing = (
    master.isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing.to_string())


# ============================================================
# 12. SENSOR COVERAGE IN MASTER DATASET
# ============================================================

print("\n" + "=" * 70)
print("MASTER SENSOR COVERAGE")
print("=" * 70)


coverage = (
    master
    .groupby("Sensor")["Date"]
    .agg(
        first_date="min",
        last_date="max",
        days="nunique"
    )
)

print(coverage.to_string())


# ============================================================
# 13. SAVE
# ============================================================

output_file = PROCESSED_DIR / "master_daily.csv"

master.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"Master dataset saved to:\n{output_file}"
)