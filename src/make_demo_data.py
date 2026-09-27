from pathlib import Path
import numpy as np
import pandas as pd

out = Path("data/raw/demo_environment.csv")
out.parent.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
dates = pd.date_range("2021-01-01", "2025-12-31", freq="D")
districts = ["Patna", "Gaya", "Nalanda", "Muzaffarpur", "Bhagalpur"]

rows = []
for district in districts:
    base = rng.uniform(20, 45)
    rain = rng.gamma(0.8, 8, len(dates))
    monsoon = dates.month.isin([6, 7, 8, 9])
    rain += monsoon * rng.gamma(1.5, 10, len(dates))
    rain[rng.random(len(dates)) < 0.70] *= 0.15

    sm = np.zeros(len(dates))
    sm[0] = np.clip(base + rain[0] * 0.15, 5, 70)
    for i in range(1, len(dates)):
        sm[i] = np.clip(0.96*sm[i-1] + 0.30*rain[i] + rng.normal(0, 1.5), 5, 80)

    rows.extend(zip(dates, [district]*len(dates), rain, sm))

df = pd.DataFrame(rows, columns=["date","district","rainfall_mm","soil_moisture"])
df.to_csv(out, index=False)
print(f"Created {out}: {len(df):,} rows")
