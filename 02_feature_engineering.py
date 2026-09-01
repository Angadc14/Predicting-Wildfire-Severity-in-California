"""
Feature engineering — county × season × year panel.
Target: severe_fires (count of ≥300-acre fires per county-season-year).
All lag/rolling features use only prior years to prevent data leakage.
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_PATH  = os.path.join(BASE_DIR, "..", "Data", "ca_fires_clean.csv")
OUT_DIR  = os.path.join(BASE_DIR, "Data")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_PATH, low_memory=False)
df = df.dropna(subset=["FIPS_NAME","SEASON","FIRE_YEAR","SEVERE","LATITUDE","LONGITUDE","FIRE_SIZE"])
df["FIRE_YEAR"] = df["FIRE_YEAR"].astype(int)
print(f"Loaded {len(df):,} fires | {int(df['FIRE_YEAR'].min())}–{int(df['FIRE_YEAR'].max())}")

# -- Aggregate to county × season × year panel ---------------------------------
panel = (df.groupby(["FIPS_NAME","SEASON","FIRE_YEAR"])
           .agg(total_fires   = ("FOD_ID",     "count"),
                severe_fires  = ("SEVERE",      "sum"),
                lat           = ("LATITUDE",    "mean"),
                lon           = ("LONGITUDE",   "mean"),
                mean_fire_size= ("FIRE_SIZE",   "mean"))
           .reset_index()
           .sort_values(["FIPS_NAME","SEASON","FIRE_YEAR"]))

panel["severe_rate"] = panel["severe_fires"] / panel["total_fires"]
print(f"Panel: {len(panel):,} rows | "
      f"{panel['FIPS_NAME'].nunique()} counties | "
      f"{panel['FIRE_YEAR'].min()}–{panel['FIRE_YEAR'].max()}")

# -- Lag features (grouped by county-season, shift to avoid leakage) -----------
g = panel.groupby(["FIPS_NAME","SEASON"])

panel["lag1_severe"]    = g["severe_fires"].shift(1)
panel["lag2_severe"]    = g["severe_fires"].shift(2)
panel["rolling3_severe"]= g["severe_fires"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())
panel["log_total_fires"]= np.log1p(panel["total_fires"])

# -- Cumulative historical severe rate (prior years only) ----------------------
panel["cum_severe"] = g["severe_fires"].cumsum().shift(1)
panel["cum_fires"]  = g["total_fires"].cumsum().shift(1)
panel["hist_severe_rate"] = (panel["cum_severe"] / panel["cum_fires"]).fillna(0)

# -- ENSO climate features (Jacob's table, sourced from NOAA/CPC) --------------
# JJA = June-July-August Niño 3.4 anomaly (fire-season window).
# Lagged scores capture the fuel-loading effect: wet El Niño years grow dense
# vegetation that dries and ignites in subsequent years.
ENSO_TABLE = {
    1990:('Neutral', 0.33), 1991:('Neutral', 0.73), 1992:('SE',    0.37),
    1993:('Neutral', 0.32), 1994:('ME',      0.44), 1995:('ML',   -0.24),
    1996:('Neutral',-0.27), 1997:('VSE',     1.60), 1998:('SL',   -0.78),
    1999:('SL',    -1.10),  2000:('WL',     -0.55), 2001:('Neutral',-0.08),
    2002:('ME',     0.79),  2003:('Neutral', 0.08), 2004:('WE',    0.47),
    2005:('WL',    -0.06),  2006:('WE',      0.10), 2007:('SL',   -0.56),
    2008:('WL',    -0.37),  2009:('ME',      0.45), 2010:('SL',   -1.05),
    2011:('ML',    -0.43),  2012:('Neutral', 0.30), 2013:('Neutral',-0.35),
    2014:('WE',     0.10),  2015:('VSE',     1.57), 2016:('WL',   -0.31),
    2017:('WL',     0.19),  2018:('WE',      0.14), 2019:('WE',    0.33),
    2020:('ML',    -0.36),  2021:('ML',     -0.35), 2022:('WL',   -0.76),
    2023:('SE',     1.12),  2024:('Neutral', 0.08), 2025:('Neutral', 0.08),
    2026:('Neutral', 0.00),
}
ENSO_SCORE = {'VSE':3,'SE':2,'ME':1,'WE':0.5,'Neutral':0,'WL':-0.5,'ML':-1,'SL':-2}

enso_rows = [{"FIRE_YEAR": yr, "enso_score": ENSO_SCORE[t], "enso_jja": jja}
             for yr,(t,jja) in sorted(ENSO_TABLE.items())]
enso_df = pd.DataFrame(enso_rows)
enso_df["enso_score_lag1"] = enso_df["enso_score"].shift(1).fillna(0)
enso_df["enso_score_lag2"] = enso_df["enso_score"].shift(2).fillna(0)

panel = panel.merge(enso_df[["FIRE_YEAR","enso_score","enso_jja",
                              "enso_score_lag1","enso_score_lag2"]],
                    on="FIRE_YEAR", how="left")

# -- Season dummies ------------------------------------------------------------
for s in ["Winter","Spring","Summer","Fall"]:
    panel[f"season_{s}"] = (panel["SEASON"] == s).astype(int)

# -- County label encoding (saved so forecast can reuse same mapping) ----------
le = LabelEncoder()
panel["county_enc"] = le.fit_transform(panel["FIPS_NAME"])

# -- Save ----------------------------------------------------------------------
FEATURES = ["lag1_severe","lag2_severe","rolling3_severe","hist_severe_rate",
            "log_total_fires","FIRE_YEAR","county_enc",
            "season_Winter","season_Spring","season_Summer","season_Fall",
            "enso_score","enso_jja","enso_score_lag1","enso_score_lag2"]
TARGET = "severe_fires"

panel.to_csv(os.path.join(OUT_DIR, "county_season_panel.csv"), index=False)

import pickle
with open(os.path.join(OUT_DIR, "county_label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

with open(os.path.join(OUT_DIR, "feature_list.txt"), "w") as f:
    f.write("\n".join(FEATURES))

print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"Saved panel -> Data/county_season_panel.csv")
print(f"  Rows with any NaN in features (will be dropped at training): "
      f"{panel[FEATURES].isna().any(axis=1).sum()}")
