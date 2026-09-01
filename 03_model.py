"""
Model: predict county-season severe fire count.
Train: 1997-2012  |  Test: 2013-2015  |  Forecast: 2016-2026
Models: Poisson Regression, Random Forest, Gradient Boosting
"""
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
OUT_DIR  = os.path.join(BASE_DIR, "Output")
os.makedirs(OUT_DIR, exist_ok=True)

panel = pd.read_csv(os.path.join(DATA_DIR, "county_season_panel.csv"))
with open(os.path.join(DATA_DIR, "county_label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)
with open(os.path.join(DATA_DIR, "feature_list.txt")) as f:
    FEATURES = [l.strip() for l in f if l.strip()]

TARGET     = "severe_fires"
TRAIN_END  = 2012
TEST_START = 2013
TEST_END   = 2015

panel = panel.sort_values(["FIPS_NAME","SEASON","FIRE_YEAR"])
panel_model = panel.dropna(subset=FEATURES + [TARGET]).copy()

train = panel_model[panel_model["FIRE_YEAR"] <= TRAIN_END]
test  = panel_model[(panel_model["FIRE_YEAR"] >= TEST_START) &
                    (panel_model["FIRE_YEAR"] <= TEST_END)]

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"Train rows: {len(train):,} | Test rows: {len(test):,}")
print(f"Train severe/row: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

# =============================================================================
# Models
# =============================================================================
models = {
    "Poisson Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  PoissonRegressor(alpha=0.5, max_iter=500)),
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=3,
        random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = np.maximum(0, model.predict(X_test))
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    results[name] = {"model": model, "pred": pred, "MAE": mae, "RMSE": rmse}
    print(f"  {name:25s}  MAE={mae:.3f}  RMSE={rmse:.3f}")

best_name = min(results, key=lambda n: results[n]["MAE"])
best      = results[best_name]
print(f"\nBest model: {best_name}  MAE={best['MAE']:.3f}")

# =============================================================================
# Model comparison chart
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, res) in zip(axes, results.items()):
    pred = res["pred"]
    ax.scatter(y_test, pred, alpha=0.4, s=20, color="#1976D2")
    lim = max(y_test.max(), pred.max()) + 1
    ax.plot([0, lim], [0, lim], "r--", linewidth=1)
    ax.set_xlabel("Actual severe fires")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{name}\nMAE={res['MAE']:.3f}  RMSE={res['RMSE']:.3f}",
                 fontweight="bold")
plt.suptitle("Model Comparison — Actual vs Predicted (Test 2013–2015)",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# =============================================================================
# Feature importance
# =============================================================================
best_model = best["model"]
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
elif hasattr(best_model, "named_steps") and hasattr(best_model.named_steps["model"], "coef_"):
    imp = pd.Series(np.abs(best_model.named_steps["model"].coef_),
                    index=FEATURES).sort_values(ascending=True)
else:
    imp = None

if imp is not None:
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot.barh(ax=ax, color="#2196F3")
    ax.set_title(f"Feature Importance — {best_name}", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

# =============================================================================
# Forecast 2016-2026 (rolling: each year's prediction feeds the next)
# =============================================================================
# ENSO lookup for forecast years (sourced from Jacob's notebook / NOAA)
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
def enso_vals(year):
    t, jja   = ENSO_TABLE.get(year, ('Neutral', 0.0))
    t1, _    = ENSO_TABLE.get(year-1, ('Neutral', 0.0))
    t2, _    = ENSO_TABLE.get(year-2, ('Neutral', 0.0))
    return ENSO_SCORE[t], jja, ENSO_SCORE[t1], ENSO_SCORE[t2]

FORECAST_YEARS = list(range(2016, 2027))

# County-season centroids and average fire volume (last 3 known years)
centroids = (panel.groupby("FIPS_NAME")
             .agg(lat=("lat","mean"), lon=("lon","mean"))
             .reset_index())
avg_fires = (panel[panel["FIRE_YEAR"].between(2013, 2015)]
             .groupby(["FIPS_NAME","SEASON"])["total_fires"]
             .mean().reset_index()
             .rename(columns={"total_fires":"avg_total_fires"}))

# Extend panel iteratively
panel_ext = panel.copy()
panel_ext["is_forecast"] = False
panel_ext["predicted_severe"] = np.nan

# Fill predicted_severe for test rows now
test_pred_df = test.copy()
test_pred_df["predicted_severe"] = best["pred"]
test_pred_df["is_forecast"] = False
for idx in test_pred_df.index:
    panel_ext.loc[panel_ext.index == idx, "predicted_severe"] = \
        test_pred_df.loc[idx, "predicted_severe"]

for year in FORECAST_YEARS:
    new_rows = []
    for _, avg_row in avg_fires.iterrows():
        county = avg_row["FIPS_NAME"]
        season = avg_row["SEASON"]

        grp = (panel_ext[
                   (panel_ext["FIPS_NAME"]==county) &
                   (panel_ext["SEASON"]==season)]
               .sort_values("FIRE_YEAR"))
        if grp.empty:
            continue

        # Use predicted or actual severe for lag computation
        sev_series = grp.apply(
            lambda r: r["predicted_severe"] if not np.isnan(r["predicted_severe"])
                      else r["severe_fires"], axis=1).values

        lag1    = float(sev_series[-1]) if len(sev_series) >= 1 else 0.0
        lag2    = float(sev_series[-2]) if len(sev_series) >= 2 else 0.0
        roll3   = float(np.mean(sev_series[-3:])) if len(sev_series) >= 1 else 0.0
        cum_sev = float(sev_series.sum())
        cum_cnt = float(grp["total_fires"].sum())
        hist_rate = cum_sev / cum_cnt if cum_cnt > 0 else 0.0
        avg_tf  = float(avg_row["avg_total_fires"])

        county_enc = int(le.transform([county])[0]) if county in le.classes_ else 0

        es, ejja, es_l1, es_l2 = enso_vals(year)
        new_rows.append({
            "FIPS_NAME":   county,
            "SEASON":      season,
            "FIRE_YEAR":   year,
            "total_fires": avg_tf,
            "severe_fires":np.nan,
            "lat":         float(grp["lat"].iloc[-1]),
            "lon":         float(grp["lon"].iloc[-1]),
            "lag1_severe":     lag1,
            "lag2_severe":     lag2,
            "rolling3_severe": roll3,
            "hist_severe_rate":hist_rate,
            "log_total_fires": np.log1p(avg_tf),
            "county_enc":      county_enc,
            "season_Winter":   int(season=="Winter"),
            "season_Spring":   int(season=="Spring"),
            "season_Summer":   int(season=="Summer"),
            "season_Fall":     int(season=="Fall"),
            "enso_score":      es,
            "enso_jja":        ejja,
            "enso_score_lag1": es_l1,
            "enso_score_lag2": es_l2,
            "is_forecast":     True,
            "predicted_severe":np.nan,
        })

    if not new_rows:
        continue
    new_df = pd.DataFrame(new_rows)
    X_new  = new_df[FEATURES]
    new_df["predicted_severe"] = np.maximum(0, best_model.predict(X_new))

    panel_ext = pd.concat([panel_ext, new_df], ignore_index=True)
    preds_sum = new_df["predicted_severe"].sum()
    print(f"  Forecast {year}: {preds_sum:.1f} total predicted severe fires across all counties/seasons")

# =============================================================================
# County risk ranking: average predicted severe fires per year (2016-2026)
# =============================================================================
forecast_only = panel_ext[panel_ext["is_forecast"]==True]
county_rank = (forecast_only.groupby("FIPS_NAME")
               .agg(avg_predicted_severe = ("predicted_severe","mean"),
                    total_predicted_severe= ("predicted_severe","sum"),
                    lat = ("lat","mean"),
                    lon = ("lon","mean"))
               .reset_index()
               .sort_values("avg_predicted_severe", ascending=False)
               .reset_index(drop=True))
county_rank.index = range(1, len(county_rank) + 1)
county_rank.index.name = "Rank"
county_rank = county_rank.round(3)

print(f"\nTop 10 counties — avg predicted severe fires/season (2016–2026):")
print(county_rank.head(10).to_string())
county_rank.to_csv(os.path.join(OUT_DIR, "county_risk_ranking.csv"))

# County ranking chart
top20 = county_rank.head(20).reset_index()
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#d32f2f" if i<5 else "#f57c00" if i<10 else "#1976D2"
          for i in range(len(top20))]
ax.barh(top20["FIPS_NAME"][::-1], top20["avg_predicted_severe"][::-1],
        color=colors[::-1])
ax.set_xlabel("Avg predicted severe fires per season (2016–2026)")
ax.set_title("Top 20 Highest-Risk Counties — Forecast 2016–2026\n"
             "(County-season model, ≥300 acres)", fontweight="bold")
for i, row in enumerate(top20[::-1].itertuples()):
    ax.text(row.avg_predicted_severe + 0.01, i,
            f"{row.avg_predicted_severe:.2f}", va="center", fontsize=8)
ax.legend(handles=[
    mpatches.Patch(color="#d32f2f", label="Critical (Top 5)"),
    mpatches.Patch(color="#f57c00", label="High (6–10)"),
    mpatches.Patch(color="#1976D2", label="Elevated (11–20)"),
], loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "county_risk_ranking.png"), dpi=150, bbox_inches="tight")
plt.close()

# =============================================================================
# Forecast trend chart — total predicted severe fires per year
# =============================================================================
annual_forecast = (panel_ext[panel_ext["is_forecast"]==True]
                   .groupby("FIRE_YEAR")["predicted_severe"].sum().reset_index())
annual_actual   = (panel[panel["FIRE_YEAR"] <= TEST_END]
                   .groupby("FIRE_YEAR")["severe_fires"].sum().reset_index())

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(annual_actual["FIRE_YEAR"], annual_actual["severe_fires"],
       color="#FF9800", alpha=0.7, label="Actual severe fires")
ax.plot(annual_forecast["FIRE_YEAR"], annual_forecast["predicted_severe"],
        color="#d32f2f", linewidth=2.5, marker="o", markersize=5,
        label=f"Forecast — {best_name}")
ax.axvline(2015.5, color="gray", linestyle="--", linewidth=1, label="Forecast boundary")
ax.set_xlabel("Year")
ax.set_ylabel("Total severe fires (all counties & seasons)")
ax.set_title("California Severe Wildfires — Historical & Forecast 2016–2026",
             fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "forecast_trend.png"), dpi=150, bbox_inches="tight")
plt.close()

# =============================================================================
# Save full panel (historical + forecast) for map
# =============================================================================
panel_ext.to_csv(os.path.join(OUT_DIR, "county_forecasts.csv"), index=False)
print(f"\nAll outputs saved to Output/")
print(f"  model_comparison.png")
print(f"  feature_importance.png")
print(f"  county_risk_ranking.csv / .png")
print(f"  forecast_trend.png")
print(f"  county_forecasts.csv")
