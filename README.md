# Predicting-Wildfire-Severity-in-California

 Predictive Modelling · Smith School of Business, Queen's University 

A county-level forecasting model that predicts severe wildfire activity across California's 59 counties from 2016–2026, built on 189,550 historical fire records and anchored to published NOAA climate data — designed to help emergency management agencies allocate limited pre-season resources before fires start, not after.

[**View the full analytical report**](https://angadc14.github.io/Predicting-Wildfire-Severity-in-California/summary_report.html) &nbsp;·&nbsp; [View the interactive risk map](https://angadc14.github.io/Predicting-Wildfire-Severity-in-California/wildfire_risk_map.html) &nbsp;·&nbsp; [View the presentation](https://github.com/Angadc14/Predicting-Wildfire-Severity-in-California/blob/main/MMA867_Team_Harbour_Presentation%20(1).pptx)

<img width="1279" height="692" alt="Screenshot 2026-09-01 at 6 23 13 PM" src="https://github.com/user-attachments/assets/96b33940-c5cf-4d39-acd9-c00d691f018f" />

---

## Overview

Emergency response agencies have to commit to pre-season decisions — where to pre-position crews, how to allocate aerial suppression resources, which counties need mutual-aid agreements — months before a wildfire ignites. Existing tools mostly describe fire behavior once ignition has already happened; they don't answer the planning question emergency managers actually face: **given historical fire patterns and current climate conditions, which California counties are forecast to see the most severe wildfire activity next season, and how should a limited budget be allocated across all 59 counties?**

California alone spends roughly $850 million a year on wildfire response, not counting property damage or loss of life, and raw metrics like "total acres burned" don't translate directly into a resourcing decision. This project reframes the problem as a **count-forecasting task**: predict the number of severe fires (≥300 acres) a given county will see in a given season, using 24 years of US Forest Service fire records combined with NOAA's El Niño–Southern Oscillation (ENSO) climate index as a forward-looking signal.

## Key Findings

- **Recent fire history is by far the strongest predictor of future severity.** In the selected Gradient Boosting model, lag and rolling features — a county's severe fire count in the prior 1–3 years — dominate feature importance, ahead of ENSO climate signals or seasonal indicators. Severe wildfire activity is highly spatially and temporally persistent: the single best predictor of next year's severe fires in a county is last year's severe fires in that same county.
- **Gradient Boosting outperformed Random Forest and Poisson Regression on validation data (2013–2015)**, achieving the lowest Mean Absolute Error (MAE = 0.418 vs. 0.424 and 0.425 respectively), and was selected as the forecasting model. RMSE was comparatively high across all three models, largely driven by one major outlier — Riverside County's 2015 season.
- **ENSO climate data adds real, if modest, explanatory power (~15%)** despite the short 24-year time series, and — critically — it's the only feature in the model that isn't dependent on USFS historical fire data, which means it's what allows the model to generate genuine forward-looking forecasts (2016–2026) rather than a flat extrapolation of the past.
- **Risk is heavily concentrated: five counties account for a disproportionate share of forecast severity.** Riverside (3.19 severe fires/season), Siskiyou (1.11), Placer (0.99), Mendocino (0.93), and Lassen (0.90) are flagged as Critical risk for 2016–2026 — Riverside alone is forecast at roughly 3x the next-highest county.
- **The county risk ranking was independently cross-checked against a second model.** Counties flagged as high-risk by the Gradient Boosting count forecast (Tulare, Los Angeles, San Diego, Kern) were also independently flagged by a separate fire-level Random Forest severity classifier, giving convergent evidence for where those resources should go first.
- **Severity is getting more extreme even where frequency isn't rising as fast.** The 2000s had the highest severe-fire *rate* (2.3%), but the 2010s produced fewer, larger events — including a single fire that burned over 315,000 acres — consistent with climate research on larger, faster-spreading fires under increased heat stress.

## Methodology

**Data:** 189,550 individual wildfire records (1992–2015) from the US Forest Service's Fire Program Analysis Fire-Occurrence Database (FPA-FOD), restricted to California, combined with monthly NOAA/CPC ENSO climate data extended through 2026.

**Panel construction:** raw fire-level records were aggregated into a **county × season × year panel** of 2,881 rows, converting an individual-event classification problem into a count-forecasting problem more suited to resource planning. Each row captures total fires, severe fire count, mean location, and a set of engineered features.

**Feature engineering (15 features), including:**

| Feature | What it captures |
|---|---|
| `lag1_severe`, `lag2_severe` | Severe fires in the prior 1–2 years (persistent spatial fire regimes) |
| `rolling3_severe` | 3-year rolling average, smoothing short-term volatility |
| `hist_severe_rate` | A county's intrinsic historical severity baseline |
| `enso_score`, `enso_jja` | Current-year ENSO climate signal (ordinal El Niño/La Niña scale; June–Aug Niño 3.4 anomaly) |
| `enso_score_lag1/2` | 1–2 year lagged ENSO — captures the delayed fuel-loading mechanism (wet El Niño years → dense vegetation growth → drier fuel 1–2 years later) |
| `log_total_fires`, `FIRE_YEAR`, `county_enc`, season dummies | Fire volume control, long-run trend, county fixed effect, seasonal regime |

**Train / validation / forecast split (strict temporal, no leakage):**

`1992–1996 (warm-up for lag features) → 1997–2012 (training, 1,784 rows) → 2013–2015 (validation) → 2016–2026 (forecast)`

**Models compared:** Poisson Regression, Random Forest, and Gradient Boosting — all evaluated against a naïve county-season mean baseline, all outperforming it. Gradient Boosting was selected for its lowest MAE.

<img width="2000" height="1125" alt="model-comparison" src="https://github.com/user-attachments/assets/fde74bbe-148c-4ebc-ab9a-79578bc07b3b" />


**Forecasting approach:** predictions are rolled forward year-by-year — each year's predicted severe count becomes the lag input for the following year — preserving the temporal dependency structure that makes the lag features meaningful, rather than a flat historical extrapolation. ENSO conditions for 2016–2025 use published NOAA values; 2026 assumes a neutral ENSO state, which is why the forecast horizon ends there.

<img width="2000" height="1125" alt="seasonal-patterns" src="https://github.com/user-attachments/assets/a69a6031-01d2-48f2-8a15-1fd8da2964dc" />


## County Risk Ranking & Resource Allocation

The final forecast distills into a ranked risk tier for all 59 counties, designed to directly support pre-season budget and staffing decisions — Critical (top 5), High (6–10), and Elevated (11–20) — with counties independently corroborated by a second model flagged for extra confidence.

<img width="2000" height="1125" alt="county-risk-ranking" src="https://github.com/user-attachments/assets/a7bf1117-a4a2-479e-b800-b2ac7b2cbb68" />


## Interactive Deliverables

Two standalone, self-contained HTML artifacts accompany the slide deck:

- **`summary_report.html`** — a full analytical write-up (executive summary, trend analysis, methodology, model comparison, forecast, and county rankings) designed to be opened directly in a browser, no server required.
- **`wildfire_risk_map.html`** — an interactive Leaflet/Folium map of California with a year slider and seasonal filters, letting a user explore county-level forecasted risk geographically rather than as a table.

## Tech Stack

- **Python** (pandas, scikit-learn, statsmodels) — panel construction, feature engineering, and all three predictive models
- **Folium / Leaflet** — interactive geographic risk map
- **Matplotlib** — static charts embedded in the report and deck
- **PowerPoint** — stakeholder-facing report deck

## Repository Structure

```
├── MMA867_Team_Harbour_Presentation.pptx   # Stakeholder-facing report deck
├── summary_report.html                     # Full self-contained analytical report
├── wildfire_risk_map.html                  # Interactive county risk map (Folium/Leaflet)
├── fires_model_ready.csv                   # Cleaned fire-level dataset (56K+ rows) with engineered features
├── county_season_panel.csv                 # Aggregated county × season × year panel (2,881 rows) used for modeling
├── assets/                                 # README screenshots
└── README.md
```

## Limitations

- No drought index (PDSI/NDVI) is included — ENSO captures inter-annual climate cycles but not local soil moisture conditions.
- Training data ends in 2015; recent extreme seasons (2017 Wine Country fires, 2018 Camp Fire, 2020–2021 mega-fires) are not reflected in the model.
- Fire volume for forecast years is fixed at the 2013–2015 average and does not account for a potential increase in ignition frequency under continued climate change.
- County centroids are approximated from fire location averages rather than official administrative boundaries.
- ENSO for 2026 uses a neutral assumption, so forecast uncertainty increases toward the outer edge of the horizon.
- The model forecasts the *count* of severe fires per county-season; it does not predict whether or where an individual fire will ignite.

## Next Steps

- Incorporate the Palmer Drought Severity Index (PDSI) as a monthly county-level covariate.
- Retrain on post-2015 data to capture the recent mega-fire regime shift.
- Add prediction intervals (bootstrap sampling or quantile regression) to give county-level forecasts a confidence range for decision-making.
- Integrate a fire-size classification pipeline for real-time dispatch recommendations alongside the planning-horizon forecast.
- Deploy the interactive map as a hosted web application for emergency management agencies.



## License

Academic project — shared for portfolio purposes. Please reach out before reusing the model, data pipeline, or report.
