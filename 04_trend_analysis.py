import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_PATH  = os.path.join(BASE_DIR, "..", "Data", "ca_fires_clean.csv")
OUT_DIR  = os.path.join(BASE_DIR, "Output")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_PATH, low_memory=False)
df = df.dropna(subset=["FIRE_YEAR","SEVERE","SEASON","COUNTY","FIPS_NAME"])
df["DECADE"] = (df["FIRE_YEAR"] // 10 * 10).astype(str) + "s"
print(f"Loaded {len(df):,} fires  |  {int(df['FIRE_YEAR'].min())}–{int(df['FIRE_YEAR'].max())}")

DECADE_ORDER  = ["1990s","2000s","2010s"]
SEASON_ORDER  = ["Winter","Spring","Summer","Fall"]
SEASON_COLORS = {"Winter":"#5C85D6","Spring":"#4CAF50","Summer":"#FF9800","Fall":"#8D6E63"}

# 1. Annual severe fire count & rate
annual = (df.groupby("FIRE_YEAR")
            .agg(total=("SEVERE","count"), severe=("SEVERE","sum"))
            .reset_index())
annual["severe_rate"] = annual["severe"] / annual["total"]

fig, ax1 = plt.subplots(figsize=(11, 4))
ax2 = ax1.twinx()
ax1.bar(annual["FIRE_YEAR"], annual["severe"], color="#FF9800", alpha=0.6, label="Severe fires")
ax2.plot(annual["FIRE_YEAR"], annual["severe_rate"]*100, color="#d32f2f",
         linewidth=2, marker="o", markersize=4, label="Severe rate (%)")
ax1.set_xlabel("Year"); ax1.set_ylabel("Severe fire count", color="#FF9800")
ax2.set_ylabel("Severe rate (%)", color="#d32f2f")
ax1.set_title("California Severe Wildfires 1992–2015 — Annual Count & Rate", fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "trend_annual.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  trend_annual.png")

# 2. Decade × Season heatmap
dec_sea = (df.groupby(["DECADE","SEASON"])
             .agg(total=("SEVERE","count"), severe=("SEVERE","sum"))
             .reset_index())
dec_sea["severe_rate"] = dec_sea["severe"] / dec_sea["total"]
pivot = (dec_sea.pivot(index="DECADE", columns="SEASON", values="severe_rate")
         .reindex(index=DECADE_ORDER, columns=SEASON_ORDER))

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(SEASON_ORDER))); ax.set_xticklabels(SEASON_ORDER)
ax.set_yticks(range(len(DECADE_ORDER))); ax.set_yticklabels(DECADE_ORDER)
ax.set_title("Severe Fire Rate by Decade & Season", fontweight="bold")
plt.colorbar(im, ax=ax, label="Severe rate")
for i in range(len(DECADE_ORDER)):
    for j in range(len(SEASON_ORDER)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color="black" if val < 0.04 else "white")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "trend_decade_season_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  trend_decade_season_heatmap.png")

# 3. Severe rate by season across years
yr_sea = (df.groupby(["FIRE_YEAR","SEASON"])
            .agg(total=("SEVERE","count"), severe=("SEVERE","sum"))
            .reset_index())
yr_sea["severe_rate"] = yr_sea["severe"] / yr_sea["total"]

fig, ax = plt.subplots(figsize=(11, 5))
for season in SEASON_ORDER:
    sub = yr_sea[yr_sea["SEASON"] == season].sort_values("FIRE_YEAR")
    ax.plot(sub["FIRE_YEAR"], sub["severe_rate"]*100,
            label=season, color=SEASON_COLORS[season], linewidth=2, marker="o", markersize=3)
ax.set_xlabel("Year"); ax.set_ylabel("Severe fire rate (%)")
ax.set_title("Severe Fire Rate by Season 1992–2015", fontweight="bold")
ax.legend(); ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "trend_season_lines.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  trend_season_lines.png")

# 4. Top 10 counties by decade
top_counties = (df[df["SEVERE"]==1].groupby("FIPS_NAME")["SEVERE"].sum()
                .nlargest(10).index.tolist())
dec_co = (df[df["FIPS_NAME"].isin(top_counties)]
          .groupby(["FIPS_NAME","DECADE"])["SEVERE"].sum().reset_index())

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(top_counties)); width = 0.25
for i, decade in enumerate(DECADE_ORDER):
    vals = [dec_co[(dec_co["FIPS_NAME"]==c)&(dec_co["DECADE"]==decade)]["SEVERE"].sum()
            for c in top_counties]
    ax.bar(x + i*width, vals, width, label=decade,
           color=["#5C85D6","#FF9800","#d32f2f"][i])
ax.set_xticks(x + width); ax.set_xticklabels(top_counties, rotation=20, ha="right")
ax.set_ylabel("Severe fire count")
ax.set_title("Top 10 Counties — Severe Fires by Decade", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "trend_top_counties_decade.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  trend_top_counties_decade.png")

# 5. Decade summary table
summary = (df.groupby("DECADE")
             .agg(total_fires=("SEVERE","count"),
                  severe_fires=("SEVERE","sum"),
                  severe_rate=("SEVERE","mean"),
                  mean_fire_size=("FIRE_SIZE","mean"),
                  max_fire_size=("FIRE_SIZE","max"))
             .round(3))
summary.to_csv(os.path.join(OUT_DIR, "trend_decade_summary.csv"))
print("  trend_decade_summary.csv")
print(summary.to_string())
