"""
Interactive wildfire risk map — California 1997-2026.
- Year slider + All/Summer/Fall/Spring/Winter season buttons
- Heatmap updates dynamically by year and season
- County markers rank dynamically by count for the displayed view
- Popup shows count + severe rate %
"""
import pandas as pd
import numpy as np
import os, re, json
import folium
from folium.plugins import HeatMap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "Output")

fires   = pd.read_csv(os.path.join(BASE_DIR, "..", "Data", "ca_fires_clean.csv"),
                      low_memory=False)
panel   = pd.read_csv(os.path.join(OUT_DIR, "county_forecasts.csv"))
ranking = pd.read_csv(os.path.join(OUT_DIR, "county_risk_ranking.csv"))

ranking["Rank"] = ranking["Rank"].astype(int)
panel["FIRE_YEAR"] = panel["FIRE_YEAR"].astype(int)
fires = fires.dropna(subset=["FIRE_YEAR","LATITUDE","LONGITUDE","SEVERE","SEASON"])
fires["FIRE_YEAR"] = fires["FIRE_YEAR"].astype(int)
print(f"Loaded {len(fires):,} fires | panel: {len(panel):,} rows")

SEASONS    = ["Summer","Fall","Spring","Winter"]
ALL_LABEL  = "All"
YEARS      = sorted(panel["FIRE_YEAR"].unique())
FORE_START = 2016

# =============================================================================
# Build heatmap data per year × season from panel county centroids.
# Using centroids (not raw fire coords) ensures every county with fires shows
# a heat signal regardless of county size or fire dispersion.
# Weight = total_fires (volume signal); severe fires add extra weight.
# =============================================================================
panel_hist = panel[panel["is_forecast"] == False].copy()
panel_hist["FIRE_YEAR"] = panel_hist["FIRE_YEAR"].astype(int)
panel_fc   = panel[panel["is_forecast"] == True].copy()
panel_fc["FIRE_YEAR"]   = panel_fc["FIRE_YEAR"].astype(int)

heatmap_data = {}
for year in YEARS:
    yr_str = str(year)
    heatmap_data[yr_str] = {}
    is_fc = year >= FORE_START
    yp = panel_fc[panel_fc["FIRE_YEAR"] == year] if is_fc \
         else panel_hist[panel_hist["FIRE_YEAR"] == year]

    for season in SEASONS + [ALL_LABEL]:
        if season == ALL_LABEL:
            sp = yp.groupby("FIPS_NAME").agg(
                lat=("lat","mean"), lon=("lon","mean"),
                total_fires=("total_fires","sum"),
                severe_fires=("severe_fires","sum") if not is_fc else ("predicted_severe","sum"),
                predicted_severe=("predicted_severe","sum") if is_fc else ("severe_fires","sum"),
            ).reset_index()
        else:
            sp = yp[yp["SEASON"] == season]
        if len(sp) == 0:
            heatmap_data[yr_str][season] = []
            continue
        pts = []
        for r in sp.itertuples():
            total  = float(r.total_fires) if hasattr(r, 'total_fires') and r.total_fires > 0 else 0
            severe = float(r.predicted_severe) if is_fc else float(r.severe_fires if hasattr(r,'severe_fires') and not np.isnan(r.severe_fires) else 0)
            weight = max(0.5, total * 0.5 + severe * 10)
            if total > 0 or severe > 0:
                pts.append([round(float(r.lat), 4), round(float(r.lon), 4), round(weight, 1)])
        heatmap_data[yr_str][season] = pts

heatmap_js = json.dumps(heatmap_data, separators=(",",":"))
print(f"Heatmap data built for {len(YEARS)} years")

# =============================================================================
# Build timeline: {year: {season|"All": {county: {...}}}}
# =============================================================================
def build_county_dict(grp_rows):
    out = {}
    for _, row in grp_rows.iterrows():
        county  = row["FIPS_NAME"]
        is_fc   = bool(row.get("is_forecast", False))
        actual  = int(row["severe_fires"]) if not np.isnan(row.get("severe_fires", float("nan"))) else None
        pred    = float(row["predicted_severe"]) if not np.isnan(row.get("predicted_severe", float("nan"))) else None
        total   = int(row["total_fires"]) if not np.isnan(row.get("total_fires", float("nan"))) else 0
        count   = pred if is_fc else (float(actual) if actual is not None else 0.0)
        rate    = round(count / total * 100, 2) if total > 0 else 0.0
        out[county] = {
            "lat":    round(float(row["lat"]), 4),
            "lon":    round(float(row["lon"]), 4),
            "count":  round(count, 2),
            "actual": actual,
            "total":  total,
            "rate":   rate,
            "fc":     is_fc,
        }
    return out

timeline = {}
for (year, season), grp in panel.groupby(["FIRE_YEAR","SEASON"]):
    yr_str = str(year)
    if yr_str not in timeline:
        timeline[yr_str] = {}
    timeline[yr_str][season] = build_county_dict(grp)

# Build "All" season by merging all seasons per county per year
for yr_str in timeline:
    merged = {}
    for season in SEASONS:
        for county, d in timeline[yr_str].get(season, {}).items():
            if county not in merged:
                merged[county] = {
                    "lat": d["lat"], "lon": d["lon"],
                    "count": 0.0, "actual": 0, "total": 0,
                    "rate": 0.0, "fc": d["fc"],
                }
            merged[county]["count"]  += d["count"]
            merged[county]["total"]  += d["total"]
            if d["actual"] is not None:
                merged[county]["actual"] += d["actual"]
    # Recompute rate on merged totals
    for county in merged:
        t = merged[county]["total"]
        merged[county]["rate"] = round(merged[county]["count"] / t * 100, 2) if t > 0 else 0.0
        merged[county]["count"] = round(merged[county]["count"], 2)
    timeline[yr_str][ALL_LABEL] = merged

timeline_js = json.dumps(timeline, separators=(",",":"))

max_count_global = max(
    (d["count"] for yr in timeline.values()
     for sea in yr.values() for d in sea.values()),
    default=1.0
)
print(f"Timeline: {YEARS[0]}–{YEARS[-1]} | max count: {max_count_global:.1f}")

# =============================================================================
# Folium base map — one dummy HeatMap to pull in leaflet-heat script
# =============================================================================
m = folium.Map(location=[37.5, -119.5], zoom_start=6,
               tiles="cartodbpositron", prefer_canvas=True)

# Single minimal HeatMap just to include the leaflet-heat.js dependency
dummy = fires[fires["FIRE_YEAR"]==2015][["LATITUDE","LONGITUDE","SEVERE"]].head(1).values.tolist()
fg_dummy = folium.FeatureGroup(name="heat_init", show=False)
HeatMap(dummy, radius=1).add_to(fg_dummy)
fg_dummy.add_to(m)

folium.LayerControl(position="topright").add_to(m)

# =============================================================================
# Sidebar HTML (pure HTML/CSS — all JS is in the post-processed block)
# =============================================================================
first_year   = FORE_START
first_season = ALL_LABEL

season_btns = ""
for s in [ALL_LABEL] + SEASONS:
    active = 'style="background:#b71c1c;color:white;border-color:#b71c1c"' if s == first_season else ""
    season_btns += f'<button id="btn-{s}" onclick="setSeason(\'{s}\')" {active}>{s}</button>\n'

sidebar_html = f"""
<div id="sidebar" style="position:fixed;top:0;left:0;width:310px;height:100%;
  background:#1a1a2e;color:#eee;z-index:1000;overflow-y:auto;
  box-shadow:3px 0 12px rgba(0,0,0,0.4);font-family:'Segoe UI',Arial,sans-serif">

  <div style="background:#b71c1c;padding:14px 14px 10px">
    <div style="font-size:14px;font-weight:700;color:white">County Risk Ranking</div>
    <div style="font-size:10px;color:rgba(255,255,255,.75);margin-top:2px">
      Predicted severe fires per county-season
    </div>
  </div>

  <div style="padding:12px 14px;background:#16213e;border-bottom:1px solid #0f3460">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
      <span style="font-size:11px;color:#aaa">Year</span>
      <span id="year-label" style="font-size:18px;font-weight:700;color:#FF9800">{first_year}</span>
    </div>
    <input id="year-slider" type="range"
      min="{YEARS[0]}" max="{YEARS[-1]}" value="{first_year}" step="1"
      oninput="setYear(this.value)"
      style="width:100%;accent-color:#FF9800;cursor:pointer">
    <div style="display:flex;justify-content:space-between;font-size:9px;color:#666;margin-top:2px">
      <span>{YEARS[0]} historical</span>
      <span style="color:#FF9800">2016 forecast &rarr;</span>
      <span>{YEARS[-1]}</span>
    </div>
  </div>

  <div style="padding:10px 14px;background:#16213e;border-bottom:1px solid #0f3460">
    <div style="font-size:11px;color:#aaa;margin-bottom:6px">Season</div>
    <div id="season-btns" style="display:flex;gap:4px;flex-wrap:wrap">
      {season_btns}
    </div>
  </div>

  <div style="padding:7px 14px;background:#0f3460;font-size:10px;
              display:flex;gap:10px;align-items:center;border-bottom:1px solid #1a1a2e">
    <span><span style="color:#d32f2f">&#9679;</span> Critical (1-5)</span>
    <span><span style="color:#FF9800">&#9679;</span> High (6-10)</span>
    <span><span style="color:#1976D2">&#9679;</span> Elevated (11-20)</span>
  </div>

  <div id="forecast-badge" style="display:none;margin:8px 14px;padding:7px 10px;
    background:#1a1a2e;border:1px dashed #FF9800;border-radius:6px;
    font-size:10px;color:#FF9800;text-align:center">
    &#9888; FORECAST &mdash; Gradient Boosting model.<br>
    Trained 1997&ndash;2012, validated 2013&ndash;2015.
  </div>

  <table style="width:100%;border-collapse:collapse;font-size:11px">
    <thead>
      <tr style="background:#0f3460;color:#ccc">
        <th style="padding:7px 5px;text-align:center">#</th>
        <th style="padding:7px 5px;text-align:left">County</th>
        <th style="padding:7px 5px;text-align:center" id="col-count">Count</th>
        <th style="padding:7px 5px;text-align:center">Rate %</th>
      </tr>
    </thead>
    <tbody id="rank-body"></tbody>
  </table>

  <div style="padding:10px 14px;font-size:9px;color:#555;border-top:1px solid #0f3460">
    Gradient Boosting (MAE=0.385) &middot; Click rows to fly to county
  </div>
</div>

<style>
  #season-btns button {{
    flex:1;padding:5px 3px;font-size:10px;border:1px solid #444;
    background:#1a1a2e;color:#ccc;border-radius:4px;cursor:pointer;
  }}
  #season-btns button:hover {{ background:#0f3460;color:white; }}
  .folium-map {{ margin-left:310px !important; }}
  #map        {{ margin-left:310px !important; }}
  .leaflet-top.leaflet-left  {{ margin-left:320px !important; }}
  .leaflet-top.leaflet-right {{ right:8px; }}
</style>
"""

title_html = """
<div style="position:fixed;top:10px;left:calc(50% + 155px);transform:translateX(-50%);
  z-index:999;background:white;padding:9px 16px;border-radius:8px;
  box-shadow:2px 2px 6px rgba(0,0,0,0.25);font-family:Arial;
  font-size:13px;font-weight:bold;text-align:center">
  California Wildfire Severity &amp; Forecast<br>
  <span style="font-size:10px;font-weight:normal">
    Historical 1997&ndash;2015 &nbsp;|&nbsp; Forecast 2016&ndash;2026
  </span>
</div>
"""

m.get_root().html.add_child(folium.Element(sidebar_html))
m.get_root().html.add_child(folium.Element(title_html))

out_path = os.path.join(OUT_DIR, "wildfire_risk_map.html")
m.save(out_path)

# =============================================================================
# Post-process: append JS after all Folium scripts
# =============================================================================
with open(out_path, "r", encoding="utf-8") as f:
    html = f.read()

match  = re.search(r'var\s+(map_[a-zA-Z0-9]+)\s*=\s*L\.map\(', html)
map_var = match.group(1) if match else "null"
print(f"  Found map variable: {map_var}")

js = """
<script>
(function() {
  var timelineData = """ + timeline_js + """;
  var heatmapData  = """ + heatmap_js  + """;
  var maxCount     = """ + str(round(max_count_global, 2)) + """;
  var MAP_VAR_NAME = '""" + map_var + """';
  var FORE_START   = 2016;
  var curYear      = """ + str(first_year) + """;
  var curSeason    = '""" + first_season + """';
  var mkLayer      = null;
  var heatLayer    = null;
  var theMap       = null;

  function rankColor(r) { return r<=5?'#d32f2f':r<=10?'#FF9800':r<=20?'#1976D2':'#90A4AE'; }
  function rowBg(r)     { return r<=5?'#FFEBEE':r<=10?'#FFF3E0':r<=20?'#F0F4F8':'#FAFAFA'; }
  function rowBgH(r)    { return r<=5?'#FFCDD2':r<=10?'#FFE0B2':'#DBEAFE'; }

  // Rank counties by count descending for the current view
  function dynRanked(data) {
    var arr = Object.keys(data).map(function(c) {
      return { county: c, d: data[c] };
    });
    arr.sort(function(a,b) { return b.d.count - a.d.count; });
    arr.forEach(function(item, i) { item.rank = i + 1; });
    return arr;
  }

  function updateHeatmap(year, season) {
    if (!theMap) return;
    if (heatLayer) { theMap.removeLayer(heatLayer); heatLayer = null; }
    var pts = ((heatmapData[String(year)] || {})[season]) || [];
    if (pts.length === 0) return;
    heatLayer = L.heatLayer(pts, {
      radius: 40, blur: 35, maxZoom: 12,
      gradient: {0.1:'#aec6f0', 0.4:'#ff9800', 0.7:'#e53935', 1.0:'#7b0000'}
    }).addTo(theMap);
  }

  function updateMarkers(ranked, isForecast) {
    if (!theMap) return;
    if (mkLayer) { theMap.removeLayer(mkLayer); mkLayer = null; }
    mkLayer = L.layerGroup().addTo(theMap);
    ranked.forEach(function(item) {
      var c = item.county, d = item.d, r = item.rank;
      if (d.count <= 0) return;
      var circle = L.circleMarker([d.lat, d.lon], {
        radius:      Math.max(5, 26 * d.count / maxCount),
        color:       isForecast ? '#FF9800' : 'white',
        weight:      isForecast ? 2 : 0.8,
        dashArray:   isForecast ? '5 4' : null,
        fillColor:   rankColor(r),
        fillOpacity: 0.88
      });
      var cnt  = isForecast ? d.count.toFixed(1) : (d.actual !== null ? d.actual : Math.round(d.count));
      var rate = d.rate.toFixed(2) + '%';
      var pop  = '<b>' + c + '</b><br>Rank: #' + r + '<br>';
      pop += isForecast
        ? '<span style="color:#e65c00">&#9888; Forecast</span><br>Predicted severe: ' + cnt + '<br>Severe rate: ' + rate
        : 'Actual severe: ' + cnt + '<br>Severe rate: ' + rate + '<br>Total fires: ' + d.total;
      circle.bindPopup(pop);
      circle.bindTooltip(c + ' #' + r + ' | ' + (isForecast?'Pred: ':'Act: ') + cnt + ' (' + rate + ')');
      circle._cn = c;
      circle.addTo(mkLayer);
    });
  }

  function updateTable(ranked, isForecast) {
    var top20 = ranked.slice(0, 20);
    var h = '';
    top20.forEach(function(item) {
      var c=item.county, d=item.d, r=item.rank;
      var bg=rowBg(r), bgh=rowBgH(r), dot=rankColor(r);
      var cnt  = isForecast ? d.count.toFixed(1) : (d.actual !== null ? d.actual : Math.round(d.count));
      var rate = d.rate.toFixed(2) + '%';
      var cntColor = isForecast ? '#e65c00' : '#333';
      var cs = c.replace(/'/g,"\\\\'");
      h += '<tr style="background:'+bg+';color:#111;cursor:pointer"' +
           ' onmouseover="this.style.background=\\''+bgh+'\\'"' +
           ' onmouseout="this.style.background=\\''+bg+'\\'"' +
           ' onclick="window._sc(\\''+cs+'\\','+r+')">' +
           '<td style="padding:6px 4px;text-align:center;font-weight:700;color:'+dot+'">'+r+'</td>' +
           '<td style="padding:6px 4px;color:#111;font-weight:600">'+c+'</td>' +
           '<td style="padding:6px 4px;text-align:center;color:'+cntColor+'">'+cnt+'</td>' +
           '<td style="padding:6px 4px;text-align:center;color:#555;font-size:10px">'+rate+'</td>' +
           '</tr>';
    });
    var tb = document.getElementById('rank-body');
    if (tb) tb.innerHTML = h || '<tr><td colspan="4" style="text-align:center;color:#666;padding:12px">No data</td></tr>';
    var hdr = document.getElementById('col-count');
    if (hdr) hdr.innerText = isForecast ? 'Forecast' : 'Actual';
  }

  function refresh() {
    var isForecast = curYear >= FORE_START;
    var data   = ((timelineData[String(curYear)] || {})[curSeason]) || {};
    var ranked = dynRanked(data);

    updateHeatmap(curYear, curSeason);
    updateMarkers(ranked, isForecast);
    updateTable(ranked, isForecast);

    var badge = document.getElementById('forecast-badge');
    if (badge) badge.style.display = isForecast ? 'block' : 'none';
    var lbl = document.getElementById('year-label');
    if (lbl) { lbl.innerText = curYear; lbl.style.color = isForecast ? '#FF9800' : '#4FC3F7'; }
  }

  window.setYear = function(v) { curYear = parseInt(v); refresh(); };

  window.setSeason = function(s) {
    curSeason = s;
    document.querySelectorAll('#season-btns button').forEach(function(b) {
      b.style.background='#1a1a2e'; b.style.color='#ccc'; b.style.borderColor='#444';
    });
    var btn = document.getElementById('btn-' + s);
    if (btn) { btn.style.background='#b71c1c'; btn.style.color='white'; btn.style.borderColor='#b71c1c'; }
    refresh();
  };

  window._sc = function(name) {
    var data = ((timelineData[String(curYear)] || {})[curSeason]) || {};
    var d = data[name];
    if (!d || !theMap) return;
    theMap.flyTo([d.lat, d.lon], 9, {duration:1.2});
    if (mkLayer) {
      setTimeout(function() {
        mkLayer.eachLayer(function(l) { if (l._cn===name) l.openPopup(); });
      }, 1300);
    }
  };

  window.addEventListener('load', function() {
    theMap = window[MAP_VAR_NAME];
    if (!theMap) { console.error('Map not found: ' + MAP_VAR_NAME); return; }
    // Hide the dummy static heat layer Folium added
    theMap.eachLayer(function(l) {
      if (l._heat) { theMap.removeLayer(l); }
    });
    setTimeout(refresh, 300);
  });
})();
</script>
"""

with open(out_path, "w", encoding="utf-8") as f:
    f.write(html + js)

print(f"Interactive map saved to: {out_path}")
