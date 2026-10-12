# USE CASE 20 – AIR QUALITY INSIGHT
**Understand Air Quality. Discover Drivers. Support Cleaner Cities.**

**Organisation:** EcoAir Analytics Centre
**Sector:** Environment
**Round:** 2 – EDA and Feature Engineering (30 Marks)

---

## Project Abstract
This project analyses hourly air pollution monitoring data across six city stations
(Jan–Mar 2025). Weather conditions, traffic volume, station location, and calendar
information are combined to identify the key drivers of air quality changes and to
prepare a clean, feature-rich dataset for downstream modelling.

---

## Problem Statement
Urban air pollution is a growing public health concern. EcoAir Analytics Centre
needs to understand which stations suffer the worst air quality, when pollution
peaks occur, and how weather and traffic relate to pollutant levels — so that
city planners can take targeted action.

---

## Objectives
1. Load, inspect, and clean five heterogeneous CSV datasets.
2. Integrate them into a single master dataset.
3. Answer six business questions through EDA and visualisations.
4. Engineer meaningful features for machine-learning readiness.
5. Select the most informative features for predicting AQI.

---

## Dataset Description

| File | Rows | Key Columns |
|------|------|-------------|
| `air_quality.csv` | 12,972 | timestamp, station, pm2_5, pm10, no2, aqi |
| `stations.csv` | 6 | station, location, land_use_type |
| `weather.csv` | 2,165 | timestamp, wind, temperature, humidity, rain |
| `traffic.csv` | 2,160 | timestamp, traffic_volume, peak_flag |
| `calendar.csv` | 2,160 | timestamp, weekday, holiday, season |

**Known data quality issues handled:**
- Inconsistent station names (e.g. `" st02 "` with whitespace) — stripped and uppercased.
- Extreme sensor outliers (pm2_5 up to 900, aqi up to 600) — capped using 3×IQR.
- Missing values in pm2_5 (80), pm10 (60), no2 (55), aqi (45) — filled with per-station median.
- Missing weather readings (wind: 35, humidity: 25) — forward-filled then median-filled.
- 12 duplicate rows in air_quality, 5 in weather — removed.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, merging |
| `numpy` | Numerical operations |
| `matplotlib` | Chart creation |
| `seaborn` | Heatmap and styled plots |
| `scikit-learn` | SelectKBest feature selection |

---

## Implementation Steps

1. **Data Loading** — Read all five CSVs; print shape, dtypes, missing counts, duplicates.
2. **Data Cleaning** — Standardise names, parse timestamps, remove duplicates, cap outliers (3×IQR), impute missing values.
3. **Data Integration** — Merge air quality + stations on `station`; merge weather, traffic, calendar on `timestamp`; validate no row multiplication.
4. **EDA** — Answer six business questions with grouped statistics and eight charts.
5. **Feature Engineering** — Create 14 new features (time, binary flags, label encodings).
6. **Feature Selection** — Rank features by Pearson correlation and F-score; select top 10.

---

## How to Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Execute the Project

```bash
py air_quality_analysis.py
```

All outputs are saved automatically to the `outputs/` folder.

---

## Output Files

| File | Description |
|------|-------------|
| `cleaned_air_quality.csv` | Cleaned air quality data (outliers capped, missing filled) |
| `master_dataset.csv` | All five datasets merged into one (12,960 rows × 17 cols) |
| `engineered_features.csv` | Master dataset + 14 engineered features |
| `feature_selection.csv` | Feature ranking by Pearson correlation and F-score |
| `eda_summary.txt` | Text summary of all six EDA answers |
| `q1_avg_aqi_by_station.png` | Bar chart – average AQI per station |
| `q2_hourly_aqi_trend.png` | Line chart – AQI by hour of day |
| `q3_pm25_vs_wind.png` | Scatter – PM2.5 vs wind speed |
| `q4_aqi_vs_traffic.png` | Scatter – AQI vs traffic volume |
| `q5_pm25_rain_boxplot.png` | Box plot – PM2.5 on rainy vs dry days |
| `q6_seasonal_aqi.png` | Bar chart – AQI by season |
| `correlation_heatmap.png` | Heatmap of all numerical feature correlations |

---

## Key Findings (from actual execution)

### Q1 – Which stations have poorer air quality?
| Station | Avg AQI | Location | Land Use |
|---------|---------|----------|----------|
| **ST02** | **138.86** | Industrial Zone | Industrial |
| **ST04** | **122.65** | Highway Junction | Transport |
| ST06 | 107.71 | Commercial District | Commercial |
| ST01 | 98.09 | Central City | Commercial |
| ST03 | 67.77 | Residential North | Residential |
| ST05 | 40.71 | Urban Park | Green Space |

**ST02 (Industrial Zone) has the worst air quality**, nearly 3.5× higher than the
cleanest station ST05 (Urban Park). Industrial activity and highway proximity are
clear drivers.

### Q2 – What times show higher pollution?
AQI spikes during two daily windows:
- **Morning peak: 07:00–10:00** (avg AQI ~106)
- **Evening peak: 16:00–20:00** (avg AQI ~106)

Off-peak hours (00:00–06:00, 11:00–15:00, 21:00–23:00) average ~89–90 AQI.
These windows align exactly with commuter traffic rush hours.

### Q3 – How does wind speed relate to PM2.5?
Pearson correlation = **−0.138** (weak negative).
Higher wind speeds slightly disperse PM2.5, but the relationship is not strong,
suggesting other factors (industrial emissions, traffic) dominate.

### Q4 – Does traffic volume align with AQI?
Pearson correlation = **+0.244** (moderate positive).
Peak-hour traffic produces a mean AQI of **106.38** vs **89.72** during off-peak —
a difference of ~18 AQI points, confirming traffic is a meaningful contributor.

### Q5 – How does rain affect pollution?
| Condition | Mean PM2.5 | Median PM2.5 |
|-----------|-----------|--------------|
| No rain (0) | 29.31 | 29.62 |
| Rain (1) | **22.85** | **22.24** |

Rain reduces mean PM2.5 by ~22%, consistent with wet deposition washing
particulates out of the air.

### Q6 – Which seasons show recurring pollution peaks?
Both winter and summer show nearly identical average AQI (~95.8–96.0).
The dataset covers only Jan–Mar 2025 (winter = Jan–Feb, summer = Mar),
so seasonal differences are minimal within this short window.

### Top 10 Features for Predicting AQI (SelectKBest F-score)
1. `pm2_5` (F = 79,436) — strongest predictor
2. `no2` (F = 19,118)
3. `high_pm25` (F = 9,383) — binary flag
4. `pm10` (F = 9,069)
5. `station_enc` (F = 891) — location matters
6. `traffic_volume` (F = 823)
7. `is_peak_hour` (F = 653)
8. `peak_flag_enc` (F = 653)
9. `land_use_enc` (F = 340)
10. `time_period_enc` (F = 181)

> **Note:** pm2_5, pm10, and no2 are co-pollutants measured alongside AQI —
> they are included for completeness but would be excluded from a predictive
> model to avoid target leakage. The most actionable external predictors are
> `station_enc`, `traffic_volume`, `is_peak_hour`, and `land_use_enc`.
