# Air Quality Insight – Practice Dataset

IMPORTANT: These CSV files are synthetically generated for education and demonstration.
They are NOT real measurements from EcoAir Analytics Centre or any city.

Files:
- air_quality.csv: hourly pollutant/AQI readings by station (90 days, 6 stations), with intentionally injected missing readings, duplicate records, inconsistent station labels, and extreme values.
- stations.csv: station location and land-use lookup.
- weather.csv: hourly wind, temperature, humidity, and rain, with intentional missing values and duplicate timestamps.
- traffic.csv: hourly traffic volume and peak/off-peak flag.
- calendar.csv: hourly weekday, holiday, and season fields.

Join keys:
- air_quality.station -> stations.station (normalize station names first)
- timestamp -> weather.timestamp, traffic.timestamp, calendar.timestamp
Resolve duplicate timestamps in weather before merging to avoid row multiplication.
Example column names use `pm2_5` for PM2.5 and `land_use_type` for land-use type.

Recommended tasks:
1. Inspect shape, types, nulls, duplicate rows, and ranges.
2. Normalize station names and timestamps.
3. Investigate outliers before deciding whether they are sensor errors.
4. Aggregate/resolve duplicate weather timestamps before joins.
5. Merge datasets and perform EDA, visualization, feature engineering, and feature selection.
