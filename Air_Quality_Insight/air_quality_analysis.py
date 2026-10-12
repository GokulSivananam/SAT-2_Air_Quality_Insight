"""
USE CASE 20 – AIR QUALITY INSIGHT
EcoAir Analytics Centre | Round 2 – EDA and Feature Engineering
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'Data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

REQUIRED_FILES = {
    'air_quality': os.path.join(DATA_DIR, 'air_quality.csv'),
    'stations':    os.path.join(DATA_DIR, 'stations.csv'),
    'weather':     os.path.join(DATA_DIR, 'weather.csv'),
    'traffic':     os.path.join(DATA_DIR, 'traffic.csv'),
    'calendar':    os.path.join(DATA_DIR, 'calendar.csv'),
}

# ─── 1. DATA LOADING ──────────────────────────────────────────────────────────
def load_data():
    """Load all five CSV files and print basic info."""
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING")
    print("="*60)

    for name, path in REQUIRED_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    dfs = {name: pd.read_csv(path) for name, path in REQUIRED_FILES.items()}

    for name, df in dfs.items():
        print(f"\n--- {name}.csv ---")
        print(f"  Shape : {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Dtypes :\n{df.dtypes.to_string()}")
        print(f"  First 5 rows:\n{df.head().to_string()}")
        print(f"  Missing values:\n{df.isnull().sum().to_string()}")
        print(f"  Duplicate rows: {df.duplicated().sum()}")

    return dfs

# ─── 2. DATA CLEANING ─────────────────────────────────────────────────────────
def clean_data(dfs):
    """Standardise, fix types, handle missing values and outliers."""
    print("\n" + "="*60)
    print("STEP 2: DATA CLEANING")
    print("="*60)

    # ── air_quality ──────────────────────────────────────────────────────────
    aq = dfs['air_quality'].copy()
    aq.columns = aq.columns.str.strip().str.lower()
    aq['station'] = aq['station'].str.strip().str.upper()   # fix " st02 "
    aq['timestamp'] = pd.to_datetime(aq['timestamp'])
    aq.drop_duplicates(inplace=True)

    # Outlier detection: IQR per column
    for col in ['pm2_5', 'pm10', 'no2', 'aqi']:
        q1, q3 = aq[col].quantile(0.25), aq[col].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 3 * iqr          # 3×IQR – conservative; keeps real peaks
        n_out = (aq[col] > upper).sum()
        print(f"  {col}: {n_out} extreme outliers (>{upper:.1f}) -> capped")
        aq[col] = aq[col].clip(upper=upper)

    # Fill missing sensor readings with per-station median
    for col in ['pm2_5', 'pm10', 'no2', 'aqi']:
        aq[col] = aq.groupby('station')[col].transform(
            lambda x: x.fillna(x.median()))

    print(f"  air_quality after cleaning: {aq.shape}, missing: {aq.isnull().sum().sum()}")

    # ── stations ─────────────────────────────────────────────────────────────
    st = dfs['stations'].copy()
    st.columns = st.columns.str.strip().str.lower()
    st['station'] = st['station'].str.strip().str.upper()
    print(f"  stations: {st.shape}")

    # ── weather ──────────────────────────────────────────────────────────────
    wx = dfs['weather'].copy()
    wx.columns = wx.columns.str.strip().str.lower()
    wx['timestamp'] = pd.to_datetime(wx['timestamp'])
    wx.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    wx.sort_values('timestamp', inplace=True)
    # Fill missing weather with forward-fill then median
    for col in ['wind', 'temperature', 'humidity']:
        wx[col] = wx[col].ffill().fillna(wx[col].median())
    print(f"  weather after cleaning: {wx.shape}, missing: {wx.isnull().sum().sum()}")

    # ── traffic ──────────────────────────────────────────────────────────────
    tr = dfs['traffic'].copy()
    tr.columns = tr.columns.str.strip().str.lower()
    tr['timestamp'] = pd.to_datetime(tr['timestamp'])
    tr.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    print(f"  traffic after cleaning: {tr.shape}")

    # ── calendar ─────────────────────────────────────────────────────────────
    cal = dfs['calendar'].copy()
    cal.columns = cal.columns.str.strip().str.lower()
    cal['timestamp'] = pd.to_datetime(cal['timestamp'])
    cal.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    print(f"  calendar after cleaning: {cal.shape}")

    # Save cleaned air quality
    aq.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_air_quality.csv'), index=False)
    print("  Saved: cleaned_air_quality.csv")

    return aq, st, wx, tr, cal

# ─── 3. DATA INTEGRATION ──────────────────────────────────────────────────────
def integrate_data(aq, st, wx, tr, cal):
    """Merge all datasets into one master dataset."""
    print("\n" + "="*60)
    print("STEP 3: DATA INTEGRATION")
    print("="*60)

    # Merge auxiliary tables on timestamp (one row per hour, shared across stations)
    aux = wx.merge(tr,  on='timestamp', how='left') \
             .merge(cal, on='timestamp', how='left')

    # Merge air quality with station metadata
    master = aq.merge(st, on='station', how='left')

    # Merge with auxiliary (timestamp-based)
    master = master.merge(aux, on='timestamp', how='left')

    print(f"  Master dataset shape: {master.shape}")
    print(f"  Columns: {list(master.columns)}")
    print(f"  Missing values:\n{master.isnull().sum()[master.isnull().sum()>0].to_string()}")

    unmatched_stations = master[master['location'].isnull()]['station'].unique()
    if len(unmatched_stations):
        print(f"  WARNING: Unmatched stations: {unmatched_stations}")
    else:
        print("  All stations matched successfully.")

    master.to_csv(os.path.join(OUTPUT_DIR, 'master_dataset.csv'), index=False)
    print("  Saved: master_dataset.csv")
    return master

# ─── 4. EDA ───────────────────────────────────────────────────────────────────
def run_eda(master):
    """Answer the six business questions with stats and charts."""
    print("\n" + "="*60)
    print("STEP 4: EXPLORATORY DATA ANALYSIS")
    print("="*60)

    summary_lines = []

    def note(text):
        print("  " + text)
        summary_lines.append(text)

    # ── Q1: Which stations have poorer air quality? ───────────────────────────
    note("Q1: Average AQI by station")
    q1 = master.groupby('station')['aqi'].agg(['mean','median']).round(2).sort_values('mean', ascending=False)
    note(q1.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    q1['mean'].plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title('Average AQI by Station', fontsize=14)
    ax.set_xlabel('Station')
    ax.set_ylabel('Average AQI')
    ax.tick_params(axis='x', rotation=0)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', (p.get_x()+p.get_width()/2, p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q1_avg_aqi_by_station.png'), dpi=150)
    plt.close()

    # ── Q2: What times show higher pollution? ────────────────────────────────
    note("\nQ2: Hourly AQI trends")
    master['hour'] = master['timestamp'].dt.hour
    q2 = master.groupby('hour')['aqi'].mean().round(2)
    note(q2.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    q2.plot(ax=ax, marker='o', color='darkorange')
    ax.set_title('Average AQI by Hour of Day', fontsize=14)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Average AQI')
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q2_hourly_aqi_trend.png'), dpi=150)
    plt.close()

    # ── Q3: Wind speed vs PM2.5 ──────────────────────────────────────────────
    note("\nQ3: Wind speed vs PM2.5 correlation")
    corr_wind = master[['wind', 'pm2_5']].corr().iloc[0, 1]
    note(f"  Pearson correlation (wind vs pm2_5): {corr_wind:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(master['wind'], master['pm2_5'], alpha=0.15, s=10, color='teal')
    ax.set_title(f'PM2.5 vs Wind Speed  (r = {corr_wind:.2f})', fontsize=13)
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('PM2.5 (µg/m³)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q3_pm25_vs_wind.png'), dpi=150)
    plt.close()

    # ── Q4: Traffic volume vs AQI ────────────────────────────────────────────
    note("\nQ4: Traffic volume vs AQI")
    corr_traf = master[['traffic_volume', 'aqi']].corr().iloc[0, 1]
    note(f"  Pearson correlation (traffic_volume vs aqi): {corr_traf:.3f}")
    q4 = master.groupby('peak_flag')['aqi'].mean().round(2)
    note(f"  Mean AQI by peak_flag:\n{q4.to_string()}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(master['traffic_volume'], master['aqi'], alpha=0.15, s=10, color='crimson')
    ax.set_title(f'AQI vs Traffic Volume  (r = {corr_traf:.2f})', fontsize=13)
    ax.set_xlabel('Traffic Volume')
    ax.set_ylabel('AQI')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q4_aqi_vs_traffic.png'), dpi=150)
    plt.close()

    # ── Q5: Rain effect on PM2.5 ─────────────────────────────────────────────
    note("\nQ5: Rain effect on PM2.5")
    q5 = master.groupby('rain')['pm2_5'].agg(['mean','median']).round(2)
    note(q5.to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    master.boxplot(column='pm2_5', by='rain', ax=ax,
                   boxprops=dict(color='navy'), medianprops=dict(color='red'))
    ax.set_title('PM2.5 Distribution: Rainy vs Non-Rainy', fontsize=13)
    ax.set_xlabel('Rain (0=No, 1=Yes)')
    ax.set_ylabel('PM2.5 (µg/m³)')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q5_pm25_rain_boxplot.png'), dpi=150)
    plt.close()

    # ── Q6: Seasonal pollution peaks ─────────────────────────────────────────
    note("\nQ6: Seasonal AQI comparison")
    q6 = master.groupby('season')['aqi'].agg(['mean','median','max']).round(2)
    note(q6.to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    q6['mean'].plot(kind='bar', ax=ax, color=['royalblue','tomato'], edgecolor='black')
    ax.set_title('Average AQI by Season', fontsize=13)
    ax.set_xlabel('Season')
    ax.set_ylabel('Average AQI')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q6_seasonal_aqi.png'), dpi=150)
    plt.close()

    # ── Correlation heatmap ───────────────────────────────────────────────────
    num_cols = ['pm2_5', 'pm10', 'no2', 'aqi', 'wind', 'temperature', 'humidity', 'rain', 'traffic_volume']
    corr_matrix = master[num_cols].corr().round(2)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5)
    ax.set_title('Correlation Matrix – Numerical Features', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()

    # ── Missing value chart ───────────────────────────────────────────────────
    missing = master.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        fig, ax = plt.subplots(figsize=(8, 4))
        missing.plot(kind='bar', ax=ax, color='salmon', edgecolor='black')
        ax.set_title('Missing Values per Column (Master Dataset)', fontsize=13)
        ax.set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'missing_values_chart.png'), dpi=150)
        plt.close()
    else:
        note("  No missing values in master dataset.")

    # Save EDA summary
    with open(os.path.join(OUTPUT_DIR, 'eda_summary.txt'), 'w') as f:
        f.write("EDA SUMMARY – AIR QUALITY INSIGHT\n")
        f.write("="*50 + "\n\n")
        f.write("\n".join(summary_lines))
    print("  Saved: eda_summary.txt")

    return master

# ─── 5. FEATURE ENGINEERING ───────────────────────────────────────────────────
def engineer_features(master):
    """Create meaningful features for modelling."""
    print("\n" + "="*60)
    print("STEP 5: FEATURE ENGINEERING")
    print("="*60)

    df = master.copy()

    # Time-based features
    df['hour']       = df['timestamp'].dt.hour          # hour of day (0-23)
    df['day']        = df['timestamp'].dt.day           # day of month
    df['month']      = df['timestamp'].dt.month         # month number
    df['weekday_num']= df['timestamp'].dt.dayofweek     # 0=Mon … 6=Sun
    df['is_weekend'] = (df['weekday_num'] >= 5).astype(int)  # 1 if Sat/Sun

    # Time period: day (6-21) vs night (22-5)
    df['time_period'] = df['hour'].apply(lambda h: 'day' if 6 <= h <= 21 else 'night')

    # Peak hour flag (morning 7-10, evening 16-20)
    df['is_peak_hour'] = df['hour'].apply(
        lambda h: 1 if (7 <= h <= 10) or (16 <= h <= 20) else 0)

    # High PM2.5 flag (above 75th percentile)
    p75 = df['pm2_5'].quantile(0.75)
    df['high_pm25'] = (df['pm2_5'] > p75).astype(int)

    # Rain indicator (already 0/1 but rename for clarity)
    df['rain_indicator'] = df['rain'].astype(int)

    # ── Encoding categorical columns ─────────────────────────────────────────
    # Label-encode station
    station_map = {s: i for i, s in enumerate(sorted(df['station'].unique()))}
    df['station_enc'] = df['station'].map(station_map)

    # Label-encode land_use_type
    lut_map = {v: i for i, v in enumerate(sorted(df['land_use_type'].dropna().unique()))}
    df['land_use_enc'] = df['land_use_type'].map(lut_map)

    # Label-encode season
    season_map = {'winter': 0, 'summer': 1, 'spring': 2, 'autumn': 3}
    df['season_enc'] = df['season'].map(season_map).fillna(0).astype(int)

    # Label-encode peak_flag
    df['peak_flag_enc'] = (df['peak_flag'] == 'peak').astype(int)

    # Label-encode time_period
    df['time_period_enc'] = (df['time_period'] == 'day').astype(int)

    print("  Engineered features:")
    new_cols = ['hour','day','month','weekday_num','is_weekend','time_period',
                'is_peak_hour','high_pm25','rain_indicator',
                'station_enc','land_use_enc','season_enc','peak_flag_enc','time_period_enc']
    for c in new_cols:
        print(f"    {c}: {df[c].dtype} — {df[c].nunique()} unique values")

    df.to_csv(os.path.join(OUTPUT_DIR, 'engineered_features.csv'), index=False)
    print("  Saved: engineered_features.csv")
    return df

# ─── 6. FEATURE SELECTION ─────────────────────────────────────────────────────
def select_features(df):
    """Rank features by correlation with AQI and apply SelectKBest."""
    print("\n" + "="*60)
    print("STEP 6: FEATURE SELECTION")
    print("="*60)

    feature_cols = [
        'pm2_5','pm10','no2','wind','temperature','humidity','rain_indicator',
        'traffic_volume','hour','is_weekend','is_peak_hour','high_pm25',
        'station_enc','land_use_enc','season_enc','peak_flag_enc','time_period_enc'
    ]
    target = 'aqi'

    sub = df[feature_cols + [target]].dropna()
    X = sub[feature_cols]
    y = sub[target]

    # Pearson correlation with AQI
    corr_with_aqi = X.corrwith(y).abs().sort_values(ascending=False).round(4)
    print("  Feature correlations with AQI (absolute Pearson):")
    print(corr_with_aqi.to_string())

    # SelectKBest (F-regression)
    selector = SelectKBest(score_func=f_regression, k=10)
    selector.fit(X, y)
    scores = pd.Series(selector.scores_, index=feature_cols).sort_values(ascending=False).round(2)
    print("\n  SelectKBest F-scores:")
    print(scores.to_string())

    selected = scores.head(10).index.tolist()
    print(f"\n  Top 10 selected features: {selected}")

    result = pd.DataFrame({
        'feature': feature_cols,
        'pearson_corr_with_aqi': corr_with_aqi.reindex(feature_cols).values,
        'f_score': scores.reindex(feature_cols).values
    }).sort_values('f_score', ascending=False)

    result.to_csv(os.path.join(OUTPUT_DIR, 'feature_selection.csv'), index=False)
    print("  Saved: feature_selection.csv")
    return result

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  AIR QUALITY INSIGHT – EDA & FEATURE ENGINEERING")
    print("  EcoAir Analytics Centre | Round 2")
    print("="*60)

    dfs    = load_data()
    aq, st, wx, tr, cal = clean_data(dfs)
    master = integrate_data(aq, st, wx, tr, cal)
    master = run_eda(master)
    df_eng = engineer_features(master)
    select_features(df_eng)

    print("\n" + "="*60)
    print("  ALL OUTPUTS SAVED TO: outputs/")
    print("="*60)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    {f}")

if __name__ == '__main__':
    main()
