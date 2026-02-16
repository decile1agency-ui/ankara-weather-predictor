"""
Fetch historical METAR data from Iowa State Mesonet for LTAC (Ankara Esenboga)
and parse into features for temperature prediction.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os


STATION = "LTAC"
BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def fetch_metar_data(station=STATION, years=5):
    """Fetch historical METAR data from Iowa State Mesonet."""
    end = datetime.utcnow()
    start = end - timedelta(days=365 * years)

    params = {
        "station": station,
        "data": "all",
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "empty",
        "trace": "0.0001",
        "direct": "no",
        "report_type": "3",  # METAR and SPECI
        "year1": start.year,
        "month1": start.month,
        "day1": start.day,
        "year2": end.year,
        "month2": end.month,
        "day2": end.day,
    }

    print(f"Fetching {years} years of METAR data for {station}...")
    resp = requests.get(BASE_URL, params=params, timeout=120)
    resp.raise_for_status()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, f"{station}_raw.csv")
    with open(raw_path, "w") as f:
        f.write(resp.text)
    print(f"Saved raw data to {raw_path}")
    return raw_path


def cloud_cover_score(metar_text):
    """Extract cloud cover as a numeric score from METAR text."""
    if not isinstance(metar_text, str):
        return 0
    cover_map = {"FEW": 1, "SCT": 3, "BKN": 6, "OVC": 8, "VV": 8}
    score = 0
    for code, val in cover_map.items():
        if code in metar_text:
            score = max(score, val)
    if "CAVOK" in metar_text or "CLR" in metar_text or "SKC" in metar_text:
        score = 0
    return score


def lowest_cloud_base(metar_text):
    """Extract lowest cloud base in hundreds of feet."""
    if not isinstance(metar_text, str):
        return 999
    matches = re.findall(r'(?:FEW|SCT|BKN|OVC|VV)(\d{3})', metar_text)
    if matches:
        return min(int(m) for m in matches)
    if "CAVOK" in metar_text or "CLR" in metar_text:
        return 999
    return 999


def has_precipitation(metar_text):
    """Check if METAR reports precipitation."""
    if not isinstance(metar_text, str):
        return 0
    precip_patterns = ["-RA", "RA", "+RA", "-SN", "SN", "+SN",
                       "-SHRA", "SHRA", "+SHRA", "TSRA", "DZ", "-DZ",
                       "FZRA", "-FZRA", "SG", "GR", "GS"]
    for p in precip_patterns:
        if p in metar_text:
            return 1
    return 0


def wind_direction_components(drct):
    """Convert wind direction to u,v components."""
    if pd.isna(drct) or drct == 0:
        return 0.0, 0.0
    rad = np.radians(drct)
    u = -np.sin(rad)  # zonal (west-east)
    v = -np.cos(rad)  # meridional (south-north)
    return u, v


def parse_and_engineer_features(raw_path):
    """Parse raw METAR CSV and engineer features for prediction."""
    print("Parsing METAR data and engineering features...")

    df = pd.read_csv(raw_path, low_memory=False)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Parse datetime
    df['valid'] = pd.to_datetime(df['valid'], errors='coerce')
    df = df.dropna(subset=['valid'])

    # Convert numeric columns
    for col in ['tmpf', 'dwpf', 'drct', 'sknt', 'alti', 'mslp',
                'p01i', 'gust', 'relh', 'feel']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert F to C if data is in Fahrenheit
    if 'tmpf' in df.columns:
        df['tmpc'] = (df['tmpf'] - 32) * 5.0 / 9.0
    if 'dwpf' in df.columns:
        df['dwpc'] = (df['dwpf'] - 32) * 5.0 / 9.0

    df = df.dropna(subset=['tmpc'])
    df['date'] = df['valid'].dt.date
    df['hour'] = df['valid'].dt.hour

    # Extract metar text features
    metar_col = 'metar' if 'metar' in df.columns else None
    if metar_col:
        df['cloud_score'] = df[metar_col].apply(cloud_cover_score)
        df['cloud_base'] = df[metar_col].apply(lowest_cloud_base)
        df['has_precip'] = df[metar_col].apply(has_precipitation)
    else:
        df['cloud_score'] = 0
        df['cloud_base'] = 999
        df['has_precip'] = 0

    # Wind components
    if 'drct' in df.columns:
        wind_comps = df['drct'].apply(lambda d: pd.Series(wind_direction_components(d), index=['wind_u', 'wind_v']))
        df = pd.concat([df, wind_comps], axis=1)
    else:
        df['wind_u'] = 0
        df['wind_v'] = 0

    if 'sknt' not in df.columns:
        df['sknt'] = 0
    else:
        df['sknt'] = df['sknt'].fillna(0)
    if 'gust' not in df.columns:
        df['gust'] = 0
    else:
        df['gust'] = df['gust'].fillna(0)

    # Dewpoint depression
    if 'dwpc' in df.columns:
        df['dewpoint_depression'] = df['tmpc'] - df['dwpc']
    else:
        df['dewpoint_depression'] = np.nan

    # Pressure tendency (change over recent observations)
    pressure_col = 'mslp' if 'mslp' in df.columns else ('alti' if 'alti' in df.columns else None)
    if pressure_col:
        df['pressure'] = df[pressure_col]
    else:
        df['pressure'] = np.nan

    # Day of year (seasonality)
    df['day_of_year'] = df['valid'].dt.dayofyear

    # Daily max temperature (target)
    daily_max = df.groupby('date')['tmpc'].max().reset_index()
    daily_max.columns = ['date', 'max_temp']

    # All timestamps are UTC (Zulu). Ankara = UTC+3.
    # Morning window: 04-08 UTC = 07-11 local Ankara time
    # This captures early/mid morning before peak heating (~12-14 UTC / 15-17 local)
    MORNING_START_UTC = 4
    MORNING_END_UTC = 8
    morning = df[(df['hour'] >= MORNING_START_UTC) & (df['hour'] <= MORNING_END_UTC)].copy()

    if len(morning) == 0:
        print(f"Warning: No morning observations in {MORNING_START_UTC}-{MORNING_END_UTC} UTC window")
        morning = df[(df['hour'] >= 3) & (df['hour'] <= 10)].copy()

    morning_agg = morning.groupby('date').agg(
        morning_temp=('tmpc', 'mean'),
        morning_temp_max=('tmpc', 'max'),
        morning_dewpoint_depression=('dewpoint_depression', 'mean'),
        morning_wind_speed=('sknt', 'mean'),
        morning_gust=('gust', 'max'),
        morning_wind_u=('wind_u', 'mean'),
        morning_wind_v=('wind_v', 'mean'),
        morning_cloud_score=('cloud_score', 'mean'),
        morning_cloud_base=('cloud_base', 'min'),
        morning_precip=('has_precip', 'max'),
        morning_pressure=('pressure', 'mean'),
        day_of_year=('day_of_year', 'first'),
    ).reset_index()

    # Previous day max temp
    daily_max_sorted = daily_max.sort_values('date')
    daily_max_sorted['prev_day_max'] = daily_max_sorted['max_temp'].shift(1)

    # Merge morning features with daily max
    dataset = morning_agg.merge(daily_max_sorted, on='date', how='inner')
    dataset = dataset.dropna(subset=['max_temp'])

    # Pressure tendency (morning pressure vs previous day)
    dataset = dataset.sort_values('date')
    dataset['pressure_tendency'] = dataset['morning_pressure'].diff()

    # Temperature trend (morning temp vs previous morning)
    dataset['temp_trend'] = dataset['morning_temp'].diff()

    # Fill remaining NaN with reasonable defaults before dropping
    dataset['pressure_tendency'] = dataset['pressure_tendency'].fillna(0)
    dataset['temp_trend'] = dataset['temp_trend'].fillna(0)
    dataset['prev_day_max'] = dataset['prev_day_max'].fillna(dataset['max_temp'])
    dataset['morning_pressure'] = dataset['morning_pressure'].fillna(dataset['morning_pressure'].median())
    dataset = dataset.dropna(subset=['max_temp', 'morning_temp']).reset_index(drop=True)

    features_path = os.path.join(os.path.dirname(raw_path), "features.csv")
    dataset.to_csv(features_path, index=False)
    print(f"Engineered {len(dataset)} daily samples, saved to {features_path}")

    return dataset


if __name__ == "__main__":
    raw_path = fetch_metar_data(STATION, years=5)
    dataset = parse_and_engineer_features(raw_path)
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Date range: {dataset['date'].min()} to {dataset['date'].max()}")
    print(f"\nFeature columns: {list(dataset.columns)}")
    print(f"\nTarget (max_temp) stats:\n{dataset['max_temp'].describe()}")
