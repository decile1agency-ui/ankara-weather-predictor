"""
Predict today's max temperature from current METAR observations.
All times are UTC (Zulu). Ankara local = UTC+3.

Morning window: 03-06 UTC (06-09 local Ankara)
Peak heating:   09-12 UTC (12-15 local Ankara)
"""

import numpy as np
import xgboost as xgb
import json
import re
import os
import sys


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Morning window in UTC (must match training in metar_fetcher.py)
# Local Ankara 06-09 = UTC 03-06
MORNING_START_UTC = 3
MORNING_END_UTC = 6

FEATURE_COLS = [
    'morning_temp',
    'morning_temp_max',
    'morning_dewpoint_depression',
    'morning_wind_speed',
    'morning_gust',
    'morning_wind_u',
    'morning_wind_v',
    'morning_cloud_score',
    'morning_cloud_base',
    'morning_precip',
    'morning_pressure',
    'day_of_year',
    'prev_day_max',
    'pressure_tendency',
    'temp_trend',
]


def extract_zulu_hour(metar_text):
    """Extract the UTC hour from METAR timestamp (DDHHMMz format)."""
    match = re.search(r'\d{2}(\d{2})\d{2}Z', metar_text)
    if match:
        return int(match.group(1))
    return None


def filter_morning_metars(metar_list):
    """Filter METARs to only include morning UTC window (04-08Z).
    Returns (morning_metars, all_parsed_with_hours) for reporting."""
    morning = []
    rejected = []
    for m in metar_list:
        hour = extract_zulu_hour(m)
        if hour is not None and MORNING_START_UTC <= hour <= MORNING_END_UTC:
            morning.append(m)
        else:
            rejected.append((m, hour))
    return morning, rejected


def parse_single_metar(metar_text):
    """Parse a single METAR string into feature values."""
    features = {}

    # UTC hour
    features['zulu_hour'] = extract_zulu_hour(metar_text)

    # Temperature and dewpoint (format: TT/DD or MTT/MDD)
    temp_match = re.search(r'\s(M?\d{2})/(M?\d{2})\s', metar_text)
    if temp_match:
        t_str, d_str = temp_match.group(1), temp_match.group(2)
        temp = -int(t_str[1:]) if t_str.startswith('M') else int(t_str)
        dewp = -int(d_str[1:]) if d_str.startswith('M') else int(d_str)
        features['temp'] = temp
        features['dewpoint'] = dewp
        features['dewpoint_depression'] = temp - dewp

    # Wind direction and speed
    wind_match = re.search(r'(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT', metar_text)
    if wind_match:
        drct_str, spd_str, gust_str = wind_match.groups()
        features['wind_speed'] = int(spd_str)
        features['gust'] = int(gust_str) if gust_str else 0
        if drct_str == 'VRB':
            features['wind_u'] = 0.0
            features['wind_v'] = 0.0
        else:
            drct = int(drct_str)
            rad = np.radians(drct)
            features['wind_u'] = -np.sin(rad)
            features['wind_v'] = -np.cos(rad)
    else:
        features['wind_speed'] = 0
        features['gust'] = 0
        features['wind_u'] = 0.0
        features['wind_v'] = 0.0

    # Pressure (QNH)
    qnh_match = re.search(r'Q(\d{4})', metar_text)
    if qnh_match:
        features['pressure'] = int(qnh_match.group(1))
    altimeter_match = re.search(r'A(\d{4})', metar_text)
    if altimeter_match and 'pressure' not in features:
        features['pressure'] = float(altimeter_match.group(1)) / 100.0 * 33.8639

    # Cloud cover
    cover_map = {"FEW": 1, "SCT": 3, "BKN": 6, "OVC": 8, "VV": 8}
    cloud_score = 0
    for code, val in cover_map.items():
        if code in metar_text:
            cloud_score = max(cloud_score, val)
    if "CAVOK" in metar_text or "CLR" in metar_text or "SKC" in metar_text:
        cloud_score = 0
    features['cloud_score'] = cloud_score

    # Lowest cloud base
    cloud_bases = re.findall(r'(?:FEW|SCT|BKN|OVC|VV)(\d{3})', metar_text)
    features['cloud_base'] = min(int(b) for b in cloud_bases) if cloud_bases else 999

    # Precipitation
    precip_patterns = ["-RA", "RA ", "+RA", "-SN", "SN ", "+SN",
                       "-SHRA", "SHRA", "+SHRA", "TSRA", "DZ", "-DZ"]
    features['has_precip'] = 1 if any(p in metar_text for p in precip_patterns) else 0

    return features


def predict_from_metars(metar_list, prev_day_max=None, prev_morning_temp=None,
                        prev_pressure=None, day_of_year=None):
    """
    Predict daily max temperature from a list of METAR strings.
    Automatically filters to morning UTC window (04-08Z = 07-11 local).

    Args:
        metar_list: List of METAR strings (any time range, will be filtered)
        prev_day_max: Yesterday's max temperature (C)
        prev_morning_temp: Yesterday morning's avg temp (C, for trend)
        prev_pressure: Yesterday morning's pressure (hPa, for tendency)
        day_of_year: Day of year (1-366), auto-detected if None
    """
    model_path = os.path.join(MODEL_DIR, "max_temp_predictor.json")
    medians_path = os.path.join(MODEL_DIR, "feature_medians.json")

    if not os.path.exists(model_path):
        print("ERROR: Model not found. Run train_predictor.py first.")
        sys.exit(1)

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    with open(medians_path) as f:
        medians = json.load(f)

    # Filter to morning UTC window
    morning_metars, rejected = filter_morning_metars(metar_list)

    print("=" * 60)
    print("  LTAC Ankara - Daily Max Temperature Prediction")
    print("  All times UTC (Zulu). Ankara local = UTC+3")
    print("=" * 60)
    print(f"\n  Total METARs provided:  {len(metar_list)}")
    print(f"  Morning window:         {MORNING_START_UTC:02d}Z - {MORNING_END_UTC:02d}Z "
          f"({MORNING_START_UTC+3:02d} - {MORNING_END_UTC+3:02d} local)")
    print(f"  METARs in window:       {len(morning_metars)}")
    print(f"  METARs outside window:  {len(rejected)}")

    if not morning_metars:
        print("\n  WARNING: No METARs in morning window. Using all provided METARs.")
        morning_metars = metar_list

    # Parse morning METARs
    parsed = [parse_single_metar(m) for m in morning_metars]

    if not parsed:
        print("  ERROR: No METAR data parsed.")
        return None

    # Show which METARs are being used
    print(f"\n  Morning METARs used:")
    for m in morning_metars:
        hour = extract_zulu_hour(m)
        temp_match = re.search(r'\s(M?\d{2})/(M?\d{2})\s', m)
        temp_str = temp_match.group(0).strip() if temp_match else "??/??"
        print(f"    {hour:02d}Z ({hour+3:02d}L)  {temp_str}")

    # Aggregate morning features
    temps = [p['temp'] for p in parsed if 'temp' in p]
    dewp_dep = [p['dewpoint_depression'] for p in parsed if 'dewpoint_depression' in p]
    speeds = [p['wind_speed'] for p in parsed]
    gusts = [p['gust'] for p in parsed]
    wind_us = [p['wind_u'] for p in parsed]
    wind_vs = [p['wind_v'] for p in parsed]
    cloud_scores = [p['cloud_score'] for p in parsed]
    cloud_bases = [p['cloud_base'] for p in parsed]
    precips = [p['has_precip'] for p in parsed]
    pressures = [p['pressure'] for p in parsed if 'pressure' in p]

    morning_temp = np.mean(temps) if temps else medians.get('morning_temp', 10)
    morning_temp_max = max(temps) if temps else medians.get('morning_temp_max', 12)
    morning_pressure = np.mean(pressures) if pressures else medians.get('morning_pressure', 1013)

    features = {
        'morning_temp': morning_temp,
        'morning_temp_max': morning_temp_max,
        'morning_dewpoint_depression': np.mean(dewp_dep) if dewp_dep else medians.get('morning_dewpoint_depression', 5),
        'morning_wind_speed': np.mean(speeds),
        'morning_gust': max(gusts),
        'morning_wind_u': np.mean(wind_us),
        'morning_wind_v': np.mean(wind_vs),
        'morning_cloud_score': np.mean(cloud_scores),
        'morning_cloud_base': min(cloud_bases),
        'morning_precip': max(precips),
        'morning_pressure': morning_pressure,
        'day_of_year': day_of_year if day_of_year else medians.get('day_of_year', 47),
        'prev_day_max': prev_day_max if prev_day_max is not None else medians.get('prev_day_max', 14),
        'pressure_tendency': (morning_pressure - prev_pressure) if prev_pressure else 0,
        'temp_trend': (morning_temp - prev_morning_temp) if prev_morning_temp else 0,
    }

    X = np.array([[features[col] for col in FEATURE_COLS]])
    prediction = model.predict(X)[0]

    print(f"\n  Aggregated Morning Features (04-08Z):")
    print(f"    Temperature:      {morning_temp:.1f}C (max: {morning_temp_max:.1f}C)")
    print(f"    Dewpoint Dep:     {features['morning_dewpoint_depression']:.1f}C")
    print(f"    Wind Speed:       {features['morning_wind_speed']:.0f} kt")
    print(f"    Max Gust:         {features['morning_gust']:.0f} kt")
    print(f"    Cloud Score:      {features['morning_cloud_score']:.1f}/8")
    print(f"    Cloud Base:       {features['morning_cloud_base']*100} ft")
    print(f"    Precipitation:    {'Yes' if features['morning_precip'] else 'No'}")
    print(f"    Pressure:         {features['morning_pressure']:.1f} hPa")
    print(f"    Pressure Trend:   {features['pressure_tendency']:+.1f} hPa")
    print(f"    Prev Day Max:     {features['prev_day_max']:.1f}C")
    print(f"    Day of Year:      {features['day_of_year']}")
    print(f"\n  {'=' * 44}")
    print(f"  PREDICTED MAX TEMP:  {prediction:.1f}C")
    print(f"  {'=' * 44}")

    return prediction


# ---- Predict for today (Feb 16) using the provided METARs ----
if __name__ == "__main__":
    # All timestamps are Zulu (UTC). Model filters to 04-08Z morning window.
    today_metars = [
        "METAR LTAC 161150Z 21009KT 9999 FEW040 BKN100 15/M01 Q1010 NOSIG",
        "METAR LTAC 161120Z 22012KT 9999 FEW040 BKN100 15/M01 Q1010 NOSIG",
        "METAR LTAC 161050Z 22017KT 9999 FEW040 BKN180 15/M02 Q1011 NOSIG",
        "METAR LTAC 161020Z 23014KT 9999 FEW040 BKN180 14/M01 Q1011 NOSIG",
        "METAR LTAC 160950Z 22009KT 180V240 9999 FEW040 BKN180 14/M02 Q1012 NOSIG",
        "METAR LTAC 160920Z 25006KT 200V270 9999 FEW040 BKN180 14/M02 Q1012 NOSIG",
        "METAR LTAC 160850Z 23010KT 9999 FEW040 BKN180 13/M03 Q1013 NOSIG",
        "METAR LTAC 160820Z 22010KT 9999 FEW040 BKN180 13/M02 Q1013 NOSIG",
        "METAR LTAC 160750Z 21011KT 9999 SCT040 BKN180 12/M03 Q1013 NOSIG",
        "METAR LTAC 160720Z 21008KT 9999 SCT040 BKN180 11/M01 Q1013 NOSIG",
        "METAR LTAC 160650Z 21007KT 9999 SCT040 BKN180 10/M00 Q1013 NOSIG",
        "METAR LTAC 160620Z 00000KT 9999 FEW040 BKN180 08/00 Q1013 NOSIG",
        "METAR LTAC 160550Z VRB02KT CAVOK 07/M00 Q1013 NOSIG",
        "METAR LTAC 160520Z 17005KT CAVOK 08/M01 Q1013 NOSIG",
        "METAR LTAC 160450Z VRB02KT CAVOK 07/M00 Q1013 NOSIG",
        "METAR LTAC 160420Z 18007KT CAVOK 08/M01 Q1013 NOSIG",
        "METAR LTAC 160350Z VRB02KT CAVOK 07/M00 Q1013 NOSIG",
        "METAR LTAC 160320Z VRB03KT CAVOK 07/M00 Q1013 NOSIG",
        "METAR LTAC 160250Z 17004KT 140V200 CAVOK 07/M00 Q1013 NOSIG",
        "METAR LTAC 160220Z 17008KT CAVOK 07/00 Q1013 NOSIG",
        "METAR LTAC 160150Z 17007KT CAVOK 07/01 Q1013 NOSIG",
        "METAR LTAC 160120Z 14003KT 100V180 CAVOK 08/01 Q1013 NOSIG",
        "METAR LTAC 160050Z VRB02KT CAVOK 07/01 Q1013 NOSIG",
        "METAR LTAC 160020Z VRB02KT CAVOK 08/01 Q1013 NOSIG",
    ]

    # Yesterday (Feb 15) context from METAR data
    # Feb 15 morning (04-08Z): temps ranged 4-8C, avg ~5C
    # Feb 15 max: 14C (observed ~1350-1420Z = 1650-1720 local)
    # Feb 15 morning pressure: ~1014 hPa
    prev_day_max = 14.0
    prev_morning_temp = 5.0
    prev_pressure = 1014.0
    day_of_year = 47  # Feb 16

    predict_from_metars(
        today_metars,
        prev_day_max=prev_day_max,
        prev_morning_temp=prev_morning_temp,
        prev_pressure=prev_pressure,
        day_of_year=day_of_year,
    )
