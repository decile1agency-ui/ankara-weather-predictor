"""
LTAC Ankara Max Temperature Predictor - Web Dashboard
All times UTC (Zulu). Ankara local = UTC+3.
"""

import os
import json
import re
import numpy as np
import xgboost as xgb
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MORNING_START_UTC = 3
MORNING_END_UTC = 6

FEATURE_COLS = [
    'morning_temp', 'morning_temp_max', 'morning_dewpoint_depression',
    'morning_wind_speed', 'morning_gust', 'morning_wind_u', 'morning_wind_v',
    'morning_cloud_score', 'morning_cloud_base', 'morning_precip',
    'morning_pressure', 'day_of_year', 'prev_day_max',
    'pressure_tendency', 'temp_trend',
]

# Load model and medians at startup
model = xgb.XGBRegressor()
model.load_model(os.path.join(MODEL_DIR, "max_temp_predictor.json"))

with open(os.path.join(MODEL_DIR, "feature_medians.json")) as f:
    MEDIANS = json.load(f)

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    METADATA = json.load(f)


def extract_zulu_hour(metar_text):
    match = re.search(r'\d{2}(\d{2})\d{2}Z', metar_text)
    return int(match.group(1)) if match else None


def extract_zulu_time(metar_text):
    match = re.search(r'(\d{2})(\d{2})(\d{2})Z', metar_text)
    if match:
        return f"{match.group(1)}d {match.group(2)}:{match.group(3)}Z"
    return None


def parse_single_metar(metar_text):
    features = {}
    features['zulu_hour'] = extract_zulu_hour(metar_text)

    temp_match = re.search(r'\s(M?\d{2})/(M?\d{2})\s', metar_text)
    if temp_match:
        t_str, d_str = temp_match.group(1), temp_match.group(2)
        temp = -int(t_str[1:]) if t_str.startswith('M') else int(t_str)
        dewp = -int(d_str[1:]) if d_str.startswith('M') else int(d_str)
        features['temp'] = temp
        features['dewpoint'] = dewp
        features['dewpoint_depression'] = temp - dewp

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

    qnh_match = re.search(r'Q(\d{4})', metar_text)
    if qnh_match:
        features['pressure'] = int(qnh_match.group(1))

    cover_map = {"FEW": 1, "SCT": 3, "BKN": 6, "OVC": 8, "VV": 8}
    cloud_score = 0
    for code, val in cover_map.items():
        if code in metar_text:
            cloud_score = max(cloud_score, val)
    if "CAVOK" in metar_text or "CLR" in metar_text or "SKC" in metar_text:
        cloud_score = 0
    features['cloud_score'] = cloud_score

    cloud_bases = re.findall(r'(?:FEW|SCT|BKN|OVC|VV)(\d{3})', metar_text)
    features['cloud_base'] = min(int(b) for b in cloud_bases) if cloud_bases else 999

    precip_patterns = ["-RA", "RA ", "+RA", "-SN", "SN ", "+SN",
                       "-SHRA", "SHRA", "+SHRA", "TSRA", "DZ", "-DZ"]
    features['has_precip'] = 1 if any(p in metar_text for p in precip_patterns) else 0

    return features


def predict_max_temp(metar_list, prev_day_max, prev_morning_temp, prev_pressure, day_of_year):
    # Filter morning window
    morning_metars = []
    outside_metars = []
    all_parsed = []

    for m in metar_list:
        hour = extract_zulu_hour(m)
        parsed = parse_single_metar(m)
        entry = {
            'raw': m.strip(),
            'hour_z': hour,
            'hour_local': (hour + 3) % 24 if hour is not None else None,
            'temp': parsed.get('temp'),
            'dewpoint': parsed.get('dewpoint'),
            'wind_speed': parsed.get('wind_speed', 0),
            'gust': parsed.get('gust', 0),
            'cloud_score': parsed.get('cloud_score', 0),
            'pressure': parsed.get('pressure'),
            'in_window': False,
        }
        if hour is not None and MORNING_START_UTC <= hour <= MORNING_END_UTC:
            morning_metars.append(parsed)
            entry['in_window'] = True
        else:
            outside_metars.append(parsed)
        all_parsed.append(entry)

    if not morning_metars:
        morning_metars = [parse_single_metar(m) for m in metar_list]
        warning = "No METARs in 04-08Z window. Using all provided METARs."
    else:
        warning = None

    temps = [p['temp'] for p in morning_metars if 'temp' in p]
    dewp_dep = [p['dewpoint_depression'] for p in morning_metars if 'dewpoint_depression' in p]
    speeds = [p['wind_speed'] for p in morning_metars]
    gusts = [p['gust'] for p in morning_metars]
    wind_us = [p['wind_u'] for p in morning_metars]
    wind_vs = [p['wind_v'] for p in morning_metars]
    cloud_scores = [p['cloud_score'] for p in morning_metars]
    cloud_bases = [p['cloud_base'] for p in morning_metars]
    precips = [p['has_precip'] for p in morning_metars]
    pressures = [p['pressure'] for p in morning_metars if 'pressure' in p]

    morning_temp = np.mean(temps) if temps else MEDIANS.get('morning_temp', 10)
    morning_temp_max = max(temps) if temps else MEDIANS.get('morning_temp_max', 12)
    morning_pressure = np.mean(pressures) if pressures else MEDIANS.get('morning_pressure', 1013)

    features = {
        'morning_temp': round(float(morning_temp), 1),
        'morning_temp_max': round(float(morning_temp_max), 1),
        'morning_dewpoint_depression': round(float(np.mean(dewp_dep)) if dewp_dep else MEDIANS.get('morning_dewpoint_depression', 5), 1),
        'morning_wind_speed': round(float(np.mean(speeds)), 1),
        'morning_gust': round(float(max(gusts)), 1),
        'morning_wind_u': round(float(np.mean(wind_us)), 3),
        'morning_wind_v': round(float(np.mean(wind_vs)), 3),
        'morning_cloud_score': round(float(np.mean(cloud_scores)), 1),
        'morning_cloud_base': int(min(cloud_bases)),
        'morning_precip': int(max(precips)),
        'morning_pressure': round(float(morning_pressure), 1),
        'day_of_year': int(day_of_year),
        'prev_day_max': float(prev_day_max),
        'pressure_tendency': round(float(morning_pressure - prev_pressure), 1) if prev_pressure else 0,
        'temp_trend': round(float(morning_temp - prev_morning_temp), 1) if prev_morning_temp else 0,
    }

    X = np.array([[features[col] for col in FEATURE_COLS]])
    prediction = round(float(model.predict(X)[0]), 1)

    return {
        'prediction': prediction,
        'features': features,
        'metars_total': len(metar_list),
        'metars_in_window': len([e for e in all_parsed if e['in_window']]),
        'metars_outside': len([e for e in all_parsed if not e['in_window']]),
        'metar_details': all_parsed,
        'morning_window': f"{MORNING_START_UTC:02d}Z-{MORNING_END_UTC:02d}Z ({MORNING_START_UTC+3:02d}-{MORNING_END_UTC+3:02d} local)",
        'warning': warning,
        'model_info': {
            'mae': METADATA.get('cv_mae', 'N/A'),
            'rmse': METADATA.get('cv_rmse', 'N/A'),
            'r2': METADATA.get('cv_r2', 'N/A'),
            'samples': METADATA.get('n_samples', 'N/A'),
            'date_range': METADATA.get('date_range', 'N/A'),
        }
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    metar_text = data.get('metars', '').strip()
    prev_day_max = float(data.get('prev_day_max', 14))
    prev_morning_temp = float(data.get('prev_morning_temp', 5))
    prev_pressure = float(data.get('prev_pressure', 1013))
    day_of_year = int(data.get('day_of_year', 47))

    # Split METARs by line, filter empty lines
    lines = [l.strip() for l in metar_text.split('\n') if l.strip()]

    if not lines:
        return jsonify({'error': 'No METAR data provided'}), 400

    result = predict_max_temp(lines, prev_day_max, prev_morning_temp, prev_pressure, day_of_year)
    return jsonify(result)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'loaded', 'station': 'LTAC'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
