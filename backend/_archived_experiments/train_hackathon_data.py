#!/usr/bin/env python3
"""
Smart Jal - Training on Hackathon Data
Uses the actual hackathon-provided 138 piezometers (1997-2025)
Train: 1997-2023, Validate: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import rasterio
import datetime
import warnings
import sys

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).parent.parent


def load_hackathon_water_levels():
    """Load the 138 piezometers from hackathon data."""
    file_path = BASE_DIR / "SmartJal_extracted" / "SmartJal" / "WaterLevels_Krishna" / "master data_updated.xlsx"
    df = pd.read_excel(file_path)

    # Identify metadata columns vs date columns
    date_cols = [c for c in df.columns if isinstance(c, datetime.datetime)]
    meta_cols = [c for c in df.columns if not isinstance(c, datetime.datetime)]

    print(f"Loaded {len(df)} piezometers")
    print(f"Date range: {min(date_cols).strftime('%Y-%m')} to {max(date_cols).strftime('%Y-%m')}")
    print(f"Total months: {len(date_cols)}")

    # Melt to long format
    df_meta = df[meta_cols].copy()
    df_meta.columns = df_meta.columns.astype(str).str.strip()

    # Rename key columns
    rename_map = {
        'ID': 'piezo_id',
        'Village Name': 'village',
        'Mandal Name': 'mandal',
        'Latitude \n(Decimal Degrees)': 'lat',
        'Longitude \n(Decimal Degrees)': 'lon',
        'Total \nDepth \nin m': 'total_depth',
        'Principal Aquifer': 'aquifer',
        'MSL in meters': 'msl'
    }
    df_meta = df_meta.rename(columns=rename_map)

    # Melt date columns to long format
    records = []
    for idx, row in df.iterrows():
        meta = df_meta.iloc[idx]
        for date_col in date_cols:
            value = row[date_col]
            try:
                value = float(value)
                if pd.notna(value) and value > 0 and value < 100:  # Valid water level
                    records.append({
                        'piezo_id': meta.get('piezo_id'),
                        'village': meta.get('village'),
                        'mandal': meta.get('mandal'),
                        'lat': meta.get('lat'),
                        'lon': meta.get('lon'),
                        'total_depth': meta.get('total_depth'),
                        'aquifer': meta.get('aquifer'),
                        'msl': meta.get('msl'),
                        'date': date_col,
                        'water_level': value
                    })
            except (ValueError, TypeError):
                pass

    df_long = pd.DataFrame(records)
    df_long['date'] = pd.to_datetime(df_long['date'])

    print(f"Total records (long format): {len(df_long)}")
    return df_long


def load_grace_data():
    """Load GRACE groundwater storage data."""
    file_path = BASE_DIR / "downloaded_data" / "grace" / "grace_krishna_proxy.csv"
    grace = pd.read_csv(file_path)
    grace['date'] = pd.to_datetime(grace['date'])
    grace = grace.set_index('date')
    print(f"GRACE data: {grace.index.min().strftime('%Y-%m')} to {grace.index.max().strftime('%Y-%m')}")
    return grace


def load_satellite_files():
    """Load satellite raster file paths."""
    sat_dir = BASE_DIR / "downloaded_data" / "satellite"
    satellite_files = {'modis_et': {}, 'ndvi': {}, 'smap': {}}

    for sat_type in ['modis_et', 'ndvi', 'smap']:
        type_dir = sat_dir / sat_type
        parts_idx = 2 if sat_type == 'modis_et' else 1
        if type_dir.exists():
            for f in type_dir.glob("*.tif"):
                parts = f.stem.split('_')
                if len(parts) > parts_idx:
                    year_month = parts[parts_idx].split('.')
                    if len(year_month) == 2:
                        year, month = int(year_month[0]), int(year_month[1])
                        satellite_files[sat_type][(year, month)] = f

    for sat_type in satellite_files:
        print(f"  {sat_type}: {len(satellite_files[sat_type])} months")

    return satellite_files


def batch_extract_raster(raster_path, coords):
    """Extract values from raster for multiple coordinates."""
    results = [np.nan] * len(coords)
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            for i, (lon, lat) in enumerate(coords):
                try:
                    py, px = src.index(lon, lat)
                    if 0 <= py < src.height and 0 <= px < src.width:
                        val = data[py, px]
                        if src.nodata is None or val != src.nodata:
                            results[i] = float(val)
                except:
                    pass
    except:
        pass
    return results


def prepare_features(df, grace_df, satellite_files):
    """Prepare features for the model."""
    print("\nPreparing features...")

    df = df.copy().sort_values(['piezo_id', 'date'])

    # Compute station statistics
    print("  Computing station statistics...")
    station_stats = df.groupby('piezo_id').agg({
        'water_level': ['mean', 'std', 'min', 'max', 'count']
    })
    station_stats.columns = ['station_mean', 'station_std', 'station_min', 'station_max', 'n_obs']
    station_stats['station_range'] = station_stats['station_max'] - station_stats['station_min']

    # Filter stations with enough data
    valid_stations = station_stats[station_stats['n_obs'] >= 12].index
    df = df[df['piezo_id'].isin(valid_stations)]
    print(f"  Stations with 12+ observations: {len(valid_stations)}")

    # Create lag features
    print("  Creating lag features...")
    df['prev_level'] = df.groupby('piezo_id')['water_level'].shift(1)
    df['prev_level_2'] = df.groupby('piezo_id')['water_level'].shift(2)
    df['prev_level_3'] = df.groupby('piezo_id')['water_level'].shift(3)

    # Merge station stats
    df = df.merge(station_stats[['station_mean', 'station_std', 'station_range']],
                  left_on='piezo_id', right_index=True)

    # Drop rows without previous observation
    df = df.dropna(subset=['prev_level'])
    print(f"  Records with lag features: {len(df)}")

    # Build feature matrix
    features = pd.DataFrame(index=df.index)

    # Station features
    features['station_mean'] = df['station_mean']
    features['station_std'] = df['station_std'].fillna(1)
    features['station_range'] = df['station_range']
    features['total_depth'] = pd.to_numeric(df['total_depth'], errors='coerce').fillna(50)

    # Autoregressive features
    features['prev_level'] = df['prev_level']
    features['prev_level_2'] = df['prev_level_2']
    features['prev_level_3'] = df['prev_level_3']
    features['prev_anomaly'] = df['prev_level'] - df['station_mean']

    # GRACE features
    print("  Adding GRACE features...")
    features['grace_tws'] = np.nan
    features['grace_trend'] = np.nan
    for idx, row in df.iterrows():
        dt = row['date'].replace(day=1)
        if dt in grace_df.index:
            features.loc[idx, 'grace_tws'] = grace_df.loc[dt, 'tws_anomaly_cm']
            features.loc[idx, 'grace_trend'] = grace_df.loc[dt, 'trend_component']

    # Satellite features (for records where available)
    print("  Adding satellite features...")
    dates = df['date']
    coords = list(zip(df['lon'], df['lat']))

    for sat_type in ['modis_et', 'ndvi', 'smap']:
        features[sat_type] = np.nan
        for (year, month), group in df.groupby([dates.dt.year, dates.dt.month]):
            key = (year, month)
            if key in satellite_files.get(sat_type, {}):
                group_coords = list(zip(group['lon'], group['lat']))
                values = batch_extract_raster(satellite_files[sat_type][key], group_coords)
                features.loc[group.index, sat_type] = values

    # Seasonal features
    print("  Adding seasonal features...")
    months = dates.dt.month
    features['month_sin'] = np.sin(2 * np.pi * months / 12)
    features['month_cos'] = np.cos(2 * np.pi * months / 12)
    features['is_monsoon'] = months.isin([6, 7, 8, 9, 10]).astype(int)
    features['is_summer'] = months.isin([3, 4, 5]).astype(int)
    features['year'] = dates.dt.year - 1997

    # Interaction features
    features['grace_x_summer'] = features['grace_tws'].fillna(0) * features['is_summer']

    # Fill NaN
    for col in features.columns:
        if features[col].isna().any():
            median = features[col].median()
            features[col] = features[col].fillna(median if pd.notna(median) else 0)

    print(f"  Final: {len(features)} samples, {len(features.columns)} features")

    return features, df['water_level'], df['date'], df


def main():
    print("=" * 70)
    print("SMART JAL - TRAINING ON HACKATHON DATA")
    print("Train: 1997-2023 | Validate: 2024")
    print("=" * 70)

    # Load data
    df = load_hackathon_water_levels()
    grace_df = load_grace_data()

    print("\nLoading satellite data...")
    satellite_files = load_satellite_files()

    # Prepare features
    X, y, dates, df_full = prepare_features(df, grace_df, satellite_files)

    print(f"\nFeatures: {list(X.columns)}")

    # Time-based split: Train on 1997-2023, Test on 2024
    train_end = pd.Timestamp('2023-12-31')

    train_mask = dates <= train_end
    test_mask = dates > train_end

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"\n{'=' * 70}")
    print("DATA SPLIT")
    print(f"{'=' * 70}")
    print(f"Train: {len(X_train)} samples (1997-2023)")
    print(f"Test:  {len(X_test)} samples (2024)")

    if len(X_test) == 0:
        print("ERROR: No test samples!")
        return

    # Train model
    print(f"\n{'=' * 70}")
    print("TRAINING MODEL")
    print(f"{'=' * 70}")

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )

    print("Training Gradient Boosting...")
    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Baseline
    baseline_r2 = r2_score(y_test, X_test['prev_level'])
    baseline_rmse = np.sqrt(mean_squared_error(y_test, X_test['prev_level']))

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"\nTraining (1997-2023):")
    print(f"  R²:   {train_r2:.3f}")
    print(f"  RMSE: {train_rmse:.2f}m")

    print(f"\nValidation (2024) - THIS IS THE KEY METRIC:")
    print(f"  R²:   {test_r2:.3f} ({test_r2*100:.1f}% variance explained)")
    print(f"  RMSE: {test_rmse:.2f}m")
    print(f"  MAE:  {test_mae:.2f}m")

    print(f"\nBaseline (just use previous month's level):")
    print(f"  R²:   {baseline_r2:.3f}")
    print(f"  RMSE: {baseline_rmse:.2f}m")

    improvement = (test_r2 - baseline_r2) / baseline_r2 * 100
    print(f"\n>>> Model improves {improvement:.1f}% over baseline")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n{'=' * 70}")
    print("FEATURE IMPORTANCE (Top 10)")
    print(f"{'=' * 70}")
    print(importance.head(10).to_string(index=False))

    # Save model
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "hackathon_model_2024.joblib"
    joblib.dump({
        'model': model,
        'feature_cols': list(X.columns),
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'baseline_r2': baseline_r2,
        'train_period': '1997-2023',
        'test_period': '2024'
    }, model_path)

    print(f"\nModel saved: {model_path}")

    return model, test_r2, test_rmse


if __name__ == '__main__':
    main()
