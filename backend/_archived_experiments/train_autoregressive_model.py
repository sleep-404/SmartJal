#!/usr/bin/env python3
"""
Smart Jal - Autoregressive Model Training
Uses previous water level measurement to predict next level.

Key insight: Water levels have strong autocorrelation.
If we know the previous level, we can predict the next one better.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import rasterio
import warnings
import sys

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from data_processing.load_cgwb_data import load_cgwb_data, get_krishna_neighbors_data
from data_processing.load_data import load_all_data


def batch_extract_from_raster(raster_path: Path, coords: list) -> list:
    """Extract values from raster for multiple coordinates at once."""
    results = [np.nan] * len(coords)
    try:
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            for i, (lon, lat) in enumerate(coords):
                try:
                    py, px = src.index(lon, lat)
                    if 0 <= py < src.height and 0 <= px < src.width:
                        val = raster_data[py, px]
                        if src.nodata is None or val != src.nodata:
                            results[i] = float(val)
                except:
                    pass
    except:
        pass
    return results


def load_satellite_files(satellite_dir: Path) -> dict:
    """Load paths to satellite raster files."""
    satellite_files = {'modis_et': {}, 'ndvi': {}, 'smap': {}}

    for sat_type in ['modis_et', 'ndvi', 'smap']:
        sat_dir = satellite_dir / sat_type
        parts_idx = 2 if sat_type == 'modis_et' else 1

        if sat_dir.exists():
            for f in sat_dir.glob("*.tif"):
                parts = f.stem.split('_')
                if len(parts) > parts_idx:
                    year_month = parts[parts_idx].split('.')
                    if len(year_month) == 2:
                        year, month = int(year_month[0]), int(year_month[1])
                        satellite_files[sat_type][(year, month)] = f

    return satellite_files


def create_autoregressive_features(cgwb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create autoregressive features - previous water level for each station.
    """
    df = cgwb_df.copy()
    df = df.sort_values(['station_name', 'date'])

    # Create lag features per station
    df['prev_level'] = df.groupby('station_name')['currentlevel'].shift(1)
    df['prev_level_2'] = df.groupby('station_name')['currentlevel'].shift(2)

    # Time since last measurement
    df['days_since_prev'] = df.groupby('station_name')['date'].diff().dt.days

    # Level change from previous
    df['level_change'] = df['currentlevel'] - df['prev_level']

    # Station mean level (known from history)
    station_means = df.groupby('station_name')['currentlevel'].transform('mean')
    df['station_mean'] = station_means

    # Drop rows without previous measurement
    df = df.dropna(subset=['prev_level'])

    print(f"  Records with previous measurement: {len(df)}")
    print(f"  Unique stations: {df['station_name'].nunique()}")

    return df


def prepare_autoregressive_dataset(
    cgwb_df: pd.DataFrame,
    rainfall_files: dict,
    satellite_files: dict
) -> tuple:
    """
    Prepare dataset with autoregressive features.
    """
    print(f"\nPreparing autoregressive dataset...")

    # Create lag features
    df = create_autoregressive_features(cgwb_df)

    dates = df['date']
    coords = list(zip(df['longitude'], df['latitude']))

    # Initialize features
    features = pd.DataFrame(index=df.index)

    # Key autoregressive features
    features['prev_level'] = df['prev_level']
    features['prev_level_2'] = df['prev_level_2']
    features['station_mean'] = df['station_mean']
    features['days_since_prev'] = df['days_since_prev']

    # Rainfall features
    print("  Extracting rainfall...")
    features['rainfall_current'] = np.nan
    for (year, month), group in df.groupby([dates.dt.year, dates.dt.month]):
        key = (year, month)
        if key in rainfall_files:
            group_coords = list(zip(group['longitude'], group['latitude']))
            values = batch_extract_from_raster(rainfall_files[key], group_coords)
            features.loc[group.index, 'rainfall_current'] = values

    for lag in [1, 2, 3]:
        col = f'rainfall_lag{lag}'
        features[col] = np.nan
        for (year, month), group in df.groupby([dates.dt.year, dates.dt.month]):
            lag_month = month - lag
            lag_year = year
            if lag_month <= 0:
                lag_month += 12
                lag_year -= 1
            key = (lag_year, lag_month)
            if key in rainfall_files:
                group_coords = list(zip(group['longitude'], group['latitude']))
                values = batch_extract_from_raster(rainfall_files[key], group_coords)
                features.loc[group.index, col] = values

    features['rainfall_cumulative'] = (
        features['rainfall_current'].fillna(0) +
        features['rainfall_lag1'].fillna(0) +
        features['rainfall_lag2'].fillna(0)
    )

    # Satellite features
    print("  Extracting satellite features...")
    for sat_type in ['modis_et', 'ndvi', 'smap']:
        features[sat_type] = np.nan
        for (year, month), group in df.groupby([dates.dt.year, dates.dt.month]):
            key = (year, month)
            if key in satellite_files[sat_type]:
                group_coords = list(zip(group['longitude'], group['latitude']))
                values = batch_extract_from_raster(satellite_files[sat_type][key], group_coords)
                features.loc[group.index, sat_type] = values

    # Seasonal encoding
    months = dates.dt.month
    features['month_sin'] = np.sin(2 * np.pi * months / 12)
    features['month_cos'] = np.cos(2 * np.pi * months / 12)
    features['is_monsoon'] = months.isin([6, 7, 8, 9, 10]).astype(int)

    # Fill NaN
    for col in features.columns:
        if features[col].isna().any():
            median_val = features[col].median()
            features[col] = features[col].fillna(median_val if pd.notna(median_val) else 0)

    print(f"  Dataset ready: {len(features)} samples, {len(features.columns)} features")

    return features, df['currentlevel'], dates, df


def main():
    print("=" * 70)
    print("AUTOREGRESSIVE MODEL TRAINING")
    print("Using previous water level as key feature")
    print("=" * 70)

    # Load data
    cgwb_df = get_krishna_neighbors_data(min_date='2018-01-01')
    print(f"\nCGWB data: {len(cgwb_df)} records, {cgwb_df['station_name'].nunique()} stations")

    # Load rasters
    print("\nLoading raster data...")
    raw_data = load_all_data()
    rainfall_files = raw_data['rainfall_files']

    satellite_dir = Path(__file__).parent.parent / "downloaded_data" / "satellite"
    satellite_files = load_satellite_files(satellite_dir)

    # Prepare dataset
    X, y, dates, df = prepare_autoregressive_dataset(
        cgwb_df, rainfall_files, satellite_files
    )

    print(f"\nFeatures: {list(X.columns)}")

    # Time-based split
    train_end = pd.Timestamp('2020-12-31')
    train_mask = dates <= train_end
    test_mask = dates > train_end

    X_train = X[train_mask].fillna(0)
    y_train = y[train_mask]
    X_test = X[test_mask].fillna(0)
    y_test = y[test_mask]

    print(f"\nTime-based split:")
    print(f"  Train: {len(X_train)} samples (before 2020-12-31)")
    print(f"  Test: {len(X_test)} samples (after 2020-12-31)")

    # Train Gradient Boosting
    print("\n" + "=" * 70)
    print("GRADIENT BOOSTING MODEL")
    print("=" * 70)

    model_gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )

    model_gb.fit(X_train, y_train)

    y_train_pred = model_gb.predict(X_train)
    y_test_pred = model_gb.predict(X_test)

    print(f"\nResults:")
    print(f"  Train R²: {r2_score(y_train, y_train_pred):.3f}")
    print(f"  Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.3f}m")
    print(f"  Test R²: {r2_score(y_test, y_test_pred):.3f}")
    print(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.3f}m")
    print(f"  Test MAE: {mean_absolute_error(y_test, y_test_pred):.3f}m")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model_gb.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(importance.to_string(index=False))

    # Compare with simple baseline: predict prev_level
    baseline_pred = X_test['prev_level']
    baseline_r2 = r2_score(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)
    print(f"\nBaseline (just use previous level):")
    print(f"  Test R²: {baseline_r2:.3f}")
    print(f"  Test RMSE: {baseline_rmse:.3f}m")

    print(f"\nAutoregressive model:")
    print(f"  Test R²: {r2_score(y_test, y_test_pred):.3f}")
    print(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.3f}m")

    improvement = r2_score(y_test, y_test_pred) - baseline_r2
    print(f"\n>>> Model improves R² by {improvement:.3f} over naive baseline")

    if improvement > 0.05:
        print(">>> The environmental features add predictive value!")
    else:
        print(">>> Environmental features add minimal value beyond previous level")
        print(">>> Water levels are mostly persistent with some drift")

    # Save model
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "autoregressive_model.joblib"
    joblib.dump({
        'model': model_gb,
        'feature_cols': list(X.columns),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'baseline_r2': baseline_r2,
        'description': 'Autoregressive model using previous water level'
    }, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == '__main__':
    main()
