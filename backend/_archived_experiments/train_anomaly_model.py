#!/usr/bin/env python3
"""
Smart Jal - Anomaly-Based Model Training
Predicts water level ANOMALIES (deviation from station mean) instead of absolute levels.

This removes the location bias and forces the model to learn:
- How rainfall affects water levels
- Seasonal patterns
- Temporal dynamics

Key insight: The model should learn "when it rains, levels rise" not "well X is 15m deep".
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
    except Exception as e:
        print(f"  Warning: Could not read {raster_path}: {e}")
    return results


def load_satellite_files(satellite_dir: Path) -> dict:
    """Load paths to satellite raster files."""
    satellite_files = {'modis_et': {}, 'ndvi': {}, 'smap': {}}

    for sat_type, pattern, parts_idx in [
        ('modis_et', 'modis_et/*.tif', 2),
        ('ndvi', 'ndvi/*.tif', 1),
        ('smap', 'smap/*.tif', 1)
    ]:
        sat_dir = satellite_dir / sat_type.replace('modis_et', 'modis_et')
        if sat_type == 'modis_et':
            sat_dir = satellite_dir / 'modis_et'
        elif sat_type == 'ndvi':
            sat_dir = satellite_dir / 'ndvi'
        elif sat_type == 'smap':
            sat_dir = satellite_dir / 'smap'

        if sat_dir.exists():
            for f in sat_dir.glob("*.tif"):
                parts = f.stem.split('_')
                if len(parts) > parts_idx:
                    year_month = parts[parts_idx].split('.')
                    if len(year_month) == 2:
                        year, month = int(year_month[0]), int(year_month[1])
                        satellite_files[sat_type][(year, month)] = f

    return satellite_files


def compute_station_anomalies(cgwb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute water level anomalies for each measurement.

    Anomaly = actual level - station mean level
    Positive anomaly = deeper than usual (worse)
    Negative anomaly = shallower than usual (better)
    """
    df = cgwb_df.copy()

    # Calculate station statistics
    station_stats = df.groupby('station_name')['currentlevel'].agg(['mean', 'std', 'count'])
    station_stats.columns = ['station_mean', 'station_std', 'n_measurements']

    # Filter to stations with at least 5 measurements
    valid_stations = station_stats[station_stats['n_measurements'] >= 5].index
    df = df[df['station_name'].isin(valid_stations)]

    # Merge stats back
    df = df.merge(station_stats[['station_mean', 'station_std']],
                  left_on='station_name', right_index=True)

    # Compute anomaly
    df['anomaly'] = df['currentlevel'] - df['station_mean']

    # Also compute normalized anomaly (z-score)
    df['anomaly_zscore'] = df['anomaly'] / df['station_std'].replace(0, 1)

    print(f"  Stations with 5+ measurements: {len(valid_stations)}")
    print(f"  Total records: {len(df)}")
    print(f"  Anomaly range: {df['anomaly'].min():.2f}m to {df['anomaly'].max():.2f}m")

    return df


def prepare_anomaly_dataset(
    cgwb_df: pd.DataFrame,
    dem_path: Path,
    rainfall_files: dict,
    satellite_files: dict
) -> tuple:
    """
    Prepare dataset for anomaly prediction.
    """
    print(f"\nPreparing anomaly dataset...")

    # Compute anomalies
    df = compute_station_anomalies(cgwb_df)

    coords = list(zip(df['longitude'], df['latitude']))
    dates = df['date']

    # Initialize features
    features = pd.DataFrame(index=df.index)

    # Rainfall features (these should correlate with anomalies)
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

    features['rainfall_cumulative_3m'] = (
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
    print("  Adding seasonal features...")
    months = dates.dt.month
    features['month_sin'] = np.sin(2 * np.pi * months / 12)
    features['month_cos'] = np.cos(2 * np.pi * months / 12)
    features['is_monsoon'] = months.isin([6, 7, 8, 9, 10]).astype(int)
    features['is_post_monsoon'] = months.isin([11, 12, 1]).astype(int)

    # Add rainfall anomaly (deviation from typical for that location)
    # This captures "more rain than usual" vs "less rain than usual"
    station_mean_rainfall = features.groupby(df['station_name'])['rainfall_current'].transform('mean')
    features['rainfall_anomaly'] = features['rainfall_current'] - station_mean_rainfall

    # Fill NaN
    for col in features.columns:
        if features[col].isna().any():
            median_val = features[col].median()
            features[col] = features[col].fillna(median_val if pd.notna(median_val) else 0)

    print(f"  Dataset ready: {len(features)} samples, {len(features.columns)} features")

    return features, df['anomaly'], dates, df


def train_anomaly_model(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_end_date: str = '2020-12-31'
) -> tuple:
    """Train model to predict water level anomalies."""
    train_end = pd.Timestamp(train_end_date)

    train_mask = dates <= train_end
    test_mask = dates > train_end

    X_train = X[train_mask].fillna(0)
    y_train = y[train_mask]
    X_test = X[test_mask].fillna(0)
    y_test = y[test_mask]

    print(f"\nTime-based split:")
    print(f"  Train: {len(X_train)} samples (before {train_end_date})")
    print(f"  Test: {len(X_test)} samples (after {train_end_date})")

    if len(X_test) == 0:
        print("  WARNING: No test samples!")
        return None, {}

    # Use Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )

    print("\nTraining anomaly model...")
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
    }

    return model, metrics


def main():
    print("=" * 70)
    print("ANOMALY-BASED MODEL TRAINING")
    print("Predicting deviation from station mean (not absolute levels)")
    print("=" * 70)

    # Load data
    cgwb_df = get_krishna_neighbors_data(min_date='2018-01-01')
    print(f"\nCGWB data: {len(cgwb_df)} records, {cgwb_df['station_name'].nunique()} stations")

    # Load rasters
    print("\nLoading raster data...")
    raw_data = load_all_data()
    dem_path = raw_data['dem_path']
    rainfall_files = raw_data['rainfall_files']

    satellite_dir = Path(__file__).parent.parent / "downloaded_data" / "satellite"
    satellite_files = load_satellite_files(satellite_dir)

    # Prepare anomaly dataset
    X, y_anomaly, dates, df = prepare_anomaly_dataset(
        cgwb_df, dem_path, rainfall_files, satellite_files
    )

    print(f"\nFeatures: {list(X.columns)}")
    print(f"\nAnomaly statistics:")
    print(f"  Mean: {y_anomaly.mean():.3f}m")
    print(f"  Std: {y_anomaly.std():.3f}m")

    # Train model
    model, metrics = train_anomaly_model(X, y_anomaly, dates)

    if model:
        print("\n" + "=" * 70)
        print("ANOMALY MODEL RESULTS")
        print("=" * 70)
        print(f"  Train R²: {metrics['train_r2']:.3f}")
        print(f"  Train RMSE: {metrics['train_rmse']:.3f}m")
        print(f"  Test R²: {metrics['test_r2']:.3f}")
        print(f"  Test RMSE: {metrics['test_rmse']:.3f}m")
        print(f"  Test MAE: {metrics['test_mae']:.3f}m")

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nFeature Importance:")
            print(importance.to_string(index=False))

        # Interpretation
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)
        if metrics['test_r2'] > 0.3:
            print(f">>> The model explains {metrics['test_r2']*100:.1f}% of water level CHANGES")
            print(">>> This indicates the model is learning temporal patterns!")
        elif metrics['test_r2'] > 0.1:
            print(f">>> The model explains {metrics['test_r2']*100:.1f}% of water level changes")
            print(">>> Some learning, but other factors dominate (pumping, geology)")
        else:
            print(f">>> The model explains only {metrics['test_r2']*100:.1f}% of changes")
            print(">>> Water level changes are driven by factors we can't observe")
            print(">>> (pumping patterns, local geology, aquifer connectivity)")

        # Save model
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        model_path = output_dir / "anomaly_model.joblib"
        joblib.dump({
            'model': model,
            'feature_cols': list(X.columns),
            'metrics': metrics,
            'description': 'Predicts water level anomaly (deviation from station mean)'
        }, model_path)
        print(f"\nModel saved to: {model_path}")


if __name__ == '__main__':
    main()
