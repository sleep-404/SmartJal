#!/usr/bin/env python3
"""
Smart Jal - Model with Dynamic (Time-Varying) Proxies
Uses GRACE, dry-season NDVI, and ET-rainfall gap as extraction proxies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import rasterio
import warnings
import sys

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from data_processing.load_cgwb_data import get_krishna_neighbors_data
from data_processing.load_data import load_all_data


def batch_extract_from_raster(raster_path: Path, coords: list) -> list:
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


def prepare_dataset_with_dynamic_proxies(
    cgwb_df: pd.DataFrame,
    dem_path: Path,
    rainfall_files: dict,
    satellite_files: dict,
    grace_df: pd.DataFrame
) -> tuple:
    """
    Prepare dataset with time-varying extraction proxies.
    """
    print("\nPreparing dataset with dynamic proxies...")

    df = cgwb_df.copy().sort_values(['station_name', 'date'])

    # Station stats
    station_stats = df.groupby('station_name').agg({
        'currentlevel': ['mean', 'std', 'min', 'max', 'count'],
        'latitude': 'first',
        'longitude': 'first'
    })
    station_stats.columns = ['station_mean', 'station_std', 'station_min',
                             'station_max', 'n_measurements', 'lat', 'lon']
    station_stats['station_range'] = station_stats['station_max'] - station_stats['station_min']

    valid_stations = station_stats[station_stats['n_measurements'] >= 4].index
    df = df[df['station_name'].isin(valid_stations)]

    # Lag features
    df['prev_level'] = df.groupby('station_name')['currentlevel'].shift(1)
    df['prev_level_2'] = df.groupby('station_name')['currentlevel'].shift(2)
    df['days_since_prev'] = df.groupby('station_name')['date'].diff().dt.days

    df = df.merge(station_stats[['station_mean', 'station_std', 'station_range']],
                  left_on='station_name', right_index=True)
    df = df.dropna(subset=['prev_level'])

    print(f"  Records with history: {len(df)}")

    dates = df['date']
    coords = list(zip(df['longitude'], df['latitude']))

    features = pd.DataFrame(index=df.index)

    # Station features
    features['station_mean'] = df['station_mean']
    features['station_std'] = df['station_std'].fillna(1)
    features['station_range'] = df['station_range']
    features['prev_level'] = df['prev_level']
    features['prev_level_2'] = df['prev_level_2']
    features['days_since_prev'] = df['days_since_prev']
    features['prev_anomaly'] = df['prev_level'] - df['station_mean']

    # Elevation
    print("  Extracting elevation...")
    features['elevation'] = batch_extract_from_raster(dem_path, coords)

    # Rainfall
    print("  Extracting rainfall...")
    for col_name, lag in [('rainfall_current', 0), ('rainfall_lag1', 1),
                          ('rainfall_lag2', 2), ('rainfall_lag3', 3)]:
        features[col_name] = np.nan
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
                features.loc[group.index, col_name] = values

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
            if key in satellite_files.get(sat_type, {}):
                group_coords = list(zip(group['longitude'], group['latitude']))
                values = batch_extract_from_raster(satellite_files[sat_type][key], group_coords)
                features.loc[group.index, sat_type] = values

    # === DYNAMIC PROXIES ===

    # 1. GRACE TWS Anomaly (regional groundwater indicator)
    print("  Adding GRACE TWS anomaly...")
    grace_df['date'] = pd.to_datetime(grace_df['date'])
    grace_df = grace_df.set_index('date')

    features['grace_tws'] = np.nan
    features['grace_trend'] = np.nan
    for idx, row in df.iterrows():
        dt = row['date']
        # Find nearest month in GRACE
        grace_month = dt.replace(day=1)
        if grace_month in grace_df.index:
            features.loc[idx, 'grace_tws'] = grace_df.loc[grace_month, 'tws_anomaly_cm']
            features.loc[idx, 'grace_trend'] = grace_df.loc[grace_month, 'trend_component']

    # 2. ET - Rainfall gap (irrigation proxy)
    print("  Computing ET-rainfall gap...")
    features['et_rainfall_gap'] = features['modis_et'].fillna(0) - features['rainfall_current'].fillna(0)
    # Positive gap = ET exceeds rainfall = irrigation happening
    features['et_rainfall_gap'] = features['et_rainfall_gap'].clip(lower=0)

    # 3. Dry season NDVI (irrigation activity)
    print("  Computing dry season NDVI proxy...")
    months = dates.dt.month
    features['is_dry_season'] = months.isin([3, 4, 5]).astype(int)
    # High NDVI in dry season = irrigation
    features['dry_season_ndvi'] = features['ndvi'] * features['is_dry_season']

    # 4. SMAP anomaly (soil moisture deviation)
    smap_mean = features['smap'].mean()
    features['smap_anomaly'] = features['smap'] - smap_mean

    # Seasonal
    features['month_sin'] = np.sin(2 * np.pi * months / 12)
    features['month_cos'] = np.cos(2 * np.pi * months / 12)
    features['is_monsoon'] = months.isin([6, 7, 8, 9, 10]).astype(int)
    features['is_post_monsoon'] = months.isin([11, 12, 1]).astype(int)
    features['is_summer'] = months.isin([3, 4, 5]).astype(int)
    features['year'] = dates.dt.year - 2018

    # Interactions
    features['grace_x_summer'] = features['grace_tws'].fillna(0) * features['is_summer']
    features['et_gap_x_summer'] = features['et_rainfall_gap'] * features['is_summer']

    # Fill NaN
    for col in features.columns:
        if features[col].isna().any():
            median_val = features[col].median()
            features[col] = features[col].fillna(median_val if pd.notna(median_val) else 0)

    print(f"  Final dataset: {len(features)} samples, {len(features.columns)} features")

    return features, df['currentlevel'], dates


def main():
    print("="*70)
    print("SMART JAL - MODEL WITH DYNAMIC EXTRACTION PROXIES")
    print("="*70)

    cgwb_df = get_krishna_neighbors_data(min_date='2018-01-01')
    print(f"\nCGWB data: {len(cgwb_df)} records")

    raw_data = load_all_data()
    dem_path = raw_data['dem_path']
    rainfall_files = raw_data['rainfall_files']
    grace_df = raw_data['grace']

    satellite_dir = Path(__file__).parent.parent / "downloaded_data" / "satellite"
    satellite_files = load_satellite_files(satellite_dir)

    X, y, dates = prepare_dataset_with_dynamic_proxies(
        cgwb_df, dem_path, rainfall_files, satellite_files, grace_df
    )

    print(f"\nFeatures ({len(X.columns)}):")
    for i in range(0, len(X.columns), 5):
        print(f"  {list(X.columns)[i:i+5]}")

    # Time split
    train_end = pd.Timestamp('2020-12-31')
    train_mask = dates <= train_end
    test_mask = dates > train_end

    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # Train
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )

    print("\nTraining...")
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    baseline_r2 = r2_score(y_test, X_test['prev_level'])

    print(f"\n{'='*60}")
    print("RESULTS WITH DYNAMIC PROXIES")
    print(f"{'='*60}")
    print(f"Train R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Test RMSE: {test_rmse:.2f}m")
    print(f"Test MAE: {test_mae:.2f}m")
    print(f"Baseline R²: {baseline_r2:.3f}")
    print(f"Improvement over baseline: {(test_r2 - baseline_r2)/baseline_r2*100:.1f}%")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Top 15)")
    print(f"{'='*60}")
    print(importance.head(15).to_string(index=False))

    # Dynamic proxy contribution
    dynamic_features = ['grace_tws', 'grace_trend', 'et_rainfall_gap',
                        'dry_season_ndvi', 'smap_anomaly', 'grace_x_summer',
                        'et_gap_x_summer']
    dynamic_importance = importance[importance['feature'].isin(dynamic_features)]
    print(f"\nDynamic proxy features:")
    print(dynamic_importance.to_string(index=False))
    total_dynamic = dynamic_importance['importance'].sum() * 100
    print(f"Total dynamic proxy contribution: {total_dynamic:.1f}%")

    # Save
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "model_with_dynamic_proxies.joblib"
    joblib.dump({
        'model': model,
        'feature_cols': list(X.columns),
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'dynamic_proxy_contribution': total_dynamic
    }, model_path)
    print(f"\nModel saved: {model_path}")


if __name__ == '__main__':
    main()
