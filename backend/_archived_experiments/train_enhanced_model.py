#!/usr/bin/env python3
"""
Smart Jal - Enhanced Model Training
Uses CGWB data with satellite features and time-based split.

Optimized for speed with batch raster extraction.
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

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from data_processing.load_cgwb_data import load_cgwb_data, get_krishna_neighbors_data
from data_processing.load_data import load_all_data


def load_satellite_files(satellite_dir: Path) -> dict:
    """Load paths to satellite raster files organized by type and date."""
    satellite_files = {
        'modis_et': {},
        'ndvi': {},
        'smap': {}
    }

    # MODIS ET
    modis_dir = satellite_dir / "modis_et"
    if modis_dir.exists():
        for f in modis_dir.glob("*.tif"):
            parts = f.stem.split('_')
            if len(parts) >= 3:
                year_month = parts[2].split('.')
                if len(year_month) == 2:
                    year, month = int(year_month[0]), int(year_month[1])
                    satellite_files['modis_et'][(year, month)] = f

    # NDVI
    ndvi_dir = satellite_dir / "ndvi"
    if ndvi_dir.exists():
        for f in ndvi_dir.glob("*.tif"):
            parts = f.stem.split('_')
            if len(parts) >= 2:
                year_month = parts[1].split('.')
                if len(year_month) == 2:
                    year, month = int(year_month[0]), int(year_month[1])
                    satellite_files['ndvi'][(year, month)] = f

    # SMAP
    smap_dir = satellite_dir / "smap"
    if smap_dir.exists():
        for f in smap_dir.glob("*.tif"):
            parts = f.stem.split('_')
            if len(parts) >= 2:
                year_month = parts[1].split('.')
                if len(year_month) == 2:
                    year, month = int(year_month[0]), int(year_month[1])
                    satellite_files['smap'][(year, month)] = f

    print(f"Loaded satellite files: MODIS ET={len(satellite_files['modis_et'])}, "
          f"NDVI={len(satellite_files['ndvi'])}, SMAP={len(satellite_files['smap'])}")

    return satellite_files


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


def prepare_training_dataset(
    cgwb_df: pd.DataFrame,
    dem_path: Path,
    rainfall_files: dict,
    satellite_files: dict,
    include_location: bool = False
) -> tuple:
    """
    Prepare training dataset with optimized batch raster extraction.
    """
    print(f"\nPreparing dataset from {len(cgwb_df)} CGWB records...")

    # Create coordinate list
    coords = list(zip(cgwb_df['longitude'], cgwb_df['latitude']))
    dates = cgwb_df['date']
    targets = cgwb_df['currentlevel']

    # Initialize features dataframe
    features = pd.DataFrame(index=cgwb_df.index)

    # 1. Extract elevation (one raster read)
    print("  Extracting elevation...")
    features['elevation'] = batch_extract_from_raster(dem_path, coords)

    # 2. Extract rainfall for each unique (year, month) in the data
    print("  Extracting rainfall...")
    unique_periods = cgwb_df[['date']].copy()
    unique_periods['year'] = unique_periods['date'].dt.year
    unique_periods['month'] = unique_periods['date'].dt.month

    # Current rainfall
    features['rainfall_current'] = np.nan
    for (year, month), group in cgwb_df.groupby([dates.dt.year, dates.dt.month]):
        key = (year, month)
        if key in rainfall_files:
            group_coords = list(zip(group['longitude'], group['latitude']))
            values = batch_extract_from_raster(rainfall_files[key], group_coords)
            features.loc[group.index, 'rainfall_current'] = values

    # Rainfall lags (simplified - use previous months' data)
    for lag in [1, 2, 3]:
        col = f'rainfall_lag{lag}'
        features[col] = np.nan
        for (year, month), group in cgwb_df.groupby([dates.dt.year, dates.dt.month]):
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

    # Cumulative rainfall
    features['rainfall_cumulative_3m'] = (
        features['rainfall_current'].fillna(0) +
        features['rainfall_lag1'].fillna(0) +
        features['rainfall_lag2'].fillna(0)
    )

    # 3. Extract satellite features
    print("  Extracting satellite features...")

    # MODIS ET
    features['modis_et'] = np.nan
    for (year, month), group in cgwb_df.groupby([dates.dt.year, dates.dt.month]):
        key = (year, month)
        if key in satellite_files['modis_et']:
            group_coords = list(zip(group['longitude'], group['latitude']))
            values = batch_extract_from_raster(satellite_files['modis_et'][key], group_coords)
            features.loc[group.index, 'modis_et'] = values

    # NDVI
    features['ndvi'] = np.nan
    for (year, month), group in cgwb_df.groupby([dates.dt.year, dates.dt.month]):
        key = (year, month)
        if key in satellite_files['ndvi']:
            group_coords = list(zip(group['longitude'], group['latitude']))
            values = batch_extract_from_raster(satellite_files['ndvi'][key], group_coords)
            features.loc[group.index, 'ndvi'] = values

    # SMAP Soil Moisture
    features['smap_soil_moisture'] = np.nan
    for (year, month), group in cgwb_df.groupby([dates.dt.year, dates.dt.month]):
        key = (year, month)
        if key in satellite_files['smap']:
            group_coords = list(zip(group['longitude'], group['latitude']))
            values = batch_extract_from_raster(satellite_files['smap'][key], group_coords)
            features.loc[group.index, 'smap_soil_moisture'] = values

    # 4. Seasonal encoding
    print("  Adding seasonal features...")
    months = dates.dt.month
    features['month_sin'] = np.sin(2 * np.pi * months / 12)
    features['month_cos'] = np.cos(2 * np.pi * months / 12)
    features['is_monsoon'] = months.isin([6, 7, 8, 9, 10]).astype(int)
    features['is_post_monsoon'] = months.isin([11, 12, 1]).astype(int)
    features['is_pre_monsoon'] = months.isin([3, 4, 5]).astype(int)

    # 5. Optionally add location
    if include_location:
        features['latitude'] = cgwb_df['latitude']
        features['longitude'] = cgwb_df['longitude']

    # Fill NaN with medians
    for col in features.columns:
        if features[col].isna().any():
            median_val = features[col].median()
            if pd.isna(median_val):
                median_val = 0
            features[col] = features[col].fillna(median_val)

    print(f"  Dataset ready: {len(features)} samples, {len(features.columns)} features")

    return features, targets, dates


def train_with_time_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_end_date: str = '2020-12-31',
    model_type: str = 'gradient_boosting'
) -> tuple:
    """Train model using time-based split."""
    train_end = pd.Timestamp(train_end_date)

    train_mask = dates <= train_end
    test_mask = dates > train_end

    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()

    print(f"\nTime-based split:")
    print(f"  Train: {len(X_train)} samples (before {train_end_date})")
    print(f"  Test: {len(X_test)} samples (after {train_end_date})")

    if len(X_test) == 0:
        print("  WARNING: No test samples!")
        return None, {}

    # Handle any remaining NaN
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    y_train = y_train.fillna(y_train.median())
    y_test = y_test.fillna(y_test.median())

    # Initialize model
    if model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"\nTraining {model_type}...")
    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    return model, metrics


def main():
    print("=" * 70)
    print("ENHANCED MODEL TRAINING")
    print("Using CGWB data + satellite features + time-based split")
    print("=" * 70)

    # Load CGWB data for Krishna and neighbors
    cgwb_df = get_krishna_neighbors_data(min_date='2018-01-01')
    print(f"\nCGWB data: {len(cgwb_df)} records, {cgwb_df['station_name'].nunique()} stations")

    # Load raster data
    print("\nLoading raster data...")
    raw_data = load_all_data()
    dem_path = raw_data['dem_path']
    rainfall_files = raw_data['rainfall_files']

    # Load satellite data
    satellite_dir = Path(__file__).parent.parent / "downloaded_data" / "satellite"
    satellite_files = load_satellite_files(satellite_dir)

    # EXPERIMENT 1: Without location features
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Without location features (forces learning)")
    print("=" * 70)

    X_no_loc, y, dates = prepare_training_dataset(
        cgwb_df, dem_path, rainfall_files, satellite_files,
        include_location=False
    )

    print(f"\nFeatures: {list(X_no_loc.columns)}")

    model_no_loc, metrics_no_loc = train_with_time_split(
        X_no_loc, y, dates,
        train_end_date='2020-12-31',
        model_type='gradient_boosting'
    )

    if model_no_loc:
        print("\n" + "-" * 40)
        print("Results WITHOUT location features:")
        print(f"  Train R²: {metrics_no_loc['train_r2']:.3f}")
        print(f"  Train RMSE: {metrics_no_loc['train_rmse']:.3f}m")
        print(f"  Test R²: {metrics_no_loc['test_r2']:.3f}")
        print(f"  Test RMSE: {metrics_no_loc['test_rmse']:.3f}m")
        print(f"  Test MAE: {metrics_no_loc['test_mae']:.3f}m")

        if hasattr(model_no_loc, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X_no_loc.columns,
                'importance': model_no_loc.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nFeature Importance:")
            print(importance.to_string(index=False))

    # EXPERIMENT 2: With location features
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: With location features (risk of memorization)")
    print("=" * 70)

    X_with_loc, y, dates = prepare_training_dataset(
        cgwb_df, dem_path, rainfall_files, satellite_files,
        include_location=True
    )

    model_with_loc, metrics_with_loc = train_with_time_split(
        X_with_loc, y, dates,
        train_end_date='2020-12-31',
        model_type='gradient_boosting'
    )

    if model_with_loc:
        print("\n" + "-" * 40)
        print("Results WITH location features:")
        print(f"  Train R²: {metrics_with_loc['train_r2']:.3f}")
        print(f"  Train RMSE: {metrics_with_loc['train_rmse']:.3f}m")
        print(f"  Test R²: {metrics_with_loc['test_r2']:.3f}")
        print(f"  Test RMSE: {metrics_with_loc['test_rmse']:.3f}m")
        print(f"  Test MAE: {metrics_with_loc['test_mae']:.3f}m")

        if hasattr(model_with_loc, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X_with_loc.columns,
                'importance': model_with_loc.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nFeature Importance:")
            print(importance.to_string(index=False))

    # Summary
    if model_no_loc and model_with_loc:
        print("\n" + "=" * 70)
        print("SUMMARY: Learning vs Memorization")
        print("=" * 70)
        print(f"\nWithout location (actual learning):")
        print(f"  Test R² = {metrics_no_loc['test_r2']:.3f}, RMSE = {metrics_no_loc['test_rmse']:.3f}m")
        print(f"\nWith location (may memorize):")
        print(f"  Test R² = {metrics_with_loc['test_r2']:.3f}, RMSE = {metrics_with_loc['test_rmse']:.3f}m")

        gap = metrics_with_loc['test_r2'] - metrics_no_loc['test_r2']
        if gap > 0.1:
            print(f"\n>>> Location adds {gap:.3f} R² - model is likely memorizing locations!")
        else:
            print(f"\n>>> Location adds only {gap:.3f} R² - model is learning physics!")

        # Save best model
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        model_path = output_dir / "enhanced_model.joblib"
        joblib.dump({
            'model': model_no_loc,
            'feature_cols': list(X_no_loc.columns),
            'metrics': metrics_no_loc,
            'description': 'Model without location features - actual learning'
        }, model_path)
        print(f"\nModel saved to: {model_path}")


if __name__ == '__main__':
    main()
