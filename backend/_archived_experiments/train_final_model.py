#!/usr/bin/env python3
"""
Smart Jal - Final Production Model
Combines all learnings into a practical groundwater prediction system.

What this model DOES:
- Predicts water levels with R² ~0.7 on future data
- Uses station history + environmental features + seasonal patterns
- Provides useful forecasts for water management

This is a WORKING model for the hackathon.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
    """Extract values from raster for multiple coordinates."""
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
    """Load satellite raster file paths."""
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


def prepare_comprehensive_dataset(
    cgwb_df: pd.DataFrame,
    dem_path: Path,
    rainfall_files: dict,
    satellite_files: dict
) -> tuple:
    """
    Prepare a comprehensive dataset with all useful features.
    """
    print("\nPreparing comprehensive dataset...")

    # Sort by station and date for lag features
    df = cgwb_df.copy()
    df = df.sort_values(['station_name', 'date'])

    # Create station-level features
    print("  Computing station statistics...")
    station_stats = df.groupby('station_name').agg({
        'currentlevel': ['mean', 'std', 'min', 'max', 'count'],
        'latitude': 'first',
        'longitude': 'first'
    })
    station_stats.columns = ['station_mean', 'station_std', 'station_min',
                             'station_max', 'n_measurements', 'lat', 'lon']
    station_stats['station_range'] = station_stats['station_max'] - station_stats['station_min']

    # Filter to stations with enough data
    valid_stations = station_stats[station_stats['n_measurements'] >= 4].index
    df = df[df['station_name'].isin(valid_stations)]
    print(f"  Stations with 4+ measurements: {len(valid_stations)}")

    # Create lag features
    print("  Creating lag features...")
    df['prev_level'] = df.groupby('station_name')['currentlevel'].shift(1)
    df['prev_level_2'] = df.groupby('station_name')['currentlevel'].shift(2)
    df['days_since_prev'] = df.groupby('station_name')['date'].diff().dt.days

    # Merge station stats
    df = df.merge(station_stats[['station_mean', 'station_std', 'station_range']],
                  left_on='station_name', right_index=True)

    # Drop rows without previous measurement
    df = df.dropna(subset=['prev_level'])
    print(f"  Records with history: {len(df)}")

    dates = df['date']
    coords = list(zip(df['longitude'], df['latitude']))

    # Build feature matrix
    features = pd.DataFrame(index=df.index)

    # Station features
    features['station_mean'] = df['station_mean']
    features['station_std'] = df['station_std'].fillna(1)
    features['station_range'] = df['station_range']

    # Autoregressive features
    features['prev_level'] = df['prev_level']
    features['prev_level_2'] = df['prev_level_2']
    features['days_since_prev'] = df['days_since_prev']

    # Deviation from mean
    features['prev_anomaly'] = df['prev_level'] - df['station_mean']

    # Extract elevation
    print("  Extracting elevation...")
    features['elevation'] = batch_extract_from_raster(dem_path, coords)

    # Extract rainfall
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

    # Extract satellite features
    print("  Extracting satellite features...")
    for sat_type in ['modis_et', 'ndvi', 'smap']:
        features[sat_type] = np.nan
        for (year, month), group in df.groupby([dates.dt.year, dates.dt.month]):
            key = (year, month)
            if key in satellite_files.get(sat_type, {}):
                group_coords = list(zip(group['longitude'], group['latitude']))
                values = batch_extract_from_raster(satellite_files[sat_type][key], group_coords)
                features.loc[group.index, sat_type] = values

    # Seasonal features
    print("  Adding seasonal features...")
    months = dates.dt.month
    features['month'] = months
    features['month_sin'] = np.sin(2 * np.pi * months / 12)
    features['month_cos'] = np.cos(2 * np.pi * months / 12)
    features['is_monsoon'] = months.isin([6, 7, 8, 9, 10]).astype(int)
    features['is_post_monsoon'] = months.isin([11, 12, 1]).astype(int)
    features['is_summer'] = months.isin([3, 4, 5]).astype(int)

    # Year trend
    features['year'] = dates.dt.year - 2018  # Normalized

    # Fill NaN
    for col in features.columns:
        if features[col].isna().any():
            median_val = features[col].median()
            features[col] = features[col].fillna(median_val if pd.notna(median_val) else 0)

    print(f"  Final dataset: {len(features)} samples, {len(features.columns)} features")

    return features, df['currentlevel'], dates, df


def train_and_evaluate(X, y, dates, train_end='2020-12-31'):
    """Train model and evaluate on held-out future data."""

    train_end = pd.Timestamp(train_end)
    train_mask = dates <= train_end
    test_mask = dates > train_end

    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()

    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*60}")
    print(f"Train: {len(X_train)} samples (up to {train_end.date()})")
    print(f"Test: {len(X_test)} samples (after {train_end.date()})")

    # Train Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )

    print("\nTraining Gradient Boosting...")
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nTraining Performance:")
    print(f"  R²: {train_r2:.3f}")
    print(f"  RMSE: {train_rmse:.2f}m")

    print(f"\nTest Performance (Future Prediction):")
    print(f"  R²: {test_r2:.3f}")
    print(f"  RMSE: {test_rmse:.2f}m")
    print(f"  MAE: {test_mae:.2f}m")

    # Baseline comparison
    baseline_pred = X_test['prev_level']
    baseline_r2 = r2_score(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

    print(f"\nBaseline (Previous Level Only):")
    print(f"  R²: {baseline_r2:.3f}")
    print(f"  RMSE: {baseline_rmse:.2f}m")

    improvement = test_r2 - baseline_r2
    print(f"\n>>> Model improves R² by {improvement:.3f} ({improvement/baseline_r2*100:.1f}% relative improvement)")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n{'='*60}")
    print("TOP 10 FEATURE IMPORTANCE")
    print(f"{'='*60}")
    print(importance.head(10).to_string(index=False))

    return model, {
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'baseline_r2': baseline_r2,
        'improvement': improvement
    }, importance


def main():
    print("="*70)
    print("SMART JAL - FINAL GROUNDWATER PREDICTION MODEL")
    print("="*70)

    # Load CGWB data
    cgwb_df = get_krishna_neighbors_data(min_date='2018-01-01')
    print(f"\nData: {len(cgwb_df)} records from {cgwb_df['station_name'].nunique()} stations")
    print(f"Districts: {cgwb_df['district_name'].unique().tolist()}")
    print(f"Date range: {cgwb_df['date'].min().date()} to {cgwb_df['date'].max().date()}")

    # Load raster data
    print("\nLoading spatial data...")
    raw_data = load_all_data()
    dem_path = raw_data['dem_path']
    rainfall_files = raw_data['rainfall_files']

    satellite_dir = Path(__file__).parent.parent / "downloaded_data" / "satellite"
    satellite_files = load_satellite_files(satellite_dir)
    print(f"Satellite data: ET={len(satellite_files['modis_et'])}, "
          f"NDVI={len(satellite_files['ndvi'])}, SMAP={len(satellite_files['smap'])} months")

    # Prepare dataset
    X, y, dates, df = prepare_comprehensive_dataset(
        cgwb_df, dem_path, rainfall_files, satellite_files
    )

    print(f"\nFeatures ({len(X.columns)}):")
    print(f"  {list(X.columns)}")

    # Train and evaluate
    model, metrics, importance = train_and_evaluate(X, y, dates)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY FOR HACKATHON")
    print(f"{'='*70}")
    print(f"""
This model predicts groundwater levels with:
  - Test R² = {metrics['test_r2']:.3f} (explains {metrics['test_r2']*100:.1f}% of variance)
  - Test RMSE = {metrics['test_rmse']:.2f}m (average error)
  - {metrics['improvement']/metrics['baseline_r2']*100:.0f}% improvement over naive baseline

The model learns from:
  1. Station history (mean levels, variability)
  2. Recent measurements (autoregressive)
  3. Seasonal patterns (monsoon effects)
  4. Environmental factors (rainfall, elevation)

This is a WORKING prediction system for water management.
""")

    # Save model
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "final_model.joblib"
    joblib.dump({
        'model': model,
        'feature_cols': list(X.columns),
        'metrics': metrics,
        'feature_importance': importance.to_dict(),
        'description': 'Production groundwater prediction model'
    }, model_path)
    print(f"Model saved to: {model_path}")

    return model, metrics


if __name__ == '__main__':
    model, metrics = main()
