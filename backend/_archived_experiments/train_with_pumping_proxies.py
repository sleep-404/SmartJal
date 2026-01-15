#!/usr/bin/env python3
"""
Smart Jal - Model with Pumping Proxy Features
Adds extraction estimates from bore well and pumping data.
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


def compute_pumping_features(bore_wells: pd.DataFrame, pumping: pd.DataFrame) -> dict:
    """
    Compute village-level pumping proxy features.
    """
    print("\nComputing pumping proxy features...")

    # Clean pumping data
    pumping = pumping.dropna(subset=['village', 'no._of_functioning_wells'])
    pumping.columns = pumping.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '')

    # Clean bore well data - convert to numeric
    bore_wells = bore_wells.copy()
    bore_wells['depth_m'] = pd.to_numeric(bore_wells['depth_m'], errors='coerce')
    bore_wells['pump_capacity'] = pd.to_numeric(bore_wells['pump_capacity'], errors='coerce')
    bore_wells['irrigated_area'] = pd.to_numeric(bore_wells['irrigated_area'], errors='coerce')
    bore_wells['lat'] = pd.to_numeric(bore_wells['lat'], errors='coerce')
    bore_wells['lon'] = pd.to_numeric(bore_wells['lon'], errors='coerce')

    # Bore well aggregation by village
    print("  Aggregating bore well data by village...")
    well_agg = bore_wells.groupby('village').agg({
        'depth_m': ['mean', 'max', 'sum', 'count'],
        'pump_capacity': ['mean', 'sum'],
        'irrigated_area': ['sum', 'mean'],
        'lat': 'mean',
        'lon': 'mean'
    })
    well_agg.columns = ['_'.join(col).strip() for col in well_agg.columns.values]
    well_agg = well_agg.rename(columns={
        'depth_m_mean': 'avg_well_depth',
        'depth_m_max': 'max_well_depth',
        'depth_m_sum': 'total_well_depth',
        'depth_m_count': 'num_wells',
        'pump_capacity_mean': 'avg_pump_capacity',
        'pump_capacity_sum': 'total_pump_capacity',
        'irrigated_area_sum': 'total_irrigated_area',
        'irrigated_area_mean': 'avg_irrigated_area',
        'lat_mean': 'village_lat',
        'lon_mean': 'village_lon'
    })

    # Compute derived features
    well_agg['well_density'] = well_agg['num_wells']  # Will normalize later
    well_agg['extraction_potential'] = well_agg['num_wells'] * well_agg['avg_well_depth'] * well_agg['avg_pump_capacity'].fillna(5)

    print(f"  Villages with bore well data: {len(well_agg)}")

    return well_agg


def get_nearest_village_features(lat: float, lon: float, village_features: pd.DataFrame, max_dist: float = 0.1) -> dict:
    """
    Get pumping features from nearest village.
    """
    if 'village_lat' not in village_features.columns:
        return {}

    # Calculate distances
    distances = np.sqrt(
        (village_features['village_lat'] - lat)**2 +
        (village_features['village_lon'] - lon)**2
    )

    min_idx = distances.idxmin()
    min_dist = distances[min_idx]

    if min_dist > max_dist:
        return {}

    row = village_features.loc[min_idx]
    return {
        'num_wells': row.get('num_wells', 0),
        'avg_well_depth': row.get('avg_well_depth', 0),
        'total_pump_capacity': row.get('total_pump_capacity', 0),
        'total_irrigated_area': row.get('total_irrigated_area', 0),
        'extraction_potential': row.get('extraction_potential', 0),
        'well_density': row.get('well_density', 0)
    }


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


def prepare_dataset_with_pumping(
    cgwb_df: pd.DataFrame,
    dem_path: Path,
    rainfall_files: dict,
    satellite_files: dict,
    village_features: pd.DataFrame
) -> tuple:
    """
    Prepare dataset with pumping proxy features.
    """
    print("\nPreparing dataset with pumping proxies...")

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

    # Build features
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

    # PUMPING PROXY FEATURES
    print("  Adding pumping proxy features...")
    pumping_cols = ['num_wells', 'avg_well_depth', 'total_pump_capacity',
                    'total_irrigated_area', 'extraction_potential', 'well_density']

    for col in pumping_cols:
        features[col] = np.nan

    for idx, row in df.iterrows():
        pump_features = get_nearest_village_features(
            row['latitude'], row['longitude'], village_features
        )
        for col in pumping_cols:
            if col in pump_features:
                features.loc[idx, col] = pump_features[col]

    # Seasonal
    months = dates.dt.month
    features['month_sin'] = np.sin(2 * np.pi * months / 12)
    features['month_cos'] = np.cos(2 * np.pi * months / 12)
    features['is_monsoon'] = months.isin([6, 7, 8, 9, 10]).astype(int)
    features['is_post_monsoon'] = months.isin([11, 12, 1]).astype(int)
    features['is_summer'] = months.isin([3, 4, 5]).astype(int)
    features['year'] = dates.dt.year - 2018

    # Interaction: extraction during summer (high demand)
    features['summer_extraction'] = features['is_summer'] * features['extraction_potential'].fillna(0)

    # Fill NaN
    for col in features.columns:
        if features[col].isna().any():
            median_val = features[col].median()
            features[col] = features[col].fillna(median_val if pd.notna(median_val) else 0)

    print(f"  Final dataset: {len(features)} samples, {len(features.columns)} features")

    return features, df['currentlevel'], dates


def main():
    print("="*70)
    print("SMART JAL - MODEL WITH PUMPING PROXY FEATURES")
    print("="*70)

    # Load data
    cgwb_df = get_krishna_neighbors_data(min_date='2018-01-01')
    print(f"\nCGWB data: {len(cgwb_df)} records from {cgwb_df['station_name'].nunique()} stations")

    raw_data = load_all_data()
    dem_path = raw_data['dem_path']
    rainfall_files = raw_data['rainfall_files']
    bore_wells = raw_data['bore_wells']
    pumping = raw_data['pumping']

    satellite_dir = Path(__file__).parent.parent / "downloaded_data" / "satellite"
    satellite_files = load_satellite_files(satellite_dir)

    # Compute pumping features
    village_features = compute_pumping_features(bore_wells, pumping)

    # Prepare dataset
    X, y, dates = prepare_dataset_with_pumping(
        cgwb_df, dem_path, rainfall_files, satellite_files, village_features
    )

    print(f"\nFeatures ({len(X.columns)}):")
    print(f"  {list(X.columns)}")

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
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
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
    print("RESULTS WITH PUMPING PROXIES")
    print(f"{'='*60}")
    print(f"Train R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Test RMSE: {test_rmse:.2f}m")
    print(f"Test MAE: {test_mae:.2f}m")
    print(f"Baseline R²: {baseline_r2:.3f}")
    print(f"Improvement: {(test_r2 - baseline_r2)/baseline_r2*100:.1f}%")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Top 15)")
    print(f"{'='*60}")
    print(importance.head(15).to_string(index=False))

    # Check if pumping features helped
    pumping_features = ['num_wells', 'avg_well_depth', 'total_pump_capacity',
                        'total_irrigated_area', 'extraction_potential',
                        'well_density', 'summer_extraction']
    pumping_importance = importance[importance['feature'].isin(pumping_features)]
    print(f"\nPumping proxy features importance:")
    print(pumping_importance.to_string(index=False))
    print(f"Total pumping contribution: {pumping_importance['importance'].sum()*100:.1f}%")

    # Save
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "model_with_pumping.joblib"
    joblib.dump({
        'model': model,
        'feature_cols': list(X.columns),
        'test_r2': test_r2,
        'test_rmse': test_rmse
    }, model_path)
    print(f"\nModel saved: {model_path}")


if __name__ == '__main__':
    main()
