#!/usr/bin/env python3
"""
Smart Jal - Feature Engineering Module
Creates feature matrix for ML model training.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
import warnings

warnings.filterwarnings('ignore')


def extract_raster_value_for_point(raster_path: Path, lon: float, lat: float) -> float:
    """
    Extract raster value at a specific point.

    Args:
        raster_path: Path to raster file
        lon: Longitude
        lat: Latitude

    Returns:
        Raster value at point
    """
    with rasterio.open(raster_path) as src:
        # Convert lat/lon to raster coordinates
        row, col = src.index(lon, lat)

        # Handle out-of-bounds
        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            return np.nan

        # Read value
        value = src.read(1)[row, col]

        # Handle nodata
        if src.nodata is not None and value == src.nodata:
            return np.nan

        return float(value)


def extract_raster_stats_for_polygon(raster_path: Path,
                                      geometry,
                                      stats: List[str] = ['mean']) -> Dict[str, float]:
    """
    Extract raster statistics for a polygon.

    Args:
        raster_path: Path to raster file
        geometry: Shapely geometry
        stats: List of statistics to compute ('mean', 'max', 'min', 'std', 'sum')

    Returns:
        Dict of statistic name to value
    """
    try:
        with rasterio.open(raster_path) as src:
            # Get the source nodata value
            src_nodata = src.nodata

            # Mask raster to polygon - use source nodata for integer types
            # Don't set nodata parameter to let it use the source's nodata
            out_image, out_transform = mask(src, [geometry], crop=True)
            data = out_image[0].astype(float)  # Convert to float for proper NaN handling

            # Replace nodata values with NaN
            if src_nodata is not None:
                data[data == src_nodata] = np.nan

            # Get valid data
            valid_data = data[~np.isnan(data)]

            if len(valid_data) == 0:
                return {s: np.nan for s in stats}

            results = {}
            for stat in stats:
                if stat == 'mean':
                    results[stat] = float(np.nanmean(valid_data))
                elif stat == 'max':
                    results[stat] = float(np.nanmax(valid_data))
                elif stat == 'min':
                    results[stat] = float(np.nanmin(valid_data))
                elif stat == 'std':
                    results[stat] = float(np.nanstd(valid_data))
                elif stat == 'sum':
                    results[stat] = float(np.nansum(valid_data))

            return results
    except Exception as e:
        return {s: np.nan for s in stats}


def calculate_slope_from_dem(dem_path: Path) -> np.ndarray:
    """
    Calculate slope from DEM using gradient.

    Args:
        dem_path: Path to DEM raster

    Returns:
        Slope array in degrees
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)

        # Handle nodata
        nodata = src.nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan

        # Calculate gradient (assuming ~30m resolution)
        cell_size = abs(src.transform[0])  # Cell size in degrees
        cell_size_m = cell_size * 111320  # Approximate meters at equator

        dy, dx = np.gradient(dem, cell_size_m)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)

        return slope_deg, src.profile


def extract_terrain_features(villages: gpd.GeoDataFrame,
                              dem_path: Path) -> gpd.GeoDataFrame:
    """
    Extract terrain features for each village.

    Features:
    - elevation_mean, elevation_min, elevation_max
    - slope_mean, slope_max
    - terrain_ruggedness
    """
    villages = villages.copy()

    if not dem_path.exists():
        print("  Warning: DEM not found, using defaults")
        villages['elevation_mean'] = 200.0
        villages['elevation_min'] = 150.0
        villages['elevation_max'] = 250.0
        villages['slope_mean'] = 2.0
        villages['slope_max'] = 10.0
        return villages

    print("  Extracting elevation statistics...")

    # Extract elevation stats
    elev_means = []
    elev_mins = []
    elev_maxs = []

    for idx, row in villages.iterrows():
        stats = extract_raster_stats_for_polygon(
            dem_path, row.geometry, ['mean', 'min', 'max']
        )
        elev_means.append(stats['mean'])
        elev_mins.append(stats['min'])
        elev_maxs.append(stats['max'])

    villages['elevation_mean'] = elev_means
    villages['elevation_min'] = elev_mins
    villages['elevation_max'] = elev_maxs

    # Fill missing with regional averages (use defaults if all NaN)
    default_elev = 50.0  # Default elevation for coastal Krishna district
    elev_median = villages['elevation_mean'].median()
    villages['elevation_mean'] = villages['elevation_mean'].fillna(elev_median if pd.notna(elev_median) else default_elev)
    villages['elevation_min'] = villages['elevation_min'].fillna(villages['elevation_min'].median() if pd.notna(villages['elevation_min'].median()) else default_elev - 10)
    villages['elevation_max'] = villages['elevation_max'].fillna(villages['elevation_max'].median() if pd.notna(villages['elevation_max'].median()) else default_elev + 10)

    # Slope from elevation range (simplified)
    villages['slope_mean'] = (villages['elevation_max'] - villages['elevation_min']) / (villages['area_km2'] * 1000).clip(lower=100) * 100
    villages['slope_max'] = villages['slope_mean'] * 2

    print(f"  Elevation range: {villages['elevation_mean'].min():.0f}m - {villages['elevation_mean'].max():.0f}m")

    return villages


def extract_rainfall_features(villages: gpd.GeoDataFrame,
                               rainfall_files: Dict[Tuple[int, int], Path],
                               target_year: int = 2023,
                               target_month: int = 10) -> gpd.GeoDataFrame:
    """
    Extract rainfall features for each village.

    Features:
    - rainfall_current: Current month rainfall
    - rainfall_lag1-3: Previous 1-3 months
    - rainfall_cumulative_3m: 3-month cumulative
    - rainfall_cumulative_monsoon: June-October total
    - rainfall_annual: Annual total
    - rainfall_anomaly: Departure from normal
    """
    villages = villages.copy()

    print(f"  Extracting rainfall for {target_year}-{target_month:02d}...")

    # Current month rainfall
    current_key = (target_year, target_month)
    if current_key in rainfall_files:
        rainfall_current = []
        raster_path = rainfall_files[current_key]

        for idx, row in villages.iterrows():
            stats = extract_raster_stats_for_polygon(raster_path, row.geometry, ['mean'])
            rainfall_current.append(stats['mean'])

        villages['rainfall_current'] = rainfall_current
    else:
        villages['rainfall_current'] = 50.0  # Default

    # Lagged rainfall (previous 1-3 months)
    for lag in range(1, 4):
        lag_month = target_month - lag
        lag_year = target_year
        if lag_month <= 0:
            lag_month += 12
            lag_year -= 1

        lag_key = (lag_year, lag_month)
        col_name = f'rainfall_lag{lag}'

        if lag_key in rainfall_files:
            rainfall_lag = []
            raster_path = rainfall_files[lag_key]

            for idx, row in villages.iterrows():
                stats = extract_raster_stats_for_polygon(raster_path, row.geometry, ['mean'])
                rainfall_lag.append(stats['mean'])

            villages[col_name] = rainfall_lag
        else:
            villages[col_name] = 50.0

    # Cumulative features
    villages['rainfall_cumulative_3m'] = (
        villages['rainfall_current'].fillna(0) +
        villages['rainfall_lag1'].fillna(0) +
        villages['rainfall_lag2'].fillna(0)
    )

    # Monsoon cumulative (June-October)
    monsoon_months = [(target_year, m) for m in range(6, min(target_month + 1, 11))]
    monsoon_total = []

    for idx, row in villages.iterrows():
        total = 0
        for key in monsoon_months:
            if key in rainfall_files:
                stats = extract_raster_stats_for_polygon(rainfall_files[key], row.geometry, ['mean'])
                total += stats['mean'] if not np.isnan(stats['mean']) else 0
        monsoon_total.append(total)

    villages['rainfall_monsoon'] = monsoon_total

    # Fill NaN
    for col in villages.columns:
        if 'rainfall' in col:
            villages[col] = villages[col].fillna(villages[col].median())

    print(f"  Current month rainfall: {villages['rainfall_current'].mean():.1f}mm avg")

    return villages


def create_temporal_features(water_levels: pd.DataFrame,
                              target_year: int = 2023,
                              target_month: int = 10) -> pd.DataFrame:
    """
    Create temporal features from water level time series.

    Features for each piezometer:
    - wl_current: Current water level (if available)
    - wl_lag1-3: Lagged values
    - wl_seasonal_avg: Same month average over years
    - wl_trend: Linear trend over past 12 months
    - wl_volatility: Std dev over past 12 months
    """
    print("  Creating temporal features from water level history...")

    # Filter to relevant time period
    target_date = pd.Timestamp(year=target_year, month=target_month, day=1)

    # Get piezometer IDs
    id_col = 'piezo_id' if 'piezo_id' in water_levels.columns else 'sno'
    piezometers = water_levels[id_col].unique()

    features = []

    for piezo in piezometers:
        piezo_data = water_levels[water_levels[id_col] == piezo].sort_values('date')

        # Current and lagged
        current = piezo_data[piezo_data['date'] == target_date]['water_level'].values
        wl_current = current[0] if len(current) > 0 else np.nan

        # Lagged values
        lag_features = {}
        for lag in range(1, 4):
            lag_date = target_date - pd.DateOffset(months=lag)
            lag_val = piezo_data[piezo_data['date'] == lag_date]['water_level'].values
            lag_features[f'wl_lag{lag}'] = lag_val[0] if len(lag_val) > 0 else np.nan

        # Seasonal average (same month over years)
        same_month = piezo_data[piezo_data['date'].dt.month == target_month]
        wl_seasonal_avg = same_month['water_level'].mean()

        # Trend over past 12 months
        past_12m = piezo_data[
            (piezo_data['date'] > target_date - pd.DateOffset(months=12)) &
            (piezo_data['date'] <= target_date)
        ]

        if len(past_12m) > 2:
            # Simple linear trend
            x = np.arange(len(past_12m))
            y = pd.to_numeric(past_12m['water_level'], errors='coerce').values
            valid = ~np.isnan(y)
            if valid.sum() > 2:
                slope, _ = np.polyfit(x[valid], y[valid], 1)
                wl_trend = slope * 12  # Annual trend
            else:
                wl_trend = 0
        else:
            wl_trend = 0

        # Volatility
        wl_volatility = past_12m['water_level'].std() if len(past_12m) > 1 else 0

        features.append({
            id_col: piezo,
            'wl_current': wl_current,
            **lag_features,
            'wl_seasonal_avg': wl_seasonal_avg,
            'wl_trend': wl_trend,
            'wl_volatility': wl_volatility
        })

    features_df = pd.DataFrame(features)

    # Fill missing with medians
    for col in features_df.columns:
        if col != id_col:
            features_df[col] = features_df[col].fillna(features_df[col].median())

    print(f"  Created features for {len(features_df)} piezometers")

    return features_df


def create_aquifer_features(villages: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create one-hot encoded aquifer features.
    """
    villages = villages.copy()

    # One-hot encode geo_class
    if 'geo_class' in villages.columns:
        aquifer_dummies = pd.get_dummies(
            villages['geo_class'],
            prefix='aquifer',
            dummy_na=True
        )
        villages = pd.concat([villages, aquifer_dummies], axis=1)

    print(f"  Created {len([c for c in villages.columns if c.startswith('aquifer_')])} aquifer features")

    return villages


def create_extraction_features(villages: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create extraction-related features.

    Features:
    - well_density: Wells per km²
    - extraction_intensity: Extraction per km²
    """
    villages = villages.copy()

    # Well density
    if 'n_wells' in villages.columns and 'area_km2' in villages.columns:
        villages['well_density'] = villages['n_wells'] / villages['area_km2'].clip(lower=0.1)
    else:
        villages['well_density'] = 0

    # Extraction intensity
    if 'monthly_extraction_ham' in villages.columns and 'area_km2' in villages.columns:
        villages['extraction_intensity'] = villages['monthly_extraction_ham'] / villages['area_km2'].clip(lower=0.1)
    else:
        villages['extraction_intensity'] = 0

    print(f"  Well density range: {villages['well_density'].min():.1f} - {villages['well_density'].max():.1f} wells/km²")

    return villages


def build_feature_matrix(data: dict,
                         target_year: int = 2023,
                         target_month: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build complete feature matrix for modeling.

    Args:
        data: Dict with preprocessed data
        target_year: Year to predict
        target_month: Month to predict

    Returns:
        Tuple of (village_features, piezometer_features)
    """
    print("=" * 60)
    print(f"Building feature matrix for {target_year}-{target_month:02d}...")
    print("=" * 60)

    villages = data['villages'].copy()

    # 1. Terrain features
    print("\n[1/5] Extracting terrain features...")
    villages = extract_terrain_features(villages, data['dem_path'])

    # 2. Rainfall features
    print("\n[2/5] Extracting rainfall features...")
    villages = extract_rainfall_features(
        villages, data['rainfall_files'],
        target_year, target_month
    )

    # 3. Aquifer features
    print("\n[3/5] Creating aquifer features...")
    villages = create_aquifer_features(villages)

    # 4. Extraction features
    print("\n[4/5] Creating extraction features...")
    villages = create_extraction_features(villages)

    # 5. Temporal features for piezometers
    print("\n[5/5] Creating temporal features...")
    piezo_features = create_temporal_features(
        data['water_levels'],
        target_year, target_month
    )

    print("\n" + "=" * 60)
    print("Feature engineering complete!")
    print("=" * 60)

    # Summary
    numeric_cols = villages.select_dtypes(include=[np.number]).columns
    print(f"\nVillage features: {len(numeric_cols)} numeric columns")
    print(f"Piezometer features: {len(piezo_features.columns)} columns")

    return villages, piezo_features


def get_feature_names() -> Dict[str, List[str]]:
    """
    Get feature names grouped by category.
    """
    return {
        'terrain': ['elevation_mean', 'elevation_min', 'elevation_max', 'slope_mean', 'slope_max'],
        'rainfall': ['rainfall_current', 'rainfall_lag1', 'rainfall_lag2', 'rainfall_lag3',
                    'rainfall_cumulative_3m', 'rainfall_monsoon'],
        'soil': ['infiltration_score', 'runoff_score'],
        'extraction': ['n_wells', 'well_density', 'avg_well_depth',
                      'monthly_extraction_ham', 'extraction_intensity'],
        'location': ['centroid_lat', 'centroid_lon', 'area_km2'],
        'temporal': ['wl_current', 'wl_lag1', 'wl_lag2', 'wl_lag3',
                    'wl_seasonal_avg', 'wl_trend', 'wl_volatility']
    }


if __name__ == '__main__':
    from .load_data import load_all_data
    from .preprocess import preprocess_all

    # Load and preprocess
    raw_data = load_all_data()
    processed = preprocess_all(raw_data)

    # Build features
    villages, piezo_features = build_feature_matrix(processed)

    print("\nFeature Matrix Summary:")
    print(f"  Villages: {len(villages)}")
    print(f"  Piezometers: {len(piezo_features)}")
    print(f"  Village feature columns: {len(villages.columns)}")
