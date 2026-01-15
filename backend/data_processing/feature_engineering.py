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

# Import satellite features module
try:
    from .satellite_features import load_satellite_rasters, extract_satellite_features
    SATELLITE_AVAILABLE = True
except ImportError:
    SATELLITE_AVAILABLE = False


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


def _get_adjacent_month_key(year: int, month: int, offset: int) -> Tuple[int, int]:
    """Get year/month tuple with offset."""
    new_month = month + offset
    new_year = year
    while new_month <= 0:
        new_month += 12
        new_year -= 1
    while new_month > 12:
        new_month -= 12
        new_year += 1
    return (new_year, new_month)


def _extract_rainfall_for_month(villages: gpd.GeoDataFrame,
                                 rainfall_files: Dict[Tuple[int, int], Path],
                                 year: int, month: int) -> List[float]:
    """
    Extract rainfall for a specific month, with interpolation fallback for missing data.

    If the requested month is missing, interpolates from adjacent months.
    """
    key = (year, month)

    # Direct lookup
    if key in rainfall_files:
        rainfall_values = []
        raster_path = rainfall_files[key]
        for _, row in villages.iterrows():
            stats = extract_raster_stats_for_polygon(raster_path, row.geometry, ['mean'])
            rainfall_values.append(stats['mean'])
        return rainfall_values

    # Try interpolation from adjacent months
    prev_key = _get_adjacent_month_key(year, month, -1)
    next_key = _get_adjacent_month_key(year, month, +1)

    prev_available = prev_key in rainfall_files
    next_available = next_key in rainfall_files

    if prev_available and next_available:
        # Interpolate from both adjacent months
        print(f"    Note: ({year}, {month}) missing - interpolating from adjacent months")
        prev_vals = []
        next_vals = []
        for _, row in villages.iterrows():
            prev_stats = extract_raster_stats_for_polygon(rainfall_files[prev_key], row.geometry, ['mean'])
            next_stats = extract_raster_stats_for_polygon(rainfall_files[next_key], row.geometry, ['mean'])
            prev_vals.append(prev_stats['mean'])
            next_vals.append(next_stats['mean'])
        return [(p + n) / 2 if not (np.isnan(p) or np.isnan(n)) else (p if not np.isnan(p) else n)
                for p, n in zip(prev_vals, next_vals)]

    elif prev_available:
        # Use previous month only
        print(f"    Note: ({year}, {month}) missing - using previous month")
        rainfall_values = []
        for _, row in villages.iterrows():
            stats = extract_raster_stats_for_polygon(rainfall_files[prev_key], row.geometry, ['mean'])
            rainfall_values.append(stats['mean'])
        return rainfall_values

    elif next_available:
        # Use next month only
        print(f"    Note: ({year}, {month}) missing - using next month")
        rainfall_values = []
        for _, row in villages.iterrows():
            stats = extract_raster_stats_for_polygon(rainfall_files[next_key], row.geometry, ['mean'])
            rainfall_values.append(stats['mean'])
        return rainfall_values

    else:
        # No adjacent data available, use regional default
        print(f"    Warning: ({year}, {month}) missing with no adjacent data - using default")
        return [50.0] * len(villages)


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

    Note: Missing months are interpolated from adjacent months when possible.
    """
    villages = villages.copy()

    print(f"  Extracting rainfall for {target_year}-{target_month:02d}...")

    # Current month rainfall
    villages['rainfall_current'] = _extract_rainfall_for_month(
        villages, rainfall_files, target_year, target_month
    )

    # Lagged rainfall (previous 1-3 months)
    for lag in range(1, 4):
        lag_month = target_month - lag
        lag_year = target_year
        if lag_month <= 0:
            lag_month += 12
            lag_year -= 1

        col_name = f'rainfall_lag{lag}'
        villages[col_name] = _extract_rainfall_for_month(
            villages, rainfall_files, lag_year, lag_month
        )

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


def extract_lulc_features(villages: gpd.GeoDataFrame,
                          lulc: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract land use / land cover features for each village.

    LULC gridcode mapping (typical NRSC classification):
    - 1: Water bodies
    - 2: Built-up/Urban
    - 3: Cropland - Kharif
    - 4: Cropland - Rabi
    - 5: Double/Triple Crop
    - 6: Plantation
    - 7: Fallow land
    - 8: Forest
    - 9: Wasteland/Barren
    - 10: Scrub/Grassland

    Features created:
    - lulc_crop_pct: Percentage of cropland (affects irrigation demand)
    - lulc_forest_pct: Percentage of forest (affects ET and recharge)
    - lulc_urban_pct: Percentage of urban (affects runoff)
    - lulc_water_pct: Percentage of water bodies
    - lulc_barren_pct: Percentage of wasteland/barren
    """
    villages = villages.copy()

    # Ensure same CRS
    if lulc.crs != villages.crs:
        lulc = lulc.to_crs(villages.crs)

    print("  Computing LULC percentages for each village...")

    # Initialize columns
    villages['lulc_crop_pct'] = 0.0
    villages['lulc_forest_pct'] = 0.0
    villages['lulc_urban_pct'] = 0.0
    villages['lulc_water_pct'] = 0.0
    villages['lulc_barren_pct'] = 0.0

    # Define gridcode groupings
    crop_codes = [3, 4, 5, 6]  # Kharif, Rabi, Double crop, Plantation
    forest_codes = [8]
    urban_codes = [2]
    water_codes = [1]
    barren_codes = [7, 9, 10]  # Fallow, Wasteland, Scrub

    # Spatial join to find overlapping LULC polygons
    for idx, village in villages.iterrows():
        try:
            # Find LULC polygons that intersect this village
            intersecting = lulc[lulc.intersects(village.geometry)]

            if len(intersecting) == 0:
                continue

            # Calculate intersection areas
            total_area = village.geometry.area
            if total_area == 0:
                continue

            crop_area = 0
            forest_area = 0
            urban_area = 0
            water_area = 0
            barren_area = 0

            for _, lulc_poly in intersecting.iterrows():
                intersection = village.geometry.intersection(lulc_poly.geometry)
                inter_area = intersection.area
                gridcode = lulc_poly['gridcode']

                if gridcode in crop_codes:
                    crop_area += inter_area
                elif gridcode in forest_codes:
                    forest_area += inter_area
                elif gridcode in urban_codes:
                    urban_area += inter_area
                elif gridcode in water_codes:
                    water_area += inter_area
                elif gridcode in barren_codes:
                    barren_area += inter_area

            # Calculate percentages
            villages.at[idx, 'lulc_crop_pct'] = 100 * crop_area / total_area
            villages.at[idx, 'lulc_forest_pct'] = 100 * forest_area / total_area
            villages.at[idx, 'lulc_urban_pct'] = 100 * urban_area / total_area
            villages.at[idx, 'lulc_water_pct'] = 100 * water_area / total_area
            villages.at[idx, 'lulc_barren_pct'] = 100 * barren_area / total_area

        except Exception:
            continue

    # Fill missing with regional averages
    for col in ['lulc_crop_pct', 'lulc_forest_pct', 'lulc_urban_pct', 'lulc_water_pct', 'lulc_barren_pct']:
        median_val = villages[col].median()
        villages[col] = villages[col].fillna(median_val if pd.notna(median_val) else 0)

    print(f"  Crop land: {villages['lulc_crop_pct'].mean():.1f}% avg")
    print(f"  Forest: {villages['lulc_forest_pct'].mean():.1f}% avg")
    print(f"  Urban: {villages['lulc_urban_pct'].mean():.1f}% avg")

    return villages


def extract_geomorphology_features(villages: gpd.GeoDataFrame,
                                    geomorphology: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract geomorphology features for each village.

    Landform affects groundwater recharge potential:
    - High recharge: Flood plain, Channel fill, Deltaic plain, River terrace, Valley fill
    - Moderate recharge: Pediment, Pediplain, Coastal plain, Beach
    - Low recharge: Residual hill, Structural hill, Denudational hill, Inselberg

    Features created:
    - geom_recharge_score: Weighted recharge potential (0-1)
    - geom_is_floodplain: Binary flag for flood plain areas
    - geom_is_hill: Binary flag for hilly terrain
    """
    villages = villages.copy()

    # Ensure same CRS
    if geomorphology.crs != villages.crs:
        geomorphology = geomorphology.to_crs(villages.crs)

    print("  Computing geomorphology features...")

    # Define recharge scores by landform type
    recharge_scores = {
        # High recharge (0.8-1.0)
        'Flood plain': 1.0,
        'Channel fill': 0.95,
        'Deltaic plain': 0.9,
        'River terrace-1': 0.85,
        'River terrace-2': 0.85,
        'River terrace-Older': 0.8,
        'Valley fill': 0.85,
        'Paleo channel': 0.8,
        'Channel bar': 0.75,
        'Point bar': 0.75,
        'Natural levee': 0.7,

        # Moderate recharge (0.4-0.7)
        'Pediment': 0.6,
        'Pediplain': 0.55,
        'Coastal plain': 0.5,
        'Beach': 0.5,
        'Beach ridge': 0.45,
        'Beach ridge and swale': 0.45,
        'Intermontane valley': 0.5,
        'Swale': 0.5,

        # Low recharge (0.1-0.4)
        'Residual hill': 0.2,
        'Structural hill': 0.2,
        'Denudational hill': 0.25,
        'Inselberg': 0.2,
        'Linear ridge': 0.3,
        'Plateau': 0.35,

        # Special areas
        'Mangrove': 0.3,
        'Salt flat': 0.1,
        'Mud flat': 0.2,
        'Tidal flat': 0.2,
        'Creek': 0.6,
        'Channel island': 0.5,
        'Offshore island': 0.3,
        'Spit': 0.3,
        'Structural valley': 0.5,
    }

    floodplain_types = ['Flood plain', 'Deltaic plain', 'Channel fill', 'Natural levee']
    hill_types = ['Residual hill', 'Structural hill', 'Denudational hill', 'Inselberg', 'Linear ridge']

    # Initialize columns
    villages['geom_recharge_score'] = 0.5  # Default moderate
    villages['geom_is_floodplain'] = 0
    villages['geom_is_hill'] = 0

    # Get the description column name
    desc_col = 'DISCRIPTIO' if 'DISCRIPTIO' in geomorphology.columns else 'DISCRIPT_1'

    for idx, village in villages.iterrows():
        try:
            # Find intersecting geomorphology polygons
            intersecting = geomorphology[geomorphology.intersects(village.geometry)]

            if len(intersecting) == 0:
                continue

            total_area = village.geometry.area
            if total_area == 0:
                continue

            # Calculate area-weighted recharge score
            weighted_score = 0
            floodplain_area = 0
            hill_area = 0

            for _, geom_poly in intersecting.iterrows():
                intersection = village.geometry.intersection(geom_poly.geometry)
                inter_area = intersection.area
                weight = inter_area / total_area

                landform = geom_poly[desc_col] if desc_col in geom_poly else ''
                score = recharge_scores.get(landform, 0.5)
                weighted_score += score * weight

                if landform in floodplain_types:
                    floodplain_area += inter_area
                if landform in hill_types:
                    hill_area += inter_area

            villages.at[idx, 'geom_recharge_score'] = weighted_score
            villages.at[idx, 'geom_is_floodplain'] = 1 if floodplain_area / total_area > 0.3 else 0
            villages.at[idx, 'geom_is_hill'] = 1 if hill_area / total_area > 0.3 else 0

        except Exception:
            continue

    # Fill missing with regional median
    villages['geom_recharge_score'] = villages['geom_recharge_score'].fillna(0.5)

    print(f"  Recharge score: {villages['geom_recharge_score'].mean():.2f} avg")
    print(f"  Floodplain villages: {villages['geom_is_floodplain'].sum()}")
    print(f"  Hilly villages: {villages['geom_is_hill'].sum()}")

    return villages


def extract_distance_to_water(villages: gpd.GeoDataFrame,
                               lulc: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate distance from each village to nearest water body.

    Water bodies affect:
    - Surface-groundwater interaction
    - Local recharge/discharge zones
    - Shallow water table areas

    Features created:
    - dist_to_water_km: Distance to nearest water body in km
    - near_water: Binary flag if within 2km of water
    """
    villages = villages.copy()

    print("  Computing distance to water bodies...")

    # Get water bodies from LULC (gridcode 1 = water)
    water_bodies = lulc[lulc['gridcode'] == 1].copy()

    if len(water_bodies) == 0:
        print("  Warning: No water bodies found in LULC")
        villages['dist_to_water_km'] = 10.0  # Default
        villages['near_water'] = 0
        return villages

    # Ensure same CRS
    if water_bodies.crs != villages.crs:
        water_bodies = water_bodies.to_crs(villages.crs)

    # Project to UTM for accurate distance calculation (meters)
    villages_utm = villages.to_crs(epsg=32644)  # UTM 44N for India
    water_utm = water_bodies.to_crs(epsg=32644)

    # Combine all water bodies into single geometry for faster distance calc
    from shapely.ops import unary_union
    water_union = unary_union(water_utm.geometry)

    # Calculate distance for each village centroid
    distances = []
    for idx, row in villages_utm.iterrows():
        centroid = row.geometry.centroid
        dist_m = centroid.distance(water_union)
        distances.append(dist_m / 1000)  # Convert to km

    villages['dist_to_water_km'] = distances
    villages['near_water'] = (villages['dist_to_water_km'] < 2.0).astype(int)

    print(f"  Distance to water: {villages['dist_to_water_km'].min():.1f} - {villages['dist_to_water_km'].max():.1f} km")
    print(f"  Villages near water (<2km): {villages['near_water'].sum()}")

    return villages


def add_seasonal_encoding(villages: gpd.GeoDataFrame,
                          target_month: int) -> gpd.GeoDataFrame:
    """
    Add cyclical encoding for month/season.

    Groundwater has strong seasonal patterns:
    - Post-monsoon (Oct-Jan): Rising water levels
    - Pre-monsoon (Apr-Jun): Lowest water levels
    - Monsoon (Jul-Sep): Recharge period

    Features created:
    - month_sin: Sine encoding of month (captures cyclical pattern)
    - month_cos: Cosine encoding of month
    - is_monsoon: Binary flag for monsoon months (Jun-Oct)
    - is_post_monsoon: Binary flag for post-monsoon (Nov-Jan)
    """
    villages = villages.copy()

    print(f"  Adding seasonal encoding for month {target_month}...")

    # Cyclical encoding (month 1-12 mapped to 0-2π)
    villages['month_sin'] = np.sin(2 * np.pi * target_month / 12)
    villages['month_cos'] = np.cos(2 * np.pi * target_month / 12)

    # Season flags
    villages['is_monsoon'] = 1 if target_month in [6, 7, 8, 9, 10] else 0
    villages['is_post_monsoon'] = 1 if target_month in [11, 12, 1] else 0
    villages['is_pre_monsoon'] = 1 if target_month in [4, 5, 6] else 0

    print(f"  Month {target_month}: sin={villages['month_sin'].iloc[0]:.3f}, cos={villages['month_cos'].iloc[0]:.3f}")
    print(f"  Is monsoon: {villages['is_monsoon'].iloc[0]}, Post-monsoon: {villages['is_post_monsoon'].iloc[0]}")

    return villages


def compute_vegetation_index(villages: gpd.GeoDataFrame,
                             lulc: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute vegetation/crop intensity index from LULC.

    This serves as a proxy for NDVI when satellite data isn't available.
    Higher crop intensity = higher irrigation demand = more extraction.

    LULC gridcodes:
    - 3: Kharif crops (monsoon)
    - 4: Rabi crops (winter)
    - 5: Double/triple crop (year-round, highest demand)
    - 6: Plantation
    - 8: Forest

    Features created:
    - vegetation_index: Weighted sum of vegetation types (0-1)
    - crop_intensity: Intensity of agricultural activity (0-1)
    - irrigation_demand_proxy: Estimated irrigation demand (0-1)
    """
    villages = villages.copy()

    print("  Computing vegetation/crop intensity index...")

    # Ensure same CRS
    if lulc.crs != villages.crs:
        lulc = lulc.to_crs(villages.crs)

    # Weights for different land use types
    # Higher weight = more water demand
    crop_weights = {
        3: 0.6,   # Kharif (monsoon crop, less irrigation)
        4: 0.8,   # Rabi (winter crop, needs irrigation)
        5: 1.0,   # Double/triple crop (highest demand)
        6: 0.5,   # Plantation
        8: 0.3,   # Forest (natural, no irrigation)
    }

    vegetation_index = []
    crop_intensity = []
    irrigation_demand = []

    for idx, village in villages.iterrows():
        try:
            intersecting = lulc[lulc.intersects(village.geometry)]

            if len(intersecting) == 0:
                vegetation_index.append(0.3)
                crop_intensity.append(0.3)
                irrigation_demand.append(0.3)
                continue

            total_area = village.geometry.area
            if total_area == 0:
                vegetation_index.append(0.3)
                crop_intensity.append(0.3)
                irrigation_demand.append(0.3)
                continue

            veg_score = 0
            crop_score = 0
            irrig_score = 0

            for _, lulc_poly in intersecting.iterrows():
                intersection = village.geometry.intersection(lulc_poly.geometry)
                weight = intersection.area / total_area
                gridcode = lulc_poly['gridcode']

                # Vegetation index (any green cover)
                if gridcode in [3, 4, 5, 6, 8]:
                    veg_score += weight

                # Crop intensity (only crops)
                if gridcode in [3, 4, 5]:
                    crop_score += weight
                    irrig_score += weight * crop_weights.get(gridcode, 0.5)

            vegetation_index.append(min(1.0, veg_score))
            crop_intensity.append(min(1.0, crop_score))
            irrigation_demand.append(min(1.0, irrig_score))

        except Exception:
            vegetation_index.append(0.3)
            crop_intensity.append(0.3)
            irrigation_demand.append(0.3)

    villages['vegetation_index'] = vegetation_index
    villages['crop_intensity'] = crop_intensity
    villages['irrigation_demand_proxy'] = irrigation_demand

    print(f"  Vegetation index: {villages['vegetation_index'].mean():.2f} avg")
    print(f"  Crop intensity: {villages['crop_intensity'].mean():.2f} avg")
    print(f"  Irrigation demand: {villages['irrigation_demand_proxy'].mean():.2f} avg")

    return villages


def compute_et_proxy(villages: gpd.GeoDataFrame,
                     lulc: gpd.GeoDataFrame,
                     target_month: int) -> gpd.GeoDataFrame:
    """
    Compute Evapotranspiration (ET) proxy from existing data.

    ET = Kc × ETo where:
    - Kc = Crop coefficient (from LULC crop types)
    - ETo = Reference ET (estimated from elevation, season)

    This approximates MODIS MOD16 ET when satellite data unavailable.

    Features created:
    - et_proxy: Estimated actual ET (mm/month)
    - eto_proxy: Reference ET (mm/month)
    - kc_weighted: Area-weighted crop coefficient
    """
    villages = villages.copy()

    print("  Computing ET proxy from LULC and terrain...")

    # Reference ET varies by month (Krishna district, ~16°N)
    # Based on FAO-56 typical values for tropical India
    monthly_eto = {
        1: 120,   # January (winter, low)
        2: 140,   # February
        3: 180,   # March (pre-summer)
        4: 200,   # April (peak summer)
        5: 210,   # May (hottest)
        6: 160,   # June (monsoon starts)
        7: 130,   # July (monsoon)
        8: 130,   # August (monsoon)
        9: 140,   # September (monsoon end)
        10: 150,  # October (post-monsoon)
        11: 130,  # November
        12: 110,  # December (winter)
    }

    # Crop coefficients by LULC gridcode (FAO-56 based)
    crop_kc = {
        1: 1.05,  # Water bodies (open water evaporation)
        2: 0.20,  # Urban (minimal ET)
        3: 0.90,  # Kharif crops (monsoon, high ET)
        4: 0.85,  # Rabi crops (winter)
        5: 0.95,  # Double/triple crop (year-round, highest)
        6: 0.80,  # Plantation
        7: 0.30,  # Fallow
        8: 0.90,  # Forest
        9: 0.20,  # Wasteland
        10: 0.40, # Scrub/grassland
    }

    # Ensure same CRS
    if lulc.crs != villages.crs:
        lulc = lulc.to_crs(villages.crs)

    base_eto = monthly_eto.get(target_month, 150)

    et_values = []
    eto_values = []
    kc_values = []

    for idx, village in villages.iterrows():
        try:
            # Adjust ETo by elevation (decreases ~6% per 100m above sea level)
            elev = village.get('elevation_mean', 50)
            eto_adjusted = base_eto * (1 - 0.0006 * max(0, elev - 50))

            # Calculate weighted Kc from LULC
            intersecting = lulc[lulc.intersects(village.geometry)]

            if len(intersecting) == 0:
                kc_weighted = 0.5  # Default
            else:
                total_area = village.geometry.area
                if total_area == 0:
                    kc_weighted = 0.5
                else:
                    kc_sum = 0
                    for _, lulc_poly in intersecting.iterrows():
                        intersection = village.geometry.intersection(lulc_poly.geometry)
                        weight = intersection.area / total_area
                        gridcode = lulc_poly.get('gridcode', 7)
                        kc_sum += weight * crop_kc.get(gridcode, 0.5)
                    kc_weighted = min(1.2, kc_sum)

            et_actual = eto_adjusted * kc_weighted

            et_values.append(et_actual)
            eto_values.append(eto_adjusted)
            kc_values.append(kc_weighted)

        except Exception:
            et_values.append(base_eto * 0.5)
            eto_values.append(base_eto)
            kc_values.append(0.5)

    villages['et_proxy'] = et_values
    villages['eto_proxy'] = eto_values
    villages['kc_weighted'] = kc_values

    print(f"  ET proxy: {villages['et_proxy'].mean():.1f} mm/month avg")
    print(f"  Reference ETo: {villages['eto_proxy'].mean():.1f} mm/month avg")
    print(f"  Crop coefficient: {villages['kc_weighted'].mean():.2f} avg")

    return villages


def extract_extended_rainfall_features(villages: gpd.GeoDataFrame,
                                        rainfall_files: Dict[Tuple[int, int], Path],
                                        target_year: int = 2023,
                                        target_month: int = 10) -> gpd.GeoDataFrame:
    """
    Extract extended rainfall features with longer lag periods.

    Groundwater response to rainfall can take 3-12 months.

    Additional features:
    - rainfall_lag4 to rainfall_lag6: 4-6 month lags
    - rainfall_cumulative_6m: 6-month cumulative
    - rainfall_annual: Full year rainfall
    - rainfall_deficit: Difference from long-term average
    """
    villages = villages.copy()

    print("  Adding extended rainfall lags (4-6 months)...")

    # Extended lags (4-6 months)
    for lag in range(4, 7):
        lag_month = target_month - lag
        lag_year = target_year
        while lag_month <= 0:
            lag_month += 12
            lag_year -= 1

        lag_key = (lag_year, lag_month)
        col_name = f'rainfall_lag{lag}'

        if lag_key in rainfall_files:
            rainfall_lag = []
            raster_path = rainfall_files[lag_key]

            for _, row in villages.iterrows():
                stats = extract_raster_stats_for_polygon(raster_path, row.geometry, ['mean'])
                rainfall_lag.append(stats['mean'])

            villages[col_name] = rainfall_lag
        else:
            villages[col_name] = villages.get('rainfall_lag3', 50.0)  # Fallback

    # 6-month cumulative
    cum_cols = ['rainfall_current', 'rainfall_lag1', 'rainfall_lag2', 'rainfall_lag3', 'rainfall_lag4', 'rainfall_lag5']
    villages['rainfall_cumulative_6m'] = sum(
        villages[col].fillna(0) for col in cum_cols if col in villages.columns
    )

    # Annual rainfall (sum of last 12 months where available)
    annual_total = []
    for _, row in villages.iterrows():
        total = 0
        for m in range(12):
            check_month = target_month - m
            check_year = target_year
            while check_month <= 0:
                check_month += 12
                check_year -= 1
            key = (check_year, check_month)
            if key in rainfall_files:
                stats = extract_raster_stats_for_polygon(rainfall_files[key], row.geometry, ['mean'])
                total += stats['mean'] if not np.isnan(stats['mean']) else 0
        annual_total.append(total)

    villages['rainfall_annual'] = annual_total

    # Fill NaN
    for col in villages.columns:
        if 'rainfall' in col:
            villages[col] = villages[col].fillna(villages[col].median())

    print(f"  6-month cumulative: {villages['rainfall_cumulative_6m'].mean():.0f}mm avg")
    print(f"  Annual rainfall: {villages['rainfall_annual'].mean():.0f}mm avg")

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
    print("\n[1/12] Extracting terrain features...")
    villages = extract_terrain_features(villages, data['dem_path'])

    # 2. Rainfall features
    print("\n[2/12] Extracting rainfall features...")
    villages = extract_rainfall_features(
        villages, data['rainfall_files'],
        target_year, target_month
    )

    # 3. Extended rainfall features (longer lags)
    print("\n[3/12] Extracting extended rainfall features...")
    villages = extract_extended_rainfall_features(
        villages, data['rainfall_files'],
        target_year, target_month
    )

    # 4. LULC features
    print("\n[4/12] Extracting LULC features...")
    if 'lulc' in data and data['lulc'] is not None:
        villages = extract_lulc_features(villages, data['lulc'])
    else:
        print("  Warning: LULC data not available, skipping")
        villages['lulc_crop_pct'] = 50.0
        villages['lulc_forest_pct'] = 10.0
        villages['lulc_urban_pct'] = 5.0
        villages['lulc_water_pct'] = 2.0
        villages['lulc_barren_pct'] = 33.0

    # 5. Geomorphology features
    print("\n[5/12] Extracting geomorphology features...")
    if 'geomorphology' in data and data['geomorphology'] is not None:
        villages = extract_geomorphology_features(villages, data['geomorphology'])
    else:
        print("  Warning: Geomorphology data not available, skipping")
        villages['geom_recharge_score'] = 0.5
        villages['geom_is_floodplain'] = 0
        villages['geom_is_hill'] = 0

    # 6. Distance to water bodies
    print("\n[6/12] Computing distance to water bodies...")
    if 'lulc' in data and data['lulc'] is not None:
        villages = extract_distance_to_water(villages, data['lulc'])
    else:
        print("  Warning: LULC not available for water distance")
        villages['dist_to_water_km'] = 5.0
        villages['near_water'] = 0

    # 7. Vegetation/crop intensity index (NDVI proxy)
    print("\n[7/12] Computing vegetation/crop intensity...")
    if 'lulc' in data and data['lulc'] is not None:
        villages = compute_vegetation_index(villages, data['lulc'])
    else:
        print("  Warning: LULC not available for vegetation index")
        villages['vegetation_index'] = 0.3
        villages['crop_intensity'] = 0.3
        villages['irrigation_demand_proxy'] = 0.3

    # 8. ET proxy (evapotranspiration estimate)
    print("\n[8/12] Computing ET proxy...")
    if 'lulc' in data and data['lulc'] is not None:
        villages = compute_et_proxy(villages, data['lulc'], target_month)
    else:
        print("  Warning: LULC not available for ET proxy")
        villages['et_proxy'] = 100.0
        villages['eto_proxy'] = 150.0
        villages['kc_weighted'] = 0.5

    # 9. Seasonal encoding
    print("\n[9/12] Adding seasonal encoding...")
    villages = add_seasonal_encoding(villages, target_month)

    # 10. Aquifer features
    print("\n[10/12] Creating aquifer features...")
    villages = create_aquifer_features(villages)

    # 11. Extraction features
    print("\n[11/12] Creating extraction features...")
    villages = create_extraction_features(villages)

    # 12. Satellite features (MODIS ET, Sentinel NDVI, SMAP)
    print("\n[12/13] Extracting satellite features...")
    if SATELLITE_AVAILABLE:
        try:
            satellite_rasters = load_satellite_rasters()
            if any(len(v) > 0 for v in satellite_rasters.values()):
                villages = extract_satellite_features(
                    villages, satellite_rasters,
                    target_year, target_month
                )
            else:
                print("  Warning: No satellite rasters found, skipping")
                villages['modis_et'] = 3.0
                villages['sentinel_ndvi'] = 0.3
        except Exception as e:
            print(f"  Warning: Satellite feature extraction failed: {e}")
            villages['modis_et'] = 3.0
            villages['sentinel_ndvi'] = 0.3
    else:
        print("  Warning: Satellite module not available")
        villages['modis_et'] = 3.0
        villages['sentinel_ndvi'] = 0.3

    # 13. Temporal features for piezometers (NOT used as village features - avoids data leakage)
    print("\n[13/13] Creating temporal features for piezometers only...")
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
                    'rainfall_lag4', 'rainfall_lag5', 'rainfall_lag6',
                    'rainfall_cumulative_3m', 'rainfall_cumulative_6m',
                    'rainfall_monsoon', 'rainfall_annual'],
        'lulc': ['lulc_crop_pct', 'lulc_forest_pct', 'lulc_urban_pct',
                'lulc_water_pct', 'lulc_barren_pct'],
        'geomorphology': ['geom_recharge_score', 'geom_is_floodplain', 'geom_is_hill'],
        'soil': ['infiltration_score', 'runoff_score'],
        'extraction': ['n_wells', 'well_density', 'avg_well_depth',
                      'monthly_extraction_ham', 'extraction_intensity'],
        'location': ['centroid_lat', 'centroid_lon', 'area_km2'],
        'temporal': ['wl_current', 'wl_lag1', 'wl_lag2', 'wl_lag3',
                    'wl_seasonal_avg', 'wl_trend', 'wl_volatility'],
        'water_proximity': ['dist_to_water_km', 'near_water'],
        'vegetation': ['vegetation_index', 'crop_intensity', 'irrigation_demand_proxy'],
        'seasonal': ['month_sin', 'month_cos', 'is_monsoon', 'is_post_monsoon', 'is_pre_monsoon'],
        'aquifer': ['aquifer_Alluvium', 'aquifer_Khondalites', 'aquifer_Shales', 'aquifer_Charnokites']
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
