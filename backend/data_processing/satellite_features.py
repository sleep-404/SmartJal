#!/usr/bin/env python3
"""
Smart Jal - Satellite Data Feature Extraction
Extracts features from MODIS ET, Sentinel-2 NDVI, and SMAP Soil Moisture.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Data directory
SATELLITE_DIR = Path(__file__).parent.parent.parent / "downloaded_data" / "satellite"


def load_satellite_rasters() -> Dict[str, Dict[Tuple[int, int], Path]]:
    """
    Load all available satellite rasters.

    Returns:
        Dict with keys 'et', 'ndvi', 'smap', each containing {(year, month): path}
    """
    rasters = {'et': {}, 'ndvi': {}, 'smap': {}}

    # MODIS ET
    et_dir = SATELLITE_DIR / "modis_et"
    if et_dir.exists():
        for f in et_dir.glob("modis_et_*.tif"):
            # Parse filename: modis_et_2023.01.tif
            parts = f.stem.split('_')[-1].split('.')
            if len(parts) == 2:
                year, month = int(parts[0]), int(parts[1])
                rasters['et'][(year, month)] = f

    # Sentinel NDVI
    ndvi_dir = SATELLITE_DIR / "ndvi"
    if ndvi_dir.exists():
        for f in ndvi_dir.glob("ndvi_*.tif"):
            parts = f.stem.split('_')[-1].split('.')
            if len(parts) == 2:
                year, month = int(parts[0]), int(parts[1])
                rasters['ndvi'][(year, month)] = f

    # SMAP Soil Moisture
    smap_dir = SATELLITE_DIR / "smap"
    if smap_dir.exists():
        for f in smap_dir.glob("smap_*.tif"):
            parts = f.stem.split('_')[-1].split('.')
            if len(parts) == 2:
                year, month = int(parts[0]), int(parts[1])
                rasters['smap'][(year, month)] = f

    print(f"Loaded satellite rasters: ET={len(rasters['et'])}, NDVI={len(rasters['ndvi'])}, SMAP={len(rasters['smap'])}")

    return rasters


def extract_raster_stats(raster_path: Path, geometry) -> Dict[str, float]:
    """Extract mean, min, max from raster for a geometry."""
    try:
        with rasterio.open(raster_path) as src:
            out_image, _ = mask(src, [geometry], crop=True)
            data = out_image[0].astype(float)

            # Handle nodata
            if src.nodata is not None:
                data[data == src.nodata] = np.nan

            valid = data[~np.isnan(data)]

            if len(valid) == 0:
                return {'mean': np.nan, 'min': np.nan, 'max': np.nan}

            return {
                'mean': float(np.nanmean(valid)),
                'min': float(np.nanmin(valid)),
                'max': float(np.nanmax(valid))
            }
    except Exception:
        return {'mean': np.nan, 'min': np.nan, 'max': np.nan}


def extract_satellite_features(villages: gpd.GeoDataFrame,
                                satellite_rasters: Dict,
                                target_year: int = 2023,
                                target_month: int = 10) -> gpd.GeoDataFrame:
    """
    Extract satellite-derived features for each village.

    Features created:
    - modis_et: MODIS evapotranspiration (kg/m²/8day)
    - modis_et_lag1: Previous month ET
    - ndvi: Sentinel-2 NDVI (-1 to 1)
    - ndvi_lag1: Previous month NDVI
    - smap_sm: SMAP soil moisture (cm³/cm³)
    """
    villages = villages.copy()

    print("  Extracting satellite features...")

    # Current month keys
    current_key = (target_year, target_month)

    # Previous month
    lag_month = target_month - 1 if target_month > 1 else 12
    lag_year = target_year if target_month > 1 else target_year - 1
    lag_key = (lag_year, lag_month)

    # MODIS ET
    et_rasters = satellite_rasters.get('et', {})
    if current_key in et_rasters:
        print(f"    Extracting MODIS ET for {target_year}-{target_month:02d}...")
        et_values = []
        for _, row in villages.iterrows():
            stats = extract_raster_stats(et_rasters[current_key], row.geometry)
            # Scale: MODIS ET is in kg/m²/8day, multiply by 0.1 to get mm/8day
            et_values.append(stats['mean'] * 0.1 if not np.isnan(stats['mean']) else np.nan)
        villages['modis_et'] = et_values
    else:
        villages['modis_et'] = np.nan

    # MODIS ET lag
    if lag_key in et_rasters:
        et_lag = []
        for _, row in villages.iterrows():
            stats = extract_raster_stats(et_rasters[lag_key], row.geometry)
            et_lag.append(stats['mean'] * 0.1 if not np.isnan(stats['mean']) else np.nan)
        villages['modis_et_lag1'] = et_lag
    else:
        villages['modis_et_lag1'] = villages.get('modis_et', np.nan)

    # Sentinel NDVI
    ndvi_rasters = satellite_rasters.get('ndvi', {})
    if current_key in ndvi_rasters:
        print(f"    Extracting Sentinel NDVI for {target_year}-{target_month:02d}...")
        ndvi_values = []
        for _, row in villages.iterrows():
            stats = extract_raster_stats(ndvi_rasters[current_key], row.geometry)
            ndvi_values.append(stats['mean'])
        villages['sentinel_ndvi'] = ndvi_values
    else:
        villages['sentinel_ndvi'] = np.nan

    # NDVI lag
    if lag_key in ndvi_rasters:
        ndvi_lag = []
        for _, row in villages.iterrows():
            stats = extract_raster_stats(ndvi_rasters[lag_key], row.geometry)
            ndvi_lag.append(stats['mean'])
        villages['sentinel_ndvi_lag1'] = ndvi_lag
    else:
        villages['sentinel_ndvi_lag1'] = villages.get('sentinel_ndvi', np.nan)

    # SMAP Soil Moisture
    smap_rasters = satellite_rasters.get('smap', {})
    if current_key in smap_rasters:
        print(f"    Extracting SMAP soil moisture for {target_year}-{target_month:02d}...")
        smap_values = []
        for _, row in villages.iterrows():
            stats = extract_raster_stats(smap_rasters[current_key], row.geometry)
            smap_values.append(stats['mean'])
        villages['smap_soil_moisture'] = smap_values
    else:
        villages['smap_soil_moisture'] = np.nan

    # Fill NaN with medians
    for col in ['modis_et', 'modis_et_lag1', 'sentinel_ndvi', 'sentinel_ndvi_lag1', 'smap_soil_moisture']:
        if col in villages.columns:
            median_val = villages[col].median()
            if pd.isna(median_val):
                # Use reasonable defaults if all NaN
                defaults = {
                    'modis_et': 3.0,  # ~3mm/8day
                    'modis_et_lag1': 3.0,
                    'sentinel_ndvi': 0.3,
                    'sentinel_ndvi_lag1': 0.3,
                    'smap_soil_moisture': 0.2
                }
                median_val = defaults.get(col, 0)
            villages[col] = villages[col].fillna(median_val)

    # Print stats
    if 'modis_et' in villages.columns:
        print(f"    MODIS ET: {villages['modis_et'].mean():.2f} mm/8day avg")
    if 'sentinel_ndvi' in villages.columns:
        print(f"    Sentinel NDVI: {villages['sentinel_ndvi'].mean():.3f} avg")
    if 'smap_soil_moisture' in villages.columns:
        print(f"    SMAP Soil Moisture: {villages['smap_soil_moisture'].mean():.3f} avg")

    return villages


if __name__ == '__main__':
    # Test loading
    rasters = load_satellite_rasters()
    print(f"\nET months: {sorted(rasters['et'].keys())}")
    print(f"NDVI months: {sorted(rasters['ndvi'].keys())}")
    print(f"SMAP months: {sorted(rasters['smap'].keys())}")
