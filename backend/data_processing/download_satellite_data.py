#!/usr/bin/env python3
"""
Smart Jal - Satellite Data Download Script
Downloads MODIS ET, Sentinel-2 NDVI, and SMAP Soil Moisture for Krishna district.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Check for Earth Engine
try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    print("Warning: earthengine-api not installed. Run: pip install earthengine-api")

import numpy as np

# Krishna district bounding box (approximate)
KRISHNA_BOUNDS = {
    'min_lon': 80.0,
    'max_lon': 81.5,
    'min_lat': 15.5,
    'max_lat': 17.0
}

# Output directory
DATA_DIR = Path(__file__).parent.parent.parent / "downloaded_data" / "satellite"


def initialize_ee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project='responsibleindian')
        print("Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"Earth Engine not authenticated. Run: earthengine authenticate")
        print(f"Error: {e}")
        return False


def get_krishna_geometry():
    """Get Krishna district geometry for Earth Engine."""
    return ee.Geometry.Rectangle([
        KRISHNA_BOUNDS['min_lon'],
        KRISHNA_BOUNDS['min_lat'],
        KRISHNA_BOUNDS['max_lon'],
        KRISHNA_BOUNDS['max_lat']
    ])


def download_modis_et(start_date: str, end_date: str, output_dir: Path):
    """
    Download MODIS MOD16A2 Evapotranspiration data.

    MOD16A2: 8-day composite, 500m resolution
    Bands: ET (evapotranspiration), PET (potential ET)
    Units: kg/m²/8day (multiply by 0.1 to get mm/8day)
    """
    print(f"\nDownloading MODIS ET for {start_date} to {end_date}...")

    geometry = get_krishna_geometry()

    # Get MODIS ET collection
    collection = ee.ImageCollection('MODIS/061/MOD16A2GF') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date) \
        .select(['ET', 'PET'])

    # Get monthly composites
    months = collection.aggregate_array('system:time_start').getInfo()

    if not months:
        print("  No MODIS ET data found for this period")
        return []

    # Create monthly averages
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    downloaded_files = []
    current = start

    while current < end:
        month_start = current.strftime('%Y-%m-%d')
        next_month = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        month_end = next_month.strftime('%Y-%m-%d')

        # Get monthly composite
        monthly = collection.filterDate(month_start, month_end).mean()

        # Check if data exists
        try:
            # Get download URL
            url = monthly.getDownloadURL({
                'name': f'modis_et_{current.year}_{current.month:02d}',
                'region': geometry,
                'scale': 500,
                'format': 'GEO_TIFF'
            })

            # Download file
            output_file = output_dir / f"modis_et_{current.year}.{current.month:02d}.tif"

            import urllib.request
            urllib.request.urlretrieve(url, output_file)

            print(f"  Downloaded: {output_file.name}")
            downloaded_files.append(output_file)

        except Exception as e:
            print(f"  Skipping {current.year}-{current.month:02d}: {e}")

        current = next_month

    return downloaded_files


def download_sentinel_ndvi(start_date: str, end_date: str, output_dir: Path):
    """
    Download Sentinel-2 NDVI data.

    Sentinel-2: 10m resolution
    NDVI = (NIR - Red) / (NIR + Red) = (B8 - B4) / (B8 + B4)
    Range: -1 to 1 (healthy vegetation > 0.3)
    """
    print(f"\nDownloading Sentinel-2 NDVI for {start_date} to {end_date}...")

    geometry = get_krishna_geometry()

    # Cloud masking function
    def mask_clouds(image):
        qa = image.select('QA60')
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(cloud_mask)

    # NDVI calculation
    def add_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    # Get Sentinel-2 collection
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
        .map(mask_clouds) \
        .map(add_ndvi) \
        .select('NDVI')

    # Create monthly composites
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    downloaded_files = []
    current = start

    while current < end:
        month_start = current.strftime('%Y-%m-%d')
        next_month = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        month_end = next_month.strftime('%Y-%m-%d')

        # Get monthly median (reduces cloud effects)
        monthly = collection.filterDate(month_start, month_end).median()

        try:
            # Get download URL (resample to 100m to reduce file size)
            url = monthly.getDownloadURL({
                'name': f'ndvi_{current.year}_{current.month:02d}',
                'region': geometry,
                'scale': 100,  # 100m instead of 10m to reduce size
                'format': 'GEO_TIFF'
            })

            output_file = output_dir / f"ndvi_{current.year}.{current.month:02d}.tif"

            import urllib.request
            urllib.request.urlretrieve(url, output_file)

            print(f"  Downloaded: {output_file.name}")
            downloaded_files.append(output_file)

        except Exception as e:
            print(f"  Skipping {current.year}-{current.month:02d}: {e}")

        current = next_month

    return downloaded_files


def download_smap_soil_moisture(start_date: str, end_date: str, output_dir: Path):
    """
    Download SMAP Soil Moisture data.

    SMAP SPL3SMP_E: 9km resolution, daily
    Band: soil_moisture_am (morning pass, more accurate)
    Units: cm³/cm³ (volumetric)
    """
    print(f"\nDownloading SMAP Soil Moisture for {start_date} to {end_date}...")

    geometry = get_krishna_geometry()

    # Get SMAP collection
    collection = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date) \
        .select(['sm_surface', 'sm_rootzone'])

    # Create monthly composites
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    downloaded_files = []
    current = start

    while current < end:
        month_start = current.strftime('%Y-%m-%d')
        next_month = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        month_end = next_month.strftime('%Y-%m-%d')

        # Get monthly mean
        monthly = collection.filterDate(month_start, month_end).mean()

        try:
            url = monthly.getDownloadURL({
                'name': f'smap_{current.year}_{current.month:02d}',
                'region': geometry,
                'scale': 9000,  # Native resolution
                'format': 'GEO_TIFF'
            })

            output_file = output_dir / f"smap_{current.year}.{current.month:02d}.tif"

            import urllib.request
            urllib.request.urlretrieve(url, output_file)

            print(f"  Downloaded: {output_file.name}")
            downloaded_files.append(output_file)

        except Exception as e:
            print(f"  Skipping {current.year}-{current.month:02d}: {e}")

        current = next_month

    return downloaded_files


def download_all(start_date: str = '2020-01-01', end_date: str = '2024-12-31'):
    """Download all satellite datasets."""

    if not EE_AVAILABLE:
        print("Earth Engine API not available. Please install: pip install earthengine-api")
        return

    if not initialize_ee():
        print("\nTo authenticate Earth Engine:")
        print("  1. Run: earthengine authenticate")
        print("  2. Follow the browser prompts")
        print("  3. Re-run this script")
        return

    # Create output directories
    et_dir = DATA_DIR / "modis_et"
    ndvi_dir = DATA_DIR / "ndvi"
    smap_dir = DATA_DIR / "smap"

    for d in [et_dir, ndvi_dir, smap_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading satellite data to: {DATA_DIR}")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 60)

    # Download each dataset
    et_files = download_modis_et(start_date, end_date, et_dir)
    ndvi_files = download_sentinel_ndvi(start_date, end_date, ndvi_dir)
    smap_files = download_smap_soil_moisture(start_date, end_date, smap_dir)

    print("\n" + "=" * 60)
    print("Download Summary:")
    print(f"  MODIS ET files: {len(et_files)}")
    print(f"  Sentinel NDVI files: {len(ndvi_files)}")
    print(f"  SMAP Soil Moisture files: {len(smap_files)}")
    print("=" * 60)

    return {
        'et': et_files,
        'ndvi': ndvi_files,
        'smap': smap_files
    }


if __name__ == '__main__':
    # Default: download 2020-2024 data
    start = sys.argv[1] if len(sys.argv) > 1 else '2020-01-01'
    end = sys.argv[2] if len(sys.argv) > 2 else '2024-12-31'

    download_all(start, end)
