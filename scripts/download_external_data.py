#!/usr/bin/env python3
"""
Smart Jal - External Data Download Script
Downloads open-source data for Krishna District, Andhra Pradesh
"""

import os
import requests
from pathlib import Path

# Krishna District Bounds
KRISHNA_BOUNDS = {
    'min_lon': 80.0,
    'max_lon': 81.5,
    'min_lat': 15.5,
    'max_lat': 17.0
}

# Output directory
DATA_DIR = Path(__file__).parent.parent / "downloaded_data"


def download_chirps_rainfall():
    """
    Download CHIRPS rainfall data for Krishna District
    Resolution: 0.05° (~5km)
    """
    print("\n=== CHIRPS Rainfall Data ===")
    print("Option 1: Use Google Earth Engine (recommended)")
    print("""
    import ee
    ee.Initialize()

    # Define Krishna district bounds
    krishna = ee.Geometry.Rectangle([80.0, 15.5, 81.5, 17.0])

    # Get CHIRPS daily data
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \\
        .filterDate('2015-01-01', '2024-12-31') \\
        .filterBounds(krishna)

    # Export monthly sum
    # Use ee.batch.Export.image.toDrive() or toCloudStorage()
    """)

    print("\nOption 2: Direct download from CHC UCSB")
    print("URL: https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/")
    print("Download files: chirps-v2.0.YYYY.MM.tif for 2015-2024")

    print("\nOption 3: Use imdlib for IMD data")
    print("""
    pip install imdlib

    import imdlib as imd

    # Download rainfall data
    start_year = 2015
    end_year = 2024

    # Downloads to current directory
    data = imd.get_data('rain', start_year, end_year, 'yearwise')

    # Or for specific region
    # imd.get_data('rain', start_year, end_year, 'yearwise',
    #              lat_min=15.5, lat_max=17.0, lon_min=80.0, lon_max=81.5)
    """)


def download_srtm_dem():
    """
    Download SRTM DEM for Krishna District
    Resolution: 30m
    """
    print("\n=== SRTM DEM (30m) ===")
    print("Option 1: USGS EarthExplorer")
    print("1. Go to https://earthexplorer.usgs.gov/")
    print("2. Register (free)")
    print("3. Search Criteria → Enter coordinates:")
    print(f"   North: {KRISHNA_BOUNDS['max_lat']}, South: {KRISHNA_BOUNDS['min_lat']}")
    print(f"   East: {KRISHNA_BOUNDS['max_lon']}, West: {KRISHNA_BOUNDS['min_lon']}")
    print("4. Data Sets → Digital Elevation → SRTM → SRTM 1 Arc-Second Global")
    print("5. Download GeoTIFF files")

    print("\nOption 2: Google Earth Engine")
    print("""
    import ee
    ee.Initialize()

    krishna = ee.Geometry.Rectangle([80.0, 15.5, 81.5, 17.0])
    dem = ee.Image('USGS/SRTMGL1_003').clip(krishna)

    # Export
    task = ee.batch.Export.image.toDrive(
        image=dem,
        description='krishna_dem',
        scale=30,
        region=krishna,
        maxPixels=1e10
    )
    task.start()
    """)

    print("\nOption 3: Online tool (easiest)")
    print("URL: https://geoprocessing.online/tool/srtm-dem-download/")
    print("Draw polygon over Krishna district and download")


def download_village_boundaries():
    """
    Download village boundaries for Krishna District
    """
    print("\n=== Village Boundaries ===")
    print("Option 1: Survey of India (Official, Recommended)")
    print("1. Go to https://onlinemaps.surveyofindia.gov.in/")
    print("2. Register (free)")
    print("3. Login → Products → Village Boundary Database")
    print("4. Select: State = Andhra Pradesh, District = Krishna")
    print("5. Add to cart → Download (Shapefile or GeoDatabase)")

    print("\nOption 2: DataMeet GitHub (Pre-processed)")
    print("URL: https://github.com/datta07/INDIAN-SHAPEFILES")
    print("Look for Andhra Pradesh sub-districts/mandals")

    print("\nOption 3: Indian Village Boundaries Project")
    print("URL: https://projects.datameet.org/indian_village_boundaries/")
    print("Download AP state file if available")


def download_validation_data():
    """
    Download groundwater validation data from CGWB/NWDP
    """
    print("\n=== VALIDATION DATA (CRITICAL!) ===")
    print("This data can be used to validate your model predictions!")

    print("\n--- National Water Data Portal ---")
    print("1. AP Groundwater Manual (Quarterly) 2021-2025:")
    print("   https://nwdp.nwic.in/en/dataset/ground-water-level-manual-quarterly-andhra-pradesh-ground-water-departments")

    print("\n2. AP Groundwater Telemetry (Hourly) 1991-2025:")
    print("   https://nwdp.nwic.in/en/dataset/ground-water-level-telemetry-hourly-andhra-pradesh-ground-water-department")

    print("\n--- CGWB Data ---")
    print("3. All-India Water Level Data (1994-2024):")
    print("   https://cgwb.gov.in/en/ground-water-level-monitoring")
    print("   - January data: 1994-2024")
    print("   - May data: 1994-2024")
    print("   - August data: 1994-2023")
    print("   - November data: 1994-2023")

    print("\n--- India-WRIS Portal ---")
    print("4. https://indiawris.gov.in/wris/#/DataDownload")
    print("   Register (free) and download groundwater data")


def download_soil_data():
    """
    Download soil data from ISRIC SoilGrids
    """
    print("\n=== Soil Data (ISRIC SoilGrids) ===")
    print("Resolution: 250m")

    print("\nOption 1: Web Interface")
    print("URL: https://soilgrids.org/")
    print("Navigate to Krishna district and download layers")

    print("\nOption 2: REST API")
    print("""
    import requests

    # Get soil properties for a point
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": 80.5,  # Krishna district center
        "lat": 16.5,
        "property": ["sand", "clay", "silt", "bdod", "cec"],
        "depth": ["0-5cm", "5-15cm", "15-30cm", "30-60cm"]
    }
    response = requests.get(url, params=params)
    data = response.json()
    """)

    print("\nOption 3: Google Earth Engine")
    print("""
    import ee
    ee.Initialize()

    # SoilGrids sand content
    sand = ee.Image("projects/soilgrids-isric/sand_mean")
    clay = ee.Image("projects/soilgrids-isric/clay_mean")
    """)


def download_grace_data():
    """
    Download GRACE satellite data for groundwater storage
    """
    print("\n=== GRACE Satellite Data (DIFFERENTIATOR!) ===")
    print("This gives regional groundwater storage - nobody else will use this!")

    print("\nOption 1: Google Earth Engine (Easiest)")
    print("""
    import ee
    ee.Initialize()

    # GRACE Tellus Land Mass Grids
    grace = ee.ImageCollection('NASA/GRACE/MASS_GRIDS/LAND')

    krishna = ee.Geometry.Rectangle([80.0, 15.5, 81.5, 17.0])

    # Get time series
    grace_krishna = grace.filterBounds(krishna) \\
        .select('lwe_thickness_csr')  # Liquid water equivalent

    # This gives monthly groundwater storage anomaly in cm
    """)

    print("\nOption 2: NASA JPL Direct Download")
    print("URL: https://grace.jpl.nasa.gov/data/get-data/")
    print("Download: GRACE-FO RL06 Mascon Solutions")

    print("\nOption 3: GLDAS for GRACE separation")
    print("URL: https://ldas.gsfc.nasa.gov/gldas")
    print("Download: GLDAS Noah 0.25° monthly")


def main():
    print("=" * 60)
    print("SMART JAL - External Data Download Guide")
    print("=" * 60)
    print(f"\nTarget Area: Krishna District, Andhra Pradesh")
    print(f"Bounds: {KRISHNA_BOUNDS}")

    # Create data directories
    dirs = ['rainfall', 'dem', 'villages', 'validation', 'soil', 'grace']
    for d in dirs:
        (DATA_DIR / d).mkdir(parents=True, exist_ok=True)
    print(f"\nData directories created in: {DATA_DIR}")

    # Show download instructions
    download_chirps_rainfall()
    download_srtm_dem()
    download_village_boundaries()
    download_validation_data()
    download_soil_data()
    download_grace_data()

    print("\n" + "=" * 60)
    print("SUMMARY - Download Priority")
    print("=" * 60)
    print("""
    CRITICAL (Do First):
    1. Village Boundaries - Survey of India (30 min)
    2. Rainfall - CHIRPS or IMD (30 min)

    IMPORTANT:
    3. Validation Data - NWDP/CGWB (30 min)
    4. DEM - SRTM (15 min)

    DIFFERENTIATORS:
    5. GRACE data (1-2 hours)
    6. Soil data (30 min)

    Total estimated time: 3-4 hours
    """)

    print("\nNote: For Google Earth Engine, sign up at:")
    print("https://earthengine.google.com/signup/")


if __name__ == "__main__":
    main()
