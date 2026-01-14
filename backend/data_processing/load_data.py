#!/usr/bin/env python3
"""
Smart Jal - Data Loading Module
Loads all raw data sources into memory.
"""

import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "downloaded_data"
OFFICIAL_DIR = BASE_DIR / "SmartJal_extracted" / "SmartJal"
USECASE_DIR = BASE_DIR / "UseCase_extracted"


def load_water_levels() -> pd.DataFrame:
    """
    Load 138 piezometers with 28 years of monthly water level data.

    Returns:
        DataFrame with piezometer metadata and monthly readings
    """
    file_path = OFFICIAL_DIR / "WaterLevels_Krishna" / "master data_updated.xlsx"
    df = pd.read_excel(file_path)

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    # Rename key columns for consistency
    rename_map = {
        'SNo': 'sno',
        'ID': 'piezo_id',
        'District': 'district',
        'Mandal Name': 'mandal',
        'Village Name': 'village',
        'Location\n(Premises)': 'location',
        'Project': 'project',
        'Total \nDepth \nin m': 'total_depth_m',
        'Principal Aquifer': 'aquifer',
        'MSL in meters': 'msl_m',
        'Latitude \n(Decimal Degrees)': 'lat',
        'Longitude \n(Decimal Degrees)': 'lon'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    print(f"Loaded water levels: {len(df)} piezometers")
    return df


def load_villages() -> gpd.GeoDataFrame:
    """
    Load 939 village boundaries.

    Returns:
        GeoDataFrame with village polygons
    """
    file_path = USECASE_DIR / "OKri_Vil.shp"
    gdf = gpd.read_file(file_path)

    # Rename columns for consistency
    rename_map = {
        'DNAME': 'district',
        'DMNAME': 'mandal',
        'DVNAME': 'village',
        'DHABCODE': 'hab_code',
        'DMV_CODE': 'village_code',
        'Area': 'area',
        'areaha': 'area_ha',
        'latitude': 'centroid_lat',
        'longitude': 'centroid_lon'
    }
    gdf = gdf.rename(columns={k: v for k, v in rename_map.items() if k in gdf.columns})

    # Ensure CRS is WGS84 (EPSG:4326)
    # Note: to_epsg() returns None for some projected CRS like Lambert Conformal Conic
    # Use is_geographic check instead for robustness
    if gdf.crs is None:
        print("  Warning: No CRS defined, assuming EPSG:4326")
        gdf = gdf.set_crs(epsg=4326)
    elif not gdf.crs.is_geographic:
        print(f"  Reprojecting from {gdf.crs.name} to EPSG:4326...")
        gdf = gdf.to_crs(epsg=4326)

    # Calculate centroids if not present
    if 'centroid_lat' not in gdf.columns:
        gdf['centroid_lon'] = gdf.geometry.centroid.x
        gdf['centroid_lat'] = gdf.geometry.centroid.y

    # Calculate area in km²
    gdf['area_km2'] = gdf.geometry.to_crs(epsg=32644).area / 1e6

    print(f"Loaded villages: {len(gdf)} polygons")
    return gdf


def load_mandals() -> gpd.GeoDataFrame:
    """
    Load 42 mandal boundaries.

    Returns:
        GeoDataFrame with mandal polygons
    """
    file_path = USECASE_DIR / "OKri_Mdl.shp"
    gdf = gpd.read_file(file_path)

    rename_map = {
        'DNAME': 'district',
        'DMNAME': 'mandal',
        'MANDAL_NAM': 'mandal_name'
    }
    gdf = gdf.rename(columns={k: v for k, v in rename_map.items() if k in gdf.columns})

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)

    print(f"Loaded mandals: {len(gdf)} polygons")
    return gdf


def load_aquifers() -> gpd.GeoDataFrame:
    """
    Load 8 aquifer boundary polygons.

    Returns:
        GeoDataFrame with aquifer zones
    """
    file_path = OFFICIAL_DIR / "Aquifers_Krishna" / "Aquifers_Krishna.shp"
    gdf = gpd.read_file(file_path)

    rename_map = {
        'AQUI_CODE': 'aquifer_code',
        'NEWCODE': 'new_code',
        'STATE': 'state',
        'area': 'area',
        'Geo_Class': 'geo_class'
    }
    gdf = gdf.rename(columns={k: v for k, v in rename_map.items() if k in gdf.columns})

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)

    print(f"Loaded aquifers: {len(gdf)} zones")
    print(f"  Aquifer types: {gdf['geo_class'].unique().tolist()}")
    return gdf


def load_soils() -> gpd.GeoDataFrame:
    """
    Load 151 soil polygons.

    Returns:
        GeoDataFrame with soil data
    """
    file_path = USECASE_DIR / "OKri_Soils.shp"
    gdf = gpd.read_file(file_path)

    rename_map = {
        'DESCRIPTIO': 'description',
        'MAPPING_UN': 'mapping_unit',
        'SOIL_TAXON': 'soil_taxon',
        'AREA': 'area',
        'AREA_1': 'area_1'
    }
    gdf = gdf.rename(columns={k: v for k, v in rename_map.items() if k in gdf.columns})

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)

    print(f"Loaded soils: {len(gdf)} polygons")
    return gdf


def load_geomorphology() -> gpd.GeoDataFrame:
    """
    Load 614 geomorphology polygons.

    Returns:
        GeoDataFrame with geomorphology data
    """
    file_path = OFFICIAL_DIR / "GM_Krishna" / "GM_Krishna.shp"
    gdf = gpd.read_file(file_path)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)

    print(f"Loaded geomorphology: {len(gdf)} polygons")
    return gdf


def load_bore_wells() -> pd.DataFrame:
    """
    Load 88,988 bore well records.

    Returns:
        DataFrame with bore well data
    """
    file_path = OFFICIAL_DIR / "GTWells_Krishna" / "GTWells" / "kris.csv"
    df = pd.read_csv(file_path, low_memory=False)

    rename_map = {
        'District Name': 'district',
        'Mandal Name': 'mandal',
        'Village Name': 'village',
        'Bore Well Working': 'status',
        'Well Type': 'well_type',
        'Bore Depth': 'depth_m',
        'Pump Capacity': 'pump_capacity',
        'Crop Type': 'crop_type',
        'Irrigation Type': 'irrigation_type',
        'Extant Land Irrigated': 'irrigated_area',
        'Lat': 'lat',
        'Long': 'lon'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    print(f"Loaded bore wells: {len(df):,} records")
    return df


def load_pumping_data() -> pd.DataFrame:
    """
    Load 714 village pumping/extraction records.

    Returns:
        DataFrame with pumping data
    """
    file_path = BASE_DIR / "Pumping Data.xlsx"
    df = pd.read_excel(file_path)

    # Clean column names
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ', '_')

    print(f"Loaded pumping data: {len(df)} villages")
    return df


def load_lulc() -> gpd.GeoDataFrame:
    """
    Load land use / land cover shapefile.

    Returns:
        GeoDataFrame with LULC polygons
    """
    file_path = OFFICIAL_DIR / "LULC_Krishna" / "LULC_Krishna1.shp"
    gdf = gpd.read_file(file_path)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)

    print(f"Loaded LULC: {len(gdf)} polygons")
    return gdf


def load_rainfall_rasters() -> Dict[Tuple[int, int], Path]:
    """
    Get paths to all rainfall raster files.

    Returns:
        Dict mapping (year, month) to file path
    """
    rainfall_dir = DATA_DIR / "rainfall" / "chirps" / "krishna_clipped"

    rainfall_files = {}
    for f in rainfall_dir.glob("chirps_krishna_*.tif"):
        # Parse year and month from filename
        parts = f.stem.split('_')
        year_month = parts[2].split('.')
        year, month = int(year_month[0]), int(year_month[1])
        rainfall_files[(year, month)] = f

    print(f"Loaded rainfall: {len(rainfall_files)} monthly rasters")
    return rainfall_files


def load_dem() -> Path:
    """
    Get path to DEM raster.

    Returns:
        Path to DEM file
    """
    dem_path = DATA_DIR / "dem" / "krishna_dem_merged.tif"

    if dem_path.exists():
        with rasterio.open(dem_path) as src:
            print(f"Loaded DEM: {src.shape}, bounds: {src.bounds}")

    return dem_path


def load_grace() -> pd.DataFrame:
    """
    Load GRACE satellite time series.

    Returns:
        DataFrame with monthly TWS anomalies
    """
    file_path = DATA_DIR / "grace" / "grace_krishna_proxy.csv"
    df = pd.read_csv(file_path, parse_dates=['date'])

    print(f"Loaded GRACE: {len(df)} months")
    return df


def load_all_data() -> dict:
    """
    Load all data sources into a single dictionary.

    Returns:
        Dict with all datasets
    """
    print("=" * 60)
    print("Loading all data sources...")
    print("=" * 60)

    data = {
        'water_levels': load_water_levels(),
        'villages': load_villages(),
        'mandals': load_mandals(),
        'aquifers': load_aquifers(),
        'soils': load_soils(),
        'geomorphology': load_geomorphology(),
        'bore_wells': load_bore_wells(),
        'pumping': load_pumping_data(),
        'lulc': load_lulc(),
        'rainfall_files': load_rainfall_rasters(),
        'dem_path': load_dem(),
        'grace': load_grace()
    }

    print("=" * 60)
    print("All data loaded successfully!")
    print("=" * 60)

    return data


if __name__ == '__main__':
    data = load_all_data()

    print("\nData Summary:")
    print(f"  Water levels: {len(data['water_levels'])} piezometers")
    print(f"  Villages: {len(data['villages'])} polygons")
    print(f"  Aquifers: {len(data['aquifers'])} zones")
    print(f"  Bore wells: {len(data['bore_wells']):,} records")
    print(f"  Rainfall: {len(data['rainfall_files'])} months")
