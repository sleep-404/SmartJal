#!/usr/bin/env python3
"""
Smart Jal - CGWB Data Integration
Loads and processes CGWB national groundwater level data for training.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from shapely.geometry import Point
import warnings

warnings.filterwarnings('ignore')

CGWB_FILE = Path(__file__).parent.parent.parent / "downloaded_data" / "cgwb" / "cgwb_water_levels.csv"


def load_cgwb_data(
    states: Optional[List[str]] = None,
    districts: Optional[List[str]] = None,
    min_date: str = '2020-01-01',
    max_date: str = '2024-12-31'
) -> pd.DataFrame:
    """
    Load CGWB groundwater level data.

    Args:
        states: List of state names to include (default: AP + Telangana)
        districts: List of district names to include (optional filter)
        min_date: Start date for filtering
        max_date: End date for filtering

    Returns:
        DataFrame with groundwater level records
    """
    if not CGWB_FILE.exists():
        print(f"CGWB data file not found: {CGWB_FILE}")
        return pd.DataFrame()

    print(f"Loading CGWB data from {CGWB_FILE}...")

    df = pd.read_csv(CGWB_FILE)
    df['date'] = pd.to_datetime(df['date'])

    print(f"  Total records: {len(df)}")

    # Default to AP + Telangana
    if states is None:
        states = ['Andhra Pradesh', 'Telangana']

    df = df[df['state_name'].isin(states)]
    print(f"  After state filter ({states}): {len(df)}")

    # Optional district filter
    if districts:
        df = df[df['district_name'].isin(districts)]
        print(f"  After district filter: {len(df)}")

    # Date filter
    df = df[(df['date'] >= min_date) & (df['date'] <= max_date)]
    print(f"  After date filter ({min_date} to {max_date}): {len(df)}")

    # Clean data
    df = df.dropna(subset=['latitude', 'longitude', 'currentlevel'])
    df = df[df['currentlevel'] > 0]  # Remove invalid levels
    df = df[df['currentlevel'] < 100]  # Remove outliers (>100m depth unlikely)

    print(f"  After cleaning: {len(df)}")
    print(f"  Unique stations: {df['station_name'].nunique()}")

    return df


def cgwb_to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert CGWB DataFrame to GeoDataFrame with point geometries."""
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    return gdf


def get_unique_stations(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Get unique CGWB stations as a GeoDataFrame.

    Returns one row per station with its location.
    """
    # Get unique stations
    stations = df.groupby('station_name').agg({
        'latitude': 'first',
        'longitude': 'first',
        'district_name': 'first',
        'state_name': 'first',
        'basin': 'first',
        'sub_basin': 'first',
        'id': 'count'  # Number of measurements
    }).reset_index()

    stations = stations.rename(columns={'id': 'n_measurements'})

    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(stations['longitude'], stations['latitude'])]
    gdf = gpd.GeoDataFrame(stations, geometry=geometry, crs='EPSG:4326')

    return gdf


def prepare_training_data(
    cgwb_df: pd.DataFrame,
    target_year: int = 2020,
    target_month: int = 11
) -> Tuple[gpd.GeoDataFrame, pd.Series]:
    """
    Prepare CGWB data for model training.

    Finds water levels for the target month/year for each station.

    Returns:
        Tuple of (stations_gdf, water_levels)
    """
    # Filter to target period (allow +/- 15 days)
    target_date = pd.Timestamp(year=target_year, month=target_month, day=1)
    date_start = target_date - pd.Timedelta(days=15)
    date_end = target_date + pd.Timedelta(days=45)

    period_df = cgwb_df[(cgwb_df['date'] >= date_start) & (cgwb_df['date'] <= date_end)]

    if len(period_df) == 0:
        print(f"  No data found for {target_year}-{target_month:02d}")
        return gpd.GeoDataFrame(), pd.Series()

    # Get one measurement per station (closest to target date)
    period_df = period_df.copy()
    period_df['date_diff'] = abs((period_df['date'] - target_date).dt.days)
    period_df = period_df.sort_values('date_diff').groupby('station_name').first().reset_index()

    print(f"  Stations with data for {target_year}-{target_month:02d}: {len(period_df)}")

    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(period_df['longitude'], period_df['latitude'])]
    gdf = gpd.GeoDataFrame(period_df, geometry=geometry, crs='EPSG:4326')

    # Add required columns for feature extraction
    gdf['centroid_lat'] = gdf['latitude']
    gdf['centroid_lon'] = gdf['longitude']
    gdf['area_km2'] = 1.0  # Placeholder for point data

    water_levels = gdf['currentlevel']

    return gdf, water_levels


def get_krishna_neighbors_data(min_date: str = '2018-01-01') -> pd.DataFrame:
    """
    Get CGWB data for Krishna and neighboring districts.

    Good for training a regional model.
    """
    districts = [
        'Krishna',
        'Guntur',
        'West Godavari',
        'East Godavari',
        'Prakasam',
        'Nalgonda',  # Telangana, similar geology
        'Khammam',   # Telangana, similar geology
    ]

    return load_cgwb_data(
        states=['Andhra Pradesh', 'Telangana'],
        districts=districts,
        min_date=min_date
    )


if __name__ == '__main__':
    # Test loading
    df = load_cgwb_data()
    print(f"\nLoaded {len(df)} records")

    stations = get_unique_stations(df)
    print(f"Unique stations: {len(stations)}")

    # Test for Krishna region
    krishna_df = get_krishna_neighbors_data()
    print(f"\nKrishna + neighbors: {len(krishna_df)} records")
    print(f"Districts: {krishna_df['district_name'].unique()}")
