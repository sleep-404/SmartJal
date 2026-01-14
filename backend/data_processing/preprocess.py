#!/usr/bin/env python3
"""
Smart Jal - Data Preprocessing Module
Cleans, transforms, and joins data for modeling.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')

from .load_data import load_all_data


def preprocess_water_levels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process water level data:
    1. Extract metadata (piezometer info)
    2. Reshape time series from wide to long format
    3. Handle missing values

    Returns:
        Tuple of (piezometer_metadata, water_levels_long)
    """
    # Identify date columns (they look like timestamps or dates)
    metadata_cols = ['sno', 'piezo_id', 'district', 'mandal', 'village', 'location',
                     'project', 'total_depth_m', 'aquifer', 'msl_m', 'lat', 'lon']

    # Get columns that exist
    existing_metadata = [c for c in metadata_cols if c in df.columns]

    # Date columns are everything else
    date_cols = [c for c in df.columns if c not in existing_metadata]

    # Extract metadata
    metadata = df[existing_metadata].copy()
    metadata['piezo_id'] = metadata['piezo_id'].fillna(metadata['sno'])

    # Reshape to long format
    water_levels_long = df.melt(
        id_vars=['piezo_id'] if 'piezo_id' in df.columns else ['sno'],
        value_vars=date_cols,
        var_name='date',
        value_name='water_level'
    )

    # Convert water_level to numeric (handle any string values)
    water_levels_long['water_level'] = pd.to_numeric(water_levels_long['water_level'], errors='coerce')

    # Parse dates
    water_levels_long['date'] = pd.to_datetime(water_levels_long['date'], errors='coerce')
    water_levels_long = water_levels_long.dropna(subset=['date'])

    # Sort by piezometer and date
    id_col = 'piezo_id' if 'piezo_id' in water_levels_long.columns else 'sno'
    water_levels_long = water_levels_long.sort_values([id_col, 'date'])

    # Handle missing values - forward fill within reasonable limits (3 months)
    water_levels_long['water_level'] = water_levels_long.groupby(id_col)['water_level'].transform(
        lambda x: x.fillna(method='ffill', limit=3)
    )

    print(f"Preprocessed water levels:")
    print(f"  Piezometers: {len(metadata)}")
    print(f"  Time series records: {len(water_levels_long):,}")
    print(f"  Date range: {water_levels_long['date'].min()} to {water_levels_long['date'].max()}")

    return metadata, water_levels_long


def assign_aquifer_to_points(points_gdf: gpd.GeoDataFrame,
                              aquifers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Spatial join to assign aquifer type to points (villages or piezometers).

    Args:
        points_gdf: GeoDataFrame with point or polygon geometries
        aquifers: GeoDataFrame with aquifer polygons

    Returns:
        GeoDataFrame with aquifer columns added
    """
    result = points_gdf.copy()

    # Get centroid for each feature
    centroids = result.geometry.centroid

    # Create temporary points GeoDataFrame for joining
    temp_points = gpd.GeoDataFrame(
        {'idx': result.index},
        geometry=centroids,
        crs=result.crs
    )

    # Spatial join with aquifers
    joined = gpd.sjoin(
        temp_points,
        aquifers[['geo_class', 'aquifer_code', 'geometry']],
        how='left',
        predicate='within'
    )

    # Handle duplicates from sjoin (take first match)
    joined = joined[~joined.index.duplicated(keep='first')]

    # Add aquifer columns to result
    result['geo_class'] = joined['geo_class'].values
    result['aquifer_code'] = joined['aquifer_code'].values

    # Handle points outside all aquifers - assign nearest
    missing_mask = result['geo_class'].isna()
    if missing_mask.any():
        print(f"  Warning: {missing_mask.sum()} points outside aquifer boundaries, assigning nearest")
        for idx in result[missing_mask].index:
            point = result.loc[idx].geometry.centroid
            distances = aquifers.geometry.distance(point)
            nearest_idx = distances.idxmin()
            result.loc[idx, 'geo_class'] = aquifers.loc[nearest_idx, 'geo_class']
            result.loc[idx, 'aquifer_code'] = aquifers.loc[nearest_idx, 'aquifer_code']

    return result


def classify_soil_types(soils: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Classify soil descriptions into infiltration and runoff classes.
    Based on official guidance from Data Issues_Clarifications.docx.

    Returns:
        GeoDataFrame with infiltration_class and runoff_class columns
    """
    soils = soils.copy()

    # Keywords for infiltration classification
    def classify_infiltration(desc):
        if pd.isna(desc):
            return 'moderate'
        desc_lower = str(desc).lower()
        if any(kw in desc_lower for kw in ['clay', 'clayey']):
            return 'low'
        elif any(kw in desc_lower for kw in ['sand', 'sandy']):
            return 'high'
        else:
            return 'moderate'

    # Keywords for runoff classification
    def classify_runoff(desc):
        if pd.isna(desc):
            return 'moderate'
        desc_lower = str(desc).lower()
        if any(kw in desc_lower for kw in ['shallow', 'less drained', 'poorly drained']):
            return 'low'
        elif any(kw in desc_lower for kw in ['deep', 'well drained']):
            return 'high'
        else:
            return 'moderate'

    soils['infiltration_class'] = soils['description'].apply(classify_infiltration)
    soils['runoff_class'] = soils['description'].apply(classify_runoff)

    # Create numeric scores
    infiltration_map = {'low': 1, 'moderate': 2, 'high': 3}
    runoff_map = {'low': 1, 'moderate': 2, 'high': 3}

    soils['infiltration_score'] = soils['infiltration_class'].map(infiltration_map)
    soils['runoff_score'] = soils['runoff_class'].map(runoff_map)

    print(f"Classified soils:")
    print(f"  Infiltration: {soils['infiltration_class'].value_counts().to_dict()}")
    print(f"  Runoff: {soils['runoff_class'].value_counts().to_dict()}")

    return soils


def assign_soil_to_villages(villages: gpd.GeoDataFrame,
                             soils: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Spatial join to assign soil type to villages.
    Uses centroid-based join for simplicity.
    """
    result = villages.copy()

    # Get centroids
    centroids = result.geometry.centroid

    # Create temporary points GeoDataFrame
    temp_points = gpd.GeoDataFrame(
        {'idx': result.index},
        geometry=centroids,
        crs=result.crs
    )

    # Spatial join with soils
    joined = gpd.sjoin(
        temp_points,
        soils[['infiltration_class', 'infiltration_score', 'runoff_class', 'runoff_score', 'geometry']],
        how='left',
        predicate='within'
    )

    # Handle duplicates
    joined = joined[~joined.index.duplicated(keep='first')]

    # Add soil columns to result
    result['infiltration_class'] = joined['infiltration_class'].values
    result['infiltration_score'] = joined['infiltration_score'].values
    result['runoff_class'] = joined['runoff_class'].values
    result['runoff_score'] = joined['runoff_score'].values

    # Handle missing - use defaults
    missing = result['infiltration_class'].isna()
    if missing.any():
        print(f"  Warning: {missing.sum()} villages outside soil polygons, using defaults")
        result.loc[missing, 'infiltration_class'] = 'moderate'
        result.loc[missing, 'infiltration_score'] = 2
        result.loc[missing, 'runoff_class'] = 'moderate'
        result.loc[missing, 'runoff_score'] = 2

    return result


def filter_bore_wells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter bore wells based on official guidance:
    - Include: Bore Well, Tube Well, Filter Point (deep wells)
    - Include: Working status only
    - Exclude: Open wells, partially working, abandoned
    """
    original_count = len(df)

    # Filter by well type (deep wells only)
    deep_well_types = ['Bore Well', 'Tube well', 'Filter Point', 'Bore well']
    df_filtered = df[df['well_type'].str.strip().isin(deep_well_types)].copy()

    # Filter by status (working only)
    working_status = ['Working', 'working']
    df_filtered = df_filtered[df_filtered['status'].str.strip().isin(working_status)]

    # Remove invalid coordinates
    df_filtered = df_filtered.dropna(subset=['lat', 'lon'])
    df_filtered = df_filtered[
        (df_filtered['lat'] >= 15.5) & (df_filtered['lat'] <= 17.5) &
        (df_filtered['lon'] >= 79.5) & (df_filtered['lon'] <= 82.0)
    ]

    print(f"Filtered bore wells:")
    print(f"  Original: {original_count:,}")
    print(f"  After filtering: {len(df_filtered):,}")
    print(f"  Removed: {original_count - len(df_filtered):,}")

    return df_filtered


def calculate_village_extraction(villages: gpd.GeoDataFrame,
                                  bore_wells: pd.DataFrame,
                                  pumping: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Calculate extraction metrics for each village using SPATIAL JOIN.
    - Number of working wells
    - Average well depth
    - Total monthly extraction (draft / 4 months)

    Includes aquifer-level imputation for villages without direct well data.
    """
    villages = villages.copy()

    # Convert depth to numeric
    bore_wells = bore_wells.copy()
    bore_wells['depth_m'] = pd.to_numeric(bore_wells['depth_m'], errors='coerce')

    # Create GeoDataFrame from bore wells
    bore_wells_valid = bore_wells.dropna(subset=['lat', 'lon'])
    wells_gdf = gpd.GeoDataFrame(
        bore_wells_valid,
        geometry=gpd.points_from_xy(bore_wells_valid['lon'], bore_wells_valid['lat']),
        crs='EPSG:4326'
    )

    # Spatial join: assign each well to a village polygon
    print("  Performing spatial join of wells to villages...")
    wells_in_villages = gpd.sjoin(
        wells_gdf,
        villages[['village', 'geometry']],
        how='inner',
        predicate='within'
    )

    # Count wells per village (using the village name from the spatial join)
    well_counts = wells_in_villages.groupby('village_right').agg(
        n_wells=('lat', 'count'),
        avg_well_depth=('depth_m', 'mean')
    ).reset_index()
    well_counts.columns = ['village', 'n_wells', 'avg_well_depth']

    # Merge with villages
    villages = villages.merge(well_counts, on='village', how='left')

    # Track which villages have direct well data
    villages['has_well_data'] = villages['n_wells'].notna() & (villages['n_wells'] > 0)

    # Monthly extraction = n_wells * avg_draft_per_well
    # Based on typical values: ~0.1 ha.m per well per season, so ~0.025 ha.m per month
    avg_draft_per_well_per_month = 0.025  # ha.m
    villages['monthly_extraction_ham'] = villages['n_wells'].fillna(0) * avg_draft_per_well_per_month

    # =========================================
    # AQUIFER-LEVEL IMPUTATION for missing data
    # =========================================
    villages_without_wells = (~villages['has_well_data']).sum()
    if villages_without_wells > 0 and 'geo_class' in villages.columns:
        print(f"  Imputing data for {villages_without_wells} villages without wells...")

        # Calculate aquifer-level statistics from villages WITH well data
        # Use per-village density to avoid division issues
        villages_with_wells = villages[villages['has_well_data']].copy()
        villages_with_wells['well_density_per_km2'] = villages_with_wells['n_wells'] / villages_with_wells['area_km2'].clip(lower=0.1)
        villages_with_wells['extraction_per_km2'] = villages_with_wells['monthly_extraction_ham'] / villages_with_wells['area_km2'].clip(lower=0.1)

        aquifer_stats = villages_with_wells.groupby('geo_class').agg(
            aquifer_well_density=('well_density_per_km2', 'median'),
            aquifer_avg_depth=('avg_well_depth', 'median'),
            aquifer_extraction_intensity=('extraction_per_km2', 'median')
        ).reset_index()

        # Apply imputation based on village area and aquifer averages
        for idx, row in villages[~villages['has_well_data']].iterrows():
            aquifer = row['geo_class']
            area_km2 = row['area_km2'] if 'area_km2' in row and pd.notna(row['area_km2']) else 10.0

            # Get aquifer stats
            aquifer_row = aquifer_stats[aquifer_stats['geo_class'] == aquifer]

            if len(aquifer_row) > 0:
                well_density = aquifer_row['aquifer_well_density'].values[0]
                avg_depth = aquifer_row['aquifer_avg_depth'].values[0]
                extraction_intensity = aquifer_row['aquifer_extraction_intensity'].values[0]

                # Handle NaN or inf values
                if pd.isna(well_density) or np.isinf(well_density):
                    well_density = 5.0  # Default wells per km2
                if pd.isna(extraction_intensity) or np.isinf(extraction_intensity):
                    extraction_intensity = 0.1  # Default extraction

                # Estimate based on area and aquifer characteristics
                estimated_wells = max(1, min(500, int(well_density * area_km2 * 0.5)))  # Conservative, capped
                villages.at[idx, 'n_wells'] = estimated_wells
                villages.at[idx, 'avg_well_depth'] = avg_depth if pd.notna(avg_depth) else 50.0
                villages.at[idx, 'monthly_extraction_ham'] = extraction_intensity * area_km2 * 0.5
            else:
                # Use regional median if aquifer stats not available
                villages.at[idx, 'n_wells'] = max(1, int(villages['n_wells'].median()))
                villages.at[idx, 'avg_well_depth'] = villages['avg_well_depth'].median()
                villages.at[idx, 'monthly_extraction_ham'] = villages['monthly_extraction_ham'].median()

        print(f"  Imputed {villages_without_wells} villages using aquifer-level averages")

    # Fill any remaining NaN with defaults
    villages['n_wells'] = villages['n_wells'].fillna(0).astype(int)
    villages['avg_well_depth'] = villages['avg_well_depth'].fillna(50)  # Default 50m
    villages['monthly_extraction_ham'] = villages['monthly_extraction_ham'].fillna(0)

    print(f"Calculated village extraction:")
    print(f"  Wells spatially matched: {len(wells_in_villages):,}")
    print(f"  Villages with direct well data: {villages['has_well_data'].sum()}")
    print(f"  Villages with imputed data: {(~villages['has_well_data']).sum()}")
    print(f"  Total wells (direct + imputed): {villages['n_wells'].sum():,}")
    print(f"  Avg wells per village: {villages['n_wells'].mean():.1f}")

    return villages


def create_piezometer_geodataframe(metadata: pd.DataFrame,
                                    aquifers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert piezometer metadata to GeoDataFrame with aquifer assignment.
    """
    # Create geometry from lat/lon
    geometry = gpd.points_from_xy(metadata['lon'], metadata['lat'])
    piezo_gdf = gpd.GeoDataFrame(metadata, geometry=geometry, crs='EPSG:4326')

    # Assign aquifers
    piezo_gdf = assign_aquifer_to_points(piezo_gdf, aquifers)

    print(f"Created piezometer GeoDataFrame:")
    print(f"  Piezometers: {len(piezo_gdf)}")
    print(f"  With aquifer: {piezo_gdf['geo_class'].notna().sum()}")

    return piezo_gdf


def preprocess_all(data: dict) -> dict:
    """
    Run all preprocessing steps.

    Returns:
        Dict with preprocessed data
    """
    print("=" * 60)
    print("Preprocessing all data...")
    print("=" * 60)

    # 1. Process water levels
    print("\n[1/6] Processing water levels...")
    piezo_metadata, water_levels_long = preprocess_water_levels(data['water_levels'])

    # 2. Classify soils
    print("\n[2/6] Classifying soils...")
    soils = classify_soil_types(data['soils'])

    # 3. Assign aquifers to villages
    print("\n[3/6] Assigning aquifers to villages...")
    villages = assign_aquifer_to_points(data['villages'], data['aquifers'])

    # 4. Assign soils to villages
    print("\n[4/6] Assigning soils to villages...")
    villages = assign_soil_to_villages(villages, soils)

    # 5. Filter bore wells
    print("\n[5/6] Filtering bore wells...")
    bore_wells_filtered = filter_bore_wells(data['bore_wells'])

    # 6. Calculate village extraction
    print("\n[6/6] Calculating village extraction...")
    villages = calculate_village_extraction(villages, bore_wells_filtered, data['pumping'])

    # Create piezometer GeoDataFrame
    print("\n[Bonus] Creating piezometer GeoDataFrame...")
    piezometers = create_piezometer_geodataframe(piezo_metadata, data['aquifers'])

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)

    return {
        'villages': villages,
        'piezometers': piezometers,
        'water_levels': water_levels_long,
        'bore_wells': bore_wells_filtered,
        'soils': soils,
        'aquifers': data['aquifers'],
        'rainfall_files': data['rainfall_files'],
        'dem_path': data['dem_path'],
        'grace': data['grace'],
        'lulc': data.get('lulc'),
        'geomorphology': data.get('geomorphology')
    }


if __name__ == '__main__':
    # Load raw data
    data = load_all_data()

    # Preprocess
    processed = preprocess_all(data)

    print("\nProcessed Data Summary:")
    print(f"  Villages: {len(processed['villages'])} with aquifer and soil")
    print(f"  Piezometers: {len(processed['piezometers'])}")
    print(f"  Water level records: {len(processed['water_levels']):,}")
    print(f"  Filtered bore wells: {len(processed['bore_wells']):,}")
