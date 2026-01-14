"""
SmartJal Feature Preprocessor

Combines data from all sources and engineers features for ML models.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point

from .loader import DataLoader, get_data_loader
from .gee_extractor import GEEFeatureExtractor, get_gee_extractor

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """
    Preprocesses and engineers features for groundwater prediction.

    Combines:
    - Local spatial data (aquifers, geomorphology, LULC)
    - Well statistics
    - Remote sensing features (terrain, rainfall, soil, vegetation)
    - Temporal features
    """

    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        gee_extractor: Optional[GEEFeatureExtractor] = None
    ):
        self.loader = data_loader or get_data_loader()
        self.gee = gee_extractor or get_gee_extractor()

        # Feature column groups
        self.aquifer_cols = ['aquifer_code', 'aquifer_type']
        self.terrain_cols = ['elevation', 'slope', 'aspect', 'twi']
        self.rainfall_cols = [
            'annual_rainfall', 'monsoon_rainfall',
            'pre_monsoon_rainfall', 'post_monsoon_rainfall', 'rainfall_days'
        ]
        self.soil_cols = ['sand_pct', 'clay_pct', 'silt_pct', 'soc', 'hydraulic_conductivity']
        self.vegetation_cols = ['ndvi_mean', 'ndvi_max', 'ndvi_min', 'ndvi_amplitude']
        self.et_cols = ['annual_et', 'monsoon_et']
        self.well_cols = [
            'well_count', 'avg_bore_depth', 'median_bore_depth',
            'max_bore_depth', 'avg_pump_capacity', 'total_irrigated_area'
        ]
        self.spatial_cols = [
            'dist_to_nearest_piezo', 'dist_to_drainage',
            'nearest_piezo_depth'
        ]
        self.temporal_cols = ['month_sin', 'month_cos', 'is_monsoon', 'is_pre_monsoon']

    def prepare_training_data(
        self,
        target_year: int = 2023
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Prepare training data from piezometer locations.

        Returns:
            Tuple of (X, y, metadata):
            - X: Feature matrix
            - y: Target variable (water levels)
            - metadata: Piezometer metadata for reference
        """
        logger.info("Preparing training data...")

        # Load piezometer data
        metadata, time_series = self.loader.load_water_levels()
        piezometers_gdf = self.loader.load_piezometers_geodataframe()

        # Filter to target year (use average for stability)
        year_data = time_series[time_series['year'] == target_year]
        if len(year_data) == 0:
            # Use most recent available year
            available_years = time_series['year'].unique()
            target_year = max(available_years)
            year_data = time_series[time_series['year'] == target_year]
            logger.info(f"Using year {target_year} (most recent available)")

        # Aggregate to annual average per piezometer
        annual_levels = year_data.groupby('piezo_id').agg({
            'water_level': 'mean'
        }).reset_index()

        # Merge with metadata
        training_meta = metadata.merge(annual_levels, on='piezo_id', how='inner')

        # Create GeoDataFrame for feature extraction
        training_gdf = gpd.GeoDataFrame(
            training_meta,
            geometry=gpd.points_from_xy(
                training_meta['lon'],
                training_meta['lat']
            ),
            crs='EPSG:4326'
        )

        # Extract all features
        features = self.extract_features_for_points(training_gdf, target_year)

        # Combine with target
        X = features
        y = training_meta['water_level']

        logger.info(f"Prepared {len(X)} training samples with {X.shape[1]} features")

        return X, y, training_meta

    def extract_features_for_points(
        self,
        points_gdf: gpd.GeoDataFrame,
        year: int = 2023
    ) -> pd.DataFrame:
        """
        Extract all features for given point locations.

        Args:
            points_gdf: GeoDataFrame with point geometries
            year: Year for temporal features

        Returns:
            DataFrame with all features
        """
        logger.info(f"Extracting features for {len(points_gdf)} points...")

        features_list = []

        # 1. Spatial features from local data
        spatial_features = self._extract_spatial_features(points_gdf)
        features_list.append(spatial_features)

        # 2. Well statistics for nearest village
        well_features = self._extract_well_features(points_gdf)
        features_list.append(well_features)

        # 3. GEE features (terrain, rainfall, soil, vegetation, ET)
        gee_features = self.gee.extract_all_features(points_gdf, year)
        features_list.append(gee_features)

        # Combine all features
        all_features = pd.concat(features_list, axis=1)

        # Handle duplicates from merging
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        # Fill missing values
        all_features = self._impute_missing(all_features)

        return all_features

    def _extract_spatial_features(
        self,
        points_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """Extract features from local spatial data"""
        features = pd.DataFrame(index=range(len(points_gdf)))

        # Assign aquifer type
        aquifers = self.loader.load_aquifers()
        points_with_aquifer = gpd.sjoin(
            points_gdf.reset_index(drop=True),
            aquifers[['aquifer_code', 'aquifer_type', 'geometry']],
            how='left',
            predicate='within'
        )

        features['aquifer_code'] = points_with_aquifer['aquifer_code'].values
        features['aquifer_type'] = points_with_aquifer['aquifer_type'].values

        # One-hot encode aquifer type
        if features['aquifer_type'].notna().any():
            aquifer_dummies = pd.get_dummies(
                features['aquifer_type'],
                prefix='aquifer'
            )
            features = pd.concat([features, aquifer_dummies], axis=1)

        # Assign geomorphology class
        try:
            geomorph = self.loader.load_geomorphology()
            points_with_geomorph = gpd.sjoin(
                points_gdf.reset_index(drop=True),
                geomorph[['geomorph_class', 'geometry']] if 'geomorph_class' in geomorph.columns else geomorph[['geometry']],
                how='left',
                predicate='within'
            )
            if 'geomorph_class' in points_with_geomorph.columns:
                features['geomorph_class'] = points_with_geomorph['geomorph_class'].values
        except Exception as e:
            logger.warning(f"Could not extract geomorphology: {e}")

        # Add coordinates
        features['lon'] = [p.x for p in points_gdf.geometry]
        features['lat'] = [p.y for p in points_gdf.geometry]

        # Distance to nearest piezometer
        piezometers = self.loader.load_piezometers_geodataframe()
        if len(piezometers) > 0:
            piezo_coords = np.array([[p.x, p.y] for p in piezometers.geometry])
            point_coords = np.array([[p.x, p.y] for p in points_gdf.geometry])

            tree = cKDTree(piezo_coords)
            distances, indices = tree.query(point_coords)

            features['dist_to_nearest_piezo'] = distances * 111  # Approx km conversion

        return features

    def _extract_well_features(
        self,
        points_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """Extract well statistics for nearest village"""
        features = pd.DataFrame(index=range(len(points_gdf)))

        try:
            # Get village-aggregated well stats
            village_wells = self.loader.aggregate_wells_by_village()

            if len(village_wells) > 0:
                # Build KDTree from village centroids
                village_coords = np.array([
                    [row['centroid_lon'], row['centroid_lat']]
                    for _, row in village_wells.iterrows()
                    if pd.notna(row['centroid_lon']) and pd.notna(row['centroid_lat'])
                ])

                if len(village_coords) > 0:
                    tree = cKDTree(village_coords)
                    point_coords = np.array([[p.x, p.y] for p in points_gdf.geometry])

                    distances, indices = tree.query(point_coords)

                    # Get well features from nearest village
                    valid_village_wells = village_wells[
                        village_wells['centroid_lon'].notna() &
                        village_wells['centroid_lat'].notna()
                    ].reset_index(drop=True)

                    for col in self.well_cols:
                        if col in valid_village_wells.columns:
                            features[col] = valid_village_wells.loc[indices, col].values

        except Exception as e:
            logger.warning(f"Could not extract well features: {e}")

        return features

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values"""
        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # Categorical columns: fill with mode or 'Unknown'
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                df[col] = df[col].fillna(fill_val)

        return df

    def add_temporal_features(
        self,
        df: pd.DataFrame,
        month: int
    ) -> pd.DataFrame:
        """Add temporal features for a specific month"""
        df = df.copy()

        # Cyclical encoding of month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)

        # Season flags
        df['is_monsoon'] = 1 if 6 <= month <= 9 else 0
        df['is_pre_monsoon'] = 1 if 3 <= month <= 5 else 0
        df['is_post_monsoon'] = 1 if month in [10, 11] else 0

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of all feature columns for modeling"""
        return (
            self.terrain_cols +
            self.rainfall_cols +
            self.soil_cols +
            self.vegetation_cols +
            self.et_cols +
            self.well_cols +
            self.spatial_cols +
            self.temporal_cols +
            ['lon', 'lat']
        )


# Singleton instance
_preprocessor = None


def get_preprocessor() -> FeaturePreprocessor:
    """Get singleton preprocessor instance"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = FeaturePreprocessor()
    return _preprocessor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    preprocessor = FeaturePreprocessor()

    print("\nPreparing training data...")
    X, y, meta = preprocessor.prepare_training_data()

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns:\n{list(X.columns)}")
    print(f"\nTarget statistics:\n{y.describe()}")
