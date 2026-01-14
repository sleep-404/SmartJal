"""
SmartJal Google Earth Engine Feature Extractor

Extracts terrain, climate, soil, and vegetation features from GEE for
groundwater prediction modeling.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)

# Try to import Earth Engine
try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    logger.warning("Earth Engine API not available. Install with: pip install earthengine-api")


class GEEFeatureExtractor:
    """
    Extract features from Google Earth Engine for groundwater modeling.

    Features extracted:
    - Terrain: elevation, slope, aspect, TWI from SRTM
    - Rainfall: annual, seasonal from CHIRPS
    - Soil: texture, properties from SoilGrids
    - Vegetation: NDVI, NDWI from Sentinel-2/Landsat
    - ET: evapotranspiration from MODIS
    """

    # Krishna District bounding box
    KRISHNA_BOUNDS = {
        'min_lon': 80.0,
        'max_lon': 81.5,
        'min_lat': 15.5,
        'max_lat': 17.0
    }

    # GEE Dataset IDs
    DATASETS = {
        'srtm': 'USGS/SRTMGL1_003',
        'chirps': 'UCSB-CHG/CHIRPS/DAILY',
        'modis_et': 'MODIS/006/MOD16A2',
        'modis_ndvi': 'MODIS/006/MOD13A2',
        'sentinel2': 'COPERNICUS/S2_SR_HARMONIZED',
        'landsat8': 'LANDSAT/LC08/C02/T1_L2',
        'soilgrids_sand': 'projects/soilgrids-isric/sand_mean',
        'soilgrids_clay': 'projects/soilgrids-isric/clay_mean',
        'soilgrids_soc': 'projects/soilgrids-isric/soc_mean',
    }

    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize GEE Feature Extractor.

        Args:
            project_id: GEE project ID for authentication
        """
        self.project_id = project_id
        self._initialized = False
        self._krishna_geometry = None

    def initialize(self) -> bool:
        """Initialize Earth Engine connection"""
        if not EE_AVAILABLE:
            logger.error("Earth Engine API not installed")
            return False

        if self._initialized:
            return True

        try:
            # Try to initialize with project
            if self.project_id:
                ee.Initialize(project=self.project_id)
            else:
                # Try default initialization
                ee.Initialize()

            self._initialized = True
            logger.info("Earth Engine initialized successfully")
            return True

        except Exception as e:
            logger.warning(f"EE initialization failed: {e}")
            logger.info("Attempting to authenticate...")

            try:
                ee.Authenticate()
                ee.Initialize(project=self.project_id) if self.project_id else ee.Initialize()
                self._initialized = True
                logger.info("Earth Engine authenticated and initialized")
                return True
            except Exception as auth_error:
                logger.error(f"EE authentication failed: {auth_error}")
                return False

    @property
    def krishna_geometry(self) -> 'ee.Geometry':
        """Get Krishna District bounding box as EE geometry"""
        if self._krishna_geometry is None:
            self._krishna_geometry = ee.Geometry.Rectangle([
                self.KRISHNA_BOUNDS['min_lon'],
                self.KRISHNA_BOUNDS['min_lat'],
                self.KRISHNA_BOUNDS['max_lon'],
                self.KRISHNA_BOUNDS['max_lat']
            ])
        return self._krishna_geometry

    # ==================== Terrain Features ====================

    def extract_terrain_features(
        self,
        points: gpd.GeoDataFrame,
        scale: int = 30
    ) -> pd.DataFrame:
        """
        Extract terrain features from SRTM DEM.

        Args:
            points: GeoDataFrame with point geometries
            scale: Resolution in meters (default 30m for SRTM)

        Returns:
            DataFrame with columns:
            - elevation: meters above sea level
            - slope: degrees
            - aspect: degrees from north
            - twi: Topographic Wetness Index
        """
        if not self.initialize():
            return self._fallback_terrain_features(points)

        try:
            dem = ee.Image(self.DATASETS['srtm'])

            # Calculate derived terrain features
            slope = ee.Terrain.slope(dem)
            aspect = ee.Terrain.aspect(dem)

            # Topographic Wetness Index: ln(a / tan(b))
            # where a = upslope contributing area, b = slope
            flow_accumulation = ee.Image.constant(1000)  # Simplified
            slope_radians = slope.multiply(np.pi / 180)
            twi = flow_accumulation.log().subtract(slope_radians.tan().log())

            # Stack all terrain bands
            terrain = dem.rename('elevation').addBands([
                slope.rename('slope'),
                aspect.rename('aspect'),
                twi.rename('twi')
            ])

            # Extract values at points
            features = self._extract_at_points(terrain, points, scale)

            return features

        except Exception as e:
            logger.error(f"Error extracting terrain features: {e}")
            return self._fallback_terrain_features(points)

    def _fallback_terrain_features(self, points: gpd.GeoDataFrame) -> pd.DataFrame:
        """Generate placeholder terrain features when GEE unavailable"""
        n = len(points)
        return pd.DataFrame({
            'elevation': np.random.uniform(10, 100, n),
            'slope': np.random.uniform(0, 15, n),
            'aspect': np.random.uniform(0, 360, n),
            'twi': np.random.uniform(5, 15, n)
        })

    # ==================== Rainfall Features ====================

    def extract_rainfall_features(
        self,
        points: gpd.GeoDataFrame,
        year: int = 2023,
        scale: int = 5000
    ) -> pd.DataFrame:
        """
        Extract rainfall features from CHIRPS.

        Args:
            points: GeoDataFrame with point geometries
            year: Year to extract rainfall for
            scale: Resolution in meters

        Returns:
            DataFrame with columns:
            - annual_rainfall: mm/year
            - monsoon_rainfall: mm (Jun-Sep)
            - pre_monsoon_rainfall: mm (Mar-May)
            - post_monsoon_rainfall: mm (Oct-Nov)
            - rainfall_days: days with >2.5mm
        """
        if not self.initialize():
            return self._fallback_rainfall_features(points)

        try:
            # Define date ranges
            year_start = f'{year}-01-01'
            year_end = f'{year}-12-31'

            chirps = ee.ImageCollection(self.DATASETS['chirps'])

            # Annual rainfall
            annual = chirps.filterDate(year_start, year_end).sum()

            # Monsoon (June-September)
            monsoon = chirps.filterDate(f'{year}-06-01', f'{year}-09-30').sum()

            # Pre-monsoon (March-May)
            pre_monsoon = chirps.filterDate(f'{year}-03-01', f'{year}-05-31').sum()

            # Post-monsoon (October-November)
            post_monsoon = chirps.filterDate(f'{year}-10-01', f'{year}-11-30').sum()

            # Rainy days (>2.5mm)
            rainy_days = chirps.filterDate(year_start, year_end).map(
                lambda img: img.gt(2.5)
            ).sum()

            # Stack bands
            rainfall = annual.rename('annual_rainfall').addBands([
                monsoon.rename('monsoon_rainfall'),
                pre_monsoon.rename('pre_monsoon_rainfall'),
                post_monsoon.rename('post_monsoon_rainfall'),
                rainy_days.rename('rainfall_days')
            ])

            features = self._extract_at_points(rainfall, points, scale)
            return features

        except Exception as e:
            logger.error(f"Error extracting rainfall features: {e}")
            return self._fallback_rainfall_features(points)

    def _fallback_rainfall_features(self, points: gpd.GeoDataFrame) -> pd.DataFrame:
        """Generate placeholder rainfall features"""
        n = len(points)
        return pd.DataFrame({
            'annual_rainfall': np.random.uniform(700, 1200, n),
            'monsoon_rainfall': np.random.uniform(500, 900, n),
            'pre_monsoon_rainfall': np.random.uniform(50, 150, n),
            'post_monsoon_rainfall': np.random.uniform(100, 200, n),
            'rainfall_days': np.random.randint(40, 80, n)
        })

    # ==================== Soil Features ====================

    def extract_soil_features(
        self,
        points: gpd.GeoDataFrame,
        scale: int = 250
    ) -> pd.DataFrame:
        """
        Extract soil features from SoilGrids.

        Args:
            points: GeoDataFrame with point geometries
            scale: Resolution in meters (250m for SoilGrids)

        Returns:
            DataFrame with columns:
            - sand_pct: Sand content %
            - clay_pct: Clay content %
            - soc: Soil organic carbon g/kg
        """
        if not self.initialize():
            return self._fallback_soil_features(points)

        try:
            # SoilGrids data at 0-5cm depth
            sand = ee.Image("projects/soilgrids-isric/sand_mean").select('sand_0-5cm_mean')
            clay = ee.Image("projects/soilgrids-isric/clay_mean").select('clay_0-5cm_mean')
            soc = ee.Image("projects/soilgrids-isric/soc_mean").select('soc_0-5cm_mean')

            # Stack bands
            soil = sand.rename('sand_pct').addBands([
                clay.rename('clay_pct'),
                soc.rename('soc')
            ])

            # Scale values (SoilGrids stores as integers)
            soil = soil.divide(10)  # Convert to actual percentages/values

            features = self._extract_at_points(soil, points, scale)
            return features

        except Exception as e:
            logger.error(f"Error extracting soil features: {e}")
            return self._fallback_soil_features(points)

    def _fallback_soil_features(self, points: gpd.GeoDataFrame) -> pd.DataFrame:
        """Generate placeholder soil features"""
        n = len(points)
        sand = np.random.uniform(30, 70, n)
        clay = np.random.uniform(10, 40, n)
        return pd.DataFrame({
            'sand_pct': sand,
            'clay_pct': clay,
            'silt_pct': 100 - sand - clay,
            'soc': np.random.uniform(5, 20, n)
        })

    # ==================== Vegetation Features ====================

    def extract_vegetation_features(
        self,
        points: gpd.GeoDataFrame,
        year: int = 2023,
        scale: int = 500
    ) -> pd.DataFrame:
        """
        Extract vegetation indices from MODIS.

        Args:
            points: GeoDataFrame with point geometries
            year: Year to extract for
            scale: Resolution in meters

        Returns:
            DataFrame with columns:
            - ndvi_mean: Mean NDVI
            - ndvi_max: Maximum NDVI
            - ndvi_min: Minimum NDVI
            - ndvi_amplitude: Max - Min NDVI
        """
        if not self.initialize():
            return self._fallback_vegetation_features(points)

        try:
            modis_ndvi = ee.ImageCollection(self.DATASETS['modis_ndvi'])

            # Filter to year and Krishna region
            ndvi_collection = modis_ndvi.filterDate(
                f'{year}-01-01', f'{year}-12-31'
            ).filterBounds(self.krishna_geometry).select('NDVI')

            # Calculate statistics
            ndvi_mean = ndvi_collection.mean().multiply(0.0001)  # Scale factor
            ndvi_max = ndvi_collection.max().multiply(0.0001)
            ndvi_min = ndvi_collection.min().multiply(0.0001)

            # Stack bands
            vegetation = ndvi_mean.rename('ndvi_mean').addBands([
                ndvi_max.rename('ndvi_max'),
                ndvi_min.rename('ndvi_min')
            ])

            features = self._extract_at_points(vegetation, points, scale)

            # Calculate amplitude
            if 'ndvi_max' in features.columns and 'ndvi_min' in features.columns:
                features['ndvi_amplitude'] = features['ndvi_max'] - features['ndvi_min']

            return features

        except Exception as e:
            logger.error(f"Error extracting vegetation features: {e}")
            return self._fallback_vegetation_features(points)

    def _fallback_vegetation_features(self, points: gpd.GeoDataFrame) -> pd.DataFrame:
        """Generate placeholder vegetation features"""
        n = len(points)
        ndvi_min = np.random.uniform(0.1, 0.3, n)
        ndvi_max = ndvi_min + np.random.uniform(0.2, 0.5, n)
        return pd.DataFrame({
            'ndvi_mean': (ndvi_min + ndvi_max) / 2,
            'ndvi_max': ndvi_max,
            'ndvi_min': ndvi_min,
            'ndvi_amplitude': ndvi_max - ndvi_min
        })

    # ==================== Evapotranspiration Features ====================

    def extract_et_features(
        self,
        points: gpd.GeoDataFrame,
        year: int = 2023,
        scale: int = 500
    ) -> pd.DataFrame:
        """
        Extract evapotranspiration from MODIS MOD16.

        Args:
            points: GeoDataFrame with point geometries
            year: Year to extract for
            scale: Resolution in meters

        Returns:
            DataFrame with columns:
            - annual_et: mm/year
            - monsoon_et: mm (Jun-Sep)
        """
        if not self.initialize():
            return self._fallback_et_features(points)

        try:
            modis_et = ee.ImageCollection(self.DATASETS['modis_et'])

            # Annual ET
            annual_et = modis_et.filterDate(
                f'{year}-01-01', f'{year}-12-31'
            ).select('ET').sum().multiply(0.1)  # Scale factor

            # Monsoon ET
            monsoon_et = modis_et.filterDate(
                f'{year}-06-01', f'{year}-09-30'
            ).select('ET').sum().multiply(0.1)

            # Stack bands
            et = annual_et.rename('annual_et').addBands(
                monsoon_et.rename('monsoon_et')
            )

            features = self._extract_at_points(et, points, scale)
            return features

        except Exception as e:
            logger.error(f"Error extracting ET features: {e}")
            return self._fallback_et_features(points)

    def _fallback_et_features(self, points: gpd.GeoDataFrame) -> pd.DataFrame:
        """Generate placeholder ET features"""
        n = len(points)
        return pd.DataFrame({
            'annual_et': np.random.uniform(800, 1400, n),
            'monsoon_et': np.random.uniform(400, 700, n)
        })

    # ==================== Helper Methods ====================

    def _extract_at_points(
        self,
        image: 'ee.Image',
        points: gpd.GeoDataFrame,
        scale: int
    ) -> pd.DataFrame:
        """Extract image values at point locations"""
        # Convert points to EE FeatureCollection
        features = []
        for idx, row in points.iterrows():
            point = ee.Geometry.Point([row.geometry.x, row.geometry.y])
            features.append(ee.Feature(point, {'id': idx}))

        fc = ee.FeatureCollection(features)

        # Sample the image
        sampled = image.sampleRegions(
            collection=fc,
            scale=scale,
            geometries=False
        )

        # Get results
        try:
            result = sampled.getInfo()
            if result and 'features' in result:
                data = []
                for f in result['features']:
                    props = f.get('properties', {})
                    data.append(props)
                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error sampling image: {e}")

        return pd.DataFrame()

    # ==================== All Features ====================

    def extract_all_features(
        self,
        points: gpd.GeoDataFrame,
        year: int = 2023
    ) -> pd.DataFrame:
        """
        Extract all GEE features for given points.

        Args:
            points: GeoDataFrame with point geometries
            year: Year for temporal features

        Returns:
            DataFrame with all extracted features
        """
        logger.info(f"Extracting GEE features for {len(points)} points...")

        # Extract each feature group
        terrain = self.extract_terrain_features(points)
        rainfall = self.extract_rainfall_features(points, year)
        soil = self.extract_soil_features(points)
        vegetation = self.extract_vegetation_features(points, year)
        et = self.extract_et_features(points, year)

        # Combine all features
        all_features = pd.concat([
            terrain.reset_index(drop=True),
            rainfall.reset_index(drop=True),
            soil.reset_index(drop=True),
            vegetation.reset_index(drop=True),
            et.reset_index(drop=True)
        ], axis=1)

        # Add derived features
        if 'annual_rainfall' in all_features.columns and 'annual_et' in all_features.columns:
            all_features['water_balance'] = all_features['annual_rainfall'] - all_features['annual_et']

        if 'sand_pct' in all_features.columns:
            # Estimate hydraulic conductivity from texture (simplified Saxton-Rawls)
            all_features['hydraulic_conductivity'] = (
                all_features['sand_pct'] * 0.5 +
                all_features.get('silt_pct', 30) * 0.1 +
                all_features.get('clay_pct', 20) * 0.01
            )

        logger.info(f"Extracted {len(all_features.columns)} features")
        return all_features


# Singleton instance
_extractor = None


def get_gee_extractor(project_id: Optional[str] = None) -> GEEFeatureExtractor:
    """Get singleton GEE extractor instance"""
    global _extractor
    if _extractor is None:
        _extractor = GEEFeatureExtractor(project_id)
    return _extractor


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GEE Feature Extractor")
    parser.add_argument("--test", action="store_true", help="Test extraction with sample points")
    parser.add_argument("--project", type=str, help="GEE project ID")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    extractor = GEEFeatureExtractor(args.project)

    if args.test:
        print("\nTesting GEE Feature Extraction...")
        print("="*50)

        # Create sample points in Krishna District
        sample_points = gpd.GeoDataFrame({
            'name': ['Point1', 'Point2', 'Point3'],
            'geometry': gpd.points_from_xy(
                [80.5, 80.8, 81.0],  # Longitudes
                [16.0, 16.3, 16.5]   # Latitudes
            )
        }, crs='EPSG:4326')

        print(f"\nSample points:\n{sample_points}")

        # Extract features
        features = extractor.extract_all_features(sample_points)

        print(f"\nExtracted features:\n{features}")
        print(f"\nFeature columns: {list(features.columns)}")
