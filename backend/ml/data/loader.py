"""
SmartJal Data Loader

Loads and processes all spatial and tabular data for the groundwater prediction system.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.config import settings

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and process all SmartJal data sources"""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self._aquifers = None
        self._geomorphology = None
        self._lulc = None
        self._wells = None
        self._water_levels = None
        self._piezometers = None

    # ==================== Aquifers ====================

    def load_aquifers(self) -> gpd.GeoDataFrame:
        """
        Load aquifer boundaries shapefile.

        Returns:
            GeoDataFrame with columns:
            - AQUI_CODE: Aquifer code (e.g., 'BG', 'ST')
            - NEWCODE: Numeric code
            - Geo_Class: Geological classification
            - area: Area in sq km
            - geometry: Polygon geometry
        """
        if self._aquifers is None:
            path = self.data_dir / "Aquifers_Krishna" / "Aquifers_Krishna.shp"
            logger.info(f"Loading aquifers from {path}")

            self._aquifers = gpd.read_file(path)

            # Standardize column names
            self._aquifers = self._aquifers.rename(columns={
                'AQUI_CODE': 'aquifer_code',
                'NEWCODE': 'aquifer_id',
                'Geo_Class': 'aquifer_type',
                'area': 'area_sqkm'
            })

            # Ensure correct CRS (WGS84 for web mapping)
            if self._aquifers.crs is None or self._aquifers.crs.to_epsg() != 4326:
                self._aquifers = self._aquifers.to_crs(epsg=4326)

            logger.info(f"Loaded {len(self._aquifers)} aquifer polygons")

        return self._aquifers

    def get_aquifer_types(self) -> Dict[str, str]:
        """Get mapping of aquifer codes to types"""
        aquifers = self.load_aquifers()
        return dict(zip(aquifers['aquifer_code'], aquifers['aquifer_type']))

    # ==================== Geomorphology ====================

    def load_geomorphology(self) -> gpd.GeoDataFrame:
        """
        Load geomorphology shapefile.

        Returns:
            GeoDataFrame with geomorphological classifications
        """
        if self._geomorphology is None:
            path = self.data_dir / "GM_Krishna" / "GM_Krishna.shp"
            logger.info(f"Loading geomorphology from {path}")

            self._geomorphology = gpd.read_file(path)

            # Use the final description column
            if 'FIN_DESC' in self._geomorphology.columns:
                self._geomorphology['geomorph_class'] = self._geomorphology['FIN_DESC']
            elif 'DISCRIPTIO' in self._geomorphology.columns:
                self._geomorphology['geomorph_class'] = self._geomorphology['DISCRIPTIO']

            # Ensure correct CRS
            if self._geomorphology.crs is None or self._geomorphology.crs.to_epsg() != 4326:
                self._geomorphology = self._geomorphology.to_crs(epsg=4326)

            logger.info(f"Loaded {len(self._geomorphology)} geomorphology features")

        return self._geomorphology

    # ==================== Land Use / Land Cover ====================

    def load_lulc(self) -> gpd.GeoDataFrame:
        """
        Load Land Use / Land Cover shapefile.

        Returns:
            GeoDataFrame with LULC classifications
        """
        if self._lulc is None:
            path = self.data_dir / "LULC_Krishna" / "LULC_Krishna1.shp"
            logger.info(f"Loading LULC from {path}")

            self._lulc = gpd.read_file(path)

            # Load LULC classification codes
            lulc_codes_path = self.data_dir / "LULC_Krishna" / "LULC GridCode.xlsx"
            if lulc_codes_path.exists():
                lulc_codes = pd.read_excel(lulc_codes_path)
                # Merge to get human-readable names
                if 'gridcode' in self._lulc.columns and 'gridcode' in lulc_codes.columns:
                    self._lulc = self._lulc.merge(
                        lulc_codes[['gridcode', 'class_name']] if 'class_name' in lulc_codes.columns else lulc_codes,
                        on='gridcode',
                        how='left'
                    )

            # Ensure correct CRS
            if self._lulc.crs is None or self._lulc.crs.to_epsg() != 4326:
                self._lulc = self._lulc.to_crs(epsg=4326)

            logger.info(f"Loaded {len(self._lulc)} LULC features")

        return self._lulc

    # ==================== Wells ====================

    def load_wells(self) -> pd.DataFrame:
        """
        Load groundwater wells data (88,988 wells).

        Returns:
            DataFrame with columns:
            - district, mandal, village
            - well_type, bore_depth, pump_capacity
            - crop_type, irrigation_type, irrigated_area
            - lat, lon
        """
        if self._wells is None:
            path = self.data_dir / "GTWells_Krishna" / "GTWells" / "kris.csv"
            logger.info(f"Loading wells from {path}")

            self._wells = pd.read_csv(path)

            # Standardize column names
            self._wells = self._wells.rename(columns={
                'District Name': 'district',
                'Mandal Name': 'mandal',
                'Village Name': 'village',
                'Bore Well Working': 'is_working',
                'Well Type': 'well_type',
                'Bore Depth': 'bore_depth',
                'Pump Capacity': 'pump_capacity',
                'Crop Type': 'crop_type',
                'Irrigation Type': 'irrigation_type',
                'Extant Land Irrigated': 'irrigated_area',
                'Lat': 'lat',
                'Long': 'lon'
            })

            # Clean data
            self._wells['bore_depth'] = pd.to_numeric(self._wells['bore_depth'], errors='coerce')
            self._wells['pump_capacity'] = pd.to_numeric(self._wells['pump_capacity'], errors='coerce')
            self._wells['irrigated_area'] = pd.to_numeric(self._wells['irrigated_area'], errors='coerce')

            # Remove rows with invalid coordinates
            self._wells = self._wells.dropna(subset=['lat', 'lon'])
            self._wells = self._wells[
                (self._wells['lat'] > 0) &
                (self._wells['lon'] > 0) &
                (self._wells['lat'] < 90) &
                (self._wells['lon'] < 180)
            ]

            logger.info(f"Loaded {len(self._wells)} wells")

        return self._wells

    def load_wells_geodataframe(self) -> gpd.GeoDataFrame:
        """Load wells as GeoDataFrame with point geometry"""
        wells = self.load_wells()
        geometry = [Point(xy) for xy in zip(wells['lon'], wells['lat'])]
        return gpd.GeoDataFrame(wells, geometry=geometry, crs='EPSG:4326')

    # ==================== Water Levels (Piezometers) ====================

    def load_water_levels(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load piezometer water level time series data.

        Returns:
            Tuple of:
            - metadata: DataFrame with piezometer locations and attributes
            - time_series: DataFrame with monthly water level readings
        """
        if self._water_levels is None:
            path = self.data_dir / "WaterLevels_Krishna" / "master data_updated.xlsx"
            logger.info(f"Loading water levels from {path}")

            # Load the main sheet
            df = pd.read_excel(path, sheet_name='meta-historical')

            # Separate metadata columns from time series columns
            metadata_cols = [
                'SNo', 'ID', 'District', 'Mandal Name', 'Village Name',
                'Location\n(Premises)', 'Project', 'Total \nDepth \nin m',
                'Principal Aquifer', 'MSL in meters',
                'Latitude \n(Decimal Degrees)', 'Longitude \n(Decimal Degrees)'
            ]

            # Find actual column names (may have variations)
            actual_metadata_cols = []
            for col in df.columns:
                if isinstance(col, str):
                    actual_metadata_cols.append(col)
                else:
                    break  # Datetime columns start here

            # Extract metadata
            metadata = df[actual_metadata_cols].copy()
            metadata.columns = [
                'sno', 'piezo_id', 'district', 'mandal', 'village',
                'location', 'project', 'total_depth', 'aquifer',
                'msl', 'lat', 'lon'
            ][:len(actual_metadata_cols)]

            # Extract time series (all datetime columns)
            time_cols = [col for col in df.columns if isinstance(col, datetime)]

            # Create long-format time series
            time_series_data = []
            for idx, row in df.iterrows():
                piezo_id = row.get('ID', idx)
                for date_col in time_cols:
                    value = row[date_col]
                    if pd.notna(value) and value != 0:
                        time_series_data.append({
                            'piezo_id': piezo_id,
                            'date': date_col,
                            'water_level': value
                        })

            time_series = pd.DataFrame(time_series_data)
            if not time_series.empty:
                time_series['date'] = pd.to_datetime(time_series['date'])
                time_series['month'] = time_series['date'].dt.month
                time_series['year'] = time_series['date'].dt.year

                # Classify season
                time_series['season'] = time_series['month'].apply(
                    lambda m: 'monsoon' if 6 <= m <= 9
                    else ('pre_monsoon' if 3 <= m <= 5 else 'post_monsoon')
                )

            self._water_levels = (metadata, time_series)
            logger.info(f"Loaded {len(metadata)} piezometers with {len(time_series)} water level readings")

        return self._water_levels

    def load_piezometers_geodataframe(self) -> gpd.GeoDataFrame:
        """Load piezometer metadata as GeoDataFrame"""
        if self._piezometers is None:
            metadata, _ = self.load_water_levels()

            # Create geometry
            metadata = metadata.dropna(subset=['lat', 'lon'])
            geometry = [Point(xy) for xy in zip(metadata['lon'], metadata['lat'])]
            self._piezometers = gpd.GeoDataFrame(metadata, geometry=geometry, crs='EPSG:4326')

        return self._piezometers

    # ==================== Feature Aggregation ====================

    def aggregate_wells_by_village(self) -> pd.DataFrame:
        """
        Aggregate well statistics by village.

        Returns:
            DataFrame with village-level well statistics:
            - well_count, well_density (approx)
            - avg_bore_depth, median_bore_depth, max_bore_depth
            - avg_pump_capacity
            - total_irrigated_area
            - dominant_crop_type
        """
        wells = self.load_wells()

        agg = wells.groupby(['district', 'mandal', 'village']).agg({
            'bore_depth': ['count', 'mean', 'median', 'max', 'std'],
            'pump_capacity': ['mean', 'sum'],
            'irrigated_area': 'sum',
            'lat': 'mean',
            'lon': 'mean'
        })

        agg.columns = [
            'well_count', 'avg_bore_depth', 'median_bore_depth',
            'max_bore_depth', 'std_bore_depth',
            'avg_pump_capacity', 'total_pump_capacity',
            'total_irrigated_area',
            'centroid_lat', 'centroid_lon'
        ]

        agg = agg.reset_index()

        # Get dominant crop type per village
        crop_mode = wells.groupby(['district', 'mandal', 'village'])['crop_type'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
        ).reset_index()
        crop_mode.columns = ['district', 'mandal', 'village', 'dominant_crop']

        agg = agg.merge(crop_mode, on=['district', 'mandal', 'village'], how='left')

        return agg

    def assign_aquifer_to_points(
        self,
        points_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Assign aquifer type to point features based on spatial join.

        Args:
            points_gdf: GeoDataFrame with point geometry

        Returns:
            GeoDataFrame with aquifer_code and aquifer_type columns added
        """
        aquifers = self.load_aquifers()

        # Spatial join
        joined = gpd.sjoin(
            points_gdf,
            aquifers[['aquifer_code', 'aquifer_type', 'geometry']],
            how='left',
            predicate='within'
        )

        # Drop duplicate columns from join
        joined = joined.drop(columns=['index_right'], errors='ignore')

        return joined

    # ==================== Summary Statistics ====================

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all data sources"""
        aquifers = self.load_aquifers()
        geomorph = self.load_geomorphology()
        lulc = self.load_lulc()
        wells = self.load_wells()
        metadata, time_series = self.load_water_levels()

        return {
            'aquifers': {
                'count': len(aquifers),
                'types': aquifers['aquifer_type'].tolist()
            },
            'geomorphology': {
                'count': len(geomorph),
                'unique_classes': geomorph['geomorph_class'].nunique() if 'geomorph_class' in geomorph.columns else 0
            },
            'lulc': {
                'count': len(lulc)
            },
            'wells': {
                'count': len(wells),
                'avg_depth': wells['bore_depth'].mean(),
                'unique_villages': wells['village'].nunique()
            },
            'piezometers': {
                'count': len(metadata),
                'water_level_readings': len(time_series),
                'date_range': {
                    'min': time_series['date'].min().isoformat() if len(time_series) > 0 else None,
                    'max': time_series['date'].max().isoformat() if len(time_series) > 0 else None
                }
            }
        }


# Singleton instance
_loader = None


def get_data_loader() -> DataLoader:
    """Get singleton DataLoader instance"""
    global _loader
    if _loader is None:
        _loader = DataLoader()
    return _loader


# CLI interface for verification
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmartJal Data Loader")
    parser.add_argument("--verify", action="store_true", help="Verify all data loads correctly")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    loader = DataLoader()

    if args.verify:
        print("\n" + "="*60)
        print("SmartJal Data Verification")
        print("="*60)

        # Load all data
        print("\n1. Loading Aquifers...")
        aquifers = loader.load_aquifers()
        print(f"   ✓ {len(aquifers)} aquifer polygons")
        print(f"   Types: {', '.join(aquifers['aquifer_type'].tolist())}")

        print("\n2. Loading Geomorphology...")
        geomorph = loader.load_geomorphology()
        print(f"   ✓ {len(geomorph)} features")

        print("\n3. Loading LULC...")
        lulc = loader.load_lulc()
        print(f"   ✓ {len(lulc)} features")

        print("\n4. Loading Wells...")
        wells = loader.load_wells()
        print(f"   ✓ {len(wells)} wells")
        print(f"   Villages: {wells['village'].nunique()}")

        print("\n5. Loading Water Levels...")
        metadata, time_series = loader.load_water_levels()
        print(f"   ✓ {len(metadata)} piezometers")
        print(f"   ✓ {len(time_series)} water level readings")

        print("\n" + "="*60)
        print("All data loaded successfully!")
        print("="*60)
    else:
        summary = loader.get_data_summary()
        print("\nData Summary:")
        import json
        print(json.dumps(summary, indent=2, default=str))
