#!/usr/bin/env python3
"""
Feature Extractor for Smart Jal
Extracts features from raster data (rainfall, DEM) at village point locations.
"""

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class FeatureExtractor:
    """Extract geospatial features for village locations."""

    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "downloaded_data"
        self.data_dir = data_dir
        self.villages = None
        self.dem = None
        self.dem_transform = None

    def load_villages(self) -> pd.DataFrame:
        """Load village centroids."""
        villages_file = self.data_dir / "villages" / "krishna_village_centroids.csv"
        self.villages = pd.read_csv(villages_file)
        print(f"Loaded {len(self.villages)} villages")
        return self.villages

    def load_dem(self) -> np.ndarray:
        """Load DEM raster."""
        dem_file = self.data_dir / "dem" / "krishna_dem_merged.tif"
        with rasterio.open(dem_file) as src:
            self.dem = src.read(1)
            self.dem_transform = src.transform
            self.dem_crs = src.crs
            self.dem_bounds = src.bounds
        print(f"Loaded DEM: {self.dem.shape}, bounds: {self.dem_bounds}")
        return self.dem

    def get_raster_value_at_point(
        self,
        raster: np.ndarray,
        transform,
        lon: float,
        lat: float
    ) -> Optional[float]:
        """Get raster value at a geographic point."""
        try:
            # Convert geographic coordinates to pixel coordinates
            col, row = ~transform * (lon, lat)
            col, row = int(col), int(row)

            # Check bounds
            if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:
                value = raster[row, col]
                # Handle nodata
                if value == -9999 or value < -1000:
                    return None
                return float(value)
            return None
        except Exception:
            return None

    def extract_elevation(self) -> pd.DataFrame:
        """Extract elevation for all villages."""
        if self.villages is None:
            self.load_villages()
        if self.dem is None:
            self.load_dem()

        elevations = []
        for _, row in self.villages.iterrows():
            elev = self.get_raster_value_at_point(
                self.dem, self.dem_transform, row['lon'], row['lat']
            )
            elevations.append(elev)

        self.villages['elevation_m'] = elevations
        valid_count = self.villages['elevation_m'].notna().sum()
        print(f"Extracted elevation for {valid_count}/{len(self.villages)} villages")
        return self.villages

    def extract_rainfall_for_month(self, year: int, month: int) -> pd.DataFrame:
        """Extract rainfall for a specific month."""
        rainfall_file = (
            self.data_dir / "rainfall" / "chirps" / "krishna_clipped" /
            f"chirps_krishna_{year}.{month:02d}.tif"
        )

        if not rainfall_file.exists():
            print(f"Rainfall file not found: {rainfall_file}")
            return self.villages

        with rasterio.open(rainfall_file) as src:
            rainfall_data = src.read(1)
            transform = src.transform

        col_name = f"rainfall_{year}_{month:02d}"
        rainfall_values = []

        for _, row in self.villages.iterrows():
            rain = self.get_raster_value_at_point(
                rainfall_data, transform, row['lon'], row['lat']
            )
            rainfall_values.append(rain)

        self.villages[col_name] = rainfall_values
        return self.villages

    def extract_all_rainfall(self, start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
        """Extract rainfall for all available months."""
        if self.villages is None:
            self.load_villages()

        rainfall_dir = self.data_dir / "rainfall" / "chirps" / "krishna_clipped"
        rainfall_files = sorted(rainfall_dir.glob("chirps_krishna_*.tif"))

        print(f"Found {len(rainfall_files)} rainfall files")

        for rf_file in rainfall_files:
            # Parse year and month from filename: chirps_krishna_YYYY.MM.tif
            parts = rf_file.stem.split('_')
            year_month = parts[2].split('.')
            year, month = int(year_month[0]), int(year_month[1])

            if start_year <= year <= end_year:
                self.extract_rainfall_for_month(year, month)

        # Calculate rainfall statistics
        rainfall_cols = [c for c in self.villages.columns if c.startswith('rainfall_')]
        if rainfall_cols:
            self.villages['rainfall_mean'] = self.villages[rainfall_cols].mean(axis=1)
            self.villages['rainfall_std'] = self.villages[rainfall_cols].std(axis=1)
            self.villages['rainfall_max'] = self.villages[rainfall_cols].max(axis=1)
            self.villages['rainfall_min'] = self.villages[rainfall_cols].min(axis=1)
            print(f"Calculated rainfall stats from {len(rainfall_cols)} months")

        return self.villages

    def calculate_terrain_features(self) -> pd.DataFrame:
        """Calculate terrain-derived features (slope proxy, relative elevation)."""
        if self.villages is None or 'elevation_m' not in self.villages.columns:
            self.extract_elevation()

        # Calculate relative elevation (compared to surrounding villages)
        # This is a simple proxy - proper slope would need DEM gradient
        mean_elev = self.villages['elevation_m'].mean()
        self.villages['elevation_relative'] = self.villages['elevation_m'] - mean_elev

        # Elevation percentile within the district
        self.villages['elevation_percentile'] = (
            self.villages['elevation_m'].rank(pct=True) * 100
        )

        print("Calculated terrain features")
        return self.villages

    def extract_all_features(self) -> pd.DataFrame:
        """Extract all available features."""
        print("=" * 50)
        print("Extracting all features for villages")
        print("=" * 50)

        self.load_villages()
        self.extract_elevation()
        self.calculate_terrain_features()
        self.extract_all_rainfall()

        print(f"\nFinal dataset: {len(self.villages)} villages, {len(self.villages.columns)} columns")
        return self.villages

    def save_features(self, output_file: Path = None) -> Path:
        """Save extracted features to CSV."""
        if output_file is None:
            output_file = self.data_dir / "processed" / "village_features.csv"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.villages.to_csv(output_file, index=False)
        print(f"Saved features to: {output_file}")
        return output_file


def main():
    """Run feature extraction pipeline."""
    extractor = FeatureExtractor()
    villages = extractor.extract_all_features()

    # Save results
    output_file = extractor.save_features()

    # Print summary
    print("\n" + "=" * 50)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Villages: {len(villages)}")
    print(f"Features: {len(villages.columns)}")
    print(f"\nColumns:")
    for col in villages.columns:
        non_null = villages[col].notna().sum()
        print(f"  {col}: {non_null}/{len(villages)} valid")

    print(f"\nOutput: {output_file}")


if __name__ == '__main__':
    main()
