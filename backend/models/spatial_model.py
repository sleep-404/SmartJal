#!/usr/bin/env python3
"""
Smart Jal - Spatial Model Module
Aquifer-stratified kriging for spatial interpolation.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Try to import pykrige, fall back to simple IDW if not available
try:
    from pykrige.ok import OrdinaryKriging
    PYKRIGE_AVAILABLE = True
except ImportError:
    PYKRIGE_AVAILABLE = False
    print("Warning: pykrige not available, using IDW fallback")


class AquiferStratifiedKriging:
    """
    Performs kriging separately for each aquifer type.

    Key insight: Water levels should NOT be interpolated across aquifer
    boundaries as different aquifer types have different hydrogeological
    properties.
    """

    def __init__(self, variogram_model: str = 'spherical'):
        """
        Initialize kriging model.

        Args:
            variogram_model: Type of variogram ('spherical', 'exponential', 'gaussian')
        """
        self.variogram_model = variogram_model
        self.aquifer_models = {}
        self.aquifer_params = {}

    def fit(self,
            piezometers: gpd.GeoDataFrame,
            water_levels: pd.DataFrame,
            target_date: pd.Timestamp) -> 'AquiferStratifiedKriging':
        """
        Fit separate kriging models for each aquifer.

        Args:
            piezometers: GeoDataFrame with piezometer locations and aquifer assignments
            water_levels: DataFrame with water level time series
            target_date: Date for which to fit models

        Returns:
            self
        """
        print("Fitting aquifer-stratified kriging models...")

        # Get water levels for target date
        id_col = 'piezo_id' if 'piezo_id' in water_levels.columns else 'sno'
        target_wl = water_levels[water_levels['date'] == target_date].copy()

        if len(target_wl) == 0:
            # Use closest date
            closest_date = water_levels['date'].iloc[(water_levels['date'] - target_date).abs().argmin()]
            target_wl = water_levels[water_levels['date'] == closest_date].copy()
            print(f"  Using closest date: {closest_date}")

        # Merge with piezometer locations
        piezo_with_wl = piezometers.merge(
            target_wl[[id_col, 'water_level']],
            on=id_col,
            how='inner'
        )

        # Remove NaN water levels
        piezo_with_wl = piezo_with_wl.dropna(subset=['water_level'])

        print(f"  Piezometers with data: {len(piezo_with_wl)}")

        # Fit model for each aquifer
        aquifer_col = 'geo_class' if 'geo_class' in piezo_with_wl.columns else 'aquifer_code'

        for aquifer in piezo_with_wl[aquifer_col].unique():
            if pd.isna(aquifer):
                continue

            aquifer_data = piezo_with_wl[piezo_with_wl[aquifer_col] == aquifer]

            if len(aquifer_data) < 3:
                print(f"  Aquifer '{aquifer}': Too few points ({len(aquifer_data)}), using mean")
                self.aquifer_models[aquifer] = {
                    'type': 'mean',
                    'value': aquifer_data['water_level'].mean()
                }
                continue

            # Extract coordinates and values
            lons = aquifer_data.geometry.centroid.x.values
            lats = aquifer_data.geometry.centroid.y.values
            values = aquifer_data['water_level'].values

            if PYKRIGE_AVAILABLE and len(aquifer_data) >= 5:
                try:
                    # Fit ordinary kriging
                    ok = OrdinaryKriging(
                        lons, lats, values,
                        variogram_model=self.variogram_model,
                        verbose=False,
                        enable_plotting=False
                    )
                    self.aquifer_models[aquifer] = {
                        'type': 'kriging',
                        'model': ok,
                        'coords': (lons, lats, values)
                    }
                    print(f"  Aquifer '{aquifer}': Kriging fitted ({len(aquifer_data)} points)")
                except Exception as e:
                    # Fall back to IDW
                    self.aquifer_models[aquifer] = {
                        'type': 'idw',
                        'coords': (lons, lats, values)
                    }
                    print(f"  Aquifer '{aquifer}': IDW fallback ({len(aquifer_data)} points)")
            else:
                # Use IDW
                self.aquifer_models[aquifer] = {
                    'type': 'idw',
                    'coords': (lons, lats, values)
                }
                print(f"  Aquifer '{aquifer}': IDW ({len(aquifer_data)} points)")

        return self

    def predict(self,
                villages: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict water levels for village centroids.

        Args:
            villages: GeoDataFrame with village polygons and aquifer assignments

        Returns:
            Tuple of (predictions, uncertainties)
        """
        print("Predicting water levels using aquifer-stratified interpolation...")

        predictions = np.zeros(len(villages))
        uncertainties = np.zeros(len(villages))

        aquifer_col = 'geo_class' if 'geo_class' in villages.columns else 'aquifer_code'

        for aquifer, model_info in self.aquifer_models.items():
            # Get villages in this aquifer
            mask = villages[aquifer_col] == aquifer
            aquifer_villages = villages[mask]

            if len(aquifer_villages) == 0:
                continue

            # Get target coordinates
            target_lons = aquifer_villages.geometry.centroid.x.values
            target_lats = aquifer_villages.geometry.centroid.y.values

            if model_info['type'] == 'mean':
                # Simple mean prediction
                predictions[mask] = model_info['value']
                uncertainties[mask] = 5.0  # High uncertainty

            elif model_info['type'] == 'kriging':
                try:
                    ok = model_info['model']
                    z, ss = ok.execute('points', target_lons, target_lats)
                    predictions[mask] = z
                    uncertainties[mask] = np.sqrt(ss)
                except Exception as e:
                    # Fall back to IDW
                    lons, lats, values = model_info['coords']
                    pred, unc = self._idw_predict(
                        lons, lats, values,
                        target_lons, target_lats
                    )
                    predictions[mask] = pred
                    uncertainties[mask] = unc

            elif model_info['type'] == 'idw':
                lons, lats, values = model_info['coords']
                pred, unc = self._idw_predict(
                    lons, lats, values,
                    target_lons, target_lats
                )
                predictions[mask] = pred
                uncertainties[mask] = unc

        # Handle villages with no aquifer assignment
        no_aquifer = villages[aquifer_col].isna()
        if no_aquifer.any():
            # Use global mean
            all_means = [m['value'] if m['type'] == 'mean' else np.mean(m['coords'][2])
                        for m in self.aquifer_models.values()]
            predictions[no_aquifer] = np.mean(all_means)
            uncertainties[no_aquifer] = 10.0

        print(f"  Predictions range: {predictions.min():.1f} - {predictions.max():.1f}m")
        print(f"  Mean uncertainty: {uncertainties.mean():.2f}m")

        return predictions, uncertainties

    def _idw_predict(self,
                     source_lons: np.ndarray,
                     source_lats: np.ndarray,
                     source_values: np.ndarray,
                     target_lons: np.ndarray,
                     target_lats: np.ndarray,
                     power: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse Distance Weighting interpolation.

        Args:
            source_lons, source_lats: Known point coordinates
            source_values: Known values
            target_lons, target_lats: Target point coordinates
            power: Distance power (typically 2)

        Returns:
            Tuple of (predictions, uncertainties)
        """
        # Calculate distances
        source_coords = np.column_stack([source_lons, source_lats])
        target_coords = np.column_stack([target_lons, target_lats])

        # Convert to approximate meters (at ~16°N latitude)
        source_coords_m = source_coords * np.array([111320 * np.cos(np.radians(16)), 110540])
        target_coords_m = target_coords * np.array([111320 * np.cos(np.radians(16)), 110540])

        distances = cdist(target_coords_m, source_coords_m)

        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)

        # Calculate weights
        weights = 1 / (distances ** power)
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_normalized = weights / weights_sum

        # Weighted average
        predictions = (weights_normalized * source_values).sum(axis=1)

        # Uncertainty based on distance to nearest point
        min_distances = distances.min(axis=1)
        uncertainties = np.log1p(min_distances / 1000)  # Log scale

        return predictions, uncertainties


class SpatialAutocorrelation:
    """
    Calculate spatial autocorrelation statistics.
    """

    @staticmethod
    def morans_i(values: np.ndarray,
                 coords: np.ndarray,
                 distance_threshold: float = 50000) -> float:
        """
        Calculate Moran's I for spatial autocorrelation.

        Args:
            values: Values to test
            coords: Coordinates (N x 2)
            distance_threshold: Max distance for neighbors (meters)

        Returns:
            Moran's I statistic
        """
        n = len(values)
        if n < 3:
            return 0.0

        # Standardize values
        z = (values - values.mean()) / values.std()

        # Calculate weights matrix
        distances = cdist(coords, coords)
        W = (distances < distance_threshold).astype(float)
        np.fill_diagonal(W, 0)

        # Normalize weights
        W_sum = W.sum()
        if W_sum == 0:
            return 0.0

        # Calculate Moran's I
        numerator = np.sum(W * np.outer(z, z))
        denominator = np.sum(z ** 2)

        I = (n / W_sum) * (numerator / denominator)

        return I


def fit_variogram(coords: np.ndarray,
                  values: np.ndarray,
                  n_lags: int = 15) -> Dict:
    """
    Fit empirical variogram to data.

    Args:
        coords: Coordinates (N x 2)
        values: Values at coordinates
        n_lags: Number of lag classes

    Returns:
        Dict with variogram parameters
    """
    # Calculate pairwise distances
    distances = cdist(coords, coords)

    # Calculate semivariance
    n = len(values)
    semivariances = []
    lag_distances = []

    max_dist = distances.max() / 2
    lag_size = max_dist / n_lags

    for i in range(n_lags):
        lag_min = i * lag_size
        lag_max = (i + 1) * lag_size

        mask = (distances > lag_min) & (distances <= lag_max)
        if mask.sum() > 0:
            # Get value differences for pairs in this lag
            idx = np.where(mask)
            diffs = values[idx[0]] - values[idx[1]]
            semivar = 0.5 * np.mean(diffs ** 2)
            semivariances.append(semivar)
            lag_distances.append((lag_min + lag_max) / 2)

    return {
        'lag_distances': np.array(lag_distances),
        'semivariances': np.array(semivariances),
        'sill': np.max(semivariances) if semivariances else 0,
        'range': lag_distances[-1] if lag_distances else 0,
        'nugget': semivariances[0] if semivariances else 0
    }


if __name__ == '__main__':
    # Test with dummy data
    print("Testing spatial model...")

    # Create dummy piezometer data
    np.random.seed(42)
    n_piezo = 50

    lons = np.random.uniform(80.0, 81.5, n_piezo)
    lats = np.random.uniform(15.5, 17.0, n_piezo)
    water_levels = 10 + 5 * (lats - 16) + np.random.normal(0, 1, n_piezo)

    piezo_gdf = gpd.GeoDataFrame(
        {
            'piezo_id': range(n_piezo),
            'geo_class': np.random.choice(['Granite', 'Basalt', 'Alluvium'], n_piezo)
        },
        geometry=gpd.points_from_xy(lons, lats),
        crs='EPSG:4326'
    )

    wl_df = pd.DataFrame({
        'piezo_id': range(n_piezo),
        'date': pd.Timestamp('2023-10-01'),
        'water_level': water_levels
    })

    # Fit model
    model = AquiferStratifiedKriging()
    model.fit(piezo_gdf, wl_df, pd.Timestamp('2023-10-01'))

    # Create dummy villages
    n_villages = 100
    village_lons = np.random.uniform(80.0, 81.5, n_villages)
    village_lats = np.random.uniform(15.5, 17.0, n_villages)

    villages_gdf = gpd.GeoDataFrame(
        {
            'village': range(n_villages),
            'geo_class': np.random.choice(['Granite', 'Basalt', 'Alluvium'], n_villages)
        },
        geometry=gpd.points_from_xy(village_lons, village_lats),
        crs='EPSG:4326'
    )

    # Predict
    predictions, uncertainties = model.predict(villages_gdf)

    print(f"\nResults:")
    print(f"  Predictions: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
    print(f"  Uncertainties: mean={uncertainties.mean():.2f}")
