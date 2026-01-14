"""
SmartJal Ensemble Model

Combines predictions from multiple models (aquifer-specific, spatial, temporal)
with learned weights and uncertainty quantification.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge

from .aquifer_model import AquiferModelManager

logger = logging.getLogger(__name__)


class SpatialInterpolator:
    """
    Simple spatial interpolation using Inverse Distance Weighting (IDW)
    constrained by aquifer boundaries.
    """

    def __init__(self, power: float = 2.0, max_neighbors: int = 10):
        self.power = power
        self.max_neighbors = max_neighbors
        self._fitted = False
        self._coords = None
        self._values = None
        self._aquifers = None
        self._tree = None

    def fit(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        aquifer_ids: Optional[np.ndarray] = None
    ):
        """
        Fit the interpolator with known points.

        Args:
            coords: Array of (lon, lat) coordinates
            values: Array of water level values
            aquifer_ids: Optional array of aquifer IDs for constrained interpolation
        """
        self._coords = np.array(coords)
        self._values = np.array(values)
        self._aquifers = aquifer_ids
        self._tree = cKDTree(self._coords)
        self._fitted = True

    def predict(
        self,
        coords: np.ndarray,
        aquifer_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate values at new locations.

        Args:
            coords: Array of (lon, lat) coordinates to predict
            aquifer_ids: Optional array of aquifer IDs for query points

        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self._fitted:
            raise ValueError("Interpolator not fitted")

        coords = np.array(coords)
        predictions = np.zeros(len(coords))
        uncertainties = np.zeros(len(coords))

        for i, (lon, lat) in enumerate(coords):
            # Find neighbors
            if self._aquifers is not None and aquifer_ids is not None:
                # Constrained: only use points in same aquifer
                query_aquifer = aquifer_ids[i]
                same_aquifer_mask = self._aquifers == query_aquifer
                same_aquifer_indices = np.where(same_aquifer_mask)[0]

                if len(same_aquifer_indices) == 0:
                    # No points in same aquifer, use all points
                    distances, indices = self._tree.query(
                        [lon, lat],
                        k=min(self.max_neighbors, len(self._coords))
                    )
                else:
                    # Build tree for same-aquifer points
                    aquifer_coords = self._coords[same_aquifer_mask]
                    aquifer_tree = cKDTree(aquifer_coords)
                    k = min(self.max_neighbors, len(aquifer_coords))
                    distances, local_indices = aquifer_tree.query([lon, lat], k=k)
                    indices = same_aquifer_indices[local_indices]
            else:
                # Unconstrained IDW
                distances, indices = self._tree.query(
                    [lon, lat],
                    k=min(self.max_neighbors, len(self._coords))
                )

            # Handle single neighbor case
            if np.isscalar(distances):
                distances = np.array([distances])
                indices = np.array([indices])

            # IDW weights
            if distances[0] < 1e-10:
                # Exact match
                predictions[i] = self._values[indices[0]]
                uncertainties[i] = 0.0
            else:
                weights = 1.0 / (distances ** self.power)
                weights /= weights.sum()

                predictions[i] = np.sum(weights * self._values[indices])

                # Uncertainty: weighted standard deviation
                residuals = self._values[indices] - predictions[i]
                uncertainties[i] = np.sqrt(np.sum(weights * residuals**2))

        return predictions, uncertainties


class EnsemblePredictor:
    """
    Ensemble model that combines:
    1. Aquifer-specific ML models
    2. Spatial interpolation (IDW)

    With learned weights and uncertainty quantification.
    """

    def __init__(
        self,
        aquifer_manager: Optional[AquiferModelManager] = None,
        models_dir: Optional[Path] = None
    ):
        self.aquifer_manager = aquifer_manager or AquiferModelManager(models_dir)
        self.spatial_interpolator = SpatialInterpolator()
        self.ensemble_weights = {'ml': 0.7, 'spatial': 0.3}
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        coords: np.ndarray,
        aquifer_column: str = 'aquifer_code'
    ):
        """
        Train all component models and learn ensemble weights.

        Args:
            X: Feature matrix
            y: Target variable
            coords: Array of (lon, lat) coordinates
            aquifer_column: Column with aquifer IDs
        """
        logger.info("Training ensemble predictor...")

        # 1. Train aquifer-specific models
        self.aquifer_manager.train_all(X, y, aquifer_column)

        # 2. Fit spatial interpolator
        aquifer_ids = X[aquifer_column].values if aquifer_column in X.columns else None
        self.spatial_interpolator.fit(coords, y.values, aquifer_ids)

        # 3. Learn optimal ensemble weights using cross-validation
        self._learn_ensemble_weights(X, y, coords, aquifer_column)

        self._fitted = True
        logger.info("Ensemble predictor trained successfully")

    def _learn_ensemble_weights(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        coords: np.ndarray,
        aquifer_column: str
    ):
        """Learn optimal weights for combining predictions"""
        # Get predictions from each model
        ml_pred = self.aquifer_manager.predict(X, aquifer_column)

        aquifer_ids = X[aquifer_column].values if aquifer_column in X.columns else None
        spatial_pred, _ = self.spatial_interpolator.predict(coords, aquifer_ids)

        # Stack predictions
        stacked = np.column_stack([ml_pred, spatial_pred])

        # Learn weights using Ridge regression
        ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
        ridge.fit(stacked, y)

        # Normalize weights
        weights = ridge.coef_
        weights = weights / weights.sum()

        self.ensemble_weights = {
            'ml': float(weights[0]),
            'spatial': float(weights[1])
        }

        logger.info(f"Learned ensemble weights: {self.ensemble_weights}")

    def predict(
        self,
        X: pd.DataFrame,
        coords: np.ndarray,
        aquifer_column: str = 'aquifer_code'
    ) -> Dict[str, Any]:
        """
        Make predictions with uncertainty estimates.

        Args:
            X: Feature matrix
            coords: Array of (lon, lat) coordinates
            aquifer_column: Column with aquifer IDs

        Returns:
            Dictionary with:
            - prediction: Final ensemble prediction
            - uncertainty: Prediction uncertainty
            - ml_prediction: ML model prediction
            - spatial_prediction: Spatial interpolation prediction
            - components: Detailed breakdown
        """
        if not self._fitted:
            raise ValueError("Ensemble not fitted")

        # ML predictions
        ml_pred = self.aquifer_manager.predict(X, aquifer_column)

        # Spatial predictions
        aquifer_ids = X[aquifer_column].values if aquifer_column in X.columns else None
        spatial_pred, spatial_uncertainty = self.spatial_interpolator.predict(
            coords, aquifer_ids
        )

        # Ensemble combination
        final_pred = (
            self.ensemble_weights['ml'] * ml_pred +
            self.ensemble_weights['spatial'] * spatial_pred
        )

        # Uncertainty: combination of model disagreement and spatial uncertainty
        model_disagreement = np.abs(ml_pred - spatial_pred)
        uncertainty = np.sqrt(
            model_disagreement**2 * 0.5 +
            spatial_uncertainty**2 * 0.5
        )

        # Classify into depth categories
        categories = self._classify_depth(final_pred)

        return {
            'prediction': final_pred,
            'uncertainty': uncertainty,
            'ml_prediction': ml_pred,
            'spatial_prediction': spatial_pred,
            'category': categories,
            'confidence': self._calculate_confidence(uncertainty),
            'ensemble_weights': self.ensemble_weights
        }

    def _classify_depth(self, depths: np.ndarray) -> np.ndarray:
        """Classify water depths into categories"""
        categories = np.empty(len(depths), dtype=object)
        categories[depths <= 3] = 'safe'
        categories[(depths > 3) & (depths <= 8)] = 'moderate'
        categories[(depths > 8) & (depths <= 20)] = 'stress'
        categories[depths > 20] = 'critical'
        return categories

    def _calculate_confidence(self, uncertainty: np.ndarray) -> np.ndarray:
        """Calculate confidence score (0-1) from uncertainty"""
        # Higher uncertainty = lower confidence
        # Normalize: uncertainty of 5m = confidence of ~0.5
        confidence = 1.0 / (1.0 + uncertainty / 5.0)
        return np.clip(confidence, 0, 1)

    def save(self, path: Path):
        """Save ensemble model"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save aquifer models
        self.aquifer_manager.save_all(path / 'aquifer_models')

        # Save spatial interpolator state
        spatial_state = {
            'coords': self.spatial_interpolator._coords.tolist() if self.spatial_interpolator._coords is not None else None,
            'values': self.spatial_interpolator._values.tolist() if self.spatial_interpolator._values is not None else None,
            'aquifers': self.spatial_interpolator._aquifers.tolist() if self.spatial_interpolator._aquifers is not None else None,
            'power': self.spatial_interpolator.power,
            'max_neighbors': self.spatial_interpolator.max_neighbors
        }
        with open(path / 'spatial_interpolator.json', 'w') as f:
            json.dump(spatial_state, f)

        # Save ensemble weights
        with open(path / 'ensemble_weights.json', 'w') as f:
            json.dump(self.ensemble_weights, f, indent=2)

        logger.info(f"Saved ensemble model to {path}")

    def load(self, path: Path):
        """Load ensemble model"""
        path = Path(path)

        # Load aquifer models
        self.aquifer_manager.load_all(path / 'aquifer_models')

        # Load spatial interpolator state
        spatial_path = path / 'spatial_interpolator.json'
        if spatial_path.exists():
            with open(spatial_path, 'r') as f:
                state = json.load(f)

            if state['coords'] is not None:
                self.spatial_interpolator._coords = np.array(state['coords'])
                self.spatial_interpolator._values = np.array(state['values'])
                self.spatial_interpolator._aquifers = np.array(state['aquifers']) if state['aquifers'] else None
                self.spatial_interpolator._tree = cKDTree(self.spatial_interpolator._coords)
                self.spatial_interpolator._fitted = True
                self.spatial_interpolator.power = state['power']
                self.spatial_interpolator.max_neighbors = state['max_neighbors']

        # Load ensemble weights
        weights_path = path / 'ensemble_weights.json'
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                self.ensemble_weights = json.load(f)

        self._fitted = True
        logger.info(f"Loaded ensemble model from {path}")


class GroundwaterPredictor:
    """
    High-level API for groundwater predictions.

    Handles:
    - Model loading/training
    - Feature extraction
    - Prediction with uncertainty
    - Result formatting
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir
        self.ensemble = EnsemblePredictor(models_dir=models_dir)
        self._ready = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        coords: np.ndarray,
        aquifer_column: str = 'aquifer_code'
    ):
        """Train the prediction model"""
        self.ensemble.fit(X, y, coords, aquifer_column)
        self._ready = True

        # Save if models_dir specified
        if self.models_dir:
            self.ensemble.save(self.models_dir)

    def load(self, path: Optional[Path] = None):
        """Load trained model"""
        path = path or self.models_dir
        if path is None:
            raise ValueError("No path specified")

        self.ensemble.load(path)
        self._ready = True

    def predict_village(
        self,
        features: Dict[str, Any],
        lon: float,
        lat: float,
        aquifer_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict groundwater level for a single village.

        Args:
            features: Dictionary of feature values
            lon: Longitude
            lat: Latitude
            aquifer_code: Aquifer code (if known)

        Returns:
            Dictionary with prediction results
        """
        if not self._ready:
            raise ValueError("Model not ready. Call train() or load() first.")

        # Create DataFrame
        X = pd.DataFrame([features])
        if aquifer_code:
            X['aquifer_code'] = aquifer_code

        coords = np.array([[lon, lat]])

        # Get prediction
        result = self.ensemble.predict(X, coords)

        return {
            'water_level_m': float(result['prediction'][0]),
            'uncertainty_m': float(result['uncertainty'][0]),
            'category': result['category'][0],
            'confidence': float(result['confidence'][0]),
            'components': {
                'ml_prediction': float(result['ml_prediction'][0]),
                'spatial_prediction': float(result['spatial_prediction'][0])
            }
        }

    def predict_batch(
        self,
        X: pd.DataFrame,
        coords: np.ndarray,
        aquifer_column: str = 'aquifer_code'
    ) -> pd.DataFrame:
        """
        Predict groundwater levels for multiple locations.

        Args:
            X: Feature matrix
            coords: Array of (lon, lat) coordinates
            aquifer_column: Column with aquifer IDs

        Returns:
            DataFrame with predictions
        """
        if not self._ready:
            raise ValueError("Model not ready. Call train() or load() first.")

        result = self.ensemble.predict(X, coords, aquifer_column)

        return pd.DataFrame({
            'water_level_m': result['prediction'],
            'uncertainty_m': result['uncertainty'],
            'category': result['category'],
            'confidence': result['confidence'],
            'ml_prediction': result['ml_prediction'],
            'spatial_prediction': result['spatial_prediction']
        })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with synthetic data
    np.random.seed(42)
    n_samples = 100

    # Create synthetic features
    X = pd.DataFrame({
        'elevation': np.random.uniform(10, 100, n_samples),
        'slope': np.random.uniform(0, 15, n_samples),
        'rainfall': np.random.uniform(700, 1200, n_samples),
        'aquifer_code': np.random.choice(['BG', 'ST', 'SH'], n_samples)
    })

    # Create synthetic target
    y = pd.Series(
        X['elevation'] * 0.1 +
        X['rainfall'] * -0.01 +
        np.where(X['aquifer_code'] == 'BG', 5, 0) +
        np.random.normal(0, 2, n_samples)
    )

    # Coordinates
    coords = np.column_stack([
        np.random.uniform(80.0, 81.5, n_samples),
        np.random.uniform(15.5, 17.0, n_samples)
    ])

    # Train ensemble
    predictor = GroundwaterPredictor()
    predictor.train(X, y, coords)

    # Test prediction
    test_features = {
        'elevation': 50,
        'slope': 5,
        'rainfall': 900
    }

    result = predictor.predict_village(
        test_features,
        lon=80.5,
        lat=16.0,
        aquifer_code='BG'
    )

    print("\n" + "="*50)
    print("Test Prediction")
    print("="*50)
    print(f"Water Level: {result['water_level_m']:.2f} m")
    print(f"Uncertainty: {result['uncertainty_m']:.2f} m")
    print(f"Category: {result['category']}")
    print(f"Confidence: {result['confidence']:.2%}")
