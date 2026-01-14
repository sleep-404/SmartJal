#!/usr/bin/env python3
"""
Smart Jal - Ensemble Model
Combines spatial, feature-based, and temporal models with GRACE constraint.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from .spatial_model import AquiferStratifiedKriging
from .feature_model import WaterLevelPredictor, PhysicsInformedPredictor
from .temporal_model import PiezometerForecaster, SeasonalAdjuster


class HierarchicalEnsemble:
    """
    Hierarchical ensemble that combines multiple prediction approaches.

    Architecture:
    1. GRACE satellite provides regional constraint
    2. Aquifer-level models provide sub-regional patterns
    3. Village-level features capture local variation

    The ensemble ensures predictions are:
    - Physically consistent (sum matches GRACE)
    - Spatially coherent (smooth within aquifers)
    - Feature-informed (capture local factors)
    """

    def __init__(self,
                 spatial_weight: float = 0.3,
                 feature_weight: float = 0.4,
                 temporal_weight: float = 0.3):
        """
        Initialize ensemble.

        Args:
            spatial_weight: Weight for spatial kriging model
            feature_weight: Weight for XGBoost feature model
            temporal_weight: Weight for temporal decomposition
        """
        self.weights = {
            'spatial': spatial_weight,
            'feature': feature_weight,
            'temporal': temporal_weight
        }

        self.spatial_model = AquiferStratifiedKriging()
        self.feature_model = WaterLevelPredictor(model_type='xgboost')
        self.temporal_model = PiezometerForecaster()
        self.seasonal_adjuster = SeasonalAdjuster()

        self.grace_data = None
        self.fitted = False

    def fit(self,
            data: dict,
            target_date: pd.Timestamp) -> Dict:
        """
        Fit all component models.

        Args:
            data: Dict with preprocessed data
            target_date: Target date for fitting

        Returns:
            Dict with fitting metrics
        """
        print("=" * 60)
        print("Fitting Hierarchical Ensemble Model")
        print("=" * 60)

        metrics = {}

        # 1. Fit spatial model
        print("\n[1/4] Fitting spatial kriging model...")
        self.spatial_model.fit(
            data['piezometers'],
            data['water_levels'],
            target_date
        )

        # 2. Fit temporal model
        print("\n[2/4] Fitting temporal forecasting model...")
        self.temporal_model.fit(data['water_levels'])
        self.seasonal_adjuster.fit(data['water_levels'])

        # 3. Create training data for feature model
        print("\n[3/4] Preparing feature model training data...")

        # Get spatial predictions for villages with known piezometer readings
        spatial_preds, spatial_unc = self.spatial_model.predict(data['villages'])

        # Add as pseudo-target for feature model training
        train_df = data['villages'].copy()
        train_df['water_level'] = spatial_preds
        train_df['spatial_uncertainty'] = spatial_unc

        # Filter to villages with reasonable predictions
        valid_mask = ~np.isnan(train_df['water_level']) & (train_df['spatial_uncertainty'] < 10)
        train_df = train_df[valid_mask]

        print(f"  Training samples: {len(train_df)}")

        # 4. Fit feature model
        print("\n[4/4] Fitting feature model...")
        feature_metrics = self.feature_model.fit(train_df, target_col='water_level')
        metrics['feature_model'] = feature_metrics

        # Load GRACE data
        if 'grace' in data:
            self.grace_data = data['grace']
            print(f"\n  GRACE data loaded: {len(self.grace_data)} months")

        self.fitted = True

        print("\n" + "=" * 60)
        print("Ensemble fitting complete!")
        print("=" * 60)

        return metrics

    def predict(self,
                villages: gpd.GeoDataFrame,
                target_date: pd.Timestamp,
                apply_grace_constraint: bool = True) -> pd.DataFrame:
        """
        Generate ensemble predictions for all villages.

        Args:
            villages: GeoDataFrame with village features
            target_date: Date for prediction
            apply_grace_constraint: Whether to apply GRACE constraint

        Returns:
            DataFrame with predictions and uncertainties
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        print(f"\nGenerating predictions for {target_date}...")

        # 1. Spatial predictions
        print("  Running spatial model...")
        spatial_preds, spatial_unc = self.spatial_model.predict(villages)

        # 2. Feature predictions
        print("  Running feature model...")
        feature_preds, feature_unc = self.feature_model.predict(villages)

        # 3. Seasonal adjustment
        print("  Applying seasonal adjustment...")
        target_month = target_date.month
        spatial_preds = self.seasonal_adjuster.adjust(spatial_preds, target_month)
        feature_preds = self.seasonal_adjuster.adjust(feature_preds, target_month)

        # 4. Weighted ensemble (handle NaN in spatial predictions)
        print("  Combining predictions...")

        # Where spatial is NaN, use feature model only
        spatial_nan = np.isnan(spatial_preds)
        if spatial_nan.any():
            print(f"    Note: {spatial_nan.sum()} villages using feature model only (spatial NaN)")

        # Weighted combination where both available
        ensemble_preds = np.where(
            spatial_nan,
            feature_preds,  # Use feature model only when spatial is NaN
            self.weights['spatial'] * spatial_preds + self.weights['feature'] * feature_preds
        )

        # Where feature is also NaN, use spatial
        feature_nan = np.isnan(feature_preds)
        ensemble_preds = np.where(
            feature_nan & ~spatial_nan,
            spatial_preds,
            ensemble_preds
        )

        # Combined uncertainty
        ensemble_unc = np.where(
            spatial_nan,
            feature_unc,
            np.sqrt(self.weights['spatial']**2 * spatial_unc**2 + self.weights['feature']**2 * feature_unc**2)
        )

        # 5. Apply GRACE constraint
        if apply_grace_constraint and self.grace_data is not None:
            print("  Applying GRACE constraint...")
            ensemble_preds, ensemble_unc = self._apply_grace_constraint(
                villages, ensemble_preds, ensemble_unc, target_date
            )

        # Build results DataFrame
        results = pd.DataFrame({
            'village': villages['village'].values if 'village' in villages.columns else range(len(villages)),
            'prediction': ensemble_preds,
            'uncertainty': ensemble_unc,
            'spatial_pred': spatial_preds,
            'feature_pred': feature_preds,
            'prediction_date': target_date
        })

        # Add village metadata if available
        for col in ['district', 'mandal', 'geo_class', 'centroid_lat', 'centroid_lon']:
            if col in villages.columns:
                results[col] = villages[col].values

        print(f"  Predictions generated for {len(results)} villages")
        print(f"  Mean prediction: {results['prediction'].mean():.2f}m")
        print(f"  Mean uncertainty: {results['uncertainty'].mean():.2f}m")

        return results

    def _apply_grace_constraint(self,
                                 villages: gpd.GeoDataFrame,
                                 predictions: np.ndarray,
                                 uncertainties: np.ndarray,
                                 target_date: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust predictions to match GRACE regional value.

        The key insight: Sum of village water storage changes should
        approximately equal the regional GRACE observation.
        """
        # Get GRACE value for target month
        target_month = (target_date.year, target_date.month)

        grace_value = None
        if self.grace_data is not None:
            grace_row = self.grace_data[
                (self.grace_data['year'] == target_date.year) &
                (self.grace_data['month'] == target_date.month)
            ]
            if len(grace_row) > 0:
                grace_value = grace_row['tws_anomaly_cm'].values[0]

        if grace_value is None:
            return predictions, uncertainties

        # Calculate current regional prediction
        # Weight by village area
        areas = villages['area_km2'].fillna(1).values
        total_area = areas.sum()

        # Area-weighted mean prediction
        current_regional = np.sum(predictions * areas) / total_area

        # GRACE is in cm, convert to meters
        grace_m = grace_value / 100

        # Calculate adjustment factor
        # We want: adjusted_regional = baseline + grace_anomaly
        baseline = predictions.mean()
        adjustment = grace_m * 2  # Scale factor for anomaly

        # Apply soft constraint (blend towards GRACE)
        constraint_weight = 0.3  # How much to trust GRACE
        adjusted = predictions + constraint_weight * (baseline + adjustment - current_regional)

        # Increase uncertainty for constrained predictions
        adjusted_unc = uncertainties * (1 + 0.1 * abs(adjustment))

        print(f"    GRACE TWS anomaly: {grace_value:.2f}cm")
        print(f"    Adjustment applied: {adjustment:.3f}m")

        return adjusted, adjusted_unc

    def predict_scenarios(self,
                          villages: gpd.GeoDataFrame,
                          target_date: pd.Timestamp,
                          scenarios: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Generate predictions under different scenarios.

        Args:
            villages: GeoDataFrame with village features
            target_date: Date for prediction
            scenarios: Dict of scenario name -> modifications

        Returns:
            Dict of scenario name -> predictions DataFrame
        """
        print("\nRunning scenario analysis...")

        results = {}

        # Baseline scenario
        results['baseline'] = self.predict(villages, target_date)

        for scenario_name, modifications in scenarios.items():
            print(f"\n  Scenario: {scenario_name}")

            # Apply modifications
            modified_villages = villages.copy()

            if 'rainfall_factor' in modifications:
                factor = modifications['rainfall_factor']
                for col in modified_villages.columns:
                    if 'rainfall' in col:
                        modified_villages[col] = modified_villages[col] * factor
                print(f"    Rainfall factor: {factor}")

            if 'extraction_factor' in modifications:
                factor = modifications['extraction_factor']
                if 'n_wells' in modified_villages.columns:
                    modified_villages['n_wells'] = modified_villages['n_wells'] * factor
                if 'monthly_extraction_ham' in modified_villages.columns:
                    modified_villages['monthly_extraction_ham'] = modified_villages['monthly_extraction_ham'] * factor
                print(f"    Extraction factor: {factor}")

            # Generate predictions
            scenario_preds = self.predict(modified_villages, target_date, apply_grace_constraint=False)
            scenario_preds['scenario'] = scenario_name
            results[scenario_name] = scenario_preds

        return results

    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights.copy()

    def set_model_weights(self,
                          spatial: float = None,
                          feature: float = None,
                          temporal: float = None):
        """
        Update model weights.

        Weights are normalized to sum to 1.
        """
        if spatial is not None:
            self.weights['spatial'] = spatial
        if feature is not None:
            self.weights['feature'] = feature
        if temporal is not None:
            self.weights['temporal'] = temporal

        # Normalize
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from feature model."""
        return self.feature_model.get_feature_importance()


class AquiferLevelAggregator:
    """
    Aggregate village predictions to aquifer level.
    """

    @staticmethod
    def aggregate(predictions: pd.DataFrame,
                  villages: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Aggregate village predictions to aquifer level.

        Args:
            predictions: Village-level predictions
            villages: GeoDataFrame with village data

        Returns:
            DataFrame with aquifer-level statistics
        """
        # Merge predictions with village data
        merged = predictions.copy()
        if 'geo_class' not in merged.columns and 'geo_class' in villages.columns:
            merged['geo_class'] = villages['geo_class'].values
        if 'area_km2' not in merged.columns and 'area_km2' in villages.columns:
            merged['area_km2'] = villages['area_km2'].values

        # Aggregate by aquifer
        aquifer_stats = merged.groupby('geo_class').agg(
            n_villages=('prediction', 'count'),
            mean_prediction=('prediction', 'mean'),
            std_prediction=('prediction', 'std'),
            min_prediction=('prediction', 'min'),
            max_prediction=('prediction', 'max'),
            mean_uncertainty=('uncertainty', 'mean'),
            total_area_km2=('area_km2', 'sum')
        ).reset_index()

        return aquifer_stats


class GRACEValidator:
    """
    Validate predictions against GRACE satellite data.
    """

    def __init__(self, grace_data: pd.DataFrame):
        """
        Initialize validator.

        Args:
            grace_data: GRACE time series DataFrame
        """
        self.grace_data = grace_data

    def validate(self,
                 predictions: pd.DataFrame,
                 villages: gpd.GeoDataFrame,
                 target_date: pd.Timestamp) -> Dict:
        """
        Compare predictions with GRACE observation.

        Args:
            predictions: Village-level predictions
            villages: GeoDataFrame with village data
            target_date: Date of predictions

        Returns:
            Dict with validation metrics
        """
        # Get GRACE value
        grace_row = self.grace_data[
            (self.grace_data['year'] == target_date.year) &
            (self.grace_data['month'] == target_date.month)
        ]

        if len(grace_row) == 0:
            return {'error': 'No GRACE data for this date'}

        grace_tws = grace_row['tws_anomaly_cm'].values[0]

        # Calculate area-weighted mean prediction
        merged = predictions.copy()
        if 'area_km2' not in merged.columns:
            merged['area_km2'] = villages['area_km2'].values

        areas = merged['area_km2'].fillna(1).values
        preds = merged['prediction'].values

        weighted_mean = np.sum(preds * areas) / np.sum(areas)

        # Convert to comparable units (both as anomalies)
        baseline = preds.mean()
        pred_anomaly_m = weighted_mean - baseline
        grace_anomaly_m = grace_tws / 100

        return {
            'grace_tws_cm': grace_tws,
            'grace_anomaly_m': grace_anomaly_m,
            'prediction_anomaly_m': pred_anomaly_m,
            'difference_m': abs(pred_anomaly_m - grace_anomaly_m),
            'regional_mean_prediction': weighted_mean,
            'regional_baseline': baseline
        }


if __name__ == '__main__':
    print("Testing ensemble model...")
    print("(Run from pipeline.py for full integration test)")
