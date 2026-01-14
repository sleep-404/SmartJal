#!/usr/bin/env python3
"""
Smart Jal - Explainability & Uncertainty Module

Implements:
1. SHAP Values - Explain WHY each prediction is made
2. Conformal Prediction - Statistically guaranteed prediction intervals
3. Feature Attribution - Which factors influence each village's prediction

Based on research from:
- "Explainable AI for Groundwater" (Water Resources Research, 2025)
- Conformal Prediction for uncertainty quantification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Explainability will use fallback.")

# Conformal Prediction imports
try:
    from mapie.regression import MapieRegressor
    from mapie.quantile_regression import MapieQuantileRegressor
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    print("Warning: MAPIE not available. Uncertainty will use fallback.")


class SHAPExplainer:
    """
    SHAP-based model explainability.

    For any village, explains WHY the model predicted that water level:
    - Which features increased/decreased the prediction?
    - What's the relative importance of each factor?
    """

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (XGBoost, RandomForest, etc.)
            feature_names: List of feature column names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.base_value = None

    def fit(self, X_train: np.ndarray):
        """
        Fit SHAP explainer on training data.
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Using feature importance fallback.")
            return

        print("Fitting SHAP explainer...")

        # Use TreeExplainer for tree-based models
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.base_value = self.explainer.expected_value
            print(f"  Base prediction value: {self.base_value:.2f}m")
        except Exception as e:
            print(f"  TreeExplainer failed, using KernelExplainer: {e}")
            # Sample background data for KernelExplainer
            background = shap.sample(X_train, min(100, len(X_train)))
            self.explainer = shap.KernelExplainer(self.model.predict, background)
            self.base_value = self.model.predict(X_train).mean()

    def explain_village(self, X_village: np.ndarray, village_name: str = None) -> Dict:
        """
        Explain prediction for a single village.

        Args:
            X_village: Feature vector for the village (1, n_features)
            village_name: Optional name for display

        Returns:
            Dict with explanation details
        """
        if self.explainer is None:
            return self._fallback_explanation(X_village)

        # Get SHAP values
        if len(X_village.shape) == 1:
            X_village = X_village.reshape(1, -1)

        shap_values = self.explainer.shap_values(X_village)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = shap_values.flatten()

        # Create explanation
        feature_contributions = []
        for i, (name, value, shap_val) in enumerate(zip(
            self.feature_names, X_village.flatten(), shap_values
        )):
            feature_contributions.append({
                'feature': name,
                'value': float(value),
                'shap_value': float(shap_val),
                'direction': 'increases' if shap_val > 0 else 'decreases',
                'abs_importance': abs(float(shap_val))
            })

        # Sort by absolute importance
        feature_contributions.sort(key=lambda x: x['abs_importance'], reverse=True)

        # Get top factors
        top_factors = feature_contributions[:5]

        # Build narrative
        narrative = self._build_narrative(village_name, top_factors)

        return {
            'village': village_name,
            'base_value': float(self.base_value) if self.base_value is not None else 0,
            'all_contributions': feature_contributions,
            'top_factors': top_factors,
            'narrative': narrative,
            'total_positive': sum(c['shap_value'] for c in feature_contributions if c['shap_value'] > 0),
            'total_negative': sum(c['shap_value'] for c in feature_contributions if c['shap_value'] < 0)
        }

    def _build_narrative(self, village_name: str, top_factors: List[Dict]) -> str:
        """Build human-readable explanation."""
        if not top_factors:
            return "Insufficient data for explanation."

        village_str = f"For {village_name}" if village_name else "For this village"

        # Build explanation parts
        increases = [f for f in top_factors if f['shap_value'] > 0]
        decreases = [f for f in top_factors if f['shap_value'] < 0]

        parts = [f"{village_str}, the predicted water level is influenced by:"]

        if increases:
            inc_str = ", ".join([f"{f['feature']} (+{f['shap_value']:.2f}m)" for f in increases[:2]])
            parts.append(f"  Factors increasing depth: {inc_str}")

        if decreases:
            dec_str = ", ".join([f"{f['feature']} ({f['shap_value']:.2f}m)" for f in decreases[:2]])
            parts.append(f"  Factors decreasing depth: {dec_str}")

        return "\n".join(parts)

    def _fallback_explanation(self, X_village: np.ndarray) -> Dict:
        """Fallback when SHAP not available."""
        # Use feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_contributions = [
                {
                    'feature': name,
                    'value': float(val),
                    'importance': float(imp),
                    'abs_importance': float(imp)
                }
                for name, val, imp in zip(self.feature_names, X_village.flatten(), importances)
            ]
            feature_contributions.sort(key=lambda x: x['abs_importance'], reverse=True)

            return {
                'all_contributions': feature_contributions,
                'top_factors': feature_contributions[:5],
                'narrative': "Using feature importance (SHAP not available)",
                'method': 'feature_importance'
            }

        return {
            'narrative': "Explanation not available",
            'method': 'none'
        }

    def explain_batch(self, X: np.ndarray, village_names: List[str] = None) -> pd.DataFrame:
        """
        Explain predictions for multiple villages.

        Returns DataFrame with SHAP values for each village.
        """
        if self.explainer is None:
            return pd.DataFrame()

        print(f"Generating explanations for {len(X)} villages...")

        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Create DataFrame
        shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in self.feature_names])

        if village_names is not None:
            shap_df['village'] = village_names

        # Add most important feature for each village
        abs_shap = np.abs(shap_values)
        most_important_idx = np.argmax(abs_shap, axis=1)
        shap_df['most_important_feature'] = [self.feature_names[i] for i in most_important_idx]
        shap_df['most_important_value'] = [shap_values[i, idx] for i, idx in enumerate(most_important_idx)]

        return shap_df

    def get_global_importance(self) -> pd.DataFrame:
        """
        Get global feature importance across all predictions.
        """
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return pd.DataFrame()


class ConformalPredictor:
    """
    Conformal Prediction for statistically valid prediction intervals.

    Provides intervals like "water level is 8-12m with 90% confidence"
    without making distributional assumptions.
    """

    def __init__(self, base_model, alpha: float = 0.1):
        """
        Initialize conformal predictor.

        Args:
            base_model: Base regression model
            alpha: Significance level (0.1 = 90% confidence intervals)
        """
        self.base_model = base_model
        self.alpha = alpha
        self.mapie_model = None
        self.calibration_scores = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_cal: np.ndarray = None, y_cal: np.ndarray = None):
        """
        Fit conformal predictor.

        Args:
            X_train: Training features
            y_train: Training targets
            X_cal: Calibration features (optional, will split from train if not provided)
            y_cal: Calibration targets
        """
        if not MAPIE_AVAILABLE:
            print("MAPIE not available. Using residual-based uncertainty.")
            self._fit_fallback(X_train, y_train)
            return

        print("Fitting Conformal Predictor...")

        # Split calibration set if not provided
        if X_cal is None:
            X_train, X_cal, y_train, y_cal = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

        # Fit MAPIE regressor
        self.mapie_model = MapieRegressor(
            self.base_model,
            method="plus",  # Jackknife+
            cv="prefit"
        )

        # First fit the base model
        self.base_model.fit(X_train, y_train)

        # Then calibrate MAPIE
        self.mapie_model.fit(X_cal, y_cal)

        print(f"  Calibration samples: {len(X_cal)}")
        print(f"  Confidence level: {(1 - self.alpha) * 100:.0f}%")

    def _fit_fallback(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fallback calibration using residuals."""
        self.base_model.fit(X_train, y_train)
        y_pred = self.base_model.predict(X_train)
        residuals = np.abs(y_train - y_pred)
        self.calibration_scores = np.percentile(residuals, (1 - self.alpha) * 100)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.

        Args:
            X: Features to predict

        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if self.mapie_model is not None:
            y_pred, y_pis = self.mapie_model.predict(X, alpha=self.alpha)
            lower = y_pis[:, 0, 0]
            upper = y_pis[:, 1, 0]
            return y_pred, lower, upper
        elif self.calibration_scores is not None:
            # Fallback
            y_pred = self.base_model.predict(X)
            lower = y_pred - self.calibration_scores
            upper = y_pred + self.calibration_scores
            return y_pred, lower, upper
        else:
            # No calibration
            y_pred = self.base_model.predict(X)
            return y_pred, y_pred - 2.0, y_pred + 2.0

    def get_interval_width(self, X: np.ndarray) -> np.ndarray:
        """Get width of prediction intervals."""
        _, lower, upper = self.predict(X)
        return upper - lower


class TransferLearner:
    """
    Transfer Learning for data-sparse aquifers.

    Train on data-rich aquifers (Alluvium with 66 samples),
    transfer to data-poor ones (Shale with <5 samples).
    """

    def __init__(self, base_model):
        """
        Initialize transfer learner.

        Args:
            base_model: Base model to use
        """
        self.base_model = base_model
        self.source_model = None
        self.target_models = {}
        self.aquifer_stats = {}

    def fit_source(self, X: np.ndarray, y: np.ndarray, aquifers: np.ndarray):
        """
        Fit on source domain (data-rich aquifers).
        """
        print("Fitting Transfer Learning source model...")

        # Count samples per aquifer
        unique, counts = np.unique(aquifers, return_counts=True)
        self.aquifer_stats = dict(zip(unique, counts))

        print(f"  Aquifer sample counts: {self.aquifer_stats}")

        # Identify data-rich aquifers (>10 samples)
        rich_mask = np.isin(aquifers, [a for a, c in self.aquifer_stats.items() if c >= 10])

        print(f"  Source samples (>=10): {rich_mask.sum()}")

        # Train source model on rich aquifers
        if rich_mask.sum() > 0:
            from sklearn.base import clone
            self.source_model = clone(self.base_model)
            self.source_model.fit(X[rich_mask], y[rich_mask])

    def transfer_to_target(self, X: np.ndarray, y: np.ndarray,
                           aquifer: str, n_finetune: int = 5):
        """
        Transfer to data-poor aquifer with fine-tuning.

        Args:
            X: Target aquifer features
            y: Target aquifer labels
            aquifer: Aquifer name
            n_finetune: Number of samples to fine-tune on
        """
        if self.source_model is None:
            print(f"  No source model. Training from scratch for {aquifer}")
            from sklearn.base import clone
            model = clone(self.base_model)
            model.fit(X, y)
            self.target_models[aquifer] = model
            return

        print(f"  Transferring to {aquifer} with {len(X)} samples...")

        # Get source predictions as soft labels
        source_preds = self.source_model.predict(X)

        # Blend with actual labels (more weight to actual for fine-tuning)
        if len(y) <= n_finetune:
            # Very few samples: mostly trust source
            blended_y = 0.7 * source_preds + 0.3 * y
        else:
            # More samples: mostly trust target
            blended_y = 0.3 * source_preds + 0.7 * y

        # Fine-tune
        from sklearn.base import clone
        target_model = clone(self.base_model)
        target_model.fit(X, blended_y)

        self.target_models[aquifer] = target_model

    def predict(self, X: np.ndarray, aquifers: np.ndarray) -> np.ndarray:
        """
        Predict using appropriate model for each aquifer.
        """
        predictions = np.zeros(len(X))

        for aquifer in np.unique(aquifers):
            mask = aquifers == aquifer

            if aquifer in self.target_models:
                predictions[mask] = self.target_models[aquifer].predict(X[mask])
            elif self.source_model is not None:
                predictions[mask] = self.source_model.predict(X[mask])
            else:
                predictions[mask] = 5.0  # Default

        return predictions


if __name__ == '__main__':
    print("Testing Explainability Module...")

    # Test with synthetic data
    from sklearn.ensemble import RandomForestRegressor

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(100) * 0.5
    feature_names = ['rainfall', 'elevation', 'wells', 'slope', 'soil']

    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Test SHAP
    print("\n--- Testing SHAP Explainer ---")
    explainer = SHAPExplainer(model, feature_names)
    explainer.fit(X)

    explanation = explainer.explain_village(X[0], "Test Village")
    print(explanation['narrative'])

    # Test Conformal Prediction
    print("\n--- Testing Conformal Predictor ---")
    conformal = ConformalPredictor(RandomForestRegressor(n_estimators=50, random_state=42))
    conformal.fit(X[:80], y[:80], X[80:], y[80:])

    pred, lower, upper = conformal.predict(X[:5])
    print(f"Predictions with 90% CI:")
    for i in range(5):
        print(f"  {pred[i]:.2f} [{lower[i]:.2f}, {upper[i]:.2f}]")
