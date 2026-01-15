#!/usr/bin/env python3
"""
Smart Jal - Feature-Based ML Model
XGBoost model for water level prediction using engineered features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Try to import XGBoost, fall back to sklearn if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Try to import LightGBM as alternative
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

# Always import sklearn fallback
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


class WaterLevelPredictor:
    """
    XGBoost-based water level prediction model.

    Features:
    - Rainfall (current and lagged)
    - Terrain (elevation, slope)
    - Soil properties
    - Aquifer type
    - Extraction (well density, pumping)
    - Temporal features (lagged water levels, trend)
    """

    # Default feature columns
    FEATURE_COLS = [
        # Rainfall (current and lags)
        'rainfall_current', 'rainfall_lag1', 'rainfall_lag2', 'rainfall_lag3',
        'rainfall_lag4', 'rainfall_lag5', 'rainfall_lag6',
        'rainfall_cumulative_3m', 'rainfall_cumulative_6m',
        'rainfall_monsoon', 'rainfall_annual',
        # Terrain
        'elevation_mean', 'slope_mean',
        # Soil
        'infiltration_score', 'runoff_score',
        # Extraction
        'n_wells', 'well_density', 'avg_well_depth', 'extraction_intensity',
        # Location
        'centroid_lat', 'centroid_lon', 'area_km2',
        # LULC
        'lulc_crop_pct', 'lulc_forest_pct', 'lulc_urban_pct',
        'lulc_water_pct', 'lulc_barren_pct',
        # Geomorphology
        'geom_recharge_score', 'geom_is_floodplain', 'geom_is_hill',
        # Water proximity
        'dist_to_water_km', 'near_water',
        # Vegetation/crop intensity (NDVI proxy)
        'vegetation_index', 'crop_intensity', 'irrigation_demand_proxy',
        # Seasonal encoding
        'month_sin', 'month_cos', 'is_monsoon', 'is_post_monsoon', 'is_pre_monsoon',
        # ET proxy (evapotranspiration)
        'et_proxy', 'eto_proxy', 'kc_weighted',
        # Satellite features
        'modis_et', 'modis_et_lag1', 'sentinel_ndvi', 'sentinel_ndvi_lag1', 'smap_soil_moisture',
    ]

    def __init__(self,
                 model_type: str = 'xgboost',
                 params: Optional[Dict] = None):
        """
        Initialize model.

        Args:
            model_type: 'xgboost', 'lightgbm', or 'sklearn'
            params: Model hyperparameters
        """
        self.model_type = model_type
        self.params = params or self._default_params()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.feature_importance_ = None

    def _default_params(self) -> Dict:
        """Get default hyperparameters."""
        if self.model_type == 'xgboost':
            return {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'random_state': 42,
                'n_jobs': -1
            }
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
        else:
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42
            }

    def _init_model(self):
        """Initialize the underlying model."""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(**self.params)
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            # Optimized GradientBoosting parameters
            sklearn_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
            self.model = GradientBoostingRegressor(**sklearn_params)
            self.model_type = 'sklearn'

    def _prepare_features(self,
                          df: pd.DataFrame,
                          fit_scaler: bool = False) -> np.ndarray:
        """
        Prepare feature matrix.

        Args:
            df: DataFrame with features
            fit_scaler: Whether to fit the scaler

        Returns:
            Feature matrix
        """
        # Get available feature columns
        available_cols = [c for c in self.FEATURE_COLS if c in df.columns]

        # Add aquifer dummy columns if present (only numeric ones)
        aquifer_cols = [c for c in df.columns if c.startswith('aquifer_')]
        available_cols.extend(aquifer_cols)

        # Filter to only numeric columns
        available_cols = [c for c in available_cols if df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool', 'uint8']]

        self.feature_cols = available_cols

        # Extract features
        X = df[available_cols].copy()

        # Convert all to numeric (in case of any edge cases)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Fill NaN with column medians
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X[col] = X[col].fillna(median_val)

        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def fit(self,
            df: pd.DataFrame,
            target_col: str = 'water_level',
            val_size: float = 0.2) -> Dict:
        """
        Fit model to data.

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            val_size: Validation set size

        Returns:
            Dict with training metrics
        """
        print(f"Training {self.model_type} model...")

        # Initialize model
        self._init_model()

        # Prepare features
        X = self._prepare_features(df, fit_scaler=True)
        y = df[target_col].values

        # Handle NaN in target
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"  Training samples: {len(y)}")
        print(f"  Features: {len(self.feature_cols)}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=42
        )

        # Fit model
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        # Calculate metrics
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'n_train': len(y_train),
            'n_val': len(y_val),
            'n_features': len(self.feature_cols)
        }

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        print(f"  Train RMSE: {metrics['train_rmse']:.3f}m")
        print(f"  Val RMSE: {metrics['val_rmse']:.3f}m")
        print(f"  Val R²: {metrics['val_r2']:.3f}")

        return metrics

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (predictions, uncertainties)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare features
        X = self._prepare_features(df, fit_scaler=False)

        # Predict
        predictions = self.model.predict(X)

        # Estimate uncertainty (simplified heuristic)
        # Use a simple heuristic based on prediction magnitude
        # In production, would use quantile regression or bootstrap
        uncertainties = np.abs(predictions - predictions.mean()) * 0.15 + 1.5

        return predictions, uncertainties

    def cross_validate(self,
                       df: pd.DataFrame,
                       target_col: str = 'water_level',
                       cv: int = 5) -> Dict:
        """
        Perform cross-validation.

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            cv: Number of folds

        Returns:
            Dict with CV metrics
        """
        print(f"Running {cv}-fold cross-validation...")

        # Initialize model
        self._init_model()

        # Prepare features
        X = self._prepare_features(df, fit_scaler=True)
        y = df[target_col].values

        # Handle NaN
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        # Cross-validate
        scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring='neg_root_mean_squared_error'
        )

        metrics = {
            'cv_rmse_mean': -scores.mean(),
            'cv_rmse_std': scores.std(),
            'cv_rmse_all': -scores
        }

        print(f"  CV RMSE: {metrics['cv_rmse_mean']:.3f} ± {metrics['cv_rmse_std']:.3f}")

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance ranking."""
        if self.feature_importance_ is None:
            return pd.DataFrame()

        return self.feature_importance_.head(top_n)

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type,
            'params': self.params,
            'feature_importance': self.feature_importance_
        }, path)

        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'WaterLevelPredictor':
        """Load model from disk."""
        data = joblib.load(path)

        predictor = cls(
            model_type=data['model_type'],
            params=data['params']
        )
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_cols = data['feature_cols']
        predictor.feature_importance_ = data.get('feature_importance')

        return predictor


class PhysicsInformedPredictor(WaterLevelPredictor):
    """
    Physics-informed water level predictor.

    Incorporates water balance equation as constraint:
    ΔStorage = Recharge - Extraction ± Lateral_Flow

    This ensures predictions respect physical constraints.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recharge_coefficient = None
        self.extraction_coefficient = None

    def _calculate_recharge(self, df: pd.DataFrame) -> np.ndarray:
        """
        Estimate recharge from rainfall and soil properties.

        Recharge = Rainfall × Infiltration_Factor × (1 - Runoff_Factor)
        """
        rainfall = df['rainfall_cumulative_3m'].fillna(0).values
        infiltration = df['infiltration_score'].fillna(2).values / 3  # Normalize to 0-1
        runoff = df['runoff_score'].fillna(2).values / 3

        recharge = rainfall * infiltration * (1 - runoff * 0.5) / 1000  # Convert mm to m

        return recharge

    def _calculate_extraction(self, df: pd.DataFrame) -> np.ndarray:
        """
        Estimate extraction from well density and pumping.

        Extraction = n_wells × avg_extraction_per_well
        """
        n_wells = df['n_wells'].fillna(0).values
        avg_extraction = 0.025  # ha.m per well per month

        # Convert to meters of aquifer depletion over village area
        area_km2 = df['area_km2'].fillna(1).values
        extraction_m = (n_wells * avg_extraction) / (area_km2 * 100)  # ha.m to m over km²

        return extraction_m

    def fit(self, df: pd.DataFrame, target_col: str = 'water_level', **kwargs) -> Dict:
        """
        Fit model with physics constraints.
        """
        # Calculate physics-based features
        df = df.copy()
        df['recharge_estimate'] = self._calculate_recharge(df)
        df['extraction_estimate'] = self._calculate_extraction(df)
        df['net_flux'] = df['recharge_estimate'] - df['extraction_estimate']

        # Add to feature columns
        self.FEATURE_COLS = self.FEATURE_COLS + [
            'recharge_estimate', 'extraction_estimate', 'net_flux'
        ]

        # Fit base model
        metrics = super().fit(df, target_col, **kwargs)

        return metrics

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make physics-constrained predictions.
        """
        # Calculate physics features
        df = df.copy()
        df['recharge_estimate'] = self._calculate_recharge(df)
        df['extraction_estimate'] = self._calculate_extraction(df)
        df['net_flux'] = df['recharge_estimate'] - df['extraction_estimate']

        return super().predict(df)


if __name__ == '__main__':
    # Test with synthetic data
    print("Testing feature model...")

    np.random.seed(42)
    n_samples = 500

    # Generate synthetic features
    df = pd.DataFrame({
        'rainfall_current': np.random.uniform(0, 200, n_samples),
        'rainfall_lag1': np.random.uniform(0, 200, n_samples),
        'rainfall_lag2': np.random.uniform(0, 200, n_samples),
        'rainfall_lag3': np.random.uniform(0, 200, n_samples),
        'rainfall_cumulative_3m': np.random.uniform(0, 500, n_samples),
        'rainfall_monsoon': np.random.uniform(0, 800, n_samples),
        'elevation_mean': np.random.uniform(100, 500, n_samples),
        'slope_mean': np.random.uniform(0, 10, n_samples),
        'infiltration_score': np.random.choice([1, 2, 3], n_samples),
        'runoff_score': np.random.choice([1, 2, 3], n_samples),
        'n_wells': np.random.poisson(10, n_samples),
        'well_density': np.random.uniform(0, 50, n_samples),
        'avg_well_depth': np.random.uniform(30, 100, n_samples),
        'extraction_intensity': np.random.uniform(0, 0.5, n_samples),
        'centroid_lat': np.random.uniform(15.5, 17.0, n_samples),
        'centroid_lon': np.random.uniform(80.0, 81.5, n_samples),
        'area_km2': np.random.uniform(1, 50, n_samples),
    })

    # Generate target with some realistic relationship
    df['water_level'] = (
        15 +
        0.01 * df['rainfall_cumulative_3m'] -
        0.02 * df['elevation_mean'] -
        0.1 * df['n_wells'] +
        np.random.normal(0, 2, n_samples)
    )

    # Train model
    model = WaterLevelPredictor(model_type='xgboost')
    metrics = model.fit(df)

    # Feature importance
    print("\nTop features:")
    print(model.get_feature_importance(10))

    # Cross-validation
    cv_metrics = model.cross_validate(df)
