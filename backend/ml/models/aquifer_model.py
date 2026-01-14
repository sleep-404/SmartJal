"""
SmartJal Aquifer-Specific Models

Trains separate LightGBM models for each aquifer type to capture
aquifer-specific groundwater behavior.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

# Try to import sklearn alternatives
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


class AquiferModel:
    """
    Aquifer-specific groundwater level prediction model.

    Uses LightGBM for each aquifer type, with fallback to sklearn
    GradientBoosting if LightGBM is not available.
    """

    DEFAULT_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 500,
        'early_stopping_rounds': 50
    }

    def __init__(
        self,
        aquifer_id: str,
        params: Optional[Dict] = None
    ):
        self.aquifer_id = aquifer_id
        self.params = params or self.DEFAULT_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.metrics = {}

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the model on aquifer-specific data.

        Args:
            X: Feature matrix
            y: Target variable (water levels)
            validation_split: Fraction for validation

        Returns:
            Dictionary of training metrics
        """
        self.feature_names = list(X.columns)

        # Split data
        n_val = int(len(X) * validation_split)
        if n_val < 2:
            # Too few samples for validation, use all for training
            X_train, y_train = X, y
            X_val, y_val = X, y
        else:
            indices = np.random.permutation(len(X))
            train_idx, val_idx = indices[n_val:], indices[:n_val]
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        if LIGHTGBM_AVAILABLE:
            self._train_lightgbm(X_train, y_train, X_val, y_val)
        else:
            self._train_sklearn(X_train, y_train)

        # Calculate metrics
        train_pred = self.predict(X_train)
        val_pred = self.predict(X_val)

        self.metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_r2': r2_score(y_val, val_pred),
            'n_samples': len(X)
        }

        logger.info(
            f"Aquifer {self.aquifer_id}: "
            f"Train RMSE={self.metrics['train_rmse']:.2f}, "
            f"Val RMSE={self.metrics['val_rmse']:.2f}"
        )

        return self.metrics

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ):
        """Train using LightGBM"""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Extract params for lgb.train
        train_params = {k: v for k, v in self.params.items()
                       if k not in ['n_estimators', 'early_stopping_rounds']}

        self.model = lgb.train(
            train_params,
            train_data,
            num_boost_round=self.params.get('n_estimators', 500),
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(self.params.get('early_stopping_rounds', 50)),
                lgb.log_evaluation(period=0)
            ]
        )

        # Store feature importance
        self.feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importance(importance_type='gain')
        ))

    def _train_sklearn(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train using sklearn GradientBoosting as fallback"""
        self.model = GradientBoostingRegressor(
            n_estimators=self.params.get('n_estimators', 100),
            learning_rate=self.params.get('learning_rate', 0.05),
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # Store feature importance
        self.feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Ensure correct column order
        X = X[self.feature_names]

        if LIGHTGBM_AVAILABLE and isinstance(self.model, lgb.Booster):
            return self.model.predict(X)
        else:
            return self.model.predict(X)

    def save(self, path: Path):
        """Save model to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path / f"{self.aquifer_id}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save metadata
        meta = {
            'aquifer_id': self.aquifer_id,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics,
            'params': self.params
        }
        meta_path = path / f"{self.aquifer_id}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: Path, aquifer_id: str) -> 'AquiferModel':
        """Load model from disk"""
        path = Path(path)

        # Load metadata
        meta_path = path / f"{aquifer_id}_meta.json"
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Create instance
        instance = cls(aquifer_id=aquifer_id, params=meta.get('params'))
        instance.feature_names = meta['feature_names']
        instance.feature_importance = meta.get('feature_importance')
        instance.metrics = meta.get('metrics', {})

        # Load model
        model_path = path / f"{aquifer_id}_model.pkl"
        with open(model_path, 'rb') as f:
            instance.model = pickle.load(f)

        return instance


class AquiferModelManager:
    """
    Manages multiple aquifer-specific models.

    Provides a unified interface for training, prediction, and
    model management across all aquifer types.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir
        self.models: Dict[str, AquiferModel] = {}
        self.global_model: Optional[AquiferModel] = None

    def train_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        aquifer_column: str = 'aquifer_code',
        min_samples: int = 5
    ) -> Dict[str, Dict]:
        """
        Train models for all aquifer types.

        Args:
            X: Feature matrix (must include aquifer_column)
            y: Target variable
            aquifer_column: Column name containing aquifer IDs
            min_samples: Minimum samples required to train aquifer-specific model

        Returns:
            Dictionary of metrics per aquifer
        """
        all_metrics = {}

        # Train global model first (fallback)
        logger.info("Training global model...")
        feature_cols = [c for c in X.columns if c != aquifer_column]
        self.global_model = AquiferModel('global')
        self.global_model.train(X[feature_cols], y)
        all_metrics['global'] = self.global_model.metrics

        # Train aquifer-specific models
        if aquifer_column in X.columns:
            for aquifer_id in X[aquifer_column].unique():
                if pd.isna(aquifer_id):
                    continue

                mask = X[aquifer_column] == aquifer_id
                n_samples = mask.sum()

                if n_samples >= min_samples:
                    logger.info(f"Training model for aquifer {aquifer_id} ({n_samples} samples)...")
                    model = AquiferModel(str(aquifer_id))
                    model.train(X.loc[mask, feature_cols], y[mask])
                    self.models[str(aquifer_id)] = model
                    all_metrics[str(aquifer_id)] = model.metrics
                else:
                    logger.info(
                        f"Skipping aquifer {aquifer_id} ({n_samples} samples < {min_samples}). "
                        "Will use global model."
                    )

        return all_metrics

    def predict(
        self,
        X: pd.DataFrame,
        aquifer_column: str = 'aquifer_code'
    ) -> np.ndarray:
        """
        Make predictions using appropriate aquifer model.

        Args:
            X: Feature matrix (must include aquifer_column for routing)

        Returns:
            Array of predictions
        """
        predictions = np.zeros(len(X))
        feature_cols = [c for c in X.columns if c != aquifer_column]

        if aquifer_column in X.columns:
            for aquifer_id in X[aquifer_column].unique():
                mask = X[aquifer_column] == aquifer_id

                if str(aquifer_id) in self.models:
                    predictions[mask] = self.models[str(aquifer_id)].predict(
                        X.loc[mask, feature_cols]
                    )
                elif self.global_model is not None:
                    predictions[mask] = self.global_model.predict(
                        X.loc[mask, feature_cols]
                    )
        else:
            # No aquifer info, use global model
            if self.global_model is not None:
                predictions = self.global_model.predict(X[feature_cols])

        return predictions

    def get_model_for_aquifer(self, aquifer_id: str) -> AquiferModel:
        """Get the appropriate model for an aquifer"""
        if aquifer_id in self.models:
            return self.models[aquifer_id]
        return self.global_model

    def save_all(self, path: Optional[Path] = None):
        """Save all models to disk"""
        path = path or self.models_dir
        if path is None:
            raise ValueError("No path specified for saving models")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save global model
        if self.global_model:
            self.global_model.save(path)

        # Save aquifer models
        for aquifer_id, model in self.models.items():
            model.save(path)

        # Save manager metadata
        meta = {
            'aquifer_ids': list(self.models.keys()),
            'has_global': self.global_model is not None
        }
        with open(path / 'manager_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved {len(self.models) + 1} models to {path}")

    def load_all(self, path: Optional[Path] = None):
        """Load all models from disk"""
        path = path or self.models_dir
        if path is None:
            raise ValueError("No path specified for loading models")

        path = Path(path)

        # Load manager metadata
        meta_path = path / 'manager_meta.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            # Load global model
            if meta.get('has_global'):
                self.global_model = AquiferModel.load(path, 'global')

            # Load aquifer models
            for aquifer_id in meta.get('aquifer_ids', []):
                try:
                    self.models[aquifer_id] = AquiferModel.load(path, aquifer_id)
                except Exception as e:
                    logger.warning(f"Could not load model for {aquifer_id}: {e}")

        logger.info(f"Loaded {len(self.models)} aquifer models")

    def get_feature_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance across all models"""
        importance_data = []

        if self.global_model and self.global_model.feature_importance:
            for feat, imp in self.global_model.feature_importance.items():
                importance_data.append({
                    'feature': feat,
                    'aquifer': 'global',
                    'importance': imp
                })

        for aquifer_id, model in self.models.items():
            if model.feature_importance:
                for feat, imp in model.feature_importance.items():
                    importance_data.append({
                        'feature': feat,
                        'aquifer': aquifer_id,
                        'importance': imp
                    })

        if importance_data:
            df = pd.DataFrame(importance_data)
            # Aggregate across aquifers
            agg = df.groupby('feature')['importance'].mean().sort_values(ascending=False)
            return agg.reset_index()

        return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with synthetic data
    np.random.seed(42)
    n_samples = 200

    # Create synthetic features
    X = pd.DataFrame({
        'elevation': np.random.uniform(10, 100, n_samples),
        'slope': np.random.uniform(0, 15, n_samples),
        'rainfall': np.random.uniform(700, 1200, n_samples),
        'well_density': np.random.uniform(1, 50, n_samples),
        'aquifer_code': np.random.choice(['BG', 'ST', 'SH'], n_samples)
    })

    # Create synthetic target (with aquifer-specific behavior)
    y = (
        X['elevation'] * 0.1 +
        X['rainfall'] * -0.01 +
        X['well_density'] * 0.2 +
        np.where(X['aquifer_code'] == 'BG', 5, 0) +
        np.where(X['aquifer_code'] == 'ST', -3, 0) +
        np.random.normal(0, 2, n_samples)
    )

    # Train models
    manager = AquiferModelManager()
    metrics = manager.train_all(X, y)

    print("\n" + "="*50)
    print("Training Results")
    print("="*50)
    for aquifer_id, m in metrics.items():
        print(f"\n{aquifer_id}:")
        print(f"  Train RMSE: {m['train_rmse']:.2f}")
        print(f"  Val RMSE: {m['val_rmse']:.2f}")
        print(f"  R²: {m['val_r2']:.3f}")

    # Feature importance
    print("\n" + "="*50)
    print("Feature Importance")
    print("="*50)
    importance = manager.get_feature_importance()
    print(importance)
