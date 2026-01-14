"""
SmartJal Training Pipeline

Complete training pipeline using only the department-provided data:
- Piezometer readings (138 stations, 28 years)
- Aquifer boundaries (8 types)
- Well data (88,988 wells)
- Geomorphology (614 features)

No external data (GEE, rainfall, DEM) required.
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.data.loader import DataLoader, get_data_loader

logger = logging.getLogger(__name__)

# Try to import LightGBM
LIGHTGBM_AVAILABLE = False
try:
    import lightgbm as lgb
    # Test if it actually works (libomp issue on macOS)
    lgb.LGBMRegressor()
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.warning(f"LightGBM not available ({e}), using sklearn GradientBoosting")
    LIGHTGBM_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor


class LocalFeatureExtractor:
    """
    Extract features using only department-provided data.

    Features:
    1. Aquifer type (one-hot encoded)
    2. Well statistics (from 88K wells within radius)
    3. Geomorphology class
    4. Temporal statistics from piezometer history
    5. Spatial features (distance to piezometers)
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.loader = data_loader or get_data_loader()
        self._well_stats_cache = None
        self._piezo_tree = None
        self._piezo_data = None

    def extract_features_for_piezometers(self, target_year: int = 2024) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Extract features for all piezometers and get their water level targets.

        Returns:
            Tuple of (features_df, target_series, metadata_df)
        """
        logger.info("Extracting features for piezometer locations...")

        # Load piezometer data
        metadata, time_series = self.loader.load_water_levels()
        piezometers = self.loader.load_piezometers_geodataframe()

        # Get target values (use recent May for pre-monsoon, November for post-monsoon)
        # Average of last 3 years for stability
        recent_years = [target_year - i for i in range(3) if target_year - i in time_series['year'].unique()]
        if not recent_years:
            recent_years = sorted(time_series['year'].unique())[-3:]

        # Pre-monsoon (May)
        pre_monsoon = time_series[
            (time_series['year'].isin(recent_years)) &
            (time_series['month'] == 5)
        ].groupby('piezo_id')['water_level'].mean()

        # Post-monsoon (November) - use this as primary target
        post_monsoon = time_series[
            (time_series['year'].isin(recent_years)) &
            (time_series['month'] == 11)
        ].groupby('piezo_id')['water_level'].mean()

        # Use annual average as fallback
        annual_avg = time_series[
            time_series['year'].isin(recent_years)
        ].groupby('piezo_id')['water_level'].mean()

        # Combine targets (prefer post-monsoon, then pre-monsoon, then annual)
        target = post_monsoon.combine_first(pre_monsoon).combine_first(annual_avg)

        # Filter to piezometers with valid targets
        valid_piezo_ids = target.dropna().index.tolist()
        valid_metadata = metadata[metadata['piezo_id'].isin(valid_piezo_ids)].copy()

        # Create GeoDataFrame for spatial operations
        valid_piezos = gpd.GeoDataFrame(
            valid_metadata,
            geometry=gpd.points_from_xy(valid_metadata['lon'], valid_metadata['lat']),
            crs='EPSG:4326'
        )

        # Extract features
        features = self._extract_all_features(valid_piezos, time_series)

        # Align target with features
        y = target.loc[valid_metadata['piezo_id']].values

        logger.info(f"Extracted {features.shape[1]} features for {len(features)} piezometers")

        return features, pd.Series(y, index=features.index), valid_metadata

    def extract_features_for_villages(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract features for all villages (defined by unique village locations in well data).

        Returns:
            Tuple of (features_df, village_info_df)
        """
        logger.info("Extracting features for village locations...")

        # Get unique villages from well data
        wells = self.loader.load_wells()

        # Aggregate to village level
        village_agg = wells.groupby(['district', 'mandal', 'village']).agg({
            'lat': 'mean',
            'lon': 'mean',
            'bore_depth': ['count', 'mean', 'median', 'max', 'std'],
            'pump_capacity': ['mean', 'sum'],
            'irrigated_area': 'sum'
        })

        village_agg.columns = [
            'lat', 'lon',
            'well_count', 'avg_bore_depth', 'median_bore_depth', 'max_bore_depth', 'std_bore_depth',
            'avg_pump_capacity', 'total_pump_capacity', 'total_irrigated_area'
        ]
        village_agg = village_agg.reset_index()

        # Filter invalid coordinates
        village_agg = village_agg[
            (village_agg['lat'] > 0) & (village_agg['lat'] < 90) &
            (village_agg['lon'] > 0) & (village_agg['lon'] < 180)
        ].copy()

        # Create GeoDataFrame
        villages_gdf = gpd.GeoDataFrame(
            village_agg,
            geometry=gpd.points_from_xy(village_agg['lon'], village_agg['lat']),
            crs='EPSG:4326'
        )

        # Load time series for temporal features
        _, time_series = self.loader.load_water_levels()

        # Extract features
        features = self._extract_all_features(villages_gdf, time_series, is_village=True)

        # Village info
        village_info = village_agg[['district', 'mandal', 'village', 'lat', 'lon']].copy()
        village_info = village_info.reset_index(drop=True)

        logger.info(f"Extracted {features.shape[1]} features for {len(features)} villages")

        return features, village_info

    def _extract_all_features(
        self,
        points_gdf: gpd.GeoDataFrame,
        time_series: pd.DataFrame,
        is_village: bool = False
    ) -> pd.DataFrame:
        """Extract all features for a set of points."""
        features_list = []

        # 1. Aquifer features
        aquifer_features = self._extract_aquifer_features(points_gdf)
        features_list.append(aquifer_features)

        # 2. Well features (already computed for villages in input)
        well_features = self._extract_well_features(points_gdf, is_village)
        features_list.append(well_features)

        # 3. Geomorphology features
        geomorph_features = self._extract_geomorphology_features(points_gdf)
        features_list.append(geomorph_features)

        # 4. Temporal features from piezometer history
        temporal_features = self._extract_temporal_features(points_gdf, time_series)
        features_list.append(temporal_features)

        # 5. Spatial features
        spatial_features = self._extract_spatial_features(points_gdf)
        features_list.append(spatial_features)

        # Combine all
        all_features = pd.concat(features_list, axis=1)

        # Remove duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        # Handle missing values
        all_features = self._handle_missing_values(all_features)

        return all_features

    def _extract_aquifer_features(self, points_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Extract aquifer type features."""
        aquifers = self.loader.load_aquifers()

        # Spatial join
        points_with_aquifer = gpd.sjoin(
            points_gdf.reset_index(drop=True),
            aquifers[['aquifer_code', 'aquifer_type', 'area_sqkm', 'geometry']],
            how='left',
            predicate='within'
        )

        features = pd.DataFrame(index=range(len(points_gdf)))
        features['aquifer_code'] = points_with_aquifer['aquifer_code'].values
        features['aquifer_type'] = points_with_aquifer['aquifer_type'].values
        features['aquifer_area_sqkm'] = points_with_aquifer['area_sqkm'].values

        # One-hot encode aquifer type
        unique_aquifers = aquifers['aquifer_code'].dropna().unique()
        for aq in unique_aquifers:
            features[f'is_aquifer_{aq}'] = (features['aquifer_code'] == aq).astype(int)

        return features

    def _extract_well_features(self, points_gdf: gpd.GeoDataFrame, is_village: bool = False) -> pd.DataFrame:
        """Extract well statistics features."""
        features = pd.DataFrame(index=range(len(points_gdf)))

        if is_village:
            # For villages, extract from the input dataframe (already aggregated)
            for col in ['well_count', 'avg_bore_depth', 'median_bore_depth',
                       'max_bore_depth', 'std_bore_depth', 'avg_pump_capacity',
                       'total_pump_capacity', 'total_irrigated_area']:
                if col in points_gdf.columns:
                    features[col] = points_gdf[col].values
        else:
            # For piezometers, compute from wells within radius
            wells = self.loader.load_wells()

            # Filter wells with valid coordinates
            wells = wells[
                wells['lon'].notna() & wells['lat'].notna() &
                np.isfinite(wells['lon']) & np.isfinite(wells['lat'])
            ]

            well_coords = np.array(list(zip(wells['lon'], wells['lat'])))
            well_tree = cKDTree(well_coords)

            point_coords = np.array([[p.x, p.y] for p in points_gdf.geometry])

            # Find wells within 5km (approx 0.05 degrees)
            radius = 0.05

            well_counts = []
            avg_depths = []
            max_depths = []
            total_irrigated = []

            for i, (lon, lat) in enumerate(point_coords):
                # Skip if point coordinates are invalid
                if not np.isfinite(lon) or not np.isfinite(lat):
                    well_counts.append(0)
                    avg_depths.append(np.nan)
                    max_depths.append(np.nan)
                    total_irrigated.append(0)
                    continue

                nearby_idx = well_tree.query_ball_point([lon, lat], radius)

                if len(nearby_idx) > 0:
                    nearby_wells = wells.iloc[nearby_idx]
                    well_counts.append(len(nearby_idx))
                    avg_depths.append(nearby_wells['bore_depth'].mean())
                    max_depths.append(nearby_wells['bore_depth'].max())
                    total_irrigated.append(nearby_wells['irrigated_area'].sum())
                else:
                    well_counts.append(0)
                    avg_depths.append(np.nan)
                    max_depths.append(np.nan)
                    total_irrigated.append(0)

            features['well_count'] = well_counts
            features['avg_bore_depth'] = avg_depths
            features['max_bore_depth'] = max_depths
            features['total_irrigated_area'] = total_irrigated

        # Well density (wells per sq km, assuming 5km radius = ~78 sq km)
        if 'well_count' in features.columns:
            features['well_density'] = features['well_count'] / 78.5

        return features

    def _extract_geomorphology_features(self, points_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Extract geomorphology features."""
        features = pd.DataFrame(index=range(len(points_gdf)))

        try:
            geomorph = self.loader.load_geomorphology()

            # Spatial join
            points_with_geomorph = gpd.sjoin(
                points_gdf.reset_index(drop=True),
                geomorph[['geomorph_class', 'geometry']] if 'geomorph_class' in geomorph.columns else geomorph[['geometry']],
                how='left',
                predicate='within'
            )

            if 'geomorph_class' in points_with_geomorph.columns:
                features['geomorph_class'] = points_with_geomorph['geomorph_class'].values

                # Recharge potential score based on geomorphology
                recharge_scores = {
                    'Flood Plain': 5,
                    'Alluvial Plain': 4,
                    'Valley Fill': 4,
                    'Pediment': 3,
                    'Buried Pediment': 2,
                    'Structural Hills': 1,
                    'Denudational Hills': 1
                }
                features['recharge_potential'] = features['geomorph_class'].map(
                    lambda x: next((v for k, v in recharge_scores.items() if k.lower() in str(x).lower()), 2.5)
                )
        except Exception as e:
            logger.warning(f"Could not extract geomorphology features: {e}")
            features['recharge_potential'] = 2.5

        return features

    def _extract_temporal_features(self, points_gdf: gpd.GeoDataFrame, time_series: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from piezometer history for points within same aquifer."""
        features = pd.DataFrame(index=range(len(points_gdf)))

        # Get aquifer assignment for each point
        aquifers = self.loader.load_aquifers()
        points_with_aquifer = gpd.sjoin(
            points_gdf.reset_index(drop=True),
            aquifers[['aquifer_code', 'geometry']],
            how='left',
            predicate='within'
        )

        # Load piezometer metadata
        metadata, _ = self.loader.load_water_levels()

        # Ensure water_level is numeric
        ts = time_series.copy()
        ts['water_level'] = pd.to_numeric(ts['water_level'], errors='coerce')
        ts = ts.dropna(subset=['water_level'])

        # Calculate statistics per piezometer from time series
        piezo_stats = ts.groupby('piezo_id')['water_level'].agg(['mean', 'std', 'min', 'max']).reset_index()
        piezo_stats.columns = ['piezo_id', 'mean_wl', 'std_wl', 'min_wl', 'max_wl']

        piezo_with_aquifer = metadata.merge(piezo_stats, on='piezo_id', how='left')

        # Rename columns if they have different names
        if 'aquifer' not in piezo_with_aquifer.columns:
            # Try to find the aquifer column
            for col in piezo_with_aquifer.columns:
                if 'aqui' in col.lower():
                    piezo_with_aquifer = piezo_with_aquifer.rename(columns={col: 'aquifer'})
                    break

        # Calculate trend per piezometer
        def calc_trend(group):
            if len(group) < 12:
                return 0
            # Simple linear trend over years
            group = group.sort_values('date')
            x = np.arange(len(group))
            y = group['water_level'].values
            valid = ~np.isnan(y)
            if valid.sum() < 12:
                return 0
            try:
                slope = np.polyfit(x[valid], y[valid], 1)[0]
                return slope * 12  # Trend per year
            except:
                return 0

        trends = ts.groupby('piezo_id').apply(calc_trend, include_groups=False)
        piezo_with_aquifer['trend_per_year'] = piezo_with_aquifer['piezo_id'].map(trends)

        # Aggregate to aquifer level - handle missing aquifer column gracefully
        if 'aquifer' in piezo_with_aquifer.columns:
            numeric_cols = ['mean_wl', 'std_wl', 'min_wl', 'max_wl', 'trend_per_year']
            existing_cols = [c for c in numeric_cols if c in piezo_with_aquifer.columns]

            try:
                aquifer_stats = piezo_with_aquifer.groupby('aquifer').agg({
                    **{c: 'mean' for c in existing_cols},
                    'piezo_id': 'count'
                })
                aquifer_stats = aquifer_stats.rename(columns={
                    'mean_wl': 'aquifer_mean_wl',
                    'std_wl': 'aquifer_std_wl',
                    'min_wl': 'aquifer_min_wl',
                    'max_wl': 'aquifer_max_wl',
                    'trend_per_year': 'aquifer_trend',
                    'piezo_id': 'aquifer_piezo_count'
                })

                # Map to points
                point_aquifer_codes = points_with_aquifer['aquifer_code'].values

                for col in aquifer_stats.columns:
                    features[col] = [
                        aquifer_stats.loc[aq, col] if pd.notna(aq) and aq in aquifer_stats.index else np.nan
                        for aq in point_aquifer_codes
                    ]
            except Exception as e:
                logger.warning(f"Could not aggregate aquifer stats: {e}")
                # Use global stats as fallback
                features['aquifer_mean_wl'] = piezo_with_aquifer['mean_wl'].mean() if 'mean_wl' in piezo_with_aquifer else np.nan
                features['aquifer_std_wl'] = piezo_with_aquifer['std_wl'].mean() if 'std_wl' in piezo_with_aquifer else np.nan
                features['aquifer_piezo_count'] = len(piezo_with_aquifer)
        else:
            # No aquifer column, use global stats
            features['aquifer_mean_wl'] = piezo_with_aquifer['mean_wl'].mean() if 'mean_wl' in piezo_with_aquifer.columns else np.nan
            features['aquifer_std_wl'] = piezo_with_aquifer['std_wl'].mean() if 'std_wl' in piezo_with_aquifer.columns else np.nan
            features['aquifer_piezo_count'] = len(piezo_with_aquifer)

        return features

    def _extract_spatial_features(self, points_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Extract spatial features."""
        features = pd.DataFrame(index=range(len(points_gdf)))

        # Coordinates
        features['lon'] = [p.x for p in points_gdf.geometry]
        features['lat'] = [p.y for p in points_gdf.geometry]

        # Distance to nearest piezometer
        try:
            piezometers = self.loader.load_piezometers_geodataframe()
            if len(piezometers) > 0:
                # Filter valid coordinates
                valid_piezos = piezometers[
                    piezometers.geometry.notna() &
                    piezometers.geometry.apply(lambda g: g is not None and g.is_valid)
                ]

                piezo_coords = np.array([[p.x, p.y] for p in valid_piezos.geometry])
                point_coords = np.array([[p.x, p.y] for p in points_gdf.geometry])

                # Filter out invalid coordinates
                valid_piezo_mask = np.isfinite(piezo_coords).all(axis=1)
                piezo_coords = piezo_coords[valid_piezo_mask]

                if len(piezo_coords) > 0:
                    tree = cKDTree(piezo_coords)

                    distances = []
                    for pc in point_coords:
                        if np.isfinite(pc).all():
                            d, _ = tree.query(pc)
                            distances.append(d)
                        else:
                            distances.append(np.nan)

                    # Convert to approximate km (1 degree ~ 111 km)
                    features['dist_to_nearest_piezo_km'] = np.array(distances) * 111
        except Exception as e:
            logger.warning(f"Could not compute distance to piezometers: {e}")
            features['dist_to_nearest_piezo_km'] = np.nan

        return features

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        df = df.copy()

        # Numeric: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # Categorical: fill with mode or 'Unknown'
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isna().any():
                mode = df[col].mode()
                df[col] = df[col].fillna(mode.iloc[0] if len(mode) > 0 else 'Unknown')

        return df


class GroundwaterTrainer:
    """
    Train groundwater prediction models using local data only.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path(__file__).parent.parent.parent / 'models'
        self.models = {}
        self.global_model = None
        self.feature_columns = []
        self.metrics = {}
        self.spatial_interpolator = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        aquifer_column: str = 'aquifer_code',
        min_samples_per_aquifer: int = 5
    ) -> Dict[str, Any]:
        """
        Train models on piezometer data.

        Args:
            X: Feature matrix
            y: Target (water levels in meters)
            aquifer_column: Column with aquifer codes
            min_samples_per_aquifer: Minimum samples to train aquifer-specific model

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training models with {len(X)} samples...")

        # Store feature columns (exclude aquifer identifiers)
        self.feature_columns = [c for c in X.columns if c not in
                               [aquifer_column, 'aquifer_type', 'geomorph_class'] and
                               not c.startswith('is_aquifer_')]

        # Include one-hot encoded aquifer columns
        aquifer_onehot = [c for c in X.columns if c.startswith('is_aquifer_')]
        self.feature_columns.extend(aquifer_onehot)

        # Prepare training data
        X_train = X[self.feature_columns].copy()

        # Train global model
        logger.info("Training global model...")
        self.global_model = self._train_model(X_train, y, 'global')

        # Train aquifer-specific models
        if aquifer_column in X.columns:
            for aquifer_code in X[aquifer_column].dropna().unique():
                mask = X[aquifer_column] == aquifer_code
                n_samples = mask.sum()

                if n_samples >= min_samples_per_aquifer:
                    logger.info(f"Training model for aquifer {aquifer_code} ({n_samples} samples)...")
                    self.models[aquifer_code] = self._train_model(
                        X_train[mask], y[mask], aquifer_code
                    )

        # Store coordinates for spatial interpolation
        if 'lon' in X.columns and 'lat' in X.columns:
            coords = X[['lon', 'lat']].values
            aquifer_ids = X[aquifer_column].values if aquifer_column in X.columns else None
            self.spatial_interpolator = {
                'coords': coords,
                'values': y.values,
                'aquifers': aquifer_ids
            }

        # Calculate cross-validation metrics
        self.metrics = self._cross_validate(X, y, aquifer_column)

        logger.info(f"Training complete. Global RMSE: {self.metrics.get('global_rmse', 'N/A'):.2f} m")

        return self.metrics

    def _train_model(self, X: pd.DataFrame, y: pd.Series, name: str):
        """Train a single model."""
        if LIGHTGBM_AVAILABLE:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'verbose': -1,
                'n_estimators': 200,
                'random_state': 42
            }
            model = lgb.LGBMRegressor(**params)
        else:
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )

        model.fit(X, y)

        # Log feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(X.columns, model.feature_importances_))
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"  Top features for {name}: {[f[0] for f in top_features]}")

        return model

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series, aquifer_column: str) -> Dict[str, float]:
        """Perform leave-one-out cross-validation."""
        logger.info("Running leave-one-out cross-validation...")

        X_train = X[self.feature_columns].copy()

        # Use LOO for small datasets
        if len(X) < 50:
            cv = LeaveOneOut()
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Cross-validate global model
        if LIGHTGBM_AVAILABLE:
            cv_model = lgb.LGBMRegressor(
                objective='regression', n_estimators=200, learning_rate=0.05,
                num_leaves=31, verbose=-1, random_state=42
            )
        else:
            cv_model = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
            )

        cv_predictions = cross_val_predict(cv_model, X_train, y, cv=cv)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, cv_predictions))
        mae = mean_absolute_error(y, cv_predictions)
        r2 = r2_score(y, cv_predictions)

        # Category accuracy
        true_cats = self._categorize_depth(y.values)
        pred_cats = self._categorize_depth(cv_predictions)
        cat_accuracy = (true_cats == pred_cats).mean()

        metrics = {
            'global_rmse': rmse,
            'global_mae': mae,
            'global_r2': r2,
            'category_accuracy': cat_accuracy,
            'n_samples': len(X)
        }

        logger.info(f"CV Results: RMSE={rmse:.2f}m, MAE={mae:.2f}m, R²={r2:.3f}, Cat.Acc={cat_accuracy:.1%}")

        return metrics

    def _categorize_depth(self, depths: np.ndarray) -> np.ndarray:
        """Categorize water depths."""
        categories = np.empty(len(depths), dtype=object)
        categories[depths <= 3] = 'safe'
        categories[(depths > 3) & (depths <= 8)] = 'moderate'
        categories[(depths > 8) & (depths <= 20)] = 'stress'
        categories[depths > 20] = 'critical'
        return categories

    def predict(
        self,
        X: pd.DataFrame,
        aquifer_column: str = 'aquifer_code',
        use_spatial: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions for new locations.

        Args:
            X: Feature matrix
            aquifer_column: Column with aquifer codes
            use_spatial: Whether to blend with spatial interpolation

        Returns:
            DataFrame with predictions
        """
        X_pred = X[self.feature_columns].copy()
        n = len(X)

        # ML predictions
        ml_predictions = np.zeros(n)

        if aquifer_column in X.columns:
            for i, row in X.iterrows():
                aquifer = row[aquifer_column]
                if aquifer in self.models:
                    ml_predictions[i] = self.models[aquifer].predict(X_pred.iloc[[i]])[0]
                elif self.global_model is not None:
                    ml_predictions[i] = self.global_model.predict(X_pred.iloc[[i]])[0]
        else:
            ml_predictions = self.global_model.predict(X_pred)

        # Spatial interpolation (IDW within same aquifer)
        spatial_predictions = np.zeros(n)
        if use_spatial and self.spatial_interpolator is not None and 'lon' in X.columns:
            query_coords = X[['lon', 'lat']].values
            query_aquifers = X[aquifer_column].values if aquifer_column in X.columns else None

            spatial_predictions = self._idw_interpolate(
                query_coords, query_aquifers
            )

        # Blend predictions (70% ML, 30% spatial)
        if use_spatial:
            final_predictions = 0.7 * ml_predictions + 0.3 * spatial_predictions
        else:
            final_predictions = ml_predictions

        # Calculate uncertainty
        uncertainty = np.abs(ml_predictions - spatial_predictions) if use_spatial else np.zeros(n)

        # Categorize
        categories = self._categorize_depth(final_predictions)

        # Confidence based on uncertainty
        confidence = 1.0 / (1.0 + uncertainty / 5.0)

        return pd.DataFrame({
            'water_level_m': final_predictions,
            'ml_prediction': ml_predictions,
            'spatial_prediction': spatial_predictions if use_spatial else None,
            'uncertainty_m': uncertainty,
            'category': categories,
            'confidence': confidence
        })

    def _idw_interpolate(
        self,
        query_coords: np.ndarray,
        query_aquifers: Optional[np.ndarray],
        power: float = 2.0,
        k: int = 10
    ) -> np.ndarray:
        """Inverse Distance Weighting interpolation within aquifer."""
        predictions = np.zeros(len(query_coords))

        ref_coords = self.spatial_interpolator['coords']
        ref_values = self.spatial_interpolator['values']
        ref_aquifers = self.spatial_interpolator['aquifers']

        for i, (lon, lat) in enumerate(query_coords):
            # Filter to same aquifer
            if query_aquifers is not None and ref_aquifers is not None:
                mask = ref_aquifers == query_aquifers[i]
                if mask.sum() == 0:
                    mask = np.ones(len(ref_coords), dtype=bool)
                coords = ref_coords[mask]
                values = ref_values[mask]
            else:
                coords = ref_coords
                values = ref_values

            # Calculate distances
            distances = np.sqrt((coords[:, 0] - lon)**2 + (coords[:, 1] - lat)**2)

            # Handle exact match
            if distances.min() < 1e-10:
                predictions[i] = values[distances.argmin()]
                continue

            # IDW weights
            k_actual = min(k, len(distances))
            idx = np.argpartition(distances, k_actual - 1)[:k_actual]

            weights = 1.0 / (distances[idx] ** power)
            weights /= weights.sum()

            predictions[i] = np.sum(weights * values[idx])

        return predictions

    def save(self, path: Optional[Path] = None):
        """Save trained models."""
        path = path or self.models_dir
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        import pickle

        # Save models
        with open(path / 'global_model.pkl', 'wb') as f:
            pickle.dump(self.global_model, f)

        with open(path / 'aquifer_models.pkl', 'wb') as f:
            pickle.dump(self.models, f)

        with open(path / 'spatial_interpolator.pkl', 'wb') as f:
            pickle.dump(self.spatial_interpolator, f)

        # Save metadata
        meta = {
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'trained_at': datetime.now().isoformat()
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Models saved to {path}")

    def load(self, path: Optional[Path] = None):
        """Load trained models."""
        path = path or self.models_dir
        path = Path(path)

        import pickle

        with open(path / 'global_model.pkl', 'rb') as f:
            self.global_model = pickle.load(f)

        with open(path / 'aquifer_models.pkl', 'rb') as f:
            self.models = pickle.load(f)

        with open(path / 'spatial_interpolator.pkl', 'rb') as f:
            self.spatial_interpolator = pickle.load(f)

        with open(path / 'metadata.json', 'r') as f:
            meta = json.load(f)

        self.feature_columns = meta['feature_columns']
        self.metrics = meta.get('metrics', {})

        logger.info(f"Models loaded from {path}")


def run_training_pipeline():
    """Run the complete training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("SmartJal Training Pipeline")
    print("="*60)

    # 1. Extract features
    print("\n[1/3] Extracting features...")
    extractor = LocalFeatureExtractor()
    X, y, metadata = extractor.extract_features_for_piezometers()

    print(f"  - Samples: {len(X)}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Target range: {y.min():.1f} - {y.max():.1f} meters")

    # 2. Train models
    print("\n[2/3] Training models...")
    trainer = GroundwaterTrainer()
    metrics = trainer.train(X, y)

    print(f"\n  Training Results:")
    print(f"  - RMSE: {metrics['global_rmse']:.2f} m")
    print(f"  - MAE: {metrics['global_mae']:.2f} m")
    print(f"  - R²: {metrics['global_r2']:.3f}")
    print(f"  - Category Accuracy: {metrics['category_accuracy']:.1%}")

    # 3. Generate predictions for villages
    print("\n[3/3] Generating village predictions...")
    village_features, village_info = extractor.extract_features_for_villages()
    predictions = trainer.predict(village_features)

    # Combine with village info
    results = pd.concat([
        village_info.reset_index(drop=True),
        predictions.reset_index(drop=True)
    ], axis=1)

    # Summary statistics
    print(f"\n  Village Predictions Summary:")
    print(f"  - Total villages: {len(results)}")
    print(f"  - Mean water level: {results['water_level_m'].mean():.1f} m")

    cat_counts = results['category'].value_counts()
    print(f"  - Categories:")
    for cat in ['safe', 'moderate', 'stress', 'critical']:
        count = cat_counts.get(cat, 0)
        pct = count / len(results) * 100
        print(f"      {cat.capitalize()}: {count} ({pct:.1f}%)")

    # Save models
    trainer.save()

    # Save predictions
    output_path = Path(__file__).parent.parent.parent / 'outputs'
    output_path.mkdir(exist_ok=True)

    results.to_csv(output_path / 'village_predictions.csv', index=False)
    print(f"\n  Predictions saved to: {output_path / 'village_predictions.csv'}")

    print("\n" + "="*60)
    print("Training Pipeline Complete!")
    print("="*60)

    return results, metrics


if __name__ == "__main__":
    run_training_pipeline()
