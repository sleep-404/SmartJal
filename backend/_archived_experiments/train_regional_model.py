#!/usr/bin/env python3
"""
Smart Jal - Regional Model Training
Trains on CGWB data from Krishna + neighboring districts.

This model learns from 500+ stations instead of 138, reducing overfitting.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings('ignore')

from data_processing.load_cgwb_data import load_cgwb_data, get_krishna_neighbors_data, prepare_training_data
from data_processing.load_data import load_all_data


def extract_features_for_points(
    points_gdf: gpd.GeoDataFrame,
    dem_path: Path,
    rainfall_files: dict,
    target_year: int,
    target_month: int
) -> pd.DataFrame:
    """
    Extract features for CGWB station points.

    Since these are points (not polygons), we extract point values from rasters.
    """
    import rasterio

    features = points_gdf.copy()

    # 1. Elevation from DEM
    print("  Extracting elevation...")
    elevations = []
    with rasterio.open(dem_path) as src:
        for idx, row in features.iterrows():
            try:
                lon, lat = row.geometry.x, row.geometry.y
                py, px = src.index(lon, lat)
                if 0 <= py < src.height and 0 <= px < src.width:
                    val = src.read(1)[py, px]
                    if src.nodata and val == src.nodata:
                        val = np.nan
                    elevations.append(float(val))
                else:
                    elevations.append(np.nan)
            except:
                elevations.append(np.nan)
    features['elevation_mean'] = elevations
    features['elevation_mean'] = features['elevation_mean'].fillna(features['elevation_mean'].median())

    # 2. Rainfall features
    print("  Extracting rainfall...")
    for lag in range(4):
        lag_month = target_month - lag
        lag_year = target_year
        if lag_month <= 0:
            lag_month += 12
            lag_year -= 1

        col_name = f'rainfall_lag{lag}' if lag > 0 else 'rainfall_current'
        key = (lag_year, lag_month)

        if key in rainfall_files:
            rainfall = []
            with rasterio.open(rainfall_files[key]) as src:
                for idx, row in features.iterrows():
                    try:
                        lon, lat = row.geometry.x, row.geometry.y
                        py, px = src.index(lon, lat)
                        if 0 <= py < src.height and 0 <= px < src.width:
                            val = src.read(1)[py, px]
                            if src.nodata and val == src.nodata:
                                val = np.nan
                            rainfall.append(float(val))
                        else:
                            rainfall.append(np.nan)
                    except:
                        rainfall.append(np.nan)
            features[col_name] = rainfall
        else:
            features[col_name] = 50.0  # Default

    # Fill NaN rainfall
    for col in features.columns:
        if 'rainfall' in col:
            features[col] = features[col].fillna(features[col].median())

    # 3. Cumulative rainfall
    features['rainfall_cumulative_3m'] = (
        features['rainfall_current'].fillna(0) +
        features['rainfall_lag1'].fillna(0) +
        features['rainfall_lag2'].fillna(0)
    )

    # 4. Seasonal encoding
    features['month_sin'] = np.sin(2 * np.pi * target_month / 12)
    features['month_cos'] = np.cos(2 * np.pi * target_month / 12)
    features['is_monsoon'] = 1 if target_month in [6, 7, 8, 9, 10] else 0
    features['is_post_monsoon'] = 1 if target_month in [11, 12, 1] else 0

    return features


def train_regional_model(
    training_periods: list = None,
    model_type: str = 'gradient_boosting'
):
    """
    Train a regional model on CGWB data.

    Args:
        training_periods: List of (year, month) tuples to use for training
        model_type: 'gradient_boosting', 'random_forest', 'ridge', 'lasso'
    """
    print("=" * 60)
    print("REGIONAL MODEL TRAINING")
    print("Using CGWB data from Krishna + neighboring districts")
    print("=" * 60)

    # Load CGWB data
    cgwb_df = get_krishna_neighbors_data(min_date='2018-01-01')
    print(f"\nCGWB data: {len(cgwb_df)} records, {cgwb_df['station_name'].nunique()} stations")

    # Load raster data for feature extraction
    print("\nLoading raster data...")
    raw_data = load_all_data()
    dem_path = raw_data['dem_path']
    rainfall_files = raw_data['rainfall_files']

    # Default training periods (CGWB measures Jan, May, Aug, Nov)
    if training_periods is None:
        training_periods = [
            (2018, 5), (2018, 8), (2018, 11),
            (2019, 1), (2019, 5), (2019, 8), (2019, 11),
            (2020, 1), (2020, 11),
            (2021, 1),
        ]

    # Collect training data from all periods
    print("\nPreparing training data from multiple time periods...")
    all_features = []
    all_targets = []

    for year, month in training_periods:
        print(f"\n  Processing {year}-{month:02d}...")
        stations_gdf, water_levels = prepare_training_data(cgwb_df, year, month)

        if len(stations_gdf) == 0:
            continue

        # Extract features
        features_df = extract_features_for_points(
            stations_gdf, dem_path, rainfall_files, year, month
        )

        all_features.append(features_df)
        all_targets.append(water_levels)

    # Combine all data
    X_df = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_targets, ignore_index=True)

    print(f"\n" + "=" * 60)
    print(f"Total training samples: {len(X_df)}")
    print(f"From {len(training_periods)} time periods")
    print("=" * 60)

    # Select feature columns
    feature_cols = [
        'centroid_lat', 'centroid_lon', 'elevation_mean',
        'rainfall_current', 'rainfall_lag1', 'rainfall_lag2', 'rainfall_lag3',
        'rainfall_cumulative_3m',
        'month_sin', 'month_cos', 'is_monsoon', 'is_post_monsoon'
    ]

    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in X_df.columns]
    X = X_df[feature_cols].copy()

    # Handle NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())

    y = pd.to_numeric(y, errors='coerce')
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"\nFeatures: {list(X.columns)}")
    print(f"Training samples after cleaning: {len(X)}")

    # Split data - hold out Krishna district for testing
    # This tests generalization to our target area
    krishna_mask = X_df.loc[valid_mask.index[valid_mask], 'district_name'] == 'Krishna'

    X_train = X[~krishna_mask]
    y_train = y[~krishna_mask]
    X_test = X[krishna_mask]
    y_test = y[krishna_mask]

    print(f"\nTrain set (other districts): {len(X_train)}")
    print(f"Test set (Krishna district): {len(X_test)}")

    # Initialize model
    if model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,  # Shallow to prevent overfitting
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"\nTraining {model_type} model...")

    # Train
    model.fit(X_train, y_train)

    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Evaluate on test set (Krishna - our target)
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nTraining (other districts):")
    print(f"  RMSE: {train_rmse:.3f}m")
    print(f"  R²: {train_r2:.3f}")

    print(f"\nTest (Krishna district - generalization):")
    print(f"  RMSE: {test_rmse:.3f}m")
    print(f"  R²: {test_r2:.3f}")
    print(f"  MAE: {test_mae:.3f}m")

    # Cross-validation on full dataset
    print("\n5-fold Cross-Validation:")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"  R² scores: {cv_scores}")
    print(f"  Mean R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature Importance:")
        print(importance.to_string(index=False))

    # Save model
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / f"regional_model_{model_type}.joblib"
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    return model, test_r2, test_rmse


if __name__ == '__main__':
    # Train with different model types
    print("\n" + "=" * 70)
    print("TRAINING GRADIENT BOOSTING")
    print("=" * 70)
    gb_model, gb_r2, gb_rmse = train_regional_model(model_type='gradient_boosting')

    print("\n" + "=" * 70)
    print("TRAINING RIDGE REGRESSION (simpler, less overfitting)")
    print("=" * 70)
    ridge_model, ridge_r2, ridge_rmse = train_regional_model(model_type='ridge')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Gradient Boosting: R² = {gb_r2:.3f}, RMSE = {gb_rmse:.3f}m")
    print(f"Ridge Regression:  R² = {ridge_r2:.3f}, RMSE = {ridge_rmse:.3f}m")
