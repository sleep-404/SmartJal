"""
Smart Jal - Constants and Configuration

All bounds, thresholds, and configuration values used across the project.
"""

# Krishna District Bounds
KRISHNA_BOUNDS = {
    'min_lat': 15.65,
    'max_lat': 17.25,
    'min_lon': 79.25,
    'max_lon': 81.35
}

# Risk Classification Thresholds (meters below ground level)
RISK_THRESHOLDS = {
    'safe': 8,        # 0-8m: Safe
    'moderate': 15,   # 8-15m: Moderate
    'stress': 20,     # 15-20m: Stress
    'critical': 100   # >20m: Critical
}

# Risk Score Thresholds
RISK_SCORE_THRESHOLDS = {
    'trend_critical': -1.5,      # m/year decline
    'trend_high': -0.5,          # m/year decline
    'anomaly_critical': -2.0,    # std deviations below expected
    'anomaly_high': -1.0,        # std deviations below expected
    'depth_critical': 0.8,       # fraction of typical well depth
    'depth_high': 0.6            # fraction of typical well depth
}

# Aquifer Types in Krishna District
AQUIFER_TYPES = {
    'alluvium': {'typical_depth': 6, 'infiltration': 'high'},
    'sandstone': {'typical_depth': 12, 'infiltration': 'moderate'},
    'shale': {'typical_depth': 15, 'infiltration': 'low'},
    'limestone': {'typical_depth': 10, 'infiltration': 'moderate'},
    'granite': {'typical_depth': 20, 'infiltration': 'low'},
    'gneiss': {'typical_depth': 18, 'infiltration': 'low'},
    'basalt': {'typical_depth': 15, 'infiltration': 'low'},
    'quartzite': {'typical_depth': 22, 'infiltration': 'very_low'}
}

# Monsoon Months (India)
MONSOON_MONTHS = [6, 7, 8, 9, 10]  # June to October
PRE_MONSOON_MONTH = 5              # May
POST_MONSOON_MONTH = 11            # November

# Model Parameters
MODEL_PARAMS = {
    'kriging': {
        'variogram_model': 'spherical',
        'nlags': 6,
        'weight': True
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'ensemble': {
        'spatial_weight': 0.4,
        'feature_weight': 0.6
    }
}

# Data Paths (relative to project root)
DATA_PATHS = {
    'piezometers': 'SmartJal_extracted/SmartJal/WaterLevels_Krishna/master data_updated.xlsx',
    'aquifers': 'Aquifers_Krishna/Aquifers_Krishna.shp',
    'villages': 'UseCase_extracted/UseCase/OKri_Vil/OKri_Vil.shp',
    'mandals': 'UseCase_extracted/UseCase/OKri_Mdl/OKri_Mdl.shp',
    'soils': 'UseCase_extracted/UseCase/OKri_Soils/OKri_Soils.shp',
    'geomorphology': 'GM_Krishna/GM_Krishna.shp',
    'bore_wells': 'GTWells_Krishna/GTWells/kris.csv',
    'lulc': 'LULC_Krishna/LULC_Krishna.shp',
    'pumping': 'Pumping Data.xlsx',
    'dem': 'downloaded_data/dem/krishna_dem.tif',
    'grace': 'downloaded_data/grace/grace_krishna_proxy.csv'
}

# Output Paths
OUTPUT_PATHS = {
    'processed_data': 'data/processed',
    'models': 'data/models',
    'predictions': 'data/processed/predictions.csv',
    'risk_classification': 'data/processed/risk_classification.csv',
    'villages_features': 'data/processed/villages_with_features.csv',
    'piezometers_processed': 'data/processed/piezometers_processed.csv'
}

# Validation Targets
VALIDATION_TARGETS = {
    'mae': 2.0,       # meters
    'rmse': 3.0,      # meters
    'r2': 0.7,        # R-squared
    'risk_accuracy': 0.8  # 80% accuracy on risk classification
}
