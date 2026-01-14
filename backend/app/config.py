"""
SmartJal Configuration Settings
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # App settings
    APP_NAME: str = "SmartJal API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # API settings
    API_V1_PREFIX: str = "/api/v1"

    # Data paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT
    MODELS_DIR: Path = PROJECT_ROOT / "models"

    # Shapefile paths
    AQUIFERS_PATH: Path = DATA_DIR / "Aquifers_Krishna" / "Aquifers_Krishna.shp"
    GEOMORPHOLOGY_PATH: Path = DATA_DIR / "GM_Krishna" / "GM_Krishna.shp"
    LULC_PATH: Path = DATA_DIR / "LULC_Krishna" / "LULC_Krishna1.shp"
    LULC_RASTER_PATH: Path = DATA_DIR / "LULC_Krishna" / "LULC_2005.tif"
    WELLS_PATH: Path = DATA_DIR / "GTWells_Krishna" / "GTWells" / "kris.csv"
    WATER_LEVELS_PATH: Path = DATA_DIR / "WaterLevels_Krishna" / "master data_updated.xlsx"

    # Google Earth Engine
    GEE_PROJECT: str = ""
    GEE_SERVICE_ACCOUNT: str = ""
    GEE_KEY_FILE: str = ""

    # Krishna District bounding box (approximate)
    KRISHNA_BOUNDS: dict = {
        "min_lon": 80.0,
        "max_lon": 81.5,
        "min_lat": 15.5,
        "max_lat": 17.0
    }

    # Model settings
    AQUIFER_MODEL_PATH: Path = MODELS_DIR / "aquifer_models"
    SPATIAL_MODEL_PATH: Path = MODELS_DIR / "spatial_model"
    TEMPORAL_MODEL_PATH: Path = MODELS_DIR / "temporal_model"
    ENSEMBLE_WEIGHTS_PATH: Path = MODELS_DIR / "ensemble_weights.json"

    # Water level categories (meters below ground)
    DEPTH_CATEGORIES: dict = {
        "safe": (0, 3),
        "moderate": (3, 8),
        "stress": (8, 20),
        "critical": (20, float("inf"))
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
