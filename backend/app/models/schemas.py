"""
SmartJal API Schemas

Pydantic models for request/response validation.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ==================== Enums ====================

class DepthCategory(str, Enum):
    SAFE = "safe"
    MODERATE = "moderate"
    STRESS = "stress"
    CRITICAL = "critical"


class Season(str, Enum):
    PRE_MONSOON = "pre_monsoon"
    MONSOON = "monsoon"
    POST_MONSOON = "post_monsoon"


# ==================== Base Models ====================

class Coordinates(BaseModel):
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    lat: float = Field(..., ge=-90, le=90, description="Latitude")


class BoundingBox(BaseModel):
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float


# ==================== Aquifer Models ====================

class AquiferBase(BaseModel):
    aquifer_code: str
    aquifer_type: str
    area_sqkm: Optional[float] = None


class AquiferResponse(AquiferBase):
    village_count: Optional[int] = None
    avg_water_level: Optional[float] = None
    piezometer_count: Optional[int] = None


class AquiferListResponse(BaseModel):
    aquifers: List[AquiferResponse]
    total: int


# ==================== Village Models ====================

class VillageBase(BaseModel):
    village_id: str
    village_name: str
    mandal: str
    district: str


class VillageFeatures(BaseModel):
    elevation: Optional[float] = None
    slope: Optional[float] = None
    annual_rainfall: Optional[float] = None
    well_count: Optional[int] = None
    avg_bore_depth: Optional[float] = None
    aquifer_code: Optional[str] = None
    aquifer_type: Optional[str] = None


class VillagePrediction(BaseModel):
    water_level_m: float = Field(..., description="Predicted water level in meters below ground")
    uncertainty_m: float = Field(..., description="Prediction uncertainty in meters")
    category: DepthCategory
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")


class VillageResponse(VillageBase):
    coordinates: Coordinates
    prediction: Optional[VillagePrediction] = None
    aquifer: Optional[AquiferBase] = None
    features: Optional[VillageFeatures] = None


class VillageListResponse(BaseModel):
    villages: List[VillageResponse]
    total: int


# ==================== Prediction Models ====================

class PredictionRequest(BaseModel):
    lon: float = Field(..., ge=79, le=82, description="Longitude (Krishna District)")
    lat: float = Field(..., ge=15, le=18, description="Latitude (Krishna District)")
    aquifer_code: Optional[str] = None
    features: Optional[Dict[str, float]] = None


class PredictionResponse(BaseModel):
    water_level_m: float
    uncertainty_m: float
    category: DepthCategory
    confidence: float
    components: Dict[str, float]
    aquifer_code: Optional[str] = None
    aquifer_type: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    locations: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


# ==================== Time Series Models ====================

class TimeSeriesPoint(BaseModel):
    date: datetime
    water_level: float
    is_forecast: bool = False


class TimeSeriesResponse(BaseModel):
    piezo_id: str
    location: str
    aquifer: Optional[str] = None
    data: List[TimeSeriesPoint]
    statistics: Dict[str, float]


# ==================== Analytics Models ====================

class DistrictSummary(BaseModel):
    total_villages: int
    villages_with_predictions: int
    avg_water_level: float
    critical_villages: int
    stress_villages: int
    moderate_villages: int
    safe_villages: int
    piezometer_count: int
    aquifer_count: int


class TrendData(BaseModel):
    year: int
    month: int
    avg_water_level: float
    sample_count: int


class TrendResponse(BaseModel):
    data: List[TrendData]
    trend_direction: str  # "improving", "stable", "declining"
    change_rate: float  # meters per year


class AnomalyResponse(BaseModel):
    piezo_id: str
    location: str
    date: datetime
    observed_level: float
    expected_level: float
    anomaly_score: float
    anomaly_type: str  # "high", "low"


# ==================== Recharge Models ====================

class RechargeZone(BaseModel):
    zone_id: str
    name: str
    recharge_potential: str  # "High", "Moderate", "Low", "Very Low"
    recharge_index: float
    area_sqkm: float
    recommended_structures: List[str]


class RechargeRecommendation(BaseModel):
    village_id: str
    village_name: str
    current_water_level: float
    target_water_level: float = 8.0
    water_deficit_mm: float
    recommended_actions: List[str]
    estimated_recharge_potential: float
    priority: str  # "High", "Medium", "Low"


# ==================== GeoJSON Models ====================

class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: Dict[str, Any]
    properties: Dict[str, Any]


class GeoJSONFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature]


# ==================== Health Check ====================

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    data_loaded: bool
    timestamp: datetime
