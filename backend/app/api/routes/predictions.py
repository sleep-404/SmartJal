"""
SmartJal Prediction API Routes
"""
import logging
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends

from ...models.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    DepthCategory
)
from ...config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def get_predictor():
    """Get predictor instance"""
    from ...main import app_state
    predictor = app_state.get('predictor')
    if predictor is None or not predictor._ready:
        # Return mock predictor for demo
        return MockPredictor()
    return predictor


def get_data_loader():
    """Get data loader instance"""
    from ...main import app_state
    return app_state.get('data_loader')


class MockPredictor:
    """Mock predictor for demo when model not trained"""

    def predict_village(self, features, lon, lat, aquifer_code=None):
        # Generate deterministic but varied predictions based on location
        np.random.seed(int(lon * 1000 + lat * 100))

        base_level = 8 + (lat - 16) * 5 + (lon - 80.5) * 3
        uncertainty = np.random.uniform(1, 3)

        water_level = max(1, min(30, base_level + np.random.normal(0, 2)))

        if water_level <= 3:
            category = 'safe'
        elif water_level <= 8:
            category = 'moderate'
        elif water_level <= 20:
            category = 'stress'
        else:
            category = 'critical'

        return {
            'water_level_m': water_level,
            'uncertainty_m': uncertainty,
            'category': category,
            'confidence': 0.75,
            'components': {
                'ml_prediction': water_level * 0.95,
                'spatial_prediction': water_level * 1.05
            }
        }

    _ready = True


@router.post("/village", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    predictor=Depends(get_predictor),
    loader=Depends(get_data_loader)
):
    """
    Predict groundwater level for a single location.

    Provide coordinates within Krishna District and optionally
    the aquifer code for more accurate predictions.
    """
    try:
        # Get aquifer from spatial join if not provided
        aquifer_code = request.aquifer_code
        aquifer_type = None

        if loader and not aquifer_code:
            try:
                import geopandas as gpd
                from shapely.geometry import Point

                aquifers = loader.load_aquifers()
                point = gpd.GeoDataFrame(
                    geometry=[Point(request.lon, request.lat)],
                    crs='EPSG:4326'
                )
                joined = gpd.sjoin(point, aquifers, how='left', predicate='within')
                if len(joined) > 0 and pd.notna(joined.iloc[0].get('aquifer_code')):
                    aquifer_code = joined.iloc[0]['aquifer_code']
                    aquifer_type = joined.iloc[0].get('aquifer_type')
            except Exception as e:
                logger.warning(f"Could not determine aquifer: {e}")

        # Make prediction
        features = request.features or {}
        result = predictor.predict_village(
            features=features,
            lon=request.lon,
            lat=request.lat,
            aquifer_code=aquifer_code
        )

        return PredictionResponse(
            water_level_m=result['water_level_m'],
            uncertainty_m=result['uncertainty_m'],
            category=DepthCategory(result['category']),
            confidence=result['confidence'],
            components=result['components'],
            aquifer_code=aquifer_code,
            aquifer_type=aquifer_type
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    predictor=Depends(get_predictor)
):
    """
    Predict groundwater levels for multiple locations.
    """
    predictions = []
    categories = {'safe': 0, 'moderate': 0, 'stress': 0, 'critical': 0}

    for loc in request.locations:
        try:
            result = predictor.predict_village(
                features=loc.features or {},
                lon=loc.lon,
                lat=loc.lat,
                aquifer_code=loc.aquifer_code
            )

            predictions.append(PredictionResponse(
                water_level_m=result['water_level_m'],
                uncertainty_m=result['uncertainty_m'],
                category=DepthCategory(result['category']),
                confidence=result['confidence'],
                components=result['components'],
                aquifer_code=loc.aquifer_code
            ))

            categories[result['category']] += 1

        except Exception as e:
            logger.error(f"Batch prediction error for {loc}: {e}")
            # Add placeholder for failed predictions
            predictions.append(PredictionResponse(
                water_level_m=-1,
                uncertainty_m=-1,
                category=DepthCategory.MODERATE,
                confidence=0,
                components={}
            ))

    avg_level = np.mean([p.water_level_m for p in predictions if p.water_level_m > 0])

    return BatchPredictionResponse(
        predictions=predictions,
        summary={
            'total': len(predictions),
            'avg_water_level': float(avg_level) if not np.isnan(avg_level) else 0,
            'categories': categories
        }
    )


@router.get("/timeseries/{piezo_id}")
async def get_timeseries(
    piezo_id: str,
    loader=Depends(get_data_loader)
):
    """
    Get historical water level time series for a piezometer.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        metadata, time_series = loader.load_water_levels()

        # Filter by piezometer ID
        piezo_data = time_series[time_series['piezo_id'] == piezo_id]

        if len(piezo_data) == 0:
            raise HTTPException(status_code=404, detail=f"Piezometer {piezo_id} not found")

        # Get metadata
        piezo_meta = metadata[metadata['piezo_id'] == piezo_id]
        location = piezo_meta.iloc[0]['village'] if len(piezo_meta) > 0 else 'Unknown'
        aquifer = piezo_meta.iloc[0].get('aquifer', 'Unknown') if len(piezo_meta) > 0 else 'Unknown'

        # Format time series
        data = [
            {
                'date': row['date'].isoformat(),
                'water_level': float(row['water_level']),
                'is_forecast': False
            }
            for _, row in piezo_data.iterrows()
        ]

        # Calculate statistics
        stats = {
            'mean': float(piezo_data['water_level'].mean()),
            'min': float(piezo_data['water_level'].min()),
            'max': float(piezo_data['water_level'].max()),
            'std': float(piezo_data['water_level'].std()),
            'count': len(piezo_data)
        }

        return {
            'piezo_id': piezo_id,
            'location': location,
            'aquifer': aquifer,
            'data': data,
            'statistics': stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Time series error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
