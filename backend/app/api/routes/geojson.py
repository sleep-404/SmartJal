"""
SmartJal GeoJSON API Routes

Provides GeoJSON endpoints for map visualization.
"""
import logging
from typing import Optional
import json

import numpy as np
from shapely.geometry import mapping
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def safe_float(val):
    """Convert value to float, returning None if NaN or invalid."""
    if val is None:
        return None
    try:
        fval = float(val)
        if np.isnan(fval) or np.isinf(fval):
            return None
        return fval
    except (TypeError, ValueError):
        return None


def safe_str(val):
    """Convert value to string, returning empty string if NaN or None."""
    if val is None:
        return ''
    if isinstance(val, float) and np.isnan(val):
        return ''
    return str(val)


def get_data_loader():
    """Get data loader instance"""
    from ...main import app_state
    return app_state.get('data_loader')


def get_predictor():
    """Get predictor instance"""
    from ...main import app_state
    return app_state.get('predictor')


@router.get("/aquifers")
async def get_aquifers_geojson(loader=Depends(get_data_loader)):
    """
    Get aquifer boundaries as GeoJSON.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        aquifers = loader.load_aquifers()

        features = []
        for _, row in aquifers.iterrows():
            feature = {
                'type': 'Feature',
                'geometry': mapping(row.geometry),
                'properties': {
                    'aquifer_code': row['aquifer_code'],
                    'aquifer_type': row['aquifer_type'],
                    'area_sqkm': float(row['area_sqkm']) if row['area_sqkm'] else None
                }
            }
            features.append(feature)

        return JSONResponse({
            'type': 'FeatureCollection',
            'features': features
        })

    except Exception as e:
        logger.error(f"Error getting aquifers GeoJSON: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/piezometers")
async def get_piezometers_geojson(loader=Depends(get_data_loader)):
    """
    Get piezometer locations as GeoJSON.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        piezometers = loader.load_piezometers_geodataframe()
        _, time_series = loader.load_water_levels()

        # Get latest water level for each piezometer
        latest = time_series.groupby('piezo_id').last().reset_index()

        features = []
        for _, row in piezometers.iterrows():
            # Skip if geometry coords are invalid
            if np.isnan(row.geometry.x) or np.isnan(row.geometry.y):
                continue

            # Get latest water level
            level_data = latest[latest['piezo_id'] == row.get('piezo_id')]
            latest_level = None
            if len(level_data) > 0:
                latest_level = safe_float(level_data['water_level'].iloc[0])

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row.geometry.x, row.geometry.y]
                },
                'properties': {
                    'piezo_id': safe_str(row.get('piezo_id', '')),
                    'village': safe_str(row.get('village', '')),
                    'mandal': safe_str(row.get('mandal', '')),
                    'aquifer': safe_str(row.get('aquifer', '')),
                    'latest_water_level': latest_level,
                    'total_depth': safe_float(row.get('total_depth'))
                }
            }
            features.append(feature)

        return JSONResponse({
            'type': 'FeatureCollection',
            'features': features
        })

    except Exception as e:
        logger.error(f"Error getting piezometers GeoJSON: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wells")
async def get_wells_geojson(
    limit: int = Query(1000, ge=100, le=10000, description="Max wells to return"),
    mandal: Optional[str] = Query(None),
    loader=Depends(get_data_loader)
):
    """
    Get well locations as GeoJSON (sampled for performance).
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        wells = loader.load_wells()

        # Filter by mandal if specified
        if mandal:
            wells = wells[wells['mandal'].str.lower() == mandal.lower()]

        # Sample if too many wells
        if len(wells) > limit:
            wells = wells.sample(n=limit, random_state=42)

        features = []
        for _, row in wells.iterrows():
            if np.isnan(row['lat']) or np.isnan(row['lon']):
                continue

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row['lon'], row['lat']]
                },
                'properties': {
                    'village': row.get('village', ''),
                    'mandal': row.get('mandal', ''),
                    'well_type': row.get('well_type', ''),
                    'bore_depth': float(row['bore_depth']) if not np.isnan(row.get('bore_depth', np.nan)) else None,
                    'crop_type': row.get('crop_type', '')
                }
            }
            features.append(feature)

        return JSONResponse({
            'type': 'FeatureCollection',
            'features': features,
            'properties': {
                'total_wells': len(wells),
                'sampled': len(features)
            }
        })

    except Exception as e:
        logger.error(f"Error getting wells GeoJSON: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/villages")
async def get_villages_geojson(
    with_predictions: bool = Query(True),
    limit: int = Query(500, ge=100, le=2000),
    loader=Depends(get_data_loader),
    predictor=Depends(get_predictor)
):
    """
    Get village centroids as GeoJSON with predictions.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        village_stats = loader.aggregate_wells_by_village()

        # Limit for performance
        if len(village_stats) > limit:
            village_stats = village_stats.head(limit)

        features = []
        for _, row in village_stats.iterrows():
            if np.isnan(row['centroid_lat']) or np.isnan(row['centroid_lon']):
                continue

            properties = {
                'village_id': f"{row['district']}_{row['mandal']}_{row['village']}".replace(' ', '_'),
                'village': row['village'],
                'mandal': row['mandal'],
                'district': row['district'],
                'well_count': int(row['well_count']),
                'avg_bore_depth': float(row['avg_bore_depth']) if not np.isnan(row.get('avg_bore_depth', np.nan)) else None
            }

            # Add mock predictions
            if with_predictions:
                np.random.seed(hash(row['village']) % 10000)
                water_level = max(1, 8 + np.random.normal(0, 5))

                if water_level <= 3:
                    category = 'safe'
                elif water_level <= 8:
                    category = 'moderate'
                elif water_level <= 20:
                    category = 'stress'
                else:
                    category = 'critical'

                properties['water_level_m'] = round(water_level, 2)
                properties['category'] = category
                properties['confidence'] = round(np.random.uniform(0.6, 0.9), 2)

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row['centroid_lon'], row['centroid_lat']]
                },
                'properties': properties
            }
            features.append(feature)

        return JSONResponse({
            'type': 'FeatureCollection',
            'features': features
        })

    except Exception as e:
        logger.error(f"Error getting villages GeoJSON: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geomorphology")
async def get_geomorphology_geojson(
    limit: int = Query(200, ge=50, le=1000),
    loader=Depends(get_data_loader)
):
    """
    Get geomorphology features as GeoJSON.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        geomorph = loader.load_geomorphology()

        # Limit features
        if len(geomorph) > limit:
            geomorph = geomorph.head(limit)

        features = []
        for _, row in geomorph.iterrows():
            feature = {
                'type': 'Feature',
                'geometry': mapping(row.geometry),
                'properties': {
                    'geomorph_class': row.get('geomorph_class', ''),
                    'description': row.get('FIN_DESC', row.get('DISCRIPTIO', ''))
                }
            }
            features.append(feature)

        return JSONResponse({
            'type': 'FeatureCollection',
            'features': features
        })

    except Exception as e:
        logger.error(f"Error getting geomorphology GeoJSON: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bounds")
async def get_district_bounds(loader=Depends(get_data_loader)):
    """
    Get bounding box for Krishna District.
    """
    if loader is None:
        # Return default bounds
        return {
            'bounds': [[15.5, 80.0], [17.0, 81.5]],
            'center': [16.25, 80.75]
        }

    try:
        aquifers = loader.load_aquifers()
        bounds = aquifers.total_bounds  # [minx, miny, maxx, maxy]

        return {
            'bounds': [[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            'center': [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        }

    except Exception as e:
        logger.error(f"Error getting bounds: {e}", exc_info=True)
        return {
            'bounds': [[15.5, 80.0], [17.0, 81.5]],
            'center': [16.25, 80.75]
        }
