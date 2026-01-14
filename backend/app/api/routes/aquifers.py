"""
SmartJal Aquifers API Routes
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends

from ...models.schemas import AquiferResponse, AquiferListResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def get_data_loader():
    """Get data loader instance"""
    from ...main import app_state
    return app_state.get('data_loader')


@router.get("", response_model=AquiferListResponse)
async def list_aquifers(loader=Depends(get_data_loader)):
    """
    List all aquifers in Krishna District.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        aquifers_gdf = loader.load_aquifers()

        aquifers = []
        for _, row in aquifers_gdf.iterrows():
            aquifers.append(AquiferResponse(
                aquifer_code=row['aquifer_code'],
                aquifer_type=row['aquifer_type'],
                area_sqkm=float(row['area_sqkm']) if row['area_sqkm'] else None
            ))

        return AquiferListResponse(
            aquifers=aquifers,
            total=len(aquifers)
        )

    except Exception as e:
        logger.error(f"Error listing aquifers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{aquifer_code}", response_model=AquiferResponse)
async def get_aquifer(
    aquifer_code: str,
    loader=Depends(get_data_loader)
):
    """
    Get details for a specific aquifer.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        aquifers_gdf = loader.load_aquifers()

        match = aquifers_gdf[aquifers_gdf['aquifer_code'] == aquifer_code]

        if len(match) == 0:
            raise HTTPException(status_code=404, detail=f"Aquifer not found: {aquifer_code}")

        row = match.iloc[0]

        # Get piezometer count in this aquifer
        piezometers = loader.load_piezometers_geodataframe()
        piezo_in_aquifer = len(piezometers[piezometers.within(row.geometry)])

        return AquiferResponse(
            aquifer_code=row['aquifer_code'],
            aquifer_type=row['aquifer_type'],
            area_sqkm=float(row['area_sqkm']) if row['area_sqkm'] else None,
            piezometer_count=piezo_in_aquifer
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting aquifer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{aquifer_code}/statistics")
async def get_aquifer_statistics(
    aquifer_code: str,
    loader=Depends(get_data_loader)
):
    """
    Get water level statistics for an aquifer.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        import geopandas as gpd

        aquifers_gdf = loader.load_aquifers()
        match = aquifers_gdf[aquifers_gdf['aquifer_code'] == aquifer_code]

        if len(match) == 0:
            raise HTTPException(status_code=404, detail=f"Aquifer not found: {aquifer_code}")

        aquifer_geom = match.iloc[0].geometry

        # Get piezometers in this aquifer
        piezometers = loader.load_piezometers_geodataframe()
        piezo_in_aquifer = piezometers[piezometers.within(aquifer_geom)]

        if len(piezo_in_aquifer) == 0:
            return {
                'aquifer_code': aquifer_code,
                'piezometer_count': 0,
                'statistics': None,
                'message': 'No piezometers in this aquifer'
            }

        # Get water level data
        _, time_series = loader.load_water_levels()
        piezo_ids = piezo_in_aquifer['piezo_id'].tolist()
        aquifer_data = time_series[time_series['piezo_id'].isin(piezo_ids)]

        if len(aquifer_data) == 0:
            return {
                'aquifer_code': aquifer_code,
                'piezometer_count': len(piezo_in_aquifer),
                'statistics': None,
                'message': 'No water level data available'
            }

        # Calculate statistics
        recent_data = aquifer_data[aquifer_data['year'] >= aquifer_data['year'].max() - 2]

        return {
            'aquifer_code': aquifer_code,
            'aquifer_type': match.iloc[0]['aquifer_type'],
            'piezometer_count': len(piezo_in_aquifer),
            'statistics': {
                'mean_water_level': float(recent_data['water_level'].mean()),
                'min_water_level': float(recent_data['water_level'].min()),
                'max_water_level': float(recent_data['water_level'].max()),
                'std_water_level': float(recent_data['water_level'].std()),
                'measurement_count': len(recent_data)
            },
            'seasonal': {
                season: {
                    'mean': float(recent_data[recent_data['season'] == season]['water_level'].mean())
                    if len(recent_data[recent_data['season'] == season]) > 0 else None
                }
                for season in ['pre_monsoon', 'monsoon', 'post_monsoon']
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting aquifer statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
