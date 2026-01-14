"""
SmartJal Villages API Routes
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Depends

from ...models.schemas import VillageResponse, VillageListResponse, Coordinates
from ...config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def get_data_loader():
    """Get data loader instance"""
    from ...main import app_state
    return app_state.get('data_loader')


@router.get("", response_model=VillageListResponse)
async def list_villages(
    mandal: Optional[str] = Query(None, description="Filter by mandal name"),
    aquifer: Optional[str] = Query(None, description="Filter by aquifer code"),
    category: Optional[str] = Query(None, description="Filter by depth category"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    loader=Depends(get_data_loader)
):
    """
    List villages in Krishna District.

    Villages are derived from well locations aggregated by village name.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        # Get village aggregates from wells
        village_stats = loader.aggregate_wells_by_village()

        # Apply filters
        if mandal:
            village_stats = village_stats[
                village_stats['mandal'].str.lower() == mandal.lower()
            ]

        # Create response
        villages = []
        for idx, row in village_stats.iloc[offset:offset+limit].iterrows():
            villages.append(VillageResponse(
                village_id=f"{row['district']}_{row['mandal']}_{row['village']}".replace(' ', '_'),
                village_name=row['village'],
                mandal=row['mandal'],
                district=row['district'],
                coordinates=Coordinates(
                    lon=row['centroid_lon'],
                    lat=row['centroid_lat']
                )
            ))

        return VillageListResponse(
            villages=villages,
            total=len(village_stats)
        )

    except Exception as e:
        logger.error(f"Error listing villages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{village_id}", response_model=VillageResponse)
async def get_village(
    village_id: str,
    loader=Depends(get_data_loader)
):
    """
    Get details for a specific village.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        # Parse village ID
        parts = village_id.replace('_', ' ').split(' ', 2)
        if len(parts) < 3:
            raise HTTPException(status_code=400, detail="Invalid village ID format")

        district, mandal, village = parts[0], parts[1], ' '.join(parts[2:])

        # Get village data
        village_stats = loader.aggregate_wells_by_village()
        match = village_stats[
            (village_stats['district'].str.lower() == district.lower()) &
            (village_stats['mandal'].str.lower() == mandal.lower()) &
            (village_stats['village'].str.lower() == village.lower())
        ]

        if len(match) == 0:
            raise HTTPException(status_code=404, detail=f"Village not found: {village_id}")

        row = match.iloc[0]

        return VillageResponse(
            village_id=village_id,
            village_name=row['village'],
            mandal=row['mandal'],
            district=row['district'],
            coordinates=Coordinates(
                lon=row['centroid_lon'],
                lat=row['centroid_lat']
            ),
            features={
                'well_count': int(row['well_count']),
                'avg_bore_depth': float(row['avg_bore_depth']) if row['avg_bore_depth'] else None,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting village: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mandals/list")
async def list_mandals(loader=Depends(get_data_loader)):
    """
    Get list of all mandals in Krishna District.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        wells = loader.load_wells()
        mandals = wells['mandal'].dropna().unique().tolist()
        mandals.sort()

        return {
            'mandals': mandals,
            'total': len(mandals)
        }

    except Exception as e:
        logger.error(f"Error listing mandals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
