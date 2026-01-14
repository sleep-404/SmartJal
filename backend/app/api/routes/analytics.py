"""
SmartJal Analytics API Routes
"""
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Depends

from ...models.schemas import DistrictSummary

logger = logging.getLogger(__name__)
router = APIRouter()


def get_data_loader():
    """Get data loader instance"""
    from ...main import app_state
    return app_state.get('data_loader')


@router.get("/summary", response_model=DistrictSummary)
async def get_district_summary(loader=Depends(get_data_loader)):
    """
    Get summary statistics for Krishna District.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        # Get village count from wells
        village_stats = loader.aggregate_wells_by_village()
        total_villages = len(village_stats)

        # Get piezometer data
        metadata, time_series = loader.load_water_levels()
        piezometer_count = len(metadata)

        # Get aquifer count
        aquifers = loader.load_aquifers()
        aquifer_count = len(aquifers)

        # Calculate water level statistics from recent data
        recent_year = time_series['year'].max()
        recent_data = time_series[time_series['year'] == recent_year]

        avg_level = recent_data['water_level'].mean()

        # Count villages by category (approximate from piezometer data)
        def categorize(level):
            if level <= 3:
                return 'safe'
            elif level <= 8:
                return 'moderate'
            elif level <= 20:
                return 'stress'
            else:
                return 'critical'

        piezo_avg = recent_data.groupby('piezo_id')['water_level'].mean()
        categories = piezo_avg.apply(categorize).value_counts()

        # Scale to village count (rough approximation)
        scale_factor = total_villages / len(piezo_avg) if len(piezo_avg) > 0 else 1

        return DistrictSummary(
            total_villages=total_villages,
            villages_with_predictions=piezometer_count,  # approximation
            avg_water_level=float(avg_level) if not np.isnan(avg_level) else 0,
            critical_villages=int(categories.get('critical', 0) * scale_factor),
            stress_villages=int(categories.get('stress', 0) * scale_factor),
            moderate_villages=int(categories.get('moderate', 0) * scale_factor),
            safe_villages=int(categories.get('safe', 0) * scale_factor),
            piezometer_count=piezometer_count,
            aquifer_count=aquifer_count
        )

    except Exception as e:
        logger.error(f"Error getting summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_trends(
    aquifer_code: Optional[str] = Query(None),
    years: int = Query(5, ge=1, le=20),
    loader=Depends(get_data_loader)
):
    """
    Get water level trends over time.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        _, time_series = loader.load_water_levels()

        # Filter by year range
        max_year = time_series['year'].max()
        min_year = max_year - years
        filtered = time_series[time_series['year'] >= min_year]

        # Group by year and month
        monthly_avg = filtered.groupby(['year', 'month']).agg({
            'water_level': 'mean'
        }).reset_index()

        # Calculate trend
        if len(monthly_avg) > 1:
            x = np.arange(len(monthly_avg))
            y = monthly_avg['water_level'].values
            slope, _ = np.polyfit(x, y, 1)
            change_rate = slope * 12  # per year

            if change_rate > 0.5:
                trend = 'declining'
            elif change_rate < -0.5:
                trend = 'improving'
            else:
                trend = 'stable'
        else:
            change_rate = 0
            trend = 'stable'

        return {
            'data': [
                {
                    'year': int(row['year']),
                    'month': int(row['month']),
                    'avg_water_level': float(row['water_level'])
                }
                for _, row in monthly_avg.iterrows()
            ],
            'trend_direction': trend,
            'change_rate_per_year': float(change_rate)
        }

    except Exception as e:
        logger.error(f"Error getting trends: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/seasonal")
async def get_seasonal_analysis(
    year: Optional[int] = Query(None),
    loader=Depends(get_data_loader)
):
    """
    Get seasonal water level analysis.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        _, time_series = loader.load_water_levels()

        if year is None:
            year = time_series['year'].max()

        year_data = time_series[time_series['year'] == year]

        seasonal_stats = {}
        for season in ['pre_monsoon', 'monsoon', 'post_monsoon']:
            season_data = year_data[year_data['season'] == season]
            if len(season_data) > 0:
                seasonal_stats[season] = {
                    'mean': float(season_data['water_level'].mean()),
                    'min': float(season_data['water_level'].min()),
                    'max': float(season_data['water_level'].max()),
                    'std': float(season_data['water_level'].std()),
                    'count': len(season_data)
                }
            else:
                seasonal_stats[season] = None

        # Calculate monsoon recharge effect
        pre = seasonal_stats.get('pre_monsoon', {})
        post = seasonal_stats.get('post_monsoon', {})

        recharge_effect = None
        if pre and post and pre.get('mean') and post.get('mean'):
            recharge_effect = pre['mean'] - post['mean']

        return {
            'year': year,
            'seasonal_statistics': seasonal_stats,
            'monsoon_recharge_effect': recharge_effect,
            'interpretation': 'Positive recharge effect indicates groundwater recovery during monsoon' if recharge_effect and recharge_effect > 0 else 'Insufficient recharge during monsoon'
        }

    except Exception as e:
        logger.error(f"Error getting seasonal analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/aquifer-comparison")
async def compare_aquifers(loader=Depends(get_data_loader)):
    """
    Compare water level statistics across aquifer types.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        import geopandas as gpd

        aquifers = loader.load_aquifers()
        piezometers = loader.load_piezometers_geodataframe()
        metadata, time_series = loader.load_water_levels()

        comparison = []

        for _, aquifer in aquifers.iterrows():
            # Find piezometers in this aquifer
            piezo_in_aquifer = piezometers[piezometers.within(aquifer.geometry)]

            if len(piezo_in_aquifer) == 0:
                comparison.append({
                    'aquifer_code': aquifer['aquifer_code'],
                    'aquifer_type': aquifer['aquifer_type'],
                    'piezometer_count': 0,
                    'statistics': None
                })
                continue

            # Get water level data
            piezo_ids = piezo_in_aquifer['piezo_id'].tolist()
            aquifer_data = time_series[time_series['piezo_id'].isin(piezo_ids)]

            if len(aquifer_data) == 0:
                comparison.append({
                    'aquifer_code': aquifer['aquifer_code'],
                    'aquifer_type': aquifer['aquifer_type'],
                    'piezometer_count': len(piezo_in_aquifer),
                    'statistics': None
                })
                continue

            # Get recent data
            recent = aquifer_data[aquifer_data['year'] >= aquifer_data['year'].max() - 1]

            comparison.append({
                'aquifer_code': aquifer['aquifer_code'],
                'aquifer_type': aquifer['aquifer_type'],
                'piezometer_count': len(piezo_in_aquifer),
                'statistics': {
                    'mean_water_level': float(recent['water_level'].mean()),
                    'min_water_level': float(recent['water_level'].min()),
                    'max_water_level': float(recent['water_level'].max()),
                    'variability': float(recent['water_level'].std())
                }
            })

        return {
            'comparison': comparison,
            'total_aquifers': len(aquifers)
        }

    except Exception as e:
        logger.error(f"Error comparing aquifers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
