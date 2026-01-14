#!/usr/bin/env python3
"""
Smart Jal - Physics-Informed Model
Implements the water balance equation from WINNING_ARCHITECTURE.md

Key Equations:
- ΔStorage = Recharge - Extraction ± Lateral_Flow
- Recharge = Rainfall × Recharge_Factor
- Recharge_Factor = f(soil_infiltration, slope, land_use)
- Water_Level(t) = Baseline + Seasonal(t) + Trend(t) + Anomaly(t)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class WaterBalanceModel:
    """
    Physics-informed model based on water balance equation.

    This is the core innovation from our research:
    Instead of just pattern matching, we model the actual physics.
    """

    # Aquifer-specific parameters (from hydrogeology research)
    AQUIFER_PARAMS = {
        'Alluvium': {'specific_yield': 0.20, 'recharge_factor': 0.25, 'K': 50},
        'Granite': {'specific_yield': 0.02, 'recharge_factor': 0.08, 'K': 1},
        'Banded Gneissic Granites': {'specific_yield': 0.02, 'recharge_factor': 0.08, 'K': 1},
        'Basalt': {'specific_yield': 0.03, 'recharge_factor': 0.10, 'K': 2},
        'Khondalites': {'specific_yield': 0.02, 'recharge_factor': 0.07, 'K': 0.5},
        'Charnokites': {'specific_yield': 0.015, 'recharge_factor': 0.05, 'K': 0.3},
        'Quartzite': {'specific_yield': 0.01, 'recharge_factor': 0.03, 'K': 0.1},
        'Sand Stones': {'specific_yield': 0.15, 'recharge_factor': 0.15, 'K': 10},
        'Shales': {'specific_yield': 0.01, 'recharge_factor': 0.02, 'K': 0.01},
        'Lime Stones': {'specific_yield': 0.10, 'recharge_factor': 0.20, 'K': 20},
    }

    # Soil infiltration factors
    INFILTRATION_FACTORS = {
        'high': 0.30,
        'moderate': 0.15,
        'low': 0.05
    }

    def __init__(self):
        self.baseline_levels = {}
        self.calibrated_params = {}

    def calculate_recharge(self,
                          rainfall_mm: float,
                          aquifer_type: str,
                          infiltration_class: str,
                          slope_deg: float,
                          area_km2: float) -> float:
        """
        Calculate groundwater recharge using physics.

        Recharge = Rainfall × Recharge_Factor × Soil_Factor × Slope_Factor × Area

        Args:
            rainfall_mm: Monthly rainfall in mm
            aquifer_type: Type of aquifer
            infiltration_class: Soil infiltration class
            slope_deg: Average slope in degrees
            area_km2: Village area in km²

        Returns:
            Recharge in ha.m (hectare-meters)
        """
        # Base recharge factor from aquifer type
        params = self.AQUIFER_PARAMS.get(aquifer_type, {'recharge_factor': 0.10})
        base_factor = params['recharge_factor']

        # Soil infiltration modifier
        soil_factor = self.INFILTRATION_FACTORS.get(infiltration_class, 0.15)

        # Slope modifier (steeper = less infiltration, more runoff)
        slope_factor = max(0.1, 1.0 - (slope_deg / 30.0))

        # Combined recharge factor
        recharge_factor = base_factor * (1 + soil_factor) * slope_factor

        # IMPORTANT: Clip area to minimum to handle edge cases
        area_km2_clipped = max(0.1, area_km2)

        # Convert rainfall (mm) to volume (ha.m)
        # rainfall_mm over area_km2 = rainfall_mm * area_km2 * 100 ha/km² / 1000 mm/m
        rainfall_volume = rainfall_mm * area_km2_clipped * 100 / 1000  # ha.m

        # Recharge volume
        recharge = rainfall_volume * recharge_factor

        return recharge

    def calculate_extraction(self,
                            n_wells: int,
                            avg_depth: float,
                            aquifer_type: str) -> float:
        """
        Calculate groundwater extraction.

        Extraction = n_wells × draft_per_well × operating_factor

        Based on official guidance from Data Issues_Clarifications.docx
        """
        # Base draft per well (ha.m per month)
        # Deeper wells and higher-yielding aquifers = more extraction
        params = self.AQUIFER_PARAMS.get(aquifer_type, {'specific_yield': 0.05})

        # Draft depends on aquifer yield and well depth
        base_draft = 0.02  # ha.m per well per month (baseline)
        depth_factor = min(2.0, avg_depth / 50.0)  # Deeper wells pump more
        yield_factor = params['specific_yield'] / 0.10  # Normalize to alluvium

        draft_per_well = base_draft * depth_factor * yield_factor

        # Total extraction
        extraction = n_wells * draft_per_well

        return extraction

    def calculate_storage_change(self,
                                 recharge: float,
                                 extraction: float,
                                 area_km2: float,
                                 specific_yield: float) -> float:
        """
        Calculate water level change from water balance.

        ΔStorage = Recharge - Extraction
        ΔLevel = ΔStorage / (Area × Specific_Yield)

        Returns:
            Change in water level (meters, positive = rising)
        """
        # Net storage change (ha.m)
        delta_storage = recharge - extraction

        # Convert to water level change
        # Area in ha = area_km2 * 100
        # IMPORTANT: Clip to minimum area to avoid division by tiny numbers
        area_km2_clipped = max(0.1, area_km2)  # Minimum 0.1 km² (10 ha)
        area_ha = area_km2_clipped * 100

        # Level change = volume change / (area × specific yield)
        if area_ha > 0 and specific_yield > 0:
            delta_level = delta_storage / (area_ha * specific_yield)
        else:
            delta_level = 0

        # Cap the level change to reasonable bounds (±10m per month)
        delta_level = max(-10.0, min(10.0, delta_level))

        return delta_level

    def predict_water_level(self,
                           baseline: float,
                           seasonal: float,
                           trend: float,
                           delta_storage: float) -> float:
        """
        Predict water level using temporal decomposition.

        Water_Level(t) = Baseline + Seasonal(t) + Trend(t) + ΔStorage(t)

        Args:
            baseline: Long-term average level
            seasonal: Seasonal component for this month
            trend: Trend component
            delta_storage: Storage change from water balance

        Returns:
            Predicted water level (meters below ground)
        """
        # Note: Higher values = deeper water = more stress
        # Positive delta_storage (net recharge) = water rises = level decreases
        predicted = baseline + seasonal + trend - delta_storage

        return max(0, predicted)  # Can't be negative (above ground)


class TemporalDecomposer:
    """
    Decompose water level time series into components.

    From research: Water_Level(t) = Baseline + Seasonal(t) + Trend(t) + Anomaly(t)
    """

    def __init__(self):
        self.components = {}

    def fit_piezometer(self,
                       piezo_id: str,
                       dates: pd.Series,
                       levels: pd.Series) -> Dict:
        """
        Fit temporal decomposition for a single piezometer.

        Uses 28 years of monthly data to extract:
        - Baseline: Long-term mean
        - Seasonal: Monthly average deviations (monsoon cycle)
        - Trend: Linear trend over time
        - Anomaly: Residuals
        """
        # Clean data
        df = pd.DataFrame({'date': dates, 'level': levels})
        df = df.dropna()

        if len(df) < 24:  # Need at least 2 years
            return None

        # Baseline: overall mean
        baseline = df['level'].mean()

        # Seasonal: average for each month
        df['month'] = df['date'].dt.month
        monthly_avg = df.groupby('month')['level'].mean()
        seasonal = (monthly_avg - baseline).to_dict()

        # Trend: linear regression over time
        df['time_idx'] = (df['date'] - df['date'].min()).dt.days / 365.25
        if len(df) > 2:
            slope, intercept = np.polyfit(df['time_idx'], df['level'], 1)
            trend_per_year = slope
        else:
            trend_per_year = 0

        # Anomaly std for uncertainty
        df['expected'] = baseline + df['month'].map(seasonal).fillna(0)
        df['anomaly'] = df['level'] - df['expected']
        anomaly_std = df['anomaly'].std()

        result = {
            'baseline': baseline,
            'seasonal': seasonal,
            'trend_per_year': trend_per_year,
            'anomaly_std': anomaly_std,
            'n_observations': len(df),
            'date_range': (df['date'].min(), df['date'].max())
        }

        self.components[piezo_id] = result
        return result

    def fit_all(self, water_levels: pd.DataFrame) -> 'TemporalDecomposer':
        """
        Fit decomposition for all piezometers.
        """
        print("Fitting temporal decomposition for all piezometers...")

        id_col = 'piezo_id' if 'piezo_id' in water_levels.columns else 'sno'

        fitted = 0
        for piezo in water_levels[id_col].unique():
            piezo_data = water_levels[water_levels[id_col] == piezo]
            result = self.fit_piezometer(
                piezo,
                piezo_data['date'],
                piezo_data['water_level']
            )
            if result is not None:
                fitted += 1

        print(f"  Fitted {fitted} piezometers with temporal decomposition")
        return self

    def get_components(self, piezo_id: str, target_month: int) -> Dict:
        """
        Get temporal components for prediction.
        """
        if piezo_id not in self.components:
            return {'baseline': 10.0, 'seasonal': 0, 'trend': 0, 'anomaly_std': 3.0}

        comp = self.components[piezo_id]
        return {
            'baseline': comp['baseline'],
            'seasonal': comp['seasonal'].get(target_month, 0),
            'trend': comp['trend_per_year'],
            'anomaly_std': comp['anomaly_std']
        }


class PhysicsInformedEnsemble:
    """
    Complete physics-informed prediction system.

    Combines:
    1. Water balance model (recharge - extraction)
    2. Temporal decomposition (baseline + seasonal + trend)
    3. Spatial interpolation (aquifer-stratified)
    4. GRACE constraint (regional validation)
    """

    def __init__(self):
        self.water_balance = WaterBalanceModel()
        self.temporal = TemporalDecomposer()
        self.piezometer_components = {}

    def fit(self,
            water_levels: pd.DataFrame,
            piezometers: gpd.GeoDataFrame) -> 'PhysicsInformedEnsemble':
        """
        Fit the physics-informed model.
        """
        print("=" * 60)
        print("Fitting Physics-Informed Model")
        print("=" * 60)

        # 1. Fit temporal decomposition for each piezometer
        print("\n[1/2] Fitting temporal decomposition (28 years of data)...")
        self.temporal.fit_all(water_levels)

        # 2. Store piezometer metadata
        print("\n[2/2] Storing piezometer components...")
        id_col = 'piezo_id' if 'piezo_id' in piezometers.columns else 'sno'

        for _, row in piezometers.iterrows():
            piezo_id = row[id_col]
            if piezo_id in self.temporal.components:
                comp = self.temporal.components[piezo_id]
                self.piezometer_components[piezo_id] = {
                    'lat': row.geometry.centroid.y,
                    'lon': row.geometry.centroid.x,
                    'aquifer': row.get('geo_class', 'Unknown'),
                    **comp
                }

        print(f"  {len(self.piezometer_components)} piezometers ready for prediction")
        print("=" * 60)

        return self

    def predict_village(self,
                       village: pd.Series,
                       target_month: int,
                       target_year: int) -> Dict:
        """
        Predict water level for a single village using physics.
        """
        # Get village properties
        aquifer = village.get('geo_class', 'Alluvium')
        infiltration = village.get('infiltration_class', 'moderate')
        slope = village.get('slope_mean', 2.0)
        area_km2 = village.get('area_km2', 10.0)
        n_wells = village.get('n_wells', 0)
        avg_depth = village.get('avg_well_depth', 50)
        rainfall = village.get('rainfall_cumulative_3m', 200)

        # Get aquifer parameters
        params = self.water_balance.AQUIFER_PARAMS.get(
            aquifer,
            {'specific_yield': 0.05, 'recharge_factor': 0.10}
        )

        # 1. Calculate recharge from rainfall
        recharge = self.water_balance.calculate_recharge(
            rainfall, aquifer, infiltration, slope, area_km2
        )

        # 2. Calculate extraction from wells
        extraction = self.water_balance.calculate_extraction(
            n_wells, avg_depth, aquifer
        )

        # 3. Calculate storage change
        delta_level = self.water_balance.calculate_storage_change(
            recharge, extraction, area_km2, params['specific_yield']
        )

        # 4. Find nearest piezometer in same aquifer for temporal component
        village_lat = village.get('centroid_lat', 16.0)
        village_lon = village.get('centroid_lon', 80.5)

        nearest_comp = self._find_nearest_piezometer(
            village_lat, village_lon, aquifer, target_month
        )

        # 5. Combine: predicted = baseline + seasonal + trend + physics_adjustment
        baseline = nearest_comp['baseline']
        seasonal = nearest_comp['seasonal']
        trend_adjustment = nearest_comp['trend'] * (target_year - 2020)  # Trend from reference year

        # Physics adjustment: storage change affects level
        physics_adjustment = -delta_level  # Negative because recharge raises water (reduces depth)

        predicted = baseline + seasonal + trend_adjustment + physics_adjustment

        # Uncertainty from anomaly std + distance uncertainty
        uncertainty = nearest_comp['anomaly_std'] * (1 + nearest_comp.get('distance_factor', 0.5))

        return {
            'prediction': max(0, predicted),
            'uncertainty': uncertainty,
            'baseline': baseline,
            'seasonal': seasonal,
            'trend': trend_adjustment,
            'physics_adjustment': physics_adjustment,
            'recharge_ham': recharge,
            'extraction_ham': extraction,
            'net_flux_ham': recharge - extraction
        }

    def _find_nearest_piezometer(self,
                                 lat: float,
                                 lon: float,
                                 aquifer: str,
                                 target_month: int) -> Dict:
        """
        Find nearest piezometer in same aquifer and get its temporal components.
        """
        min_dist = float('inf')
        nearest = None

        for piezo_id, comp in self.piezometer_components.items():
            # Prefer same aquifer
            if comp['aquifer'] == aquifer:
                dist = np.sqrt((lat - comp['lat'])**2 + (lon - comp['lon'])**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = comp

        # Fallback to any aquifer if none found
        if nearest is None:
            for piezo_id, comp in self.piezometer_components.items():
                dist = np.sqrt((lat - comp['lat'])**2 + (lon - comp['lon'])**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = comp

        if nearest is None:
            return {
                'baseline': 10.0,
                'seasonal': 0,
                'trend': 0,
                'anomaly_std': 5.0,
                'distance_factor': 1.0
            }

        seasonal = nearest['seasonal'].get(target_month, 0) if isinstance(nearest['seasonal'], dict) else 0

        return {
            'baseline': nearest['baseline'],
            'seasonal': seasonal,
            'trend': nearest['trend_per_year'],
            'anomaly_std': nearest['anomaly_std'],
            'distance_factor': min(1.0, min_dist / 0.5)  # Increases with distance
        }

    def predict_all_villages(self,
                            villages: gpd.GeoDataFrame,
                            target_date: pd.Timestamp) -> pd.DataFrame:
        """
        Predict water levels for all villages.
        """
        print(f"\nPredicting for {len(villages)} villages using physics model...")

        target_month = target_date.month
        target_year = target_date.year

        results = []
        for idx, village in villages.iterrows():
            pred = self.predict_village(village, target_month, target_year)

            results.append({
                'village': village.get('village', idx),
                'prediction': pred['prediction'],
                'uncertainty': pred['uncertainty'],
                'baseline': pred['baseline'],
                'seasonal': pred['seasonal'],
                'trend': pred['trend'],
                'physics_adjustment': pred['physics_adjustment'],
                'recharge_ham': pred['recharge_ham'],
                'extraction_ham': pred['extraction_ham'],
                'net_flux_ham': pred['net_flux_ham']
            })

        df = pd.DataFrame(results)

        print(f"  Mean prediction: {df['prediction'].mean():.2f}m")
        print(f"  Mean uncertainty: {df['uncertainty'].mean():.2f}m")
        print(f"  Net flux range: {df['net_flux_ham'].min():.3f} to {df['net_flux_ham'].max():.3f} ha.m")

        return df


if __name__ == '__main__':
    print("Testing Physics-Informed Model...")

    # Test water balance calculation
    wb = WaterBalanceModel()

    # Test village
    recharge = wb.calculate_recharge(
        rainfall_mm=200,
        aquifer_type='Alluvium',
        infiltration_class='high',
        slope_deg=2,
        area_km2=20
    )
    print(f"Recharge for 200mm rain over 20km² alluvium: {recharge:.3f} ha.m")

    extraction = wb.calculate_extraction(
        n_wells=100,
        avg_depth=60,
        aquifer_type='Alluvium'
    )
    print(f"Extraction for 100 wells: {extraction:.3f} ha.m")

    delta = wb.calculate_storage_change(
        recharge, extraction,
        area_km2=20,
        specific_yield=0.20
    )
    print(f"Level change: {delta:.3f}m")
