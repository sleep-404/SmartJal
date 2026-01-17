# Smart Jal - Implementation Progress Tracker

**Last Updated:** 2026-01-17
**Status:** Planning Phase

---

## 1. Problem Statement

### 1.1 The Core Challenge
Predict groundwater levels for **939 villages** in Krishna District, Andhra Pradesh, using data from only **138 piezometers** (monitoring wells).

### 1.2 Why This Is Hard
- **No ground truth**: We have water level readings at 138 points, but need predictions for 939 villages. There's no way to directly validate predictions for the ~800 villages without sensors.
- **Sparse coverage**: 138 piezometers across ~7,000 km² means roughly 1 sensor per 50 km².
- **Geological heterogeneity**: Different aquifer types (granite, basalt, alluvium) behave differently. Simple distance-based interpolation ignores this.

### 1.3 What Success Looks Like
1. Predictions for all 939 villages with uncertainty estimates
2. Predictions that are **physically consistent** (respect water balance)
3. Predictions that are **geologically coherent** (respect aquifer boundaries)
4. A way to **validate** predictions despite lacking ground truth at most locations

---

## 2. Data Available

### 2.1 Hackathon-Provided Data
| Dataset | Records | Description |
|---------|---------|-------------|
| Piezometer readings | 138 stations × 28 years | Monthly water level measurements |
| Village boundaries | 939 polygons | Administrative boundaries with centroids |
| Mandal boundaries | 42 polygons | Sub-district administrative units |
| Aquifer boundaries | 8 zones | Geological aquifer types |
| Bore wells | ~89,000 records | Location, depth, status, crop type |
| Soils | 151 polygons | Soil classification |
| Geomorphology | 614 polygons | Landform types |
| LULC | Polygons | Land use/land cover (2005) |
| Pumping data | 714 villages | Extraction rates by season |

### 2.2 External Data (Downloaded)
| Dataset | Source | Resolution | Use |
|---------|--------|------------|-----|
| GRACE TWS | NASA | ~50 km, monthly | Regional groundwater constraint |
| CHIRPS Rainfall | UCSB | ~5 km, monthly | Recharge estimation |
| DEM | SRTM | 30m | Terrain analysis |

### 2.3 Additional Data to Consider (from Research)
| Dataset | Source | Resolution | Use | Status |
|---------|--------|------------|-----|--------|
| MODIS NDVI | NASA | 250m | Vegetation stress proxy | Not downloaded |
| MODIS LST | NASA | 1km | Thermal anomaly proxy | Not downloaded |
| MODIS ET | NASA | 500m | Evapotranspiration | Not downloaded |
| ESA WorldCover | ESA | 10m | Updated LULC (2021) | Not downloaded |

---

## 3. Solution Architecture

### 3.1 Core Insight
The research findings identified that the key challenge is **validation without ground truth**. The solution is to use **multiple independent signals** that constrain predictions:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-SIGNAL ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CONSTRAINT 1: GRACE Satellite (Regional)                       │
│  ├── Provides total groundwater storage at ~50km scale          │
│  └── Village predictions must sum to GRACE regional value       │
│                                                                 │
│  CONSTRAINT 2: Physics (Water Balance)                          │
│  ├── ΔStorage = Recharge - Extraction                           │
│  ├── Recharge = f(Rainfall, Soil, Slope)                        │
│  └── Extraction = f(Wells, Pumping, Crops)                      │
│                                                                 │
│  CONSTRAINT 3: Geology (Aquifer Boundaries)                     │
│  ├── Interpolation respects aquifer boundaries                  │
│  └── Different models per aquifer type                          │
│                                                                 │
│  SIGNAL 1: Piezometer Readings (Direct)                         │
│  ├── 138 stations with known water levels                       │
│  └── Primary training data                                      │
│                                                                 │
│  SIGNAL 2: Vegetation Stress (Indirect)                         │
│  ├── NDVI anomalies indicate water availability                 │
│  └── Independent validation signal                              │
│                                                                 │
│  SIGNAL 3: Land Surface Temperature (Indirect)                  │
│  ├── Thermal anomalies correlate with groundwater depth         │
│  └── Independent validation signal                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Implementation Approach

Based on research findings, we will implement in this order:

#### Phase 1: Data Foundation
1. Load and validate all hackathon data
2. Download and process GRACE satellite data properly
3. Download NDVI/LST data for multi-signal validation
4. Create unified village feature dataset

#### Phase 2: Baseline Model (Spatial Interpolation)
1. Implement aquifer-stratified kriging
2. Validate spatial autocorrelation within aquifers
3. Generate baseline predictions with uncertainty

#### Phase 3: Physics Constraints
1. Implement water balance equation
2. Calculate recharge from rainfall + soil + slope
3. Calculate extraction from wells + pumping data
4. Constrain predictions to satisfy water balance

#### Phase 4: GRACE Regional Constraint
1. Process GRACE data to extract groundwater storage
2. Implement constraint: village predictions sum to GRACE
3. Adjust predictions to match regional satellite observation

#### Phase 5: Multi-Signal Validation
1. Extract NDVI anomalies per village
2. Extract LST anomalies per village
3. Cross-validate: villages with low NDVI should have deeper water
4. Use as independent validation (not training) signal

#### Phase 6: Model Selection
Choose ONE of these based on time/complexity:
- **Option A**: Aquifer-stratified kriging + physics + GRACE (simpler, robust)
- **Option B**: Graph Neural Network with geology edges (novel, higher risk)
- **Option C**: Physics-Informed Neural Network (novel, medium risk)

#### Phase 7: Explainability & Dashboard
1. Generate explanations for each prediction
2. Build simple dashboard showing predictions + uncertainty
3. Implement active learning: which villages to validate first

---

## 4. Technical Specifications

### 4.1 Key Equations

**Water Balance:**
```
ΔS = R - E ± L

Where:
  ΔS = Change in groundwater storage
  R  = Recharge = Rainfall × Infiltration_Factor × (1 - Runoff_Factor)
  E  = Extraction = n_wells × unit_draft / area
  L  = Lateral flow (usually small, can ignore for village scale)
```

**GRACE Constraint:**
```
Σ(village_prediction × village_area) ≈ GRACE_regional_value × total_area
```

**Aquifer-Stratified Interpolation:**
```
For village v in aquifer A:
  prediction(v) = Σ w_i × observation(p_i)  for all piezometers p_i in A

Where:
  w_i = 1/d(v, p_i)^2 / Σ(1/d(v, p_j)^2)  (IDW weights)
  d(v, p) = distance between village v and piezometer p
```

### 4.2 Validation Strategy

Since we can't validate at village level directly:

1. **Cross-validation on piezometers**: Leave-one-out CV on 138 known points
2. **GRACE consistency**: Check if village sums match regional satellite
3. **Physical plausibility**: Check water balance is satisfied
4. **Multi-signal correlation**: Villages with vegetation stress should have deeper water
5. **Active learning**: Recommend which villages to field-validate first

### 4.3 Output Format

For each village:
```json
{
  "village_id": "KRI_VIL_001",
  "village_name": "Gudivada",
  "mandal": "Gudivada",
  "aquifer_type": "Alluvium",
  "prediction": {
    "water_level_m": 14.2,
    "uncertainty_m": 2.1,
    "confidence": "medium"
  },
  "influences": [
    {"piezometer": "P047", "distance_km": 2.3, "weight": 0.45},
    {"piezometer": "P032", "distance_km": 5.1, "weight": 0.28}
  ],
  "risk_tier": "moderate",
  "validation_priority": 7
}
```

---

## 5. Implementation Progress

### 5.1 Progress Summary

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Data Foundation | Not Started | 0% |
| Phase 2: Baseline Model | Not Started | 0% |
| Phase 3: Physics Constraints | Not Started | 0% |
| Phase 4: GRACE Constraint | Not Started | 0% |
| Phase 5: Multi-Signal Validation | Not Started | 0% |
| Phase 6: Model Selection | Not Started | 0% |
| Phase 7: Dashboard | Not Started | 0% |

**Overall Progress: 0%**

### 5.2 Detailed Task Tracking

#### Phase 1: Data Foundation
- [ ] 1.1 Create data loading module for hackathon data
  - [ ] Load piezometer locations and metadata
  - [ ] Load water level time series
  - [ ] Load village boundaries
  - [ ] Load aquifer boundaries
  - [ ] Load soil data
  - [ ] Load bore well data
  - [ ] Load pumping data
- [ ] 1.2 Process GRACE satellite data
  - [ ] Download GRACE mascon data for Krishna region
  - [ ] Extract groundwater storage anomaly time series
  - [ ] Align temporal resolution with piezometer data
- [ ] 1.3 Download and process multi-signal data
  - [ ] Download MODIS NDVI for Krishna district
  - [ ] Download MODIS LST for Krishna district
  - [ ] Calculate per-village statistics
- [ ] 1.4 Create unified feature dataset
  - [ ] Spatial join: village ↔ aquifer
  - [ ] Spatial join: village ↔ soil
  - [ ] Calculate terrain features from DEM
  - [ ] Calculate well density per village
  - [ ] Calculate pumping intensity per village

#### Phase 2: Baseline Model
- [ ] 2.1 Exploratory spatial analysis
  - [ ] Calculate spatial autocorrelation (Moran's I)
  - [ ] Fit variograms per aquifer type
  - [ ] Visualize piezometer distribution
- [ ] 2.2 Implement aquifer-stratified kriging
  - [ ] Group piezometers by aquifer
  - [ ] Fit kriging model per aquifer
  - [ ] Handle aquifers with few piezometers (fallback to IDW)
- [ ] 2.3 Generate baseline predictions
  - [ ] Predict for all village centroids
  - [ ] Calculate prediction uncertainty
  - [ ] Validate with leave-one-out CV

#### Phase 3: Physics Constraints
- [ ] 3.1 Implement recharge calculation
  - [ ] Extract rainfall per village from CHIRPS
  - [ ] Apply soil infiltration factors
  - [ ] Apply slope/runoff factors
- [ ] 3.2 Implement extraction calculation
  - [ ] Use pumping data where available
  - [ ] Estimate from bore well density elsewhere
- [ ] 3.3 Apply water balance constraint
  - [ ] Calculate expected ΔStorage per village
  - [ ] Adjust predictions to satisfy physics

#### Phase 4: GRACE Regional Constraint
- [ ] 4.1 Process GRACE to groundwater storage
  - [ ] Download soil moisture (GLDAS)
  - [ ] Calculate: GW = TWS - Soil_Moisture
- [ ] 4.2 Implement regional constraint
  - [ ] Calculate area-weighted sum of predictions
  - [ ] Adjust to match GRACE observation
  - [ ] Propagate constraint to village level

#### Phase 5: Multi-Signal Validation
- [ ] 5.1 Extract vegetation signals
  - [ ] Calculate NDVI anomaly per village
  - [ ] Calculate VCI (Vegetation Condition Index)
- [ ] 5.2 Extract thermal signals
  - [ ] Calculate LST anomaly per village
- [ ] 5.3 Cross-validate predictions
  - [ ] Correlate predictions with NDVI (expect negative)
  - [ ] Correlate predictions with LST (expect positive)
  - [ ] Flag inconsistent villages

#### Phase 6: Model Selection & Training
- [ ] 6.1 Evaluate baseline performance
  - [ ] RMSE on leave-one-out CV
  - [ ] Physical consistency score
  - [ ] GRACE consistency score
- [ ] 6.2 Decide on advanced model (if time permits)
  - [ ] Option A: Enhanced kriging with features
  - [ ] Option B: GNN with geology edges
  - [ ] Option C: Physics-informed neural network
- [ ] 6.3 Train final model
  - [ ] Implement chosen approach
  - [ ] Validate performance
  - [ ] Generate final predictions

#### Phase 7: Explainability & Dashboard
- [ ] 7.1 Generate explanations
  - [ ] For each prediction, list top influencing piezometers
  - [ ] Show uncertainty sources
  - [ ] Calculate validation priority score
- [ ] 7.2 Build dashboard
  - [ ] Map view with village predictions
  - [ ] Risk classification display
  - [ ] Drill-down to village details
- [ ] 7.3 Active learning output
  - [ ] Rank villages by information gain
  - [ ] Generate recommended validation sequence

---

## 6. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-17 | Clean slate - removed all previous code | Previous implementation didn't align with research findings; was training ML on interpolated values (circular) |
| 2026-01-17 | Adopt multi-signal architecture | Research shows this solves the "no ground truth" validation problem |
| | | |

---

## 7. Open Questions

1. **GRACE data granularity**: Is 50km resolution useful for village-level constraint, or too coarse?
2. **Temporal alignment**: Which time period to focus on for predictions? Latest available or historical average?
3. **Aquifer boundaries accuracy**: Are the provided aquifer shapefiles accurate enough for stratification?
4. **Validation field visits**: Will the department provide any field validation data during the hackathon?

---

## 8. Files Structure (Planned)

```
SmartJal/
├── PROGRESS.md                 # This file
├── docs/                       # Research and planning docs
│   ├── RESEARCH_FINDINGS.md    # Literature review
│   └── ...
├── data/
│   ├── hackathon_provided/     # Original hackathon data
│   ├── external_downloaded/    # GRACE, rainfall, etc.
│   └── processed/              # Cleaned, joined datasets
├── src/                        # Source code (to be created)
│   ├── data/                   # Data loading and processing
│   ├── models/                 # Prediction models
│   ├── validation/             # Multi-signal validation
│   └── dashboard/              # Visualization
├── notebooks/                  # Exploration notebooks (optional)
└── outputs/                    # Predictions, reports
```

---

*This document will be updated as implementation progresses.*
