# Smart Jal - Implementation Progress Tracker

**Last Updated:** 2026-01-17 13:45 IST
**Status:** Starting Development
**Current Phase:** Phase 1 - Data Foundation

---

## 1. Problem Statement

### 1.1 The Core Challenge
Predict groundwater levels for **939 villages** in Krishna District, Andhra Pradesh, using data from only **138 piezometers** (monitoring wells).

### 1.2 Why This Is Hard
- **No ground truth**: We have water level readings at 138 points, but need predictions for 939 villages
- **Sparse coverage**: 138 piezometers across ~7,000 km² = 1 sensor per 50 km²
- **Geological heterogeneity**: Different aquifer types behave differently

### 1.3 What Success Looks Like
1. Predictions for all 939 villages with uncertainty estimates
2. Predictions that respect aquifer boundaries (geologically coherent)
3. Predictions that satisfy water balance (physically consistent)
4. Validation via leave-one-out cross-validation on piezometers

---

## 2. Data Inventory (VERIFIED)

### 2.1 Hackathon-Provided Data

| Dataset | File Path | Records | Verified |
|---------|-----------|---------|----------|
| **Water Levels** | `data/hackathon_provided/WaterLevels_Krishna/master data_updated.xlsx` | 138 piezometers × 347 months (1997-2025) | ✓ |
| **Villages** | `data/hackathon_provided/UseCase_extracted/OKri_Vil.shp` | 939 polygons | ✓ |
| **Mandals** | `data/hackathon_provided/UseCase_extracted/OKri_Mdl.shp` | 42 polygons | ✓ |
| **Aquifers** | `data/hackathon_provided/Aquifers_Krishna/Aquifers_Krishna.shp` | 8 zones | ✓ |
| **Soils** | `data/hackathon_provided/UseCase_extracted/OKri_Soils.shp` | 151 polygons | ✓ |
| **Geomorphology** | `data/hackathon_provided/GM_Krishna/GM_Krishna.shp` | 614 polygons | ✓ |
| **Bore Wells** | `data/hackathon_provided/GTWells_Krishna/GTWells/kris.csv` | 88,988 records | ✓ |
| **Pumping Data** | `data/hackathon_provided/Pumping Data.xlsx` | 713 villages | ✓ |
| **LULC** | `data/hackathon_provided/LULC_Krishna/LULC_Krishna1.shp` | Present | ✓ |

### 2.2 External Data (Downloaded - NOT from hackathon)

| Dataset | Path | Purpose | Required? |
|---------|------|---------|-----------|
| **CHIRPS Rainfall** | `data/external_downloaded/rainfall/chirps/` | Recharge calculation | YES - not in hackathon data |
| **SRTM DEM** | `data/external_downloaded/dem/` | Slope/terrain | YES - not in hackathon data |
| **GRACE** | `data/external_downloaded/grace/` | Regional constraint | OPTIONAL - for validation |

### 2.3 Data NOT Provided by Hackathon (Gap Analysis)

| Data | Mentioned in Orientation? | Our Solution |
|------|---------------------------|--------------|
| Rainfall | Yes, they said they have it | Downloaded CHIRPS as substitute |
| DEM/Elevation | Not mentioned | Downloaded SRTM |

---

## 3. Solution Approach (Simplified)

### 3.1 Core Strategy

Focus on hackathon data with minimal external supplements:

```
INPUTS (Hackathon Data)                    OUTPUT
─────────────────────                      ──────
138 Piezometers (water levels)
    ↓
Aquifer Boundaries (8 zones)      ───→     939 Village Predictions
    ↓                                      with uncertainty
Soil Types (infiltration)
    ↓
Pumping Data (extraction)
    ↓
Bore Well Density (extraction proxy)

EXTERNAL SUPPLEMENTS
────────────────────
Rainfall (CHIRPS) → Recharge calculation
DEM (SRTM) → Slope for runoff
```

### 3.2 Method: Aquifer-Stratified Interpolation + Physics

**Step 1: Spatial Interpolation (within aquifer)**
- For each village, find its aquifer type
- Only use piezometers from the SAME aquifer for interpolation
- Use IDW (Inverse Distance Weighting) or Kriging

**Step 2: Physics Adjustment**
- Calculate expected water balance: ΔStorage = Recharge - Extraction
- Recharge = Rainfall × Soil_Infiltration_Factor
- Extraction = Pumping_Data or Bore_Well_Density proxy
- Adjust interpolated values to satisfy physics

**Step 3: Validation**
- Leave-one-out CV on 138 piezometers
- Report RMSE, MAE, R²

---

## 4. Implementation Plan

### Phase 1: Data Foundation (Current)
- [ ] 1.1 Load piezometer data (locations + time series)
- [ ] 1.2 Load village boundaries
- [ ] 1.3 Load aquifer boundaries
- [ ] 1.4 Load soil data with infiltration classification
- [ ] 1.5 Load bore well data
- [ ] 1.6 Load pumping data
- [ ] 1.7 Spatial join: assign each village and piezometer to an aquifer
- [ ] 1.8 Spatial join: assign soil type to each village

### Phase 2: Aquifer-Stratified Interpolation
- [ ] 2.1 Group piezometers by aquifer type
- [ ] 2.2 For each aquifer, fit IDW/Kriging model
- [ ] 2.3 Predict water level at each village centroid
- [ ] 2.4 Calculate uncertainty based on distance to nearest piezometer
- [ ] 2.5 Leave-one-out cross-validation

### Phase 3: Physics Constraints
- [ ] 3.1 Extract rainfall per village (from CHIRPS)
- [ ] 3.2 Calculate infiltration factor per village (from soil type)
- [ ] 3.3 Calculate recharge = rainfall × infiltration
- [ ] 3.4 Calculate extraction from pumping data (713 villages have direct data)
- [ ] 3.5 Estimate extraction for remaining villages from bore well density
- [ ] 3.6 Apply water balance adjustment

### Phase 4: Output & Visualization
- [ ] 4.1 Generate predictions for all 939 villages
- [ ] 4.2 Classify risk tiers (Critical/High/Moderate/Low)
- [ ] 4.3 Create simple map visualization
- [ ] 4.4 Export results to GeoJSON/CSV

---

## 5. Progress Log

### 2026-01-17

| Time | Action | Result |
|------|--------|--------|
| 12:00 | Cleaned codebase - removed all previous code | Fresh start on main branch |
| 12:30 | Created PROGRESS.md | Planning document |
| 13:00 | Verified actual data files present | Confirmed all hackathon data exists |
| 13:30 | Identified data gap: no rainfall in hackathon data | Will use CHIRPS |
| 13:45 | Updated PROGRESS.md with verified inventory | Ready to start development |

---

## 6. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-17 | Clean slate - removed all previous code | Previous implementation trained ML on interpolated values (circular/invalid) |
| 2026-01-17 | Use hackathon data + minimal external | Rainfall not provided, need CHIRPS; DEM needed for slope |
| 2026-01-17 | Skip GRACE/NDVI for now | Focus on core solution first; add later if time permits |
| 2026-01-17 | Use IDW over Kriging initially | Simpler, fewer assumptions, easier to debug |

---

## 7. File Structure

```
SmartJal/
├── PROGRESS.md                 # This file - tracks everything
├── docs/                       # Research and reference docs
│   └── RESEARCH_FINDINGS.md    # Literature review
├── data/
│   ├── hackathon_provided/     # Original hackathon data (verified)
│   │   ├── WaterLevels_Krishna/
│   │   ├── UseCase_extracted/
│   │   ├── Aquifers_Krishna/
│   │   ├── GTWells_Krishna/
│   │   ├── GM_Krishna/
│   │   ├── LULC_Krishna/
│   │   └── Pumping Data.xlsx
│   ├── external_downloaded/    # Supplements we downloaded
│   │   ├── rainfall/chirps/
│   │   ├── dem/
│   │   └── grace/
│   └── processed/              # Output of our processing
├── src/                        # Source code (to be created)
│   ├── data_loader.py          # Load all datasets
│   ├── spatial_model.py        # Aquifer-stratified interpolation
│   ├── physics_model.py        # Water balance constraints
│   └── predict.py              # Generate predictions
└── outputs/                    # Final predictions
```

---

## 8. Key Equations

### Water Balance
```
ΔStorage = Recharge - Extraction

Where:
  Recharge = Rainfall × Infiltration_Factor × (1 - Runoff_Factor)
  Extraction = n_wells × unit_draft / area  (from Pumping Data.xlsx)
```

### Soil Infiltration Factors (from Department clarification)
| Soil Type | Infiltration | Factor |
|-----------|--------------|--------|
| Sandy | High | 0.8 |
| Loamy | Moderate | 0.5 |
| Clayey | Low | 0.2 |

### IDW Interpolation
```
For village v in aquifer A:
  prediction(v) = Σ w_i × observation(p_i)

Where:
  w_i = (1/d_i²) / Σ(1/d_j²)
  d_i = distance from village v to piezometer i
  Only use piezometers in same aquifer A
```

---

## 9. Validation Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| RMSE | < 3m | Root mean square error on leave-one-out CV |
| MAE | < 2m | Mean absolute error |
| R² | > 0.7 | Coefficient of determination |

---

## 10. Next Steps

**Immediate (Phase 1):**
1. Create `src/data_loader.py` - load all hackathon data
2. Verify data quality (missing values, CRS alignment)
3. Perform spatial joins (village ↔ aquifer, village ↔ soil)

**Then (Phase 2):**
1. Implement aquifer-stratified IDW
2. Run leave-one-out CV
3. Report baseline metrics

---

*This document is updated continuously as work progresses.*
