# Smart Jal - Implementation Progress Tracker

**Last Updated:** 2026-01-17 14:30 IST
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

### 1.3 What We Are Predicting (Clarified)
- **Target:** Water level (depth in meters) for each of 939 villages
- **Time Period:** Latest available month (to demonstrate method works)
- **Method:** Aquifer-stratified spatial interpolation from 138 piezometers
- **Extensible:** Method can be applied to any month in the 347-month dataset

### 1.4 What Success Looks Like
1. Predictions for all 939 villages with uncertainty estimates
2. Predictions that respect aquifer boundaries (geologically coherent)
3. Validation via leave-one-out cross-validation on 138 piezometers
4. Clear deliverables: GeoJSON + CSV + validation metrics

### 1.5 Final Deliverables
| Deliverable | Description | Priority |
|-------------|-------------|----------|
| `village_predictions.geojson` | 939 villages with predicted water levels, uncertainty, risk tier | REQUIRED |
| `village_predictions.csv` | Same data in tabular format | REQUIRED |
| Validation metrics | RMSE, MAE, R² from leave-one-out CV | REQUIRED |
| Simple map visualization | Villages colored by risk tier | REQUIRED |
| Streamlit dashboard | Interactive exploration | OPTIONAL |

### 1.6 Validation Limitation (Acknowledged)
- We can only validate at **piezometer locations** (138 points)
- We **cannot directly validate** village predictions (no ground truth)
- Leave-one-out CV tells us how well we predict at known locations
- Village predictions are extrapolations - their accuracy is inferred, not measured

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

## 4. Implementation Plan (Detailed)

### Phase Priority Summary

| Phase | Priority | Reason |
|-------|----------|--------|
| Phase 1: Data Foundation | **REQUIRED** | Must load and prepare all data |
| Phase 2: Spatial Interpolation | **REQUIRED** | This IS the core goal - interpolate from 138 to 939 |
| Phase 3: Physics Constraints | **OPTIONAL** | Enhancement, not core ask |
| Phase 4.1-4.3: Output & Export | **REQUIRED** | Must produce deliverables |
| Phase 4.4: Dashboard | **OPTIONAL** | Nice to have for demo |

---

### Phase 1: Data Foundation [REQUIRED] (Current)

#### 1.1 Load Piezometer Data
- [ ] 1.1.1 Parse `master data_updated.xlsx`
- [ ] 1.1.2 Extract metadata: ID, lat, lon, aquifer type, depth, MSL
- [ ] 1.1.3 Extract monthly water level time series (347 months: 1997-2025)
- [ ] 1.1.4 Convert to GeoDataFrame with Point geometry
- [ ] 1.1.5 Validate: check for missing coordinates, invalid values

#### 1.2 Load Village Boundaries
- [ ] 1.2.1 Read `OKri_Vil.shp` shapefile
- [ ] 1.2.2 Standardize column names (district, mandal, village)
- [ ] 1.2.3 Calculate village centroids
- [ ] 1.2.4 Ensure CRS is EPSG:4326 (WGS84)
- [ ] 1.2.5 Calculate village area in km²

#### 1.3 Load Aquifer Boundaries
- [ ] 1.3.1 Read `Aquifers_Krishna.shp`
- [ ] 1.3.2 Identify aquifer types from `Geo_Class` column
- [ ] 1.3.3 Ensure CRS matches villages (EPSG:4326)

#### 1.4 Load Soil Data
- [ ] 1.4.1 Read `OKri_Soils.shp`
- [ ] 1.4.2 Classify soils into Sandy/Loamy/Clayey categories
- [ ] 1.4.3 Assign infiltration factors: Sandy=0.8, Loamy=0.5, Clayey=0.2

#### 1.5 Load Bore Well Data
- [ ] 1.5.1 Read `kris.csv` (88,988 records)
- [ ] 1.5.2 Filter to working wells only (per department guidance)
- [ ] 1.5.3 Filter to bore/tube wells (exclude open wells)
- [ ] 1.5.4 Extract lat/lon and convert to GeoDataFrame

#### 1.6 Load Pumping Data
- [ ] 1.6.1 Read `Pumping Data.xlsx` (skip header row)
- [ ] 1.6.2 Parse monsoon and non-monsoon draft values
- [ ] 1.6.3 Calculate monthly extraction: draft / 4 months

#### 1.7 Perform Spatial Joins
- [ ] 1.7.1 Join villages → aquifers (assign aquifer type to each village)
- [ ] 1.7.2 Join piezometers → aquifers (assign aquifer type to each piezometer)
- [ ] 1.7.3 Join villages → soils (dominant soil type per village)
- [ ] 1.7.4 Count bore wells per village (well density)
- [ ] 1.7.5 Join pumping data → villages (by mandal + village name)

#### 1.8 Create Unified Dataset
- [ ] 1.8.1 Merge all village attributes into single GeoDataFrame
- [ ] 1.8.2 Save to `data/processed/village_features.geojson`
- [ ] 1.8.3 Save piezometers to `data/processed/piezometers.geojson`

**Phase 1 Output:**
- `village_features.geojson` (939 villages with aquifer, soil, wells, pumping)
- `piezometers.geojson` (138 piezometers with metadata and aquifer assignment)

---

### Phase 2: Spatial Interpolation [REQUIRED] - CORE GOAL

**This phase IS the main goal. Everything else supports this.**

#### 2.1 Analyze Piezometer Distribution
- [ ] 2.1.1 Count piezometers per aquifer type
- [ ] 2.1.2 Identify aquifers with <3 piezometers (need fallback strategy)
- [ ] 2.1.3 Calculate piezometer density per aquifer (piezos/km²)

#### 2.2 Select Target Time Period
- [ ] 2.2.1 Identify latest month with most complete data
- [ ] 2.2.2 Calculate average water level per piezometer (for static prediction)
- [ ] 2.2.3 Decide: predict latest month OR long-term average

#### 2.3 Implement IDW Interpolation
- [ ] 2.3.1 Create IDW function: `idw(target_point, source_points, values, power=2)`
- [ ] 2.3.2 Implement aquifer-stratified wrapper: only use piezometers from same aquifer
- [ ] 2.3.3 Handle edge case: village in aquifer with 0 piezometers → use nearest from any aquifer
- [ ] 2.3.4 Handle edge case: village in aquifer with 1-2 piezometers → blend with neighbors

#### 2.4 Generate Baseline Predictions
- [ ] 2.4.1 For each village: predict water level using aquifer-stratified IDW
- [ ] 2.4.2 Calculate distance to nearest piezometer (in same aquifer)
- [ ] 2.4.3 Calculate distance to nearest piezometer (any aquifer)
- [ ] 2.4.4 Assign uncertainty: higher if far from piezometers

#### 2.5 Leave-One-Out Cross-Validation
- [ ] 2.5.1 For each of 138 piezometers: remove it, predict its value, record error
- [ ] 2.5.2 Calculate RMSE = sqrt(mean(errors²))
- [ ] 2.5.3 Calculate MAE = mean(|errors|)
- [ ] 2.5.4 Calculate R² = 1 - SS_res/SS_tot
- [ ] 2.5.5 Analyze errors by aquifer type (which aquifers have worst predictions?)
- [ ] 2.5.6 Analyze errors by distance to nearest neighbor

**Phase 2 Output:**
- Baseline predictions for 939 villages
- Validation metrics: RMSE, MAE, R²
- Error analysis by aquifer

---

### Phase 3: Physics Constraints [OPTIONAL] - Enhancement Only

**This phase is NOT required for core deliverable. Implement only if time permits after Phase 2 is complete and validated.**

#### 3.1 Extract Rainfall Data
- [ ] 3.1.1 Load CHIRPS GeoTIFFs for target time period
- [ ] 3.1.2 For each village polygon: extract mean rainfall (zonal statistics)
- [ ] 3.1.3 Calculate annual rainfall (sum of monthly)
- [ ] 3.1.4 Calculate monsoon vs non-monsoon rainfall

#### 3.2 Calculate Recharge
- [ ] 3.2.1 Get infiltration factor for each village from soil type
- [ ] 3.2.2 Calculate: `Recharge = Rainfall × Infiltration_Factor`
- [ ] 3.2.3 Convert units to consistent format (mm → m, per area)

#### 3.3 Calculate Extraction
- [ ] 3.3.1 For villages with pumping data (713): use directly
  - Extraction = n_wells × unit_draft / village_area
- [ ] 3.3.2 For villages without pumping data (226): estimate from bore well density
  - Count wells per village, apply average draft
- [ ] 3.3.3 Validate extraction estimates against known values

#### 3.4 Apply Water Balance
- [ ] 3.4.1 Calculate: `ΔStorage = Recharge - Extraction` per village
- [ ] 3.4.2 Compare ΔStorage to interpolated water level trend
- [ ] 3.4.3 Flag villages where interpolation contradicts physics (e.g., level rising but extraction > recharge)
- [ ] 3.4.4 Adjust predictions: blend interpolation with physics expectation

**Phase 3 Output:**
- Physics-adjusted predictions for 939 villages
- Recharge and extraction estimates per village
- Physics consistency flags

---

### Phase 4: Output & Visualization [MIXED]

#### 4.1 Generate Final Predictions [REQUIRED]
- [ ] 4.1.1 Use spatial interpolation predictions from Phase 2
- [ ] 4.1.2 Calculate uncertainty = f(distance to nearest piezometer, piezometer count in aquifer)
- [ ] 4.1.3 Generate confidence tier: High (close to piezometer), Medium, Low (far from piezometer)
- [ ] 4.1.4 (Optional) If Phase 3 done: blend with physics adjustment

#### 4.2 Classify Risk Tiers [REQUIRED]
- [ ] 4.2.1 Define thresholds based on water level depth:
  - Critical: > 15m depth
  - High: 10-15m depth
  - Moderate: 5-10m depth
  - Low: < 5m depth
- [ ] 4.2.2 Assign risk tier to each village
- [ ] 4.2.3 Calculate summary: count and % of villages per tier

#### 4.3 Export Results [REQUIRED]
- [ ] 4.3.1 CSV: `outputs/village_predictions.csv`
  - Columns: village_id, name, mandal, lat, lon, predicted_level, uncertainty, risk_tier
- [ ] 4.3.2 GeoJSON: `outputs/village_predictions.geojson`
  - Full geometry + all attributes for mapping
- [ ] 4.3.3 Summary report: `outputs/prediction_report.md`
  - Validation metrics, risk distribution, methodology notes
- [ ] 4.3.4 Simple static map: `outputs/prediction_map.html`
  - Villages colored by risk tier using Folium

#### 4.4 Build Interactive Dashboard [OPTIONAL]
- [ ] 4.4.1 Create Streamlit app: `src/dashboard.py`
- [ ] 4.4.2 Add map with Folium showing village polygons
- [ ] 4.4.3 Color villages by risk tier (red/orange/yellow/green)
- [ ] 4.4.4 Add click-to-select: show village details
- [ ] 4.4.5 Show piezometer locations with their influence radius
- [ ] 4.4.6 Add summary statistics panel
- [ ] 4.4.7 Add aquifer filter dropdown

**Phase 4 Output (Required):**
- `outputs/village_predictions.csv`
- `outputs/village_predictions.geojson`
- `outputs/prediction_report.md`
- `outputs/prediction_map.html` (simple static map)

**Phase 4 Output (Optional):**
- Working Streamlit dashboard

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
