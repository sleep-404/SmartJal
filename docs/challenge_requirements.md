# SmartJal Hackathon Challenge Requirements

## Challenge Overview

**Challenge Code**: 100008
**Title**: AI-Based Mapping and Forecasting of Groundwater Levels
**Organization**: Andhra Pradesh Ground Water and Water Audit Department
**Supported by**: Ernst & Young (EY) and RTGS (Real-Time Governance Society)

---

## Problem Statement

### The Core Challenge

Andhra Pradesh is a **geologically heterogeneous** region spanning 1,62,000 sq km with:
- **14 principal aquifer types** (from recent alluvium to ancient Archeans)
- Rainfall varying from **500mm to 1000mm**
- ~40% command area (surface water irrigated), 60% non-command area
- Groundwater levels ranging from **ground level to 100+ meters deep**

### Current Monitoring Infrastructure

- **1,800 piezometers** across the state monitoring real-time groundwater levels
- Data distributed across all 14 hydrogeological aquifers
- Coverage of 748 microbasins and all mandals (blocks)

### The Pain Point

> "We are having 1,800 piezometers, and we are having 18,000+ villages. The challenge is we have interpolated the data available... we observed that more than 10,000+ villages are under groundwater stress. But it is practically not correct."

**Why simple interpolation fails:**

1. **Aquifer heterogeneity**: In the same village or neighboring villages, formations can be alluvium AND crystalline rocks
2. **Water level behavior varies by aquifer type**:
   - Alluvial areas → shallow water levels
   - Crystalline rocks → deep water levels
3. **IDW takes nearest neighbor** without respecting aquifer boundaries
4. **Result**: Alluvial areas get assigned deep water levels, crystalline areas get shallow levels

### Example from Mr. Srinivas (JD):

> "For example, our piezometer is in alluvium, and one is in sandstone, one is in crystalline rocks. While interpolating this data, it is taking nearest neighbor concept... In alluvial area, we are getting deeper water levels, and in some crystalline areas, we are getting shallow water levels."

---

## POC Scope

### Geographic Focus: Krishna District

- **~150 piezometers** (138 in provided data)
- **~800-900 villages** to predict
- Monthly water level data for ~10 years

### Required Outputs

#### Primary Deliverable
Assign a **groundwater level value (in meters below ground)** to **every village** in Krishna District:
- **Pre-monsoon** water level
- **Post-monsoon** water level

#### Secondary Deliverables
1. Identify **artificial recharge zones**
2. Identify **groundwater deficient zones** (water level > 20 meters)
3. Calculate **water requirement** to recharge aquifer and bring level to 8 meters
4. Identify suitable locations for water conservation structures:
   - Check dams
   - Percolation tanks
   - Recharge shafts
   - Farm ponds

---

## Water Level Categories

The department classifies villages based on pre-monsoon groundwater levels:

| Category | Depth Range | Action Required |
|----------|-------------|-----------------|
| Safe | 0-3 meters | Low priority |
| Moderate | 3-8 meters | Monitor |
| Stress | 8-20 meters | Plan intervention |
| Critical | >20 meters | Immediate water conservation |

---

## Factors Affecting Groundwater

Per Mr. Srinivas, groundwater availability and recharge depend on **5 key parameters**:

1. **Rainfall** - Primary recharge source
2. **Applied irrigation** - Recharge through surface water and groundwater irrigation
3. **Canal seepage** - Recharge from irrigation canals
4. **Tanks and ponds** - Recharge from surface water bodies
5. **Artificial recharge structures** - Check dams, percolation tanks, etc.

### Additional Factors to Consider

- **Aquifer type** (primary discriminator)
- **Terrain** - Elevation, slope, drainage
- **Utilization factor** - Extraction patterns vary by area
- **Command vs non-command area** - Surface water availability differs

---

## Data Provided by Department

### For Krishna District POC

| Dataset | Description | Format |
|---------|-------------|--------|
| Water Levels | Monthly data for ~10 years, 138 piezometers | Excel |
| Piezometer Metadata | Location, depth, aquifer type | Excel |
| Aquifers | Principal aquifer boundaries | Shapefile |
| Geomorphology | 614 geomorphological features | Shapefile |
| LULC | Land Use Land Cover (2005) | Shapefile + TIFF |
| GT Wells | 88,988 groundwater wells | CSV |
| Stratigraphic Succession | Lithology prioritization | Document |

### Data to be Provided (promised in session)

- Rainfall data for Krishna District
- Village/Mandal boundary layer
- Lithology around piezometers
- Major and minor irrigation structures

---

## Data to Download (Freely Available)

The department confirmed these can be downloaded from public sources:

| Data | Source | Resolution |
|------|--------|------------|
| DEM (Elevation) | SRTM | 30 meters |
| Slope | Derived from DEM | 30 meters |
| Soil | ISRIC SoilGrids | 250 meters |
| Rainfall | CHIRPS, IMD | Daily |
| Land Use | Sentinel-2 classification | 10-30 meters |

> "Most of the parameters which you are asking for are freely available... DEM from SRTM, slope derived from DEM, soil from ISRIC at 250m resolution." - Bindu Ma'am (EY)

---

## Validation Approach

### Field Verification

1. Department will select **10% of villages** randomly
2. Field officers will visit selected villages
3. Compare predictions with **actual farmer bore well readings**
4. Manual water level measurements taken on-site

### Success Criteria

- Predictions should be **realistic** for the aquifer type
- Should not assign shallow levels to crystalline areas
- Should not assign deep levels to alluvial areas
- Accuracy assessed against field measurements

---

## Technical Approach Suggested

### Why Simple IDW Fails

```
Problem: IDW interpolates across aquifer boundaries
Result: Incorrect water levels assigned to villages
```

### Required Approach

1. **Aquifer-aware interpolation** - Constrain predictions within same aquifer type
2. **Multi-parameter model** - Use rainfall, terrain, soil, extraction patterns
3. **Not just spatial** - Consider hydrogeological factors

### Key Insight from Mr. Srinivas

> "If simple interpolation is required, any of our department team can do that one. Every day, we are doing IDW interpolation. But we are not getting desired results for that village."

---

## Timeline

- **POC Duration**: 7-10 days from orientation
- **Final Demo**: All 6 shortlisted teams present
- **Evaluation**: Field verification of predictions

---

## Participating Teams

1. Spondum Enterprises Private Limited
2. JSP Private Limited
3. Spacing Technologies
4. Harrier Technologies
5. Hitlach Solutions
6. Individual - Badri Vamsi

---

## Key Contacts

- **Mr. Srinivas** - Joint Director, AP Groundwater Department
- **Dilip** - Department team (data provider)
- **Srinath** - Ernst & Young
- **Harsha** - Ernst & Young (Manager)
- **Bindu Ma'am** - Ernst & Young (Data sources guidance)

---

## Questions Raised in Session

### Q: What is the test data?
**A**: Predict for all villages in Krishna District. Department will randomly select villages and compare with field readings.

### Q: How to deliver results?
**A**: Web application acceptable. Focus on methodology first, presentation format can be discussed.

### Q: Units in the data?
**A**: Department to clarify units (pumping rate, etc.)

### Q: Old LULC data (2005)?
**A**: Use if nothing else available, or classify recent satellite imagery.

---

## Summary

> "My pain point is to identify the water level for a particular village for the purpose of planning water conservation structures in that village."

**Goal**: Enable the department to identify villages with deep water levels (>20m) so they can prioritize water conservation activities in those areas.

**Challenge**: Build an AI model that respects aquifer boundaries and assigns realistic groundwater levels to every village in Krishna District.
