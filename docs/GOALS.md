# Smart Jal Challenge - Goals & Requirements

## Challenge Overview

**Challenge Code**: 100008
**Department**: Andhra Pradesh Ground Water and Water Audit Department
**Theme**: AI for Water Resource Management

### The Core Problem

Andhra Pradesh has **~18,000 villages** but only **~1,800 piezometers** (water level sensors). Simple interpolation (IDW) doesn't work because geology varies dramatically:

- In alluvial areas: water is shallow (0-3m)
- In crystalline rock areas: water is deep (20-100m)
- Current interpolation wrongly assigns shallow readings to deep areas and vice versa

The department needs accurate village-level groundwater data to plan water conservation activities (check dams, percolation tanks, recharge shafts, etc.)

---

## Primary Goal

**Assign accurate groundwater level values to every village** based on:

- Existing piezometer data
- Aquifer type/characteristics (14 principal aquifers in AP)
- Rainfall patterns (varies 500mm to 1000mm across AP)
- Land use / land cover
- Surface water bodies (tanks, canals, irrigation)
- Extraction/pumping patterns

---

## Required Deliverables

### 1. Groundwater Level Predictions
- **Pre-monsoon** water level for each village
- **Post-monsoon** water level for each village
- Must account for hydrogeological variations

### 2. Village Classification
- Identify villages with water levels:
  - 0-3 meters (safe)
  - 3-8 meters (moderate)
  - 8-20 meters (stressed)
  - >20 meters (critical - priority for conservation)

### 3. Recharge Planning
- Calculate water requirement to recharge aquifer to 8m level
- Identify available surface water sources
- Recommend artificial recharge structures based on:
  - Surplus water availability
  - Field conditions
  - Existing conservation structures

### 4. Integration & Visualization
- Dashboard for department officials
- API-based data services
- Real-time outputs for engineers and field officers

---

## POC Scope

| Parameter | Value |
|-----------|-------|
| Focus Area | Krishna District only |
| Piezometers | ~150 |
| Villages to predict | ~700-900 |
| Validation | Field comparison with farmer borewells |

### Evaluation Method
Department field officers will:
1. Take your predicted values for random villages
2. Visit those villages
3. Measure actual water levels from farmer borewells
4. Compare predictions vs actual readings

---

## Data Provided

| Dataset | Description |
|---------|-------------|
| Piezometer Data | Monthly readings, 10 years historical |
| Aquifer Layers | 14 principal aquifers, geometry data |
| Geomorphology | Terrain characteristics |
| Geology | Rock types, stratification |
| LULC | Land use/land cover (2005 - outdated) |
| Pumping Data | Extraction rates |
| Rainfall | To be shared |
| Village Boundaries | Mandal/District layers |

### Data Gaps to Address
- LULC data is from 2005 (consider downloading recent satellite imagery)
- Soil data not provided (use ISRIC at 250m resolution or FAO)
- DEM available from SRTM (30m resolution)

---

## Key Technical Considerations

### Factors Affecting Groundwater Levels
1. **Rainfall** - Primary recharge source
2. **Applied irrigation** - Surface water and groundwater
3. **Canal seepage** - Recharge from irrigation canals
4. **Tanks and ponds** - Surface water body recharge
5. **Artificial recharge structures** - Check dams, percolation tanks

### Why Simple Interpolation Fails
- Piezometers not uniformly distributed
- Single piezometer may cover 10-15 villages with different geology
- Neighboring villages can have completely different aquifer types
- Water behavior varies by formation (alluvium vs sandstone vs crystalline)

---

## Success Criteria

- [ ] 70% positive feedback from user testing (15 scientists + 8 district officials)
- [ ] Demonstrated integration with dashboards and APIs
- [ ] Accurate predictions validated against field measurements
- [ ] Explainable and transparent AI outputs for audit purposes

---

## Timeline

| Milestone | Date |
|-----------|------|
| Orientation Session | January 6, 2026 |
| Demo Presentation | January 19-23, 2026 |
| Preparation Time | ~7-10 days |

---

## Compliance Requirements

- Adhere to data privacy and security norms for hydrological/geospatial data
- Anonymize any village-level or farmer-related data
- Ensure explainable AI outputs for validation and audit
