# Smart Jal - Data Status Summary

**Last Updated:** January 15, 2026
**Status:** ALL ESSENTIAL DATA AVAILABLE

---

## Executive Summary

We now have **complete data** from official sources. No external downloads required for core POC.

| Category | Status | Source |
|----------|--------|--------|
| Water Levels (Target) | AVAILABLE | Officials |
| Village Boundaries | AVAILABLE | Officials (UseCase.zip) |
| Aquifer Boundaries | AVAILABLE | Officials |
| Soil Data | AVAILABLE | Officials |
| DEM | AVAILABLE | Officials + Downloaded |
| Rainfall | AVAILABLE | Downloaded (CHIRPS) |
| Bore Wells | AVAILABLE | Officials |
| Pumping/Extraction | AVAILABLE | Officials |
| LULC | AVAILABLE | Officials |
| Geomorphology | AVAILABLE | Officials |

---

## Official Data (From Department)

### 1. Water Levels - TARGET VARIABLE
**File:** `SmartJal_extracted/SmartJal/WaterLevels_Krishna/master data_updated.xlsx`

| Attribute | Value |
|-----------|-------|
| Stations | 138 piezometers |
| Time Series | 347 months (~28 years) |
| Period | 1997 - 2027 |
| Format | Excel (.xlsx) |

**Columns:**
- Location: District, Mandal, Village, Lat/Long
- Well Info: Total Depth (m), Principal Aquifer, MSL
- Monthly water level readings (1997-01 to 2027-xx)

**Quality:** Excellent - This is our training/validation data

---

### 2. Village Boundaries - SPATIAL UNITS
**File:** `UseCase_extracted/OKri_Vil.shp`

| Attribute | Value |
|-----------|-------|
| Villages | 939 polygons |
| Format | Shapefile |
| CRS | EPSG:4326 |

**Columns:**
- `DNAME` - District name
- `DMNAME` - Mandal name
- `DVNAME` - Village name
- `DMV_CODE` - Unique code
- `Area`, `areaha` - Area
- `latitude`, `longitude` - Centroid
- `geometry` - Polygon boundary

**Quality:** Excellent - Official boundaries, no need for Survey of India

---

### 3. Mandal Boundaries
**File:** `UseCase_extracted/OKri_Mdl.shp`

| Attribute | Value |
|-----------|-------|
| Mandals | 42 polygons |
| Format | Shapefile |

---

### 4. Aquifer Boundaries - CRITICAL FOR GEOLOGY-AWARE MODEL
**File:** `SmartJal_extracted/SmartJal/Aquifers_Krishna/Aquifers_Krishna.shp`

| Attribute | Value |
|-----------|-------|
| Aquifers | 8 polygons |
| Format | Shapefile |

**Aquifer Types:**
| Code | Geo_Class | Description |
|------|-----------|-------------|
| AL | Alluvium | High permeability |
| BG | Banded Gneissic Granites | Crystalline |
| ST | Sand Stones | Sedimentary |
| SH | Shales | Low permeability |
| LS | Lime Stones | Karst potential |
| QZ | Quartzite | Hard rock |
| KH | Khondalites | Metamorphic |
| CK | Charnokites | Crystalline |

**Quality:** Excellent - Essential for geology-aware interpolation

---

### 5. Soil Data
**File:** `UseCase_extracted/OKri_Soils.shp`

| Attribute | Value |
|-----------|-------|
| Polygons | 151 |
| Format | Shapefile |

**Columns:**
- `DESCRIPTIO` - Soil description
- `SOIL_TAXON` - Soil taxonomy
- `MAPPING_UN` - Mapping unit

**Official Classification Guidance (from Data Issues_Clarifications.docx):**

For **Infiltration/Recharge**:
| Category | Infiltration | Soil Types |
|----------|--------------|------------|
| Low | Low | Clays, Clayey |
| Moderate | Moderate | Loam, Loamy, Others |
| High | High | Sands, Sandy |

For **Runoff**:
| Category | Runoff | Soil Types |
|----------|--------|------------|
| Low | Less | Very shallow, Shallow, Less Drained |
| Moderate | Moderate | Moderately Deep, Moderately Drained |
| High | High | Deep, Very Deep, Well Drained |

---

### 6. Bore Wells Data
**File:** `SmartJal_extracted/SmartJal/GTWells_Krishna/GTWells/kris.csv`

| Attribute | Value |
|-----------|-------|
| Records | 88,988 bore wells |
| Format | CSV + Shapefile |

**Columns:**
- Location: District, Mandal, Village, Lat, Long
- Well: Type, Depth, Pump Capacity
- Agriculture: Crop Type, Irrigation Type, Land Irrigated

**Official Filtering Guidance:**
- **Include:** Bore wells, Tube wells, Filter points (Deep wells), Working status
- **Exclude:** Open wells (shallow), Partially working, Abandoned

---

### 7. Pumping/Extraction Data
**File:** `Pumping Data.xlsx`

| Attribute | Value |
|-----------|-------|
| Records | 714 villages |
| Format | Excel |

**Columns:**
- Mandal, Village
- Structure Type
- No. of Functioning Wells
- Estimated draft per well (ha.m) - Monsoon & Non-Monsoon

**Usage:** Monthly average extraction = Unit Draft / 4 months

---

### 8. Geomorphology
**File:** `SmartJal_extracted/SmartJal/GM_Krishna/GM_Krishna.shp`

| Attribute | Value |
|-----------|-------|
| Polygons | 614 |
| Format | Shapefile |

**Key Types:**
- Structural/Residual hills (255)
- Alluvial Plain (192)
- Deltaic plain (69)
- Pediment (58)
- Valley fill (26)

---

### 9. Land Use / Land Cover
**Files:**
- `SmartJal_extracted/SmartJal/LULC_Krishna/LULC_Krishna1.shp` (4,295 polygons)
- `SmartJal_extracted/SmartJal/LULC_Krishna/LULC_2005.tif` (Raster)

**Note:** Official LULC is from 2005. We also have ESA WorldCover 2021 (10m) in `downloaded_data/landcover/`

---

## Downloaded External Data

### 10. CHIRPS Rainfall
**Directory:** `downloaded_data/rainfall/chirps/krishna_clipped/`

| Attribute | Value |
|-----------|-------|
| Files | 56 monthly TIFs |
| Period | 2020-01 to 2024-12 |
| Resolution | 0.05° (~5km) |
| Size | 224 KB total (clipped) |

---

### 11. SRTM DEM
**File:** `downloaded_data/dem/krishna_dem_merged.tif`

| Attribute | Value |
|-----------|-------|
| Resolution | 30m |
| Size | 30 MB |
| Coverage | Krishna District |

---

### 12. Processed Village Features
**File:** `downloaded_data/processed/village_features.csv`

| Attribute | Value |
|-----------|-------|
| Villages | 959 |
| Features | 69 columns |
| Includes | Elevation, 56 months rainfall, statistics |

---

## Data Files Location Summary

```
SmartJal/
├── UseCase_extracted/           # Official boundary data
│   ├── OKri_Vil.shp            # Village boundaries (939)
│   ├── OKri_Mdl.shp            # Mandal boundaries (42)
│   ├── OKri_Soils.shp          # Soil data (151)
│   ├── OKri_MIT.shp            # MIT data
│   └── OKri_DEM.*              # DEM auxiliary files
│
├── SmartJal_extracted/SmartJal/ # Official thematic data
│   ├── WaterLevels_Krishna/    # TARGET: 138 piezometers, 28 years
│   ├── Aquifers_Krishna/       # 8 aquifer boundaries
│   ├── GM_Krishna/             # 614 geomorphology polygons
│   ├── GTWells_Krishna/        # 88,988 bore wells
│   └── LULC_Krishna/           # Land use (2005)
│
├── downloaded_data/             # External data
│   ├── rainfall/chirps/        # 56 months CHIRPS
│   ├── dem/                    # 30m SRTM DEM
│   ├── villages/               # Extracted centroids
│   ├── processed/              # Feature-engineered data
│   └── landcover/              # ESA WorldCover 2021
│
├── Pumping Data.xlsx           # Village-level extraction
├── Data Issues_Clarifications.docx  # Official Q&A
└── GTWells_Krishna/            # Original bore well data
```

---

## Krishna District Bounds

| Parameter | Value |
|-----------|-------|
| Min Longitude | 80.0° E |
| Max Longitude | 81.5° E |
| Min Latitude | 15.5° N |
| Max Latitude | 17.0° N |
| EPSG | 4326 (WGS84) |

---

## Quick Start Code

### Load All Official Data
```python
import pandas as pd
import geopandas as gpd

# Water levels (target)
water_levels = pd.read_excel('SmartJal_extracted/SmartJal/WaterLevels_Krishna/master data_updated.xlsx')

# Village boundaries
villages = gpd.read_file('UseCase_extracted/OKri_Vil.shp')

# Aquifer boundaries
aquifers = gpd.read_file('SmartJal_extracted/SmartJal/Aquifers_Krishna/Aquifers_Krishna.shp')

# Soil data
soils = gpd.read_file('UseCase_extracted/OKri_Soils.shp')

# Bore wells
borewells = pd.read_csv('SmartJal_extracted/SmartJal/GTWells_Krishna/GTWells/kris.csv')

# Pumping data
pumping = pd.read_excel('Pumping Data.xlsx')

print(f"Piezometers: {len(water_levels)}")
print(f"Villages: {len(villages)}")
print(f"Aquifers: {len(aquifers)}")
print(f"Bore wells: {len(borewells):,}")
```

---

## Next Steps

1. **Data Preprocessing**
   - Clean water level time series (handle missing values)
   - Join villages with aquifer types
   - Extract soil properties per village
   - Filter bore wells (working, deep only)

2. **Feature Engineering**
   - Extract rainfall at piezometer/village locations
   - Calculate terrain features from DEM
   - Compute pumping intensity per village

3. **Model Development**
   - Train on 138 piezometers
   - Predict for 939 villages
   - Validate with held-out piezometers

---

*Document updated: January 15, 2026*
*Status: Ready for development - all data available*
