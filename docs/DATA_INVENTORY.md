# Smart Jal - Data Inventory & External Data Requirements

**Last Updated:** January 15, 2026
**Status:** ALL DATA AVAILABLE - Ready for Development

---

## Part 1: Available Data (Provided by Department)

### 1.1 Water Levels Data (PRIMARY)
**File**: `WaterLevels_Krishna/master data_updated.xlsx`

| Attribute | Value |
|-----------|-------|
| Records | 138 piezometers |
| Time span | Monthly from 1997 to ~2024 (359 columns) |
| Coverage | Krishna District |

**Columns**:
- `SNo`, `ID` - Identifiers
- `District`, `Mandal Name`, `Village Name` - Location
- `Location (Premises)` - Specific location description
- `Project` - Project name
- `Total Depth in m` - Well depth
- `Principal Aquifer` - Aquifer type
- `MSL in meters` - Mean Sea Level
- `Latitude`, `Longitude` - Coordinates (Decimal Degrees)
- Monthly water level columns (1997-01 to 2024-xx)

**Quality**: Good - this is the core training data

---

### 1.2 Aquifer Data
**File**: `Aquifers_Krishna/Aquifers_Krishna.shp`

| Attribute | Value |
|-----------|-------|
| Records | 8 aquifer polygons |
| CRS | EPSG:4044 |
| Format | Shapefile |

**Aquifer Types in Krishna**:
| Code | Geo_Class |
|------|-----------|
| BG | Banded Gneissic Granites |
| ST | Sand Stones |
| SH | Shales |
| LS | Lime Stones |
| QZ | Quartzite |
| KH | Khondalites |
| CK | Charnokites |
| AL | Alluvium |

**Quality**: Good - essential for geology-aware modeling

---

### 1.3 Geomorphology Data
**File**: `GM_Krishna/GM_Krishna.shp`

| Attribute | Value |
|-----------|-------|
| Records | 614 polygons |
| Key Column | `FIN_DESC` |
| Format | Shapefile |

**Geomorphology Types (Top 8)**:
| Type | Count |
|------|-------|
| Structural/Residual hills | 255 |
| Alluviam Plain | 192 |
| Deltaic plain | 69 |
| Pediment | 58 |
| Valley fill | 26 |
| Structural valley | 5 |
| Paleo channel | 5 |
| Pediplain | 4 |

**Quality**: Good - useful for terrain characterization

---

### 1.4 Land Use / Land Cover (LULC)
**Files**:
- `LULC_Krishna/LULC_Krishna1.shp` (4,295 polygons)
- `LULC_Krishna/LULC_2005.tif` (Raster)
- `LULC_Krishna/LULC GridCode.xlsx` (Empty - codes not documented)

| Attribute | Value |
|-----------|-------|
| Records | 4,295 polygons |
| Key Column | `gridcode` (integer codes) |
| Year | **2005 (OUTDATED!)** |

**Quality**: POOR - 20 years old, needs replacement with recent data

---

### 1.5 GT Wells (Bore Wells) Data
**File**: `GTWells_Krishna/GTWells/kris.csv` + `.shp`

| Attribute | Value |
|-----------|-------|
| Records | 88,988 bore wells |
| Coverage | Krishna District |

**Columns**:
- `District Name`, `Mandal Name`, `Village Name`
- `Bore Well Working` - Status (Working/Partially Working/etc.)
- `Well Type` - Bore Well, Open Well, etc.
- `Bore Depth` - Depth in meters
- `Pump Capacity` - HP
- `Crop Type` - Wet/Dry
- `Irrigation Type` - Flood/Drip/etc.
- `Extant Land Irrigated` - Area in acres
- `Lat`, `Long` - Coordinates

**Clarification from Department**:
- Only consider: Filter points/Tube wells/Bore wells (Deep Wells)
- Only consider: Functioning/Working wells
- Ignore: Open Wells, Partially Working, Abandoned

**Quality**: Good - valuable for extraction estimation

---

### 1.6 Pumping/Extraction Data
**File**: `Pumping Data.xlsx`

| Attribute | Value |
|-----------|-------|
| Records | 714 rows |
| Granularity | Village-level |

**Columns**:
- `Mandal`, `Village`
- `Structure Type` - Filter Points, etc.
- `No. of Functioning Wells`
- `Estimated draft per well (ha.m)` - Monsoon & Non-Monsoon

**Usage**: Monthly average extraction = (Unit Draft) / 4 months

**Quality**: Good - critical for extraction modeling

---

### 1.7 Downloaded External Data (Already Present)
**Directory**: `downloaded_data/`

| Data | File | Notes |
|------|------|-------|
| WorldCover LULC | `landcover/worldcover_N15E078.tif`, `worldcover_N15E081.tif` | ESA 10m resolution, 2021 |
| Villages | `villages/` | Empty - needs population |

**Quality**: WorldCover is excellent replacement for 2005 LULC

---

### 1.8 Other Documents
- `Data Issues_Clarifications.docx` - Q&A with department (important!)
- `challenge.json` - Problem statement
- `Smart Jal orientation session.txt` - Transcribed meeting

---

## Part 2: Data Gaps - ALL RESOLVED

### Status Update (January 15, 2026)

All previously missing data has been obtained:

| Data | Previous Status | Current Status | Source |
|------|-----------------|----------------|--------|
| **Rainfall** | Missing | AVAILABLE | Downloaded CHIRPS (56 months) |
| **Village boundaries** | Missing | AVAILABLE | UseCase.zip (939 villages) |
| **Mandal boundaries** | Missing | AVAILABLE | UseCase.zip (42 mandals) |
| **Soil data** | Missing | AVAILABLE | UseCase.zip (151 polygons) |
| **DEM/Elevation** | Missing | AVAILABLE | Downloaded SRTM (30m) |
| **Recent LULC** | Partial | AVAILABLE | WorldCover 2021 + Official 2005 |

### New Data from Officials (January 15, 2026)

**UseCase.zip** contains:
- `OKri_Vil.shp` - Village boundaries (939 polygons)
- `OKri_Mdl.shp` - Mandal boundaries (42 polygons)
- `OKri_Soils.shp` - Soil data (151 polygons)
- `OKri_MIT.shp` - Additional boundaries
- `OKri_DEM.*` - DEM auxiliary files

**SmartJal.zip** contains:
- `WaterLevels_Krishna/` - 138 piezometers, 28 years monthly data
- `Aquifers_Krishna/` - 8 aquifer boundary polygons
- `GM_Krishna/` - 614 geomorphology polygons
- `GTWells_Krishna/` - 88,988 bore wells
- `LULC_Krishna/` - Land use (2005)

### Downloaded External Data
- CHIRPS rainfall: 56 months (2020-2024), clipped to Krishna
- SRTM DEM: 30m resolution, merged for Krishna district
- ESA WorldCover: 10m LULC (2021)

---

## Part 3: Recommended External Data Sources

### 3.1 MUST HAVE (Free, Easy to Obtain)

#### Rainfall - IMD / GPM
| Source | Resolution | Access |
|--------|------------|--------|
| IMD Gridded Rainfall | 0.25° (~25km) | https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html |
| GPM (NASA) | 0.1° (~10km) | https://gpm.nasa.gov/data/directory |
| CHIRPS | 0.05° (~5km) | https://www.chc.ucsb.edu/data/chirps |

**Recommendation**: Use IMD if available (India-specific), else GPM/CHIRPS

#### DEM - SRTM
| Source | Resolution | Access |
|--------|------------|--------|
| SRTM | 30m | https://earthexplorer.usgs.gov/ |
| ALOS PALSAR | 12.5m | https://search.asf.alaska.edu/ |

**Derived products**: Slope, Aspect, Flow direction, Flow accumulation

#### Soil - ISRIC SoilGrids
| Source | Resolution | Access |
|--------|------------|--------|
| ISRIC SoilGrids | 250m | https://soilgrids.org/ |

**Key properties**: Sand/Clay/Silt %, Hydraulic conductivity, Water holding capacity

#### Village Boundaries - Survey of India / GADM
| Source | Level | Access |
|--------|-------|--------|
| Survey of India | Village | https://onlinemaps.surveyofindia.gov.in/ |
| GADM | District/Taluk | https://gadm.org/download_country.html |
| Bhuvan | Village | https://bhuvan.nrsc.gov.in/ |

---

### 3.2 HIGH VALUE (Differentiators)

#### GRACE Satellite - Groundwater Storage
| Source | Resolution | Access |
|--------|------------|--------|
| GRACE/GRACE-FO Mascons | 0.5° (~50km) | https://grace.jpl.nasa.gov/data/get-data/ |
| GLDAS (for separation) | 0.25° | https://ldas.gsfc.nasa.gov/gldas |

**Why**: Regional constraint for village predictions, validated for India

**Processing**:
```
Groundwater = GRACE_TWS - GLDAS_SoilMoisture - GLDAS_SurfaceWater - GLDAS_Snow
```

#### Sentinel-1 InSAR - Land Subsidence
| Source | Resolution | Access |
|--------|------------|--------|
| Sentinel-1 | 10m | https://scihub.copernicus.eu/ |

**Why**: Subsidence indicates extraction intensity, high resolution

---

### 3.3 NICE TO HAVE

#### MODIS - Vegetation/ET
| Product | Resolution | Use |
|---------|------------|-----|
| NDVI (MOD13Q1) | 250m | Vegetation stress = water stress |
| ET (MOD16A2) | 500m | Evapotranspiration estimates |

#### Census/Socioeconomic
| Data | Source |
|------|--------|
| Population | Census of India |
| Irrigation statistics | Dept of Agriculture |
| Cropping patterns | Bhuvan / State portal |

---

## Part 4: Data Integration Plan

### Priority 1: Essential (Must have for POC)

```
┌─────────────────────────────────────────────────────────────────┐
│                       ALREADY AVAILABLE                          │
├─────────────────────────────────────────────────────────────────┤
│ ✓ Water Levels (138 piezometers, monthly since 1997)            │
│ ✓ Aquifer boundaries (8 types)                                   │
│ ✓ Geomorphology (614 polygons)                                   │
│ ✓ GT Wells (88,988 bore wells)                                   │
│ ✓ Pumping data (714 village records)                             │
│ ✓ Recent LULC (WorldCover 2021, 10m)                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       NEED TO OBTAIN                             │
├─────────────────────────────────────────────────────────────────┤
│ ! Rainfall (IMD/GPM) - CRITICAL                                  │
│ ! Village boundaries - CRITICAL                                  │
│ ! DEM (SRTM 30m) - IMPORTANT                                     │
│ ! Soil data (ISRIC 250m) - HELPFUL                               │
└─────────────────────────────────────────────────────────────────┘
```

### Priority 2: Differentiators (For competitive edge)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DIFFERENTIATING DATA                          │
├─────────────────────────────────────────────────────────────────┤
│ ★ GRACE groundwater storage (regional constraint)                │
│ ★ GLDAS soil moisture (for GRACE separation)                     │
│ ○ Sentinel-1 InSAR (subsidence/extraction proxy)                 │
│ ○ MODIS NDVI (vegetation stress)                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 5: API/Download Instructions

### 5.1 GRACE Data Download

```python
# Option 1: Direct download from NASA
# URL: https://grace.jpl.nasa.gov/data/get-data/jpl_global_mascons/

# Option 2: Using Python
# pip install podaac-data-subscriber
from podaac.podaac import Podaac
p = Podaac()
# Download GRACE-FO RL06 Mascons

# Option 3: Google Earth Engine
import ee
ee.Initialize()
grace = ee.ImageCollection('NASA/GRACE/MASS_GRIDS/LAND')
```

### 5.2 SRTM DEM Download

```python
# Option 1: elevation package
# pip install elevation
import elevation
elevation.clip(bounds=(80, 15, 82, 17), output='krishna_dem.tif')

# Option 2: Google Earth Engine
dem = ee.Image('USGS/SRTMGL1_003')
```

### 5.3 IMD Rainfall

```python
# Manual download from IMD website
# https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html

# Or use imdlib package
# pip install imdlib
import imdlib as imd
data = imd.get_data('rain', 2015, 2024, 'yearwise')
```

### 5.4 ISRIC Soil Data

```python
# Using REST API
import requests
url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
params = {
    "lon": 80.5,
    "lat": 16.5,
    "property": ["sand", "clay", "silt"],
    "depth": ["0-5cm", "5-15cm", "15-30cm"]
}
response = requests.get(url, params=params)
```

### 5.5 Village Boundaries

```python
# Option 1: Bhuvan (may require registration)
# https://bhuvan-app1.nrsc.gov.in/state/AP

# Option 2: GADM for district boundaries
import geopandas as gpd
india = gpd.read_file("gadm41_IND_3.shp")  # Admin level 3
krishna = india[india['NAME_2'] == 'Krishna']
```

---

## Part 6: Summary

### Complete Data Inventory (All Available)

| Category | Data | Quality | Ready | Source |
|----------|------|---------|-------|--------|
| Water Levels | 138 piezometers, 28 years | Excellent | Yes | Officials |
| Village Boundaries | 939 polygons | Excellent | Yes | Officials (UseCase.zip) |
| Mandal Boundaries | 42 polygons | Excellent | Yes | Officials (UseCase.zip) |
| Aquifers | 8 types, polygons | Excellent | Yes | Officials |
| Geomorphology | 614 polygons | Good | Yes | Officials |
| Soil Data | 151 polygons | Good | Yes | Officials (UseCase.zip) |
| Bore Wells | 88,988 wells | Good | Yes | Officials |
| Pumping | 714 village records | Good | Yes | Officials |
| LULC | 4,295 polygons + WorldCover | Good | Yes | Officials + Downloaded |
| Rainfall | 56 months CHIRPS | Good | Yes | Downloaded |
| DEM | 30m SRTM | Good | Yes | Downloaded |

### Optional Differentiators (Not Yet Obtained)
| Category | Source | Priority | Effort |
|----------|--------|----------|--------|
| GRACE | NASA | Differentiator | Medium |
| InSAR Subsidence | Sentinel-1 | Differentiator | High |
| MODIS NDVI | NASA | Nice to have | Low |

### Development Status
- **Essential data:** 100% complete
- **Ready for:** Model development, feature engineering, training
- **No blockers:** All critical data available

---

## Appendix: Krishna District Bounds

For data downloads, use these approximate bounds:

| Parameter | Value |
|-----------|-------|
| Min Longitude | 80.0° E |
| Max Longitude | 81.5° E |
| Min Latitude | 15.5° N |
| Max Latitude | 17.0° N |
| EPSG | 4326 (WGS84) |
