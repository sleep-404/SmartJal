# All Available Data for SmartJal

A comprehensive inventory of ALL data that can be used - provided, public, and derived.

---

## Part 1: Data Provided by Department

### 1.1 Piezometer Water Levels
| Attribute | Value |
|-----------|-------|
| File | `WaterLevels_Krishna/master data_updated.xlsx` |
| Stations | 138 |
| Time range | January 1997 - November 2025 (28 years) |
| Frequency | Monthly |
| Fields | Station ID, Location, Mandal, Village, Aquifer Type, Depth, Lat/Lon, Monthly readings |

**What we can extract:**
- Mean water level per station
- Seasonal patterns (pre/post monsoon)
- Long-term trends (rising/falling)
- Variability (standard deviation)
- Recovery rate after monsoon
- Minimum/maximum levels

### 1.2 Aquifer Boundaries
| Attribute | Value |
|-----------|-------|
| File | `Aquifers_Krishna/Aquifers_Krishna.shp` |
| Features | 8 principal aquifer types |
| Format | Polygon shapefile |

**Aquifer types:**
- Alluvium
- Sandstone
- Shale
- Limestone
- Quartzite
- Khondalites
- Banded Gneissic Granites
- Charnockites

### 1.3 Geomorphology
| Attribute | Value |
|-----------|-------|
| File | `GM_Krishna/GM_Krishna.shp` |
| Features | 614 polygons |
| Format | Polygon shapefile |

**Landform classes (likely):**
- Flood plains
- Alluvial plains
- Pediments
- Buried pediments
- Structural hills
- Denudational hills
- Valley fills
- Plateaus

### 1.4 Wells Data
| Attribute | Value |
|-----------|-------|
| File | `GTWells_Krishna/GTWells/kris.csv` |
| Records | 88,988 wells |
| Fields | Location, bore depth, pump capacity, irrigated area, crop type, etc. |

**What we can derive:**
- Village-level well density
- Average/max bore depth per village
- Total extraction capacity
- Irrigation intensity
- Crop water demand patterns

### 1.5 Land Use Land Cover (LULC)
| Attribute | Value |
|-----------|-------|
| File | `LULC_Krishna/LULC_Krishna1.shp` + `LULC_2005.tif` |
| Year | 2005 |
| Format | Shapefile + Raster |

**Classes (typical):**
- Agricultural land
- Forest
- Urban/Built-up
- Water bodies
- Barren land
- Scrubland

---

## Part 2: Publicly Available Data (India-Specific)

### 2.1 India-WRIS (Water Resources Information System)
| URL | https://indiawris.gov.in |
| Data | Groundwater, surface water, reservoirs, canals |

**Available datasets:**
- Groundwater levels (CGWB monitoring)
- Reservoir storage levels
- Canal network
- River basin boundaries
- Watershed boundaries
- Water bodies inventory

### 2.2 Bhuvan (ISRO Geoportal)
| URL | https://bhuvan.nrsc.gov.in |
| Data | Indian satellite imagery and derived products |

**Available datasets:**
- Administrative boundaries (state, district, mandal, village)
- LULC maps (multiple years)
- Wasteland maps
- Groundwater prospect maps
- Drainage networks
- DEM for India
- Cartosat imagery

### 2.3 CGWB (Central Ground Water Board)
| URL | https://cgwb.gov.in |
| Data | Official groundwater data for India |

**Available datasets:**
- District-wise groundwater resources
- Aquifer maps
- Groundwater quality
- Stage of extraction
- Recharge estimates
- Dynamic groundwater resources

### 2.4 IMD (India Meteorological Department)
| URL | https://imdpune.gov.in |
| Data | Weather and climate data |

**Available datasets:**
- Gridded rainfall (0.25° resolution)
- District-wise rainfall
- Temperature data
- Drought indices
- Historical climate data

### 2.5 Census of India
| URL | https://censusindia.gov.in |
| Data | Village-level demographic data |

**Available datasets:**
- Village boundaries
- Village amenities (water sources)
- Population data
- Agricultural statistics

### 2.6 Andhra Pradesh State Portals
| Portal | Data Available |
|--------|----------------|
| AP Water Resources | Canal data, reservoir levels |
| AP Agriculture | Crop patterns, irrigation data |
| APWRIMS | Real-time water monitoring |

---

## Part 3: Global Public Datasets

### 3.1 Elevation / Terrain

| Dataset | Resolution | Source | Access |
|---------|------------|--------|--------|
| SRTM DEM | 30m | NASA/USGS | Free (EarthExplorer, GEE) |
| ASTER DEM | 30m | NASA | Free (EarthExplorer) |
| ALOS DEM | 30m | JAXA | Free |
| Copernicus DEM | 30m | ESA | Free |
| FABDEM | 30m | U Bristol | Free (forest/building removed) |

**Derived products:**
- Slope
- Aspect
- Curvature
- Topographic Wetness Index (TWI)
- Flow accumulation
- Drainage density
- Distance to drainage

### 3.2 Rainfall / Precipitation

| Dataset | Resolution | Temporal | Source |
|---------|------------|----------|--------|
| CHIRPS | 5km | Daily, 1981-present | UCSB |
| GPM IMERG | 10km | 30-min, 2000-present | NASA |
| TRMM | 25km | 3-hourly, 1998-2019 | NASA |
| ERA5 | 30km | Hourly, 1950-present | ECMWF |
| TerraClimate | 4km | Monthly, 1958-present | U Idaho |

**Derived products:**
- Annual rainfall
- Monsoon rainfall (June-September)
- Pre-monsoon rainfall
- Rainfall variability
- Drought indices (SPI, SPEI)
- Antecedent precipitation

### 3.3 Soil Properties

| Dataset | Resolution | Source |
|---------|------------|--------|
| ISRIC SoilGrids | 250m | ISRIC |
| FAO Harmonized World Soil | 1km | FAO |
| OpenLandMap Soil | 250m | OpenGeoHub |

**Properties available:**
- Sand/Silt/Clay content (%)
- Bulk density
- Organic carbon
- pH
- Hydraulic conductivity (derived)
- Water holding capacity
- Infiltration rate (derived)

### 3.4 Land Cover / Land Use

| Dataset | Resolution | Year | Source |
|---------|------------|------|--------|
| ESA WorldCover | 10m | 2020, 2021 | ESA |
| Dynamic World | 10m | Near real-time | Google |
| MODIS Land Cover | 500m | Annual | NASA |
| Copernicus Global LC | 100m | Annual | Copernicus |
| GlobeLand30 | 30m | 2000, 2010, 2020 | China |
| FROM-GLC | 10m | 2017 | Tsinghua |

**Derived products:**
- Agricultural area %
- Urban area %
- Forest cover %
- Water body area
- Impervious surface %
- Vegetation fraction

### 3.5 Vegetation Indices

| Dataset | Resolution | Frequency | Source |
|---------|------------|-----------|--------|
| MODIS NDVI | 250m | 16-day | NASA |
| MODIS EVI | 250m | 16-day | NASA |
| Sentinel-2 NDVI | 10m | 5-day | ESA |
| Landsat NDVI | 30m | 16-day | USGS |

**What it indicates:**
- Crop health
- Irrigation status
- Water availability proxy
- Evapotranspiration proxy

### 3.6 Evapotranspiration

| Dataset | Resolution | Source |
|---------|------------|--------|
| MODIS ET | 500m | NASA |
| SSEBop ET | 1km | USGS |
| OpenET | 30m | OpenET consortium |
| GLEAM ET | 25km | GLEAM |

**What it indicates:**
- Water consumption
- Irrigation demand
- Groundwater extraction proxy

### 3.7 Surface Water

| Dataset | Resolution | Source |
|---------|------------|--------|
| JRC Global Surface Water | 30m | EC JRC |
| HydroSHEDS | Various | WWF |
| Global River Widths | Vector | UMass |
| GRanD (dams) | Point | GRanD |

**What's available:**
- Water body extent
- Seasonal water variation
- River networks
- Dam locations
- Reservoir storage

### 3.8 Groundwater-Specific

| Dataset | Description | Source |
|---------|-------------|--------|
| GRACE/GRACE-FO | Groundwater storage anomaly | NASA |
| WHYMAP | Global groundwater resources | BGR |
| British Geological Survey | Aquifer productivity | BGS |

### 3.9 Climate / Weather

| Dataset | Resolution | Source |
|---------|------------|--------|
| ERA5-Land | 9km | ECMWF |
| MERRA-2 | 50km | NASA |
| WorldClim | 1km | WorldClim |

**Variables:**
- Temperature (min, max, mean)
- Solar radiation
- Wind speed
- Humidity
- Reference ET

---

## Part 4: Google Earth Engine Catalog

All these are accessible via GEE with a free account:

### Terrain
```
USGS/SRTMGL1_003          - SRTM 30m DEM
JAXA/ALOS/AW3D30/V3_2     - ALOS 30m DEM
COPERNICUS/DEM/GLO30      - Copernicus 30m DEM
```

### Precipitation
```
UCSB-CHG/CHIRPS/DAILY     - CHIRPS daily rainfall
NASA/GPM_L3/IMERG_V06     - GPM rainfall
ECMWF/ERA5_LAND/DAILY     - ERA5 climate
```

### Soil
```
OpenLandMap/SOL/SOL_*     - Soil properties
projects/soilgrids-isric/* - ISRIC SoilGrids
```

### Land Cover
```
ESA/WorldCover/v200/2021  - ESA WorldCover 10m
GOOGLE/DYNAMICWORLD/V1    - Dynamic World (near real-time)
MODIS/061/MCD12Q1         - MODIS Land Cover
```

### Vegetation
```
MODIS/061/MOD13A2         - MODIS NDVI/EVI
COPERNICUS/S2_SR          - Sentinel-2 imagery
LANDSAT/LC08/C02/T1_L2    - Landsat 8
```

### Water
```
JRC/GSW1_4/GlobalSurfaceWater - Surface water extent
WWF/HydroSHEDS/*          - River networks, basins
```

### Evapotranspiration
```
MODIS/061/MOD16A2         - MODIS ET
```

---

## Part 5: Derived Features (What We Can Calculate)

### From DEM
| Feature | Description | Relevance |
|---------|-------------|-----------|
| Elevation | Height above sea level | Affects water table depth |
| Slope | Terrain steepness | Affects runoff vs infiltration |
| Aspect | Direction facing | Affects solar radiation, ET |
| TWI | Topographic Wetness Index | Predicts water accumulation |
| Curvature | Surface shape | Affects flow patterns |
| Flow accumulation | Upstream area | Drainage patterns |
| Distance to drainage | Proximity to streams | Recharge potential |

### From Rainfall
| Feature | Description | Relevance |
|---------|-------------|-----------|
| Annual total | Total yearly rainfall | Primary recharge source |
| Monsoon total | June-September rainfall | Main recharge period |
| Pre-monsoon | March-May rainfall | Affects pre-monsoon levels |
| Variability (CV) | Rainfall consistency | Risk factor |
| Recent rainfall | Last 3-6 months | Current conditions |
| Drought index | SPI/SPEI | Stress indicator |

### From Land Cover
| Feature | Description | Relevance |
|---------|-------------|-----------|
| Agricultural % | Cropland fraction | Extraction intensity |
| Irrigated % | Irrigated area | Direct extraction |
| Urban % | Built-up area | Impervious surface |
| Forest % | Tree cover | Natural recharge |
| Water body % | Surface water | Recharge source |

### From Vegetation (NDVI)
| Feature | Description | Relevance |
|---------|-------------|-----------|
| Mean NDVI | Average greenness | Water availability proxy |
| NDVI trend | Greening/browning | Long-term water change |
| Seasonal amplitude | Monsoon response | Irrigation dependency |

### From Wells Data
| Feature | Description | Relevance |
|---------|-------------|-----------|
| Well density | Wells per sq km | Extraction pressure |
| Avg bore depth | Mean drilling depth | Water table proxy |
| Max bore depth | Deepest well | Minimum water table |
| Total pump capacity | Extraction potential | Demand estimate |
| Irrigated area | Area under irrigation | Water consumption |
| Crop intensity | Cropping pattern | Seasonal demand |

---

## Part 6: Summary - What to Use

### Tier 1: Essential (Use these)
| Data | Source | Why |
|------|--------|-----|
| Piezometer readings | Department | Training data |
| Aquifer boundaries | Department | Primary constraint |
| Well data | Department | Extraction proxy |
| Geomorphology | Department | Recharge potential |

### Tier 2: High Value (Strongly recommended)
| Data | Source | Why |
|------|--------|-----|
| DEM + derivatives | SRTM (free) | Terrain affects water flow |
| Recent rainfall | CHIRPS (free) | Recharge driver |
| Land cover | ESA WorldCover (free) | Extraction patterns |

### Tier 3: Enhancement (Nice to have)
| Data | Source | Why |
|------|--------|-----|
| NDVI | MODIS/Sentinel | Vegetation water use |
| ET | MODIS | Actual water consumption |
| Soil properties | ISRIC | Infiltration capacity |
| Surface water | JRC | Recharge sources |

### Tier 4: Advanced (Impressive but complex)
| Data | Source | Why |
|------|--------|-----|
| GRACE GW anomaly | NASA | Groundwater storage trends |
| High-res imagery | Sentinel-2 | Crop identification |
| Canal network | India-WRIS | Recharge from canals |

---

## Part 7: Quick Access Links

| Data | Direct Link |
|------|-------------|
| SRTM DEM | https://earthexplorer.usgs.gov |
| CHIRPS Rainfall | https://data.chc.ucsb.edu/products/CHIRPS-2.0/ |
| ISRIC Soil | https://soilgrids.org |
| ESA WorldCover | https://worldcover2021.esa.int |
| India-WRIS | https://indiawris.gov.in |
| Bhuvan | https://bhuvan.nrsc.gov.in |
| CGWB | https://cgwb.gov.in |
| IMD Rainfall | https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html |
| Google Earth Engine | https://earthengine.google.com |

---

## Part 8: Recommendation

### For Winning the Hackathon

**Must use:**
1. All department data (piezometers, aquifers, wells, geomorphology)
2. DEM + slope + TWI (easy to get, high impact)
3. Recent rainfall (CHIRPS - explains current conditions)

**Should use:**
4. Land cover (ESA WorldCover - irrigation patterns)
5. NDVI (crop water demand proxy)

**Could use (differentiator):**
6. Soil properties (infiltration)
7. Distance to rivers/canals
8. GRACE groundwater anomaly (trend validation)

This combination uses **publicly available data** that the jury might not expect, while staying grounded in the core department data.
