# Smart Jal - Open Data Sources

All data required for the POC is available freely online. **No dependency on officials required.**

---

## Quick Reference

| Data Type | Source | Resolution | Effort |
|-----------|--------|------------|--------|
| Village Boundaries | Survey of India | Village-level | 30 min |
| Rainfall | CHIRPS / IMD | 5km / 25km | 30 min |
| DEM | SRTM | 30m | 15 min |
| Soil | ISRIC SoilGrids | 250m | 30 min |
| **Validation Data** | NWDP / CGWB | Station-level | 30 min |
| GRACE (differentiator) | NASA/GEE | 50km | 1-2 hrs |

---

## 1. Village Boundaries

### Option A: Survey of India (Recommended - Official)
- **URL**: https://onlinemaps.surveyofindia.gov.in/
- **Coverage**: Village-level for all India
- **Format**: Shapefile, GeoDatabase
- **Cost**: FREE (registration required)

**Steps**:
1. Go to https://onlinemaps.surveyofindia.gov.in/
2. Click "Sign In" → Register (free)
3. After login: Products → Village Boundary Database
4. Select: State = **Andhra Pradesh**, District = **Krishna**
5. Click "Click to Buy" → Add to Cart
6. Place Order → Generate Download Link → Download

### Option B: DataMeet GitHub
- **URL**: https://github.com/datta07/INDIAN-SHAPEFILES
- **Format**: GeoJSON
- **Coverage**: States, Districts, Sub-districts
- **Note**: May not have village-level, but has mandal/taluk

### Option C: Indian Village Boundaries Project
- **URL**: https://projects.datameet.org/indian_village_boundaries/
- **Format**: GeoJSON
- **License**: Open Database License (ODbL)

---

## 2. Rainfall Data

### Option A: CHIRPS (Recommended - Higher Resolution)
- **URL**: https://data.chc.ucsb.edu/products/CHIRPS-2.0/
- **Resolution**: 0.05° (~5km)
- **Period**: 1981-present
- **Format**: GeoTIFF, NetCDF
- **Cost**: FREE (Public Domain)

**Direct Download**:
- Monthly: https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/
- Daily: https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/

**Google Earth Engine**:
```javascript
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate('2015-01-01', '2024-12-31')
  .filterBounds(krishna_geometry);
```

### Option B: IMD Gridded Rainfall
- **URL**: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html
- **Resolution**: 0.25° (~25km)
- **Period**: 1901-2024
- **Format**: NetCDF
- **Cost**: FREE

**Python Library**:
```python
pip install imdlib

import imdlib as imd
data = imd.get_data('rain', 2015, 2024, 'yearwise')
```

**GitHub Tool**: https://github.com/iamsaswata/imdlib

---

## 3. DEM (Digital Elevation Model)

### Option A: USGS EarthExplorer (Manual)
- **URL**: https://earthexplorer.usgs.gov/
- **Product**: SRTM 1 Arc-Second Global
- **Resolution**: 30m
- **Cost**: FREE (registration required)

**Steps**:
1. Go to https://earthexplorer.usgs.gov/
2. Register (free)
3. Search Criteria → Enter coordinates:
   - North: 17.0, South: 15.5
   - East: 81.5, West: 80.0
4. Data Sets → Digital Elevation → SRTM → SRTM 1 Arc-Second Global
5. Results → Download GeoTIFF

### Option B: Google Earth Engine
```javascript
var dem = ee.Image('USGS/SRTMGL1_003')
  .clip(krishna_geometry);

Export.image.toDrive({
  image: dem,
  description: 'krishna_dem',
  scale: 30,
  region: krishna_geometry
});
```

### Option C: Online Tool (Easiest)
- **URL**: https://geoprocessing.online/tool/srtm-dem-download/
- Draw polygon, download directly

---

## 4. Validation Data (CRITICAL FOR TESTING!)

### Option A: National Water Data Portal (NWDP)

**AP Groundwater - Manual Quarterly (2021-2025)**:
- **URL**: https://nwdp.nwic.in/en/dataset/ground-water-level-manual-quarterly-andhra-pradesh-ground-water-departments
- **Format**: CSV
- **Direct CSV**: https://nwdp.nwic.in/dataset/.../gwl_manual_quarterly_andhra_pradesh_gw_ap_2021_2025.csv

**AP Groundwater - Telemetry Hourly (1991-2025)**:
- **URL**: https://nwdp.nwic.in/en/dataset/ground-water-level-telemetry-hourly-andhra-pradesh-ground-water-department
- **Format**: CSV

### Option B: CGWB (Central Ground Water Board)
- **URL**: https://cgwb.gov.in/en/ground-water-level-monitoring
- **Data Available**:
  - January water level: 1994-2024
  - April/May water level: 1994-2024
  - August water level: 1994-2023
  - November water level: 1994-2023

### Option C: India-WRIS
- **URL**: https://indiawris.gov.in/wris/#/DataDownload
- **Requires**: Free registration
- **Coverage**: Pan-India groundwater data

**Testing Strategy**:
1. Train model on provided Krishna piezometer data (138 stations)
2. Validate against CGWB/NWDP data for same locations
3. Test temporal consistency with telemetry data

---

## 5. Soil Data

### ISRIC SoilGrids
- **URL**: https://soilgrids.org/
- **Resolution**: 250m
- **Properties**: Sand%, Clay%, Silt%, Bulk density, CEC, pH
- **Cost**: FREE

**REST API**:
```python
import requests

url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
params = {
    "lon": 80.5,
    "lat": 16.5,
    "property": ["sand", "clay", "silt", "bdod"],
    "depth": ["0-5cm", "5-15cm", "15-30cm"]
}
response = requests.get(url, params=params)
```

**Google Earth Engine**:
```javascript
var sand = ee.Image("projects/soilgrids-isric/sand_mean");
var clay = ee.Image("projects/soilgrids-isric/clay_mean");
```

---

## 6. GRACE Satellite Data (DIFFERENTIATOR!)

### Why Use GRACE?
- Provides **regional groundwater storage** anomaly
- Acts as constraint for village-level predictions
- **Nobody else at hackathon will use this!**

### Google Earth Engine (Easiest)
```javascript
var grace = ee.ImageCollection('NASA/GRACE/MASS_GRIDS/LAND')
  .filterBounds(krishna_geometry)
  .select('lwe_thickness_csr');

// Monthly groundwater storage anomaly in cm
```

### Direct Download
- **URL**: https://grace.jpl.nasa.gov/data/get-data/
- **Product**: GRACE-FO RL06 Mascon Solutions

### GLDAS (for GRACE separation)
- **URL**: https://ldas.gsfc.nasa.gov/gldas
- **Product**: GLDAS Noah 0.25° monthly
- **Use**: Separate soil moisture from GRACE TWS

**Formula**:
```
Groundwater_Storage = GRACE_TWS - GLDAS_SoilMoisture - GLDAS_SurfaceWater
```

---

## 7. Additional Proxy Data

### NDVI/Vegetation (Groundwater Proxy)
- **MODIS MOD13Q1**: 250m, 16-day
- **Sentinel-2**: 10m, 5-day
- **GEE Dataset**: `MODIS/006/MOD13Q1`

### Land Surface Temperature
- **MODIS MOD11A2**: 1km, 8-day
- **GEE Dataset**: `MODIS/006/MOD11A2`

### Evapotranspiration
- **MODIS MOD16A2**: 500m, 8-day
- **GEE Dataset**: `MODIS/006/MOD16A2`

---

## Krishna District Bounds

For all downloads, use these coordinates:

| Parameter | Value |
|-----------|-------|
| Min Longitude | 80.0° E |
| Max Longitude | 81.5° E |
| Min Latitude | 15.5° N |
| Max Latitude | 17.0° N |
| EPSG | 4326 (WGS84) |

---

## Google Earth Engine Setup

Many datasets are easiest to access via GEE:

1. **Sign up**: https://earthengine.google.com/signup/
2. **Code Editor**: https://code.earthengine.google.com/
3. **Python API**: `pip install earthengine-api`

```python
import ee
ee.Authenticate()
ee.Initialize()
```

---

## Download Checklist

### Day 1 (Essential)
- [ ] Village boundaries from Survey of India
- [ ] Rainfall data (CHIRPS or IMD)
- [ ] Validation data from NWDP

### Day 2 (Important)
- [ ] SRTM DEM
- [ ] Soil data from ISRIC

### Day 3 (Differentiators)
- [ ] GRACE satellite data
- [ ] NDVI time series
- [ ] LST anomaly

---

## Summary

| What | Where | Time |
|------|-------|------|
| Village Boundaries | Survey of India | 30 min |
| Rainfall | CHIRPS | 30 min |
| DEM | USGS EarthExplorer | 15 min |
| Validation Data | NWDP/CGWB | 30 min |
| Soil | ISRIC SoilGrids | 30 min |
| GRACE | GEE/NASA | 1-2 hrs |

**Total: ~4 hours to download all essential data**

---

*Document created: January 2026*
*Purpose: Data independence for Smart Jal POC*
