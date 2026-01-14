# Publicly Available Data Sources for SmartJal

This document outlines publicly available datasets that can enhance the SmartJal groundwater prediction approach beyond the provided challenge data.

---

## 1. Groundwater Data

| Source | Description | Resolution | Access |
|--------|-------------|------------|--------|
| [CGWB Data Portal](https://gwdata.cgwb.gov.in/) | National piezometer network (25,000+ stations), quarterly measurements | Point data | Free |
| [India-WRIS](https://indiawris.gov.in/wris/#/) | Integrated water resources data including groundwater | Various | Free |
| [India Data Portal - Groundwater](https://ckandev.indiadataportal.com/dataset/groundwater) | Historical groundwater level changes since 1969 | Point data | Free |
| [Nature Scientific Data (2025)](https://www.nature.com/articles/s41597-025-05899-5) | Quality-controlled GW data from 32,299 wells with specific yield | Point data | Free |

---

## 2. Rainfall & Climate Data

| Source | Description | Resolution | Access |
|--------|-------------|------------|--------|
| [IMD Gridded Rainfall](https://www.imdpune.gov.in/Clim_Pred_LRF_New/Grided_Data_Download.html) | Daily gridded data (1901-2024) | 0.25° × 0.25° | Free |
| [IMD CDSP Portal](https://cdsp.imdpune.gov.in/) | Temperature and rainfall downloads | Various | Free |
| [IMDLIB Python Library](https://www.sciencedirect.com/science/article/abs/pii/S1364815223002554) | Open-source tool for IMD data retrieval | Programmatic | Free |
| [data.gov.in Rainfall](https://www.data.gov.in/catalog/rainfall-india) | Station-wise and gridded rainfall | Various | Free |

### Citation for IMD Gridded Data
> Pai D.S., Latha Sridhar, Rajeevan M., Sreejith O.P., Satbhai N.S. and Mukhopadhyay B., 2014: Development of a new high spatial resolution (0.25° X 0.25°) Long period (1901-2010) daily gridded rainfall data set over India and its comparison with existing data sets over the region; MAUSAM, 65, 1(January 2014), pp1-18.

---

## 3. Digital Elevation Model (DEM)

| Source | Description | Resolution | Access |
|--------|-------------|------------|--------|
| [SRTM 30m Downloader](https://dwtkns.com/srtm30m/) | Simple tile-based SRTM download | 30m | Free |
| [USGS Earth Explorer](https://earthexplorer.usgs.gov/) | SRTM 1 Arc-Second Global | 30m | Free |
| [Bhuvan - Cartosat-1 DEM](https://bhuvan.nrsc.gov.in/) | India-specific DEM from ISRO | 30m | Free |
| [Google Earth Engine SRTM](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003) | Cloud-based access | 30m | Free |

### Derived Products from DEM
- Slope and aspect maps
- Drainage network and watershed delineation
- Topographic Wetness Index (TWI)
- Flow accumulation maps

---

## 4. Soil Data

| Source | Description | Resolution | Access |
|--------|-------------|------------|--------|
| [ISRIC SoilGrids](https://soilgrids.org) | pH, organic carbon, bulk density, sand/silt/clay, CEC | 250m | Free (CC-BY 4.0) |
| [SoilGrids on GEE](https://gee-community-catalog.org/projects/isric/) | Same data, cloud-accessible | 250m | Free |

### Available Soil Properties
- pH
- Soil organic carbon content
- Bulk density
- Coarse fragments content
- Sand, silt, clay content
- Cation exchange capacity (CEC)
- Total nitrogen
- Soil organic carbon density and stock

**Depth intervals:** 0-5cm, 5-15cm, 15-30cm, 30-60cm, 60-100cm, 100-200cm

### Relevance for Groundwater
Soil properties directly influence:
- Infiltration rates
- Percolation to groundwater
- Water holding capacity
- Recharge potential estimation

---

## 5. Land Use / Land Cover (LULC)

| Source | Description | Resolution | Access |
|--------|-------------|------------|--------|
| [Bhuvan LULC 250K](https://bhuvan.nrsc.gov.in/) | Annual maps since 2004-05 (AWiFS data) | 1:250,000 | Free |
| [Bhuvan LULC 50K](https://bhuvan.nrsc.gov.in/) | 2005-06, 2011-12, 2015-16 cycles (LISS-III) | 1:50,000 | Free |
| [ORNL DAAC India LULC](https://daac.ornl.gov/VEGETATION/guides/Decadal_LULC_India.html) | Decadal (1985, 1995, 2005) | 100m | Free |
| [Dynamic World (Google)](https://dynamicworld.app/) | Near real-time global LULC | 10m | Free |
| [ESA WorldCover](https://worldcover2021.esa.int/) | Global land cover 2020, 2021 | 10m | Free |

### Relevance for Groundwater
- Agricultural areas indicate extraction zones
- Urban areas indicate impervious surfaces (reduced recharge)
- Forest/vegetation cover affects evapotranspiration
- Water bodies indicate surface water availability

---

## 6. GRACE Satellite - Groundwater Storage Anomaly

| Source | Description | Resolution | Access |
|--------|-------------|------------|--------|
| [GRACE Tellus Data Portal](https://grace.jpl.nasa.gov/data/get-data/) | Total water storage anomalies | ~300km | Free |
| [GRACE Data Analysis Tool](https://grace.jpl.nasa.gov/data/data-analysis-tool/) | Interactive analysis | 1° grid | Free |
| [Drought.gov GRACE](https://www.drought.gov/data-maps-tools/groundwater-and-soil-moisture-conditions-grace-data-assimilation) | Weekly GW drought indicators | Various | Free |

### Key Findings for India
- Northwest India Aquifer shows highest global groundwater depletion rate
- Depletion rates up to ~60mm/year of groundwater storage loss
- GRACE data available from 2002-present (with gap 2017-2018)

### Use Cases
- Validate regional groundwater trends
- Independent verification of model predictions
- Long-term storage change analysis

---

## 7. Evapotranspiration (ET)

| Source | Description | Resolution | Access |
|--------|-------------|------------|--------|
| [MODIS MOD16A2](https://lpdaac.usgs.gov/products/mod16a2v006/) | 8-day ET, PET, Latent Heat | 500m | Free |
| [LAADS DAAC MOD16](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD16A2) | Same, different portal | 500m | Free |

### Water Balance Equation for Recharge
```
Recharge = Precipitation - Evapotranspiration - Runoff - ΔSoil Storage
```

Where:
- **Ri** = Recharge
- **P** = Precipitation
- **Ea** = Actual evapotranspiration
- **ΔW** = Change in soil water storage
- **Ro** = Runoff

### Algorithm
MODIS ET uses the Penman-Monteith equation with:
- Daily meteorological reanalysis data
- MODIS remotely sensed data (LAI, FPAR)

---

## 8. Geology & Lithology

| Source | Description | Resolution | Access |
|--------|-------------|------------|--------|
| [GSI Bhukosh Portal](https://bhukosh.gsi.gov.in/) | Lithological maps of India | 1:250K, 1:50K | Free |
| [data.gov.in GSI](https://www.data.gov.in/ministrydepartment/Geological%20Survey%20of%20India) | Open government geological data | Various | Free |
| [USGS South Asia Geologic Map](https://catalog.data.gov/dataset/geologic-map-of-south-asia-geo8ag) | Regional geology shapefile | Regional | Free |
| [NDEM GSI Data](https://ndem.nrsc.gov.in/geological_gsi.php) | Geological Survey data via NRSC | Various | Free |

### Available Products from GSI
- **Geological Quadrangle Maps (GQM)** - 1:250,000 scale with lithology, structure, mineral resources, stratigraphy
- **District Resources Map (DRM)** - 1:250,000 scale with geology, mineral potential, groundwater resources
- **Geological Map Series (GMS)** - 1:50,000 scale seamless maps

### Relevance for Groundwater
- Aquifer type and permeability estimation
- Recharge zone identification
- Groundwater flow direction inference
- Storage coefficient estimation

---

## 9. Administrative Boundaries (Village-level)

| Source | Description | Access |
|--------|-------------|--------|
| [DataMeet Village Boundaries](https://projects.datameet.org/indian_village_boundaries/) | Open-source village polygons | Free |
| [SHRUG Dataset](https://docs.devdatalab.org/SHRUG-Construction-Details/shrug-open-source-polygons/) | Socio-economic + boundaries | Free |
| [NASA Village-Level Data](https://data.nasa.gov/dataset/india-village-level-geospatial-socio-economic-data-set-1991-2001) | 1991, 2001 census + geo | Free |
| [Survey of India](https://onlinemaps.surveyofindia.gov.in/) | Official boundaries | Free (PDF) |
| [IGISMAP India Boundaries](https://www.igismap.com/download-india-administrative-boundary-shapefiles-states-districts-sub-districts-pincodes-constituencies/) | States, districts, sub-districts, pincodes | Free/Paid |

---

## 10. Google Earth Engine - Integrated Platform

Google Earth Engine provides a single cloud platform to access most datasets above, plus additional resources.

### Available Datasets on GEE

| Dataset | Description | Resolution |
|---------|-------------|------------|
| Sentinel-2 | Multispectral imagery | 10m |
| Landsat Collection | Historical imagery (1984+) | 30m |
| CHIRPS | Daily precipitation | 5km |
| TerraClimate | Monthly climate variables | 4km |
| Global Surface Water | Water occurrence maps | 30m |
| SRTM DEM | Elevation | 30m |
| SoilGrids | Soil properties | 250m |
| MODIS Products | Various (ET, NDVI, LST) | 250m-1km |

### Relevant Tutorials
- [GEE Groundwater Recharge Estimation](https://developers.google.com/earth-engine/tutorials/community/groundwater-recharge-estimation) - Thornthwaite-Mather procedure
- [GEE for Water Resources Management](https://courses.spatialthoughts.com/gee-water-resources-management.html) - Full course
- [Machine Learning in Earth Engine](https://developers.google.com/earth-engine/guides/machine-learning) - ML APIs documentation

### Benefits of GEE
- No data download required
- Parallel processing at scale
- Built-in ML capabilities
- Access to petabytes of geospatial data
- JavaScript and Python APIs available

---

## Recommended Data Enhancement Strategy

### Phase 1: Immediate Value Additions
1. **IMD Gridded Rainfall (0.25°)** - Direct correlation with recharge
2. **SRTM DEM** - Derive slope, aspect, drainage patterns, TWI
3. **SoilGrids** - Infiltration capacity and permeability estimation
4. **MODIS ET** - Water balance calculations

### Phase 2: Medium-term Enhancements
1. **GRACE Data** - Validate regional groundwater trends
2. **Updated LULC** - Sentinel-2 or Dynamic World for current land use
3. **GSI Lithology Maps** - Aquifer permeability estimation
4. **Village Boundaries** - Proper spatial aggregation

### Phase 3: Advanced Modeling
1. Use **Google Earth Engine** for integrated multi-source analysis
2. Combine **GRACE + Rainfall + ET** for regional recharge modeling
3. Build **ML models** with hydrogeologically-relevant features
4. Implement **Thornthwaite-Mather** procedure for recharge estimation

---

## Data Integration Considerations

### Coordinate Systems
- Most Indian government data uses **Everest Datum** (GCS_Everest_Def_1962)
- Global datasets typically use **WGS84**
- Ensure proper reprojection when combining datasets

### Temporal Alignment
- Align temporal resolution (daily, monthly, seasonal) across datasets
- Account for monsoon seasonality (June-September)
- Match pre-monsoon (May) and post-monsoon (November) observation periods

### Spatial Resolution Harmonization
- Resample all rasters to common resolution
- Consider using village polygons for zonal statistics
- Account for scale differences when combining data

---

## References

1. Pai, D.S., et al. (2014). Development of a new high spatial resolution gridded rainfall data set over India. MAUSAM, 65(1), 1-18.
2. CGWB (2024). Ground Water Year Book - India.
3. Rodell, M., et al. (2009). Satellite-based estimates of groundwater depletion in India. Nature, 460(7258), 999-1002.
4. Hengl, T., et al. (2017). SoilGrids250m: Global gridded soil information. PLOS ONE, 12(2), e0169748.
