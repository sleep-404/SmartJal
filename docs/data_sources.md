# SmartJal Data Sources

## Summary: What You Need

| # | Data | Resolution | Source | How to Get |
|---|------|------------|--------|------------|
| 1 | DEM (Elevation) | 30m | SRTM | Script / EarthExplorer |
| 2 | Rainfall (recent) | 5km | CHIRPS | Script / Direct download |
| 3 | Soil Properties | 250m | ISRIC SoilGrids | Script / REST API |
| 4 | Land Cover | 10m | ESA WorldCover | Script / Direct download |
| 5 | Village Boundaries | Vector | DataMeet | Script / GitHub |

---

## Quick Start

```bash
cd backend
source .venv/bin/activate

# Download all data
python scripts/download_data.py --all

# Or download specific datasets
python scripts/download_data.py --rainfall
python scripts/download_data.py --landcover
```

---

## 1. DEM (Digital Elevation Model)

### Why Needed
- Calculate slope, aspect, terrain wetness index (TWI)
- Identify drainage patterns and water flow

### Source: SRTM 30m
- **Resolution**: 30 meters
- **Provider**: NASA/USGS
- **Data Type**: Static (doesn't change)

### Download Options

#### Option A: OpenTopography API (Recommended)
1. Create free account at https://opentopography.org
2. Get API key from profile
3. Set environment variable:
   ```bash
   export OPENTOPO_API_KEY=your_key_here
   ```
4. Run: `python scripts/download_data.py --dem`

#### Option B: USGS EarthExplorer
1. Go to https://earthexplorer.usgs.gov
2. Search for "SRTM 1 Arc-Second Global"
3. Draw box over Krishna District (80-82°E, 15-17°N)
4. Download tiles and merge

#### Option C: Google Earth Engine
```python
import ee
ee.Initialize()

dem = ee.Image('USGS/SRTMGL1_003')
krishna = ee.Geometry.Rectangle([80.0, 15.5, 81.5, 17.0])

# Export to Drive
task = ee.batch.Export.image.toDrive(
    image=dem.clip(krishna),
    description='krishna_dem',
    scale=30,
    region=krishna
)
task.start()
```

---

## 2. Rainfall Data

### Why Needed
- Primary driver of groundwater recharge
- Seasonal patterns affect water levels
- Recent rainfall predicts current conditions

### Source: CHIRPS
- **Resolution**: ~5km (0.05°)
- **Frequency**: Daily
- **Latency**: ~2-3 weeks
- **Period**: 1981-present

### Download Options

#### Option A: IRI Data Library (Script)
```bash
python scripts/download_data.py --rainfall
```

#### Option B: CHIRPS Direct Download
1. Go to https://data.chc.ucsb.edu/products/CHIRPS-2.0/
2. Navigate to `global_daily/netcdf/p25/`
3. Download files for 2023-2025

#### Option C: Google Earth Engine
```python
import ee
ee.Initialize()

chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
    .filterDate('2024-01-01', '2025-01-01') \
    .filterBounds(krishna)

# Get monthly totals
monthly = chirps.sum()
```

### IMD (India Meteorological Department)
For more accurate local data:
1. Go to https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html
2. Download gridded rainfall data
3. Higher accuracy for India but requires registration

---

## 3. Soil Properties

### Why Needed
- Hydraulic conductivity affects infiltration
- Soil texture affects water retention
- Sand/clay content determines aquifer recharge

### Source: ISRIC SoilGrids
- **Resolution**: 250m
- **Properties**: Sand%, Clay%, Bulk Density, Organic Carbon
- **Depths**: 0-5cm, 5-15cm, 15-30cm, 30-60cm

### Download Options

#### Option A: REST API (Script)
```bash
python scripts/download_data.py --soil
```

#### Option B: SoilGrids Web Interface
1. Go to https://soilgrids.org
2. Navigate to Krishna District
3. Use download tools

#### Option C: Google Earth Engine
```python
import ee
ee.Initialize()

# Sand content 0-5cm
sand = ee.Image("projects/soilgrids-isric/sand_mean").select('sand_0-5cm_mean')

# Clay content
clay = ee.Image("projects/soilgrids-isric/clay_mean").select('clay_0-5cm_mean')
```

---

## 4. Land Use / Land Cover

### Why Needed
- Irrigated areas have different recharge patterns
- Urban areas have impervious surfaces
- Forest areas affect infiltration

### Source: ESA WorldCover 2021
- **Resolution**: 10m
- **Classes**: 11 land cover types
- **Year**: 2021 (most recent global)

### Land Cover Classes
| Code | Class |
|------|-------|
| 10 | Tree cover |
| 20 | Shrubland |
| 30 | Grassland |
| 40 | Cropland |
| 50 | Built-up |
| 60 | Bare/sparse vegetation |
| 70 | Snow and ice |
| 80 | Permanent water bodies |
| 90 | Herbaceous wetland |
| 95 | Mangroves |
| 100 | Moss and lichen |

### Download Options

#### Option A: Direct Download (Script)
```bash
python scripts/download_data.py --landcover
```

#### Option B: ESA Website
1. Go to https://worldcover2021.esa.int/download
2. Select tiles N15E078 and N15E081
3. Download GeoTIFF files

#### Option C: Google Earth Engine
```python
import ee
ee.Initialize()

worldcover = ee.Image("ESA/WorldCover/v200/2021")
krishna_lulc = worldcover.clip(krishna)
```

### Alternative: Dynamic World (Near Real-Time)
For more recent land cover:
```python
dynamic_world = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
    .filterDate('2024-01-01', '2025-01-01') \
    .filterBounds(krishna) \
    .mode()  # Most common class
```

---

## 5. Village Boundaries

### Why Needed
- Prediction targets (one value per village)
- Spatial aggregation of features
- Mapping visualization

### Source Options

#### Option A: DataMeet (Open Source)
```bash
python scripts/download_data.py --villages
```
- GitHub: https://github.com/datameet/maps
- Contains village boundaries for all states

#### Option B: Bhuvan (ISRO)
1. Go to https://bhuvan.nrsc.gov.in
2. Login (free registration)
3. Download administrative boundaries
4. Higher accuracy than DataMeet

#### Option C: Census of India
1. Go to https://censusindia.gov.in/maps
2. Download village-level maps
3. Official government source

---

## 6. Additional Data (Optional)

### Surface Water Bodies (Tanks, Reservoirs)
- **India-WRIS**: https://indiawris.gov.in
- **JRC Global Surface Water**: GEE `JRC/GSW1_4/GlobalSurfaceWater`

### Vegetation Index (NDVI)
- **MODIS**: GEE `MODIS/061/MOD13A2`
- **Sentinel-2**: GEE `COPERNICUS/S2_SR`

### Evapotranspiration
- **MODIS ET**: GEE `MODIS/061/MOD16A2`

### Groundwater Data (Government)
- **CGWB**: https://cgwb.gov.in
- **India-WRIS**: https://indiawris.gov.in/wris

---

## Google Earth Engine Setup

For the best experience, set up Google Earth Engine:

### 1. Create Account
1. Go to https://earthengine.google.com
2. Sign up (free for research/education)
3. Wait for approval (usually instant)

### 2. Authenticate
```bash
pip install earthengine-api
earthengine authenticate
```

### 3. Use in Python
```python
import ee
ee.Initialize()

# Now you can access all GEE datasets
```

### 4. Set up Service Account (for server)
1. Go to https://console.cloud.google.com
2. Create service account
3. Download JSON key
4. Set in .env:
   ```
   GEE_SERVICE_ACCOUNT=your-account@project.iam.gserviceaccount.com
   GEE_KEY_FILE=/path/to/key.json
   ```

---

## Data Recency Summary

| Data | Update Frequency | Latest Available |
|------|------------------|------------------|
| DEM | Static | N/A (terrain doesn't change) |
| Rainfall | Daily | Yesterday (2-3 week lag for CHIRPS) |
| Soil | Static | N/A (soil changes very slowly) |
| Land Cover | Annual | 2021 (WorldCover), Real-time (Dynamic World) |
| Water Levels | Monthly | November 2025 (provided data) |

---

## Recommended Approach

1. **For Quick Start**: Run the download script
   ```bash
   python scripts/download_data.py --all
   ```

2. **For Best Quality**: Use Google Earth Engine
   - All data in one place
   - Easy subsetting and processing
   - Most recent data available

3. **For Production**: Set up automated data pipelines
   - Daily rainfall updates
   - Weekly NDVI updates
   - Monthly model retraining
