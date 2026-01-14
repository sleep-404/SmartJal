# Smart Jal - Research Findings

This document collates all research findings on approaches, tools, and techniques for solving the groundwater level prediction problem. These findings will inform the final architecture decision.

---

## Table of Contents

1. [Standard/Off-the-Shelf Solutions](#1-standardoff-the-shelf-solutions)
2. [Novel/Innovative Approaches](#2-novelinnovative-approaches)
3. [Satellite Data Sources](#3-satellite-data-sources)
4. [Additional "Sensors" - Multi-Signal Approach](#4-additional-sensors---multi-signal-approach)
5. [Existing India-Specific Work](#5-existing-india-specific-work)
6. [Machine Learning Approaches for Groundwater](#6-machine-learning-approaches-for-groundwater)
7. [Visualization & Integration Tools](#7-visualization--integration-tools)
8. [Key Research Papers](#8-key-research-papers)
9. [Approach Comparison Matrix](#9-approach-comparison-matrix)
10. [Recommended Technology Stack](#10-recommended-technology-stack-for-reference)
11. [Quick Reference: Differentiation Strategies](#11-quick-reference-differentiation-strategies)
12. [Official Clarifications from Department](#12-official-clarifications-from-department) **(NEW)**

---

## 1. Standard/Off-the-Shelf Solutions

### 1.1 Python Geostatistical Libraries

#### PyKrige
- **Purpose**: Kriging interpolation (Ordinary, Universal, Regression Kriging)
- **Pros**: Simple API, well-documented, supports 2D and 3D kriging
- **Cons**: Doesn't natively respect geological boundaries
- **Install**: `pip install pykrige`
- **GitHub**: https://github.com/GeoStat-Framework/PyKrige

#### Pyinterpolate
- **Purpose**: Comprehensive spatial statistics (Kriging, Poisson Kriging, IDW)
- **Pros**: Supports stratified kriging, uncertainty quantification, area-to-point interpolation
- **Cons**: Newer library, less community support
- **Version**: 1.0.0 (June 2025)
- **Install**: `pip install pyinterpolate`
- **Docs**: https://pyinterpolate.readthedocs.io/

#### GStatSim
- **Purpose**: Geostatistical interpolation and simulation
- **Pros**: Handles non-stationary data, clustering by regions, Sequential Gaussian Simulation
- **Cons**: More complex setup
- **Paper**: "GStatSim V1.0" (Geoscientific Model Development, 2023)
- **Features**:
  - Ordinary and Simple Kriging
  - Co-kriging with secondary variables
  - Cluster-based SGS for geology-aware simulation
  - Anisotropic interpolation

#### GSTools
- **Purpose**: Geostatistical toolbox
- **Pros**: Variogram modeling, random field generation
- **GitHub**: https://github.com/GeoStat-Framework/GSTools

#### SciKit-GStat
- **Purpose**: Variogram modeling and analysis
- **Pros**: Good for exploratory spatial data analysis
- **GitHub**: https://github.com/mmaelicke/scikit-gstat

#### GeostatsPy
- **Purpose**: Python wrapper for GSLIB functions
- **Pros**: Industry-standard geostatistics
- **GitHub**: https://github.com/GeostatsGuy/GeostatsPy

#### Pastas
- **Purpose**: Groundwater time series analysis
- **Pros**: Built specifically for groundwater level analysis, handles irregular time series
- **Features**: Transfer function noise modeling, step response functions
- **GitHub**: https://github.com/pastas/pastas
- **Relevance**: HIGH - directly applicable to temporal modeling (P5)

### 1.2 GIS-Based Solutions

#### QGIS + Smart-Map Plugin
- **Purpose**: GUI-based kriging workflow
- **Pros**: No-code, visual semivariogram fitting
- **Cons**: Manual stratification needed for geology-aware interpolation
- **Workflow**:
  1. Load piezometer points
  2. Generate semivariogram
  3. Fit model (Spherical, Gaussian, Exponential)
  4. Interpolate to village centroids

#### SAGA GIS
- **Purpose**: Spatial analysis platform
- **Pros**: Better interpolation options than many GIS tools
- **Website**: www.saga-gis.org

### 1.3 Groundwater Modeling Software

#### MODFLOW (USGS)
- **Purpose**: 3D finite-difference groundwater flow model
- **Pros**: Industry standard, well-validated
- **Cons**: Requires calibration, complex setup, overkill for POC
- **Versions**: MODFLOW-6, MODFLOW-NWT, MODFLOW-USG

#### FloPy
- **Purpose**: Python interface for MODFLOW
- **Pros**: Scriptable, integrates with Python ecosystem
- **GitHub**: https://github.com/modflowpy/flopy
- **Install**: `pip install flopy`

#### ModelMuse (USGS)
- **Purpose**: Free GUI for MODFLOW
- **Pros**: Open source, USGS supported
- **Website**: https://www.usgs.gov/software/modelmuse

#### FREEWAT
- **Purpose**: QGIS plugin for MODFLOW
- **Pros**: Free, integrates with GIS
- **Cons**: Steep learning curve

#### Modflow-setup
- **Purpose**: Automated MODFLOW model construction
- **Paper**: "Modflow-setup: Robust automation of groundwater model construction" (Frontiers, 2022)
- **GitHub**: https://github.com/doi-usgs/modflow-setup

#### Other Groundwater Software
| Software | Type | Notes |
|----------|------|-------|
| iMOD | Open Source | GUI + accelerated MODFLOW |
| IWFM | Open Source | Integrated surface-groundwater |
| ParFlow | Open Source | Large-scale parallel simulation |
| Visual MODFLOW Flex | Commercial | ~$3000+ license |
| FEFLOW | Commercial | Finite element, DHI |
| GMS | Commercial | Comprehensive modeling system |

### 1.4 General ML Libraries

#### scikit-learn
- **Relevant models**: Random Forest, XGBoost, SVR, KNN
- **Pros**: Well-documented, easy to use
- **Approach**: Train on piezometer features, predict for villages

#### XGBoost / LightGBM
- **Purpose**: Gradient boosting for tabular data
- **Pros**: Handles mixed feature types, feature importance
- **Relevance**: Good for incorporating multiple covariates

---

## 2. Novel/Innovative Approaches

### 2.1 Graph Neural Networks (GNN) for Spatial Interpolation

#### Concept
Model villages and piezometers as nodes in a graph where edges connect geologically similar locations, not just nearby ones. GNNs learn to propagate information along geologically valid paths.

#### Why It's Novel
- Standard approaches use distance-based interpolation
- GNNs naturally respect aquifer boundaries without manual stratification
- Can learn complex non-linear spatial relationships

#### Key Architectures

##### NN-GLS (Neural Network - Generalized Least Squares)
- **Paper**: "Neural networks for geospatial data" (arXiv:2304.09157, 2023)
- **Key idea**: Embed neural networks within Gaussian Process geostatistical model
- **Innovation**: GLS loss accounts for spatial covariance
- **Representation**: Admits representation as special type of GNN
- **Features**:
  - Non-linear mean functions
  - Explicit spatial covariance modeling
  - Kriging predictions at new locations
  - Uncertainty bounds
- **Software**: `geospaNN` Python package

##### PE-GNN (Positional Encoder GNN)
- **Paper**: "Positional Encoder Graph Neural Networks for Geographic Data" (AISTATS 2023)
- **Authors**: Microsoft Research
- **Key idea**: Learn context-aware encoding of geographic coordinates
- **Innovation**: Predicts spatial autocorrelation as auxiliary task
- **Result**: Matches Gaussian Processes on spatial interpolation
- **Code**: https://bit.ly/3xDpfyV

##### GNNWR (Geographically Neural Network Weighted Regression)
- **Paper**: International Journal of Geographical Information Science
- **Key idea**: Combines OLS and neural networks for spatial non-stationarity
- **Innovation**: Spatial proximity grid + spatially weighted CNN

##### MSA-GNN-HMP (Multi-Scale Attention GNN)
- **Paper**: "Innovative graph neural network approach for predicting soil heavy metal content" (2024)
- **Key idea**: Multi-scale graph convolution + attention-based GNN
- **Relevance**: Demonstrated for soil properties prediction
- **Features**: Captures spatial relationships at multiple scales

#### Implementation Approach
```python
# Conceptual using PyTorch Geometric
import torch_geometric

# Build graph
# - Nodes: villages (target) + piezometers (source)
# - Node features: aquifer type, geology, rainfall, elevation, etc.
# - Edge weights: f(distance, aquifer_similarity, geology_match)

# GNN architecture
# - Graph convolution layers propagate water level information
# - Attention mechanism weights influence of each piezometer
# - Output: predicted water level + uncertainty per village
```

#### Differentiation Factor: ⭐⭐⭐⭐⭐

---

### 2.2 Physics-Informed Neural Networks (PINN)

#### Concept
Neural networks that embed physical laws (Darcy's law, groundwater flow equations) as constraints in the loss function. Predictions must obey physics.

#### Why It's Novel
- Works with sparse data because physics provides regularization
- Can infer hydraulic parameters from observations (inverse modeling)
- Handles missing/incomplete boundary conditions
- Validated for groundwater in multiple 2022-2024 papers

#### Mathematical Formulation
```
Total Loss = Data_Loss + Physics_Loss + Boundary_Loss

where:
- Data_Loss = ||predicted - observed||² at piezometer locations
- Physics_Loss = ||∂h/∂t - ∇·(K·∇h) - R||² (groundwater flow PDE)
- Boundary_Loss = constraints at domain boundaries

h = hydraulic head
K = hydraulic conductivity
R = recharge/discharge
```

#### Key Papers

##### "Forward and inverse modeling of water flow in unsaturated soils with PINNs"
- **Journal**: Hydrology and Earth System Sciences (2022)
- **Key findings**:
  - PINNs with adaptive activation comparable to traditional methods
  - Domain decomposition for layered soils
  - No strict requirement for initial/boundary conditions
  - Provides inverse solution without repeatedly solving forward problem

##### "Physics-Informed Neural Networks for solving transient unconfined groundwater flow"
- **Journal**: Computers & Geosciences (2024)
- **Relevance**: Directly applicable to unconfined aquifer modeling

##### "KLE-PINN for groundwater inverse modeling"
- **Journal**: Journal of Hydrology (2024)
- **Key idea**: Combines PINN with Karhunen-Loeve Expansion
- **Features**:
  - Robust inverse method
  - Works with missing boundary conditions
  - Handles hydraulic tomography data

##### "PINN-HYDRO: Temporally Adaptive PINNs"
- **Conference**: AMS Meeting 2025
- **Innovation**: Incorporates meteorological forcing variables
- **Features**:
  - Attention mechanism for temporal dynamics
  - Handles rainfall-groundwater lag correlations

#### Implementation Libraries
- **DeepXDE**: General PINN library
- **PyTorch**: Custom implementation
- **TensorFlow**: Custom implementation

#### Differentiation Factor: ⭐⭐⭐⭐

---

### 2.3 GRACE Satellite Data Integration

#### Concept
NASA's GRACE (Gravity Recovery and Climate Experiment) satellites measure gravity changes, which translate to groundwater storage changes at regional scale (~100km). Use as macro constraint for village-level predictions.

#### Why It's Novel
- Provides "ground truth" at regional scale
- Ensures village predictions sum to satellite-observed totals
- Already validated for India (Rodell et al., Nature 2009)
- Nobody at the hackathon will think of this

#### GRACE Data Details
- **Resolution**: 0.5° x 0.5° (~50km)
- **Temporal**: Monthly data since 2002
- **Measurement**: Terrestrial Water Storage (TWS) anomalies
- **Units**: cm equivalent water height

#### How to Extract Groundwater from GRACE
```
Groundwater Storage = TWS - Soil Moisture - Surface Water - Snow

where:
- TWS from GRACE
- Soil moisture from GLDAS (Global Land Data Assimilation System)
- Surface water from GLDAS
- Snow from GLDAS
```

#### Data Sources
- **GRACE/GRACE-FO**: https://grace.jpl.nasa.gov/data/get-data/
- **GLDAS**: https://ldas.gsfc.nasa.gov/gldas
- **Processing options**: JPL, CSR, GFZ mascon solutions

#### India-Specific Findings
- **Rodell et al. (Nature, 2009)**: Punjab/Haryana/Rajasthan losing 4.0±1.0 cm/year
- **Total loss**: 109 km³ over 2002-2008 study period
- **Validation**: Correlated with >15,000 wells across India (Bhanja et al., 2016)

#### Integration Approach
```python
# Constraint: village predictions must be consistent with GRACE
village_predictions = model.predict(villages)
regional_total = sum(village_predictions * village_areas)
grace_regional = get_grace_value(region_bounds, month)

# Add as loss term or post-hoc adjustment
constraint_loss = (regional_total - grace_regional)²
```

#### Differentiation Factor: ⭐⭐⭐⭐⭐

---

### 2.4 Sentinel-1 InSAR for Land Subsidence

#### Concept
Synthetic Aperture Radar (SAR) interferometry detects ground surface deformation. Land subsidence indicates groundwater extraction.

#### Why It's Relevant
- Subsidence is a proxy for extraction intensity
- High-resolution (~10m) vs GRACE (~50km)
- Can identify extraction hotspots

#### Data Source
- **Sentinel-1**: Free, ESA, C-band SAR
- **Processing**: SNAP software, or cloud platforms (Google Earth Engine)

#### Differentiation Factor: ⭐⭐⭐

---

### 2.5 Active Learning for Field Validation

#### Concept
Instead of random validation, model tells department WHICH villages to visit first to maximize information gain from limited field visits.

#### Why It's Novel
- Department will validate ~10% of villages
- Most teams provide predictions only
- You provide: predictions + optimal validation sequence

#### Algorithm
```python
def select_validation_villages(model, villages, n_to_select):
    scores = []
    for village in villages:
        # Uncertainty from model
        uncertainty = model.predict_uncertainty(village)

        # Diversity bonus (different aquifer types, spatial spread)
        diversity = compute_diversity_score(village, already_selected)

        # Information gain estimate
        info_gain = uncertainty * diversity
        scores.append((village, info_gain))

    # Return top-n villages
    return sorted(scores, key=lambda x: -x[1])[:n_to_select]
```

#### Practical Value
- Optimizes use of limited field validation resources
- Shows systems thinking to judges
- Provides ongoing value post-hackathon

#### Differentiation Factor: ⭐⭐⭐⭐

---

### 2.6 Attention-Based Explainability

#### Concept
Use attention mechanisms to show exactly which piezometers influenced each village's prediction and why.

#### Why It's Novel
- Department needs to trust predictions
- Most ML models are black boxes
- Attention weights provide natural explanation

#### Output Format
```
Village: Gudivada
Predicted water level: 14.2m (±2.1m)
Confidence: Medium

Top influences:
├── Piezometer P047 (same aquifer, 2.3km): 45% weight
├── Piezometer P032 (similar geology, 5.1km): 28% weight
├── Rainfall correlation (last 3 months): 18% weight
└── Geology factor (granite-gneiss): 9% weight

Similar villages: Vuyyuru (13.8m), Kaikalur (15.1m)
```

#### Implementation
- Transformer attention layers
- Graph attention networks (GAT)
- SHAP values for tree-based models

#### Differentiation Factor: ⭐⭐⭐⭐

---

### 2.7 Conformal Prediction for Uncertainty

#### Concept
Rigorous uncertainty quantification that provides prediction intervals with guaranteed coverage probability.

#### Why It's Novel
- Standard ML uncertainty is often uncalibrated
- Conformal prediction provides statistical guarantees
- Can say "90% of predictions will fall within these bounds"

#### Implementation
```python
from mapie.regression import MapieRegressor

# Wrap any sklearn regressor
mapie_model = MapieRegressor(base_model, method="plus")
mapie_model.fit(X_train, y_train)

# Predictions with confidence intervals
y_pred, y_intervals = mapie_model.predict(X_test, alpha=0.1)  # 90% CI
```

#### Differentiation Factor: ⭐⭐⭐

---

### 2.8 Transfer Learning from Data-Rich Regions

#### Concept
Train on data-rich districts or global datasets, fine-tune on Krishna district.

#### Why It's Relevant
- Only 150 piezometers in Krishna
- Other districts or countries have more data
- Transfer learning can improve generalization

#### Approaches
- Pre-train on state-wide data, fine-tune on Krishna
- Use global groundwater datasets for pre-training
- Domain adaptation techniques

#### Differentiation Factor: ⭐⭐⭐

---

### 2.9 Ensemble of Aquifer-Specific Experts

#### Concept
Train separate "expert" models for each aquifer type, use meta-model to combine.

#### Why It Makes Sense
- Different aquifers have different behavior
- Expert models can specialize
- Mimics how hydrogeologists think

#### Architecture
```
Input village features
        ↓
┌───────┴───────┐
↓               ↓
Alluvium     Granite
Expert       Expert    ... (N aquifer experts)
↓               ↓
└───────┬───────┘
        ↓
   Meta-model
   (weighted combination)
        ↓
   Final prediction
```

#### Differentiation Factor: ⭐⭐⭐

---

### 2.10 Temporal: Pre/Post Monsoon Delta Modeling

#### Concept
Instead of predicting absolute levels, predict the CHANGE between pre and post monsoon.

#### Why It's Better
- Change is more consistent across villages
- Removes baseline uncertainty
- Directly relates to recharge assessment

#### Formulation
```
Δh = h_post - h_pre = f(rainfall, aquifer_type, recharge_structures, ...)

# Then:
h_post = h_pre + Δh_predicted
```

#### Differentiation Factor: ⭐⭐⭐

---

## 3. Satellite Data Sources

### 3.1 GRACE / GRACE-FO (Groundwater Storage)
- **Provider**: NASA/DLR
- **Resolution**: ~0.5° (~50km)
- **Temporal**: Monthly since 2002
- **Data**: Terrestrial Water Storage anomalies
- **URL**: https://grace.jpl.nasa.gov/data/get-data/
- **Processing**: JPL, CSR, GFZ mascon solutions

### 3.2 Sentinel-1 (SAR/InSAR)
- **Provider**: ESA
- **Resolution**: 10m
- **Temporal**: 6-12 day repeat
- **Use**: Land subsidence detection (extraction proxy)
- **URL**: https://scihub.copernicus.eu/

### 3.3 SRTM DEM (Elevation)
- **Provider**: NASA
- **Resolution**: 30m
- **Use**: Terrain analysis, slope, flow direction
- **URL**: https://earthexplorer.usgs.gov/

### 3.4 GLDAS (Land Surface Model)
- **Provider**: NASA
- **Resolution**: 0.25° to 1°
- **Data**: Soil moisture, runoff, evapotranspiration
- **Use**: Separate soil water from GRACE TWS
- **URL**: https://ldas.gsfc.nasa.gov/gldas

### 3.5 MODIS (Vegetation/Land Use)
- **Provider**: NASA
- **Resolution**: 250m - 1km
- **Data**: NDVI, land cover, evapotranspiration
- **Use**: Updated LULC (replacing 2005 data)

### 3.6 GPM/TRMM (Rainfall)
- **Provider**: NASA/JAXA
- **Resolution**: 0.1° (~10km)
- **Temporal**: Daily/monthly
- **Use**: Rainfall input for models

### 3.7 ISRIC SoilGrids (Soil Data)
- **Provider**: ISRIC
- **Resolution**: 250m
- **Data**: Soil properties (texture, hydraulic properties)
- **URL**: https://soilgrids.org/
- **Note**: Mentioned in orientation as alternative to local soil data

### 3.8 ESA WorldCover (Land Cover)
- **Provider**: ESA
- **Resolution**: 10m
- **Year**: 2021
- **Use**: Updated LULC (replacing 2005 data)
- **Note**: Already downloaded in `downloaded_data/landcover/`

---

## 4. Additional "Sensors" - Multi-Signal Approach

The core idea: Just like multiple CCTV cameras + mic provide better surveillance than a single camera, multiple independent signals provide better groundwater estimates than piezometers alone.

### 4.1 Vegetation Stress / NDVI (Satellite)

**How it works**: Plants show water stress before humans can detect it. When groundwater depletes, crops and vegetation exhibit reduced NDVI values.

**Key Indices**:
- **NDVI** (Normalized Difference Vegetation Index): General vegetation health
- **VCI** (Vegetation Condition Index): `(NDVI - NDVI_min) / (NDVI_max - NDVI_min)`
- **NDWI** (Normalized Difference Water Index): Crop water content
- **VHI** (Vegetation Health Index): Combines NDVI + temperature

**Data Sources**:
| Source | Resolution | Revisit | Access |
|--------|------------|---------|--------|
| MODIS MOD13Q1 | 250m | 16-day | Free (NASA) |
| Sentinel-2 | 10m | 5-day | Free (ESA) |
| Landsat 8/9 | 30m | 16-day | Free (USGS) |

**Why useful**: Villages with consistently low VCI during growing season likely have depleted groundwater. Vegetation stress precedes well failure.

**Integration approach**:
```python
# Calculate seasonal VCI anomaly per village
village_vci = calculate_mean_vci(village_boundary, growing_season)
vci_anomaly = (village_vci - historical_mean) / historical_std
# Negative anomaly = water stress = potentially deeper groundwater
```

### 4.2 Land Surface Temperature (LST) & Evapotranspiration

**How it works**: Areas with shallow groundwater have higher evapotranspiration and cooler surface temperatures. Deep water table = hotter, drier surface.

**Physical basis**:
- Shallow groundwater → vegetation accesses water → evapotranspiration cools surface
- Deep groundwater → vegetation stressed → reduced ET → surface heats up

**Data Sources**:
| Product | Resolution | Use |
|---------|------------|-----|
| MODIS MOD11A2 | 1km | 8-day LST |
| Landsat Thermal | 100m | Higher resolution LST |
| MODIS MOD16A2 | 500m | Actual evapotranspiration |
| SSEBop (USGS) | 1km | ET estimates |

**Processing**:
```python
# Hot spots relative to neighbors = likely deeper water table
lst_anomaly = village_lst - regional_mean_lst
# Positive anomaly suggests water stress / deeper groundwater
```

**Why novel**: Nobody at the hackathon will think to use thermal anomalies as a groundwater proxy.

### 4.3 Electricity Consumption (Non-Satellite Proxy)

**How it works**: Agricultural electricity consumption directly correlates with groundwater pumping intensity. More pumping = deeper water levels over time.

**Why powerful**: This is a direct measure of extraction intensity, not a proxy.

**Data Sources**:
- **State electricity board** (APSPDCL for Krishna district)
- **Feeder-level consumption** (best granularity)
- **Agricultural tariff category** (separates from residential)

**Challenge**: May require RTI or department coordination to obtain.

**Integration**:
```python
# If available at village/feeder level
pumping_intensity = ag_electricity_kwh / number_of_wells
# Higher pumping intensity correlates with faster depletion
```

### 4.4 Nighttime Lights / Settlement Growth

**How it works**: Urban expansion and irrigation expansion correlate with increased groundwater demand.

**Data Sources**:
- **VIIRS DNB**: Nighttime lights (free, NASA)
- **GHSL**: Global Human Settlement Layer - built-up area

**Use case**: Identify villages with rapid growth = increased extraction pressure.

### 4.5 Crop Type Classification

**How it works**: Water-intensive crops (paddy, sugarcane) vs drought-tolerant crops indicate local water availability and extraction patterns.

**Data Sources**:
- **Sentinel-2 time series**: Classify crops by phenology
- **Bhuvan Crop Inventory**: State-level crop maps
- **MODIS temporal NDVI**: Crop calendar analysis

**Integration**:
```python
# Water-intensive crop fraction
water_intensive_fraction = area_paddy + area_sugarcane / total_ag_area
# Higher fraction = more extraction pressure
```

### 4.6 InSAR Land Subsidence (Advanced)

**How it works**: Over-extraction causes land subsidence (millimeters to centimeters). Sentinel-1 InSAR can detect this.

**Data Sources**:
- **Sentinel-1 SAR**: Free, requires processing
- **EGMS**: European Ground Motion Service (pre-processed for some areas)

**Why highly differentiating**: Almost nobody in India is using this for groundwater. It's a strong signal of chronic over-extraction.

**Processing complexity**: High - requires time series of SAR images and interferometric processing.

### 4.7 Multi-Signal Fusion Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-SIGNAL FUSION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Piezometer Data]──────┐                                       │
│    (Direct measurement) │                                       │
│                         │                                       │
│  [GRACE Satellite]──────┼───► FUSION ───► Village-Level         │
│    (Regional GW)        │      MODEL      Predictions           │
│                         │        │                              │
│  [Vegetation Stress]────┤        │                              │
│    (NDVI, VCI, NDWI)    │        ▲                              │
│                         │        │                              │
│  [LST / ET]─────────────┤     Geology                           │
│    (Thermal anomaly)    │     Constraint                        │
│                         │                                       │
│  [Pumping Proxy]────────┤                                       │
│    (Electricity/Wells)  │                                       │
│                         │                                       │
│  [Crop Water Demand]────┤                                       │
│    (Crop classification)│                                       │
│                         │                                       │
│  [Land Subsidence]──────┘                                       │
│    (InSAR - optional)                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.8 Differentiation Value of Each Signal

| Signal | Data Availability | Processing Effort | Novelty | Impact |
|--------|-------------------|-------------------|---------|--------|
| NDVI/VCI | Easy (free) | Low | Medium | Medium |
| LST anomaly | Easy (free) | Low | High | Medium |
| Electricity | Hard (need access) | Very Low | Very High | High |
| Crop classification | Medium | Medium | Medium | Medium |
| InSAR subsidence | Medium (free) | High | Very High | High |
| GRACE + GLDAS | Easy (free) | Medium | Very High | Very High |

### 4.9 Recommended Multi-Signal Strategy

**Tier 1 (Must implement)**:
- Piezometer data (provided)
- GRACE regional constraint (high impact, reasonable effort)

**Tier 2 (Should implement)**:
- Vegetation stress index (NDVI/VCI) - easy to add
- LST anomaly - differentiating, easy
- Crop type / water demand estimation

**Tier 3 (Nice to have)**:
- Electricity consumption (if obtainable)
- InSAR subsidence (if time permits)

---

## 5. Existing India-Specific Work

### 5.1 SIH-AI-enabled-Water-Well-Prediction-Model
- **Context**: Smart India Hackathon winner
- **GitHub**: https://github.com/pawan-cpu/SIH-AI-enabled-Water-Well-Prediction-Model
- **Approach**: Random Forest for well suitability prediction
- **Features predicted**: Well depth, discharge, drilling techniques, water quality
- **Relevance**: Similar problem domain, can learn from approach
- **Demo**: https://water-well-predictor-4gtfeehgyxsxkbzxwtd9kf.streamlit.app/

### 5.2 Deep Learning for Ganga Basin (2023)
- **Paper**: ISPRS Archives (2023)
- **Approach**: Vision Transformers + Sentinel-1 + GRACE
- **Result**: MAE of 1.2m with Swin-Transformer NAM
- **Relevance**: Demonstrates satellite + DL for India groundwater

### 5.3 Efficacy of ML in Agro-Ecological Zones of India
- **Paper**: Science of Total Environment (2021)
- **Models tested**: ANFIS, DNN, SVM
- **Scale**: Country-wide, 18 Agro-Ecological Zones
- **Finding**: DNN most proficient for seasonal prediction
- **Limitation**: 'Moderate' to 'poor' in some zones including Deccan Plateau

### 5.4 GRACE Validation with Indian Wells
- **Paper**: Bhanja et al. (2016)
- **Finding**: GRACE anomalies validated against >15,000 wells across India
- **Relevance**: Confirms GRACE utility for India

### 5.5 Andhra Pradesh Water Scarcity ML (2024)
- **Paper**: IRJMETS (May 2024)
- **Focus**: Andhra Pradesh specifically
- **Models**: Linear Regression, Decision Trees, Neural Networks
- **Data**: Rainfall, river flows, groundwater levels, usage statistics

---

## 6. Machine Learning Approaches for Groundwater

### 6.1 Models Tested in Literature

| Model | Typical R² | Pros | Cons |
|-------|-----------|------|------|
| Random Forest | 0.85-0.95 | Robust, feature importance | No spatial awareness |
| XGBoost | 0.85-0.95 | High accuracy, handles missing data | Black box |
| SVM/SVR | 0.75-0.90 | Works with small data | Kernel selection |
| ANFIS | 0.80-0.95 | Combines fuzzy logic + NN | Complex tuning |
| DNN | 0.85-0.95 | Non-linear patterns | Needs more data |
| LSTM | 0.80-0.90 | Temporal patterns | Needs long sequences |
| Kriging | 0.70-0.85 | Uncertainty, spatial | Stationarity assumption |

### 6.2 Feature Importance (from Literature)
Typical important features for groundwater prediction:
1. **Rainfall** (cumulative, seasonal)
2. **Geology/Aquifer type**
3. **Elevation/Slope**
4. **Distance to water bodies**
5. **Land use (irrigation intensity)**
6. **Previous water levels** (temporal autocorrelation)
7. **River discharge** (if connected aquifer)
8. **Soil properties** (infiltration capacity)

### 6.3 Validation Approaches
- **Leave-one-out cross-validation**: For sparse data
- **Spatial cross-validation**: Account for spatial autocorrelation
- **Temporal hold-out**: Train on historical, test on recent
- **Stratified by aquifer**: Ensure each aquifer type represented

---

## 7. Visualization & Integration Tools

### 7.1 Dashboard Frameworks
- **Streamlit**: Rapid prototyping, Python-native
- **Dash (Plotly)**: More customizable, React-based
- **Panel/Holoviz**: Good for geospatial

### 7.2 Mapping Libraries
- **Folium**: Leaflet.js wrapper for Python
- **Kepler.gl**: Uber's geospatial visualization
- **Mapbox**: Commercial, high quality
- **Leafmap**: Geospatial analysis + visualization

### 7.3 API Frameworks
- **FastAPI**: Modern, fast, automatic docs
- **Flask**: Simpler, more established
- **Django REST**: Full-featured

### 7.4 Geospatial Python Stack
```python
# Core stack
import geopandas as gpd  # Spatial dataframes
import rasterio          # Raster data
import shapely           # Geometry operations
import pyproj            # Coordinate transforms
import fiona             # File I/O

# Analysis
from pykrige import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor
import xarray as xr      # NetCDF handling (for GRACE)
```

---

## 8. Key Research Papers

### Foundational
1. **Rodell et al. (2009)** - "Satellite-based estimates of groundwater depletion in India" - Nature
2. **Raissi et al. (2019)** - "Physics-informed neural networks" - Journal of Computational Physics

### GNN for Spatial Data
3. **NN-GLS (2023)** - "Neural networks for geospatial data" - arXiv:2304.09157
4. **PE-GNN (2023)** - "Positional Encoder GNN for Geographic Data" - AISTATS

### PINN for Groundwater
5. **Bandai & Ghezzehei (2022)** - "Forward and inverse modeling with PINNs" - HESS
6. **KLE-PINN (2024)** - "Groundwater inverse modeling" - Journal of Hydrology

### India-Specific ML
7. **Mohapatra et al. (2021)** - "ML for groundwater in India's AEZs" - Science of Total Environment
8. **ISPRS (2023)** - "Deep Learning for Ganga Basin" - ISPRS Archives

### Geostatistics
9. **GStatSim (2023)** - "Python package for geostatistical interpolation" - GMD
10. **Pyinterpolate (2022)** - "Spatial interpolation in Python" - JOSS

---

## 9. Approach Comparison Matrix

### For Core Prediction (P3: Geology-Aware Interpolation)

| Approach | Data Needs | Geology Aware | Uncertainty | Explainability | Implementation Effort | Novelty |
|----------|------------|---------------|-------------|----------------|----------------------|---------|
| IDW | Low | No | No | High | Very Low | None |
| Ordinary Kriging | Low | No | Yes | Medium | Low | None |
| Stratified Kriging | Medium | Yes (manual) | Yes | Medium | Medium | Low |
| Random Forest | Medium | Implicit | Limited | Medium (SHAP) | Low | Low |
| GNN (NN-GLS/PE-GNN) | Medium | Yes (learned) | Yes | Medium (attention) | High | Very High |
| PINN | Medium | Yes (physics) | Yes | High (physics) | High | High |
| Ensemble Experts | Medium | Yes (explicit) | Yes | High | Medium | Medium |

### For Regional Constraint (Novel)

| Approach | Data Source | Resolution | Effort | Impact |
|----------|-------------|------------|--------|--------|
| GRACE integration | Satellite | ~50km | Medium | Very High |
| InSAR subsidence | Sentinel-1 | ~10m | High | Medium |
| None | - | - | - | Baseline |

### For Explainability (P14)

| Approach | Method | Interpretability | Effort |
|----------|--------|------------------|--------|
| SHAP values | Post-hoc | High | Low |
| Attention weights | Built-in | High | Medium |
| Physics constraints | Built-in | Very High | High |
| Feature importance | Built-in (RF) | Medium | Very Low |

---

## 10. Recommended Technology Stack (For Reference)

### Data Processing
- **GeoPandas**: Spatial data handling
- **Rasterio**: Raster processing
- **xarray**: NetCDF (GRACE data)
- **QGIS**: Visual exploration

### Core Modeling (Options)
- **PyKrige / Pyinterpolate**: Kriging baseline
- **scikit-learn**: RF/XGBoost
- **PyTorch Geometric**: GNN implementation
- **DeepXDE / PyTorch**: PINN implementation

### Visualization & API
- **Streamlit**: Dashboard
- **Folium**: Interactive maps
- **FastAPI**: REST API
- **Plotly**: Charts

### External Data
- **GRACE**: Regional groundwater storage
- **GLDAS**: Soil moisture separation
- **SRTM**: DEM
- **ISRIC**: Soil properties

---

## 11. Quick Reference: Differentiation Strategies

| Strategy | Effort | Impact | Risk |
|----------|--------|--------|------|
| GRACE satellite integration | Medium | Very High | Low |
| Graph Neural Network | High | Very High | Medium |
| Physics-Informed NN | High | High | Medium |
| Attention-based explanations | Medium | High | Low |
| Active learning for validation | Low | Medium | Very Low |
| Conformal prediction uncertainty | Low | Medium | Very Low |
| Pre/post monsoon delta modeling | Low | Medium | Very Low |
| Ensemble of aquifer experts | Medium | Medium | Low |

---

## Notes for Architecture Decision

When finalizing architecture, consider:

1. **Time constraint**: 7-10 days for POC
2. **Data availability**: What's actually in the provided datasets
3. **Validation method**: Department will field-validate
4. **User needs**: Dashboard + API required
5. **Explainability**: Officials need to understand predictions
6. **Scalability**: Should extend beyond Krishna district

The winning combination likely involves:
- One **novel technical approach** (GNN or PINN or GRACE)
- One **practical innovation** (active learning, explainability)
- **Solid baseline** (kriging or RF) for comparison
- **Clean visualization** (Streamlit + Folium)

---

## 12. Official Clarifications from Department

### Source
`Data Issues_Clarifications.docx` - Q&A responses from AP Ground Water Department

### 12.1 Soil Classification for Groundwater Modeling

The department provided guidance on how to classify the 51 soil types for groundwater modeling:

#### For Infiltration/Recharge Assessment

| Category | Infiltration Rate | Soil Types to Include |
|----------|-------------------|----------------------|
| Low | Low | Clays, Clayey soils |
| Moderate | Moderate | Loam, Loamy soils, Others |
| High | High | Sands, Sandy soils |

**Physical basis:** Infiltration of rainfall through soils leads to groundwater recharge. Sandy soils allow more water to percolate; clayey soils impede infiltration.

#### For Runoff Assessment

| Category | Runoff | Soil Types to Include |
|----------|--------|----------------------|
| Low | Less runoff | Very shallow, Shallow, Less Drained |
| Moderate | Moderate runoff | Moderately Deep, Moderately Drained |
| High | High runoff | Deep, Very Deep, Well Drained |

**Physical basis:** After soil saturation, excess rainfall becomes overland flow (runoff). Shallow soils saturate quickly leading to more runoff.

### 12.2 Bore Well Data Filtering

The department clarified which bore wells to include in analysis:

#### Include (for analysis)
- Filter points
- Tube wells
- Bore wells
- **Functioning/Working** status only

#### Exclude (from analysis)
- Open wells (shallow wells) - not representative of piezometer depths
- Partially working wells
- Abandoned/Closed wells

**Rationale:** The 138 monitoring piezometers are deep wells. Analysis should focus on similar well types.

### 12.3 Pumping/Extraction Calculation

**Data source:** `Pumping Data.xlsx`

The pumping data provides unit drafts in hectare-meters (ha.m) per well per village for two seasons:
- Monsoon season
- Non-monsoon season

**To calculate monthly average extraction per well:**
```
Monthly extraction = Unit draft / 4 months
```

**Usage:** Change in groundwater levels = Recharge (infiltration) - Extraction (pumping)

### 12.4 Area of Interest (AOI)

**Clarification from department:**
- Ultimate goal: All villages in Andhra Pradesh (18,000+ villages)
- For this pilot: Krishna District
- Minimum scope: At least 10 mandals from Krishna

**Recommendation:** Use full Krishna district (42 mandals, 939 villages) for comprehensive POC.

### 12.5 Modeling Goal

**Department's stated objective:**
> "Preliminary we want the relationship between all monthly available data sets, we need interpolation/extrapolation as smooth as possible. Rainfall available monthly / extrapolation of monthly water levels of limited stations to each village."

**Key requirements:**
1. Establish relationships between influencing parameters
2. Interpolate from 138 piezometers to 939 villages
3. Monthly temporal resolution
4. Smooth spatial interpolation/extrapolation

---

*Document updated: January 15, 2026*
*Added: Official clarifications from department (Section 12)*
*Purpose: Architecture decision support for Smart Jal hackathon*
