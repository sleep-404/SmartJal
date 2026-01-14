# Smart Jal - Prototype Implementation Plan

**Scope:** Working prototype for hackathon demonstration
**Timeline:** 5 days
**Output:** Interactive dashboard with village-level groundwater predictions

---

## Prototype Scope (What We're Building)

### In Scope
- [x] Data preprocessing pipeline
- [x] Feature engineering for villages and piezometers
- [x] Geology-aware spatial interpolation model
- [x] Temporal decomposition (trend + seasonal + anomaly)
- [x] GRACE satellite constraint integration
- [x] Risk classification system (Critical/High/Moderate/Low)
- [x] Interactive Streamlit dashboard with map
- [x] Village detail view with explainability
- [x] Basic scenario analysis (drought simulation)

### Out of Scope (Future Work)
- [ ] Real-time data ingestion
- [ ] User authentication
- [ ] Database backend (using CSV/files for prototype)
- [ ] Mobile app
- [ ] Full API deployment
- [ ] Optimal sensor placement algorithm (mention in presentation only)

---

## Technical Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Language | Python 3.11+ | Data science ecosystem |
| Data Processing | Pandas, GeoPandas, Rasterio | Geospatial standard |
| ML/Modeling | Scikit-learn, PyKrige, XGBoost | Proven, fast |
| Visualization | Streamlit, Folium, Plotly | Rapid prototyping |
| Maps | Folium + Leaflet | Interactive, free |

---

## Directory Structure

```
SmartJal/
├── backend/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── load_data.py          # Load all raw data
│   │   ├── preprocess.py         # Clean and transform
│   │   ├── feature_engineering.py # Create features
│   │   └── feature_extractor.py  # (exists) Raster extraction
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── spatial_model.py      # Aquifer-stratified kriging
│   │   ├── feature_model.py      # XGBoost with features
│   │   ├── temporal_model.py     # Time series decomposition
│   │   ├── ensemble.py           # Combine models + GRACE
│   │   └── risk_classifier.py    # Risk tier assignment
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── geo_utils.py          # Spatial operations
│   │   └── constants.py          # Bounds, thresholds
│   │
│   └── pipeline.py               # End-to-end pipeline
│
├── frontend/
│   ├── app.py                    # Streamlit main app
│   ├── pages/
│   │   ├── 1_dashboard.py        # Main dashboard
│   │   ├── 2_village_detail.py   # Village drilldown
│   │   └── 3_scenarios.py        # What-if analysis
│   │
│   ├── components/
│   │   ├── map_view.py           # Folium map
│   │   ├── charts.py             # Plotly charts
│   │   └── risk_cards.py         # Summary cards
│   │
│   └── assets/
│       └── style.css             # Custom styling
│
├── data/                         # Processed data (generated)
│   ├── processed/
│   │   ├── villages_with_features.csv
│   │   ├── piezometers_processed.csv
│   │   ├── predictions.csv
│   │   └── risk_classification.csv
│   │
│   └── models/                   # Saved models
│       ├── kriging_models/       # Per-aquifer models
│       ├── xgboost_model.pkl
│       └── ensemble_weights.json
│
├── scripts/
│   ├── run_pipeline.py           # Run full pipeline
│   └── evaluate_model.py         # Validation metrics
│
├── docs/                         # Documentation
│
└── requirements.txt              # Dependencies
```

---

## Implementation Modules

### Module 1: Data Loading (`backend/data_processing/load_data.py`)

**Purpose:** Load all raw data into memory

**Functions:**
```python
def load_water_levels() -> pd.DataFrame:
    """Load 138 piezometers with 28 years of monthly data."""
    # Returns: DataFrame with columns [id, mandal, village, lat, lon, aquifer,
    #          depth, msl, 1997-01, 1997-02, ..., 2024-12]

def load_villages() -> gpd.GeoDataFrame:
    """Load 939 village boundaries."""
    # Returns: GeoDataFrame with columns [village_name, mandal, geometry, area, centroid]

def load_aquifers() -> gpd.GeoDataFrame:
    """Load 8 aquifer boundaries."""
    # Returns: GeoDataFrame with columns [aquifer_code, geo_class, geometry]

def load_soils() -> gpd.GeoDataFrame:
    """Load 151 soil polygons."""
    # Returns: GeoDataFrame with columns [description, soil_taxon, geometry]

def load_bore_wells() -> pd.DataFrame:
    """Load 88,988 bore wells (filtered to working deep wells)."""
    # Returns: DataFrame with columns [village, lat, lon, depth, well_type, status]

def load_pumping_data() -> pd.DataFrame:
    """Load 714 village pumping records."""
    # Returns: DataFrame with columns [mandal, village, n_wells, draft_monsoon, draft_nonmonsoon]

def load_rainfall() -> dict:
    """Load 56 monthly rainfall rasters."""
    # Returns: dict {(year, month): rasterio.DatasetReader}

def load_dem() -> rasterio.DatasetReader:
    """Load 30m DEM."""
    # Returns: rasterio dataset

def load_grace() -> pd.DataFrame:
    """Load GRACE monthly time series."""
    # Returns: DataFrame with columns [date, tws_anomaly_cm, seasonal, trend]
```

**Output:** All data loaded into DataFrames/GeoDataFrames

---

### Module 2: Preprocessing (`backend/data_processing/preprocess.py`)

**Purpose:** Clean and transform raw data

**Functions:**
```python
def preprocess_water_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Reshape from wide to long format
    - Handle missing values (forward fill within limits)
    - Calculate statistics per piezometer
    """
    # Returns: DataFrame [piezo_id, date, water_level, ...]

def assign_villages_to_aquifers(villages: gpd.GeoDataFrame,
                                 aquifers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatial join villages with aquifer boundaries."""
    # Returns: villages with aquifer_code column

def classify_soils(soils: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Classify 51 soil types into:
    - infiltration_class: low/moderate/high
    - runoff_class: low/moderate/high
    Based on official guidance.
    """
    # Returns: soils with classification columns

def assign_villages_to_soils(villages: gpd.GeoDataFrame,
                              soils: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatial join villages with soil classes."""
    # Returns: villages with soil columns

def filter_bore_wells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only:
    - Well type: Bore Well, Tube Well, Filter Point
    - Status: Working
    """
    # Returns: filtered DataFrame

def calculate_village_extraction(villages: gpd.GeoDataFrame,
                                  bore_wells: pd.DataFrame,
                                  pumping: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    For each village:
    - Count working bore wells
    - Sum extraction (draft/4 months = monthly extraction)
    """
    # Returns: villages with extraction columns
```

**Output:** Clean, joined datasets ready for feature engineering

---

### Module 3: Feature Engineering (`backend/data_processing/feature_engineering.py`)

**Purpose:** Create model features for villages and piezometers

**Functions:**
```python
def extract_raster_features(points: gpd.GeoDataFrame,
                            dem: rasterio.DatasetReader,
                            rainfall: dict) -> pd.DataFrame:
    """
    For each point (village centroid or piezometer):
    - elevation_m
    - slope (from DEM gradient)
    - rainfall_monthly (56 values)
    - rainfall_mean, rainfall_std, rainfall_max
    """

def calculate_distance_features(villages: gpd.GeoDataFrame,
                                 piezometers: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    For each village:
    - distance_to_nearest_piezometer
    - distance_to_nearest_same_aquifer_piezometer
    - n_piezometers_within_10km
    """

def calculate_temporal_features(water_levels: pd.DataFrame) -> pd.DataFrame:
    """
    For each piezometer:
    - seasonal_amplitude
    - trend_per_year
    - anomaly_std
    - mean_level
    - recent_level (last 12 months avg)
    """

def create_feature_matrix(villages: gpd.GeoDataFrame, ...) -> pd.DataFrame:
    """
    Combine all features into final matrix.

    Columns:
    - village_id, mandal, lat, lon
    - aquifer_code, aquifer_type
    - soil_infiltration, soil_runoff
    - elevation_m, slope
    - rainfall_features (mean, std, seasonal)
    - n_wells, extraction_monthly
    - distance_to_piezometer
    """
    # Returns: DataFrame with all features
```

**Output:** `villages_with_features.csv`, `piezometers_processed.csv`

---

### Module 4: Spatial Model (`backend/models/spatial_model.py`)

**Purpose:** Aquifer-stratified kriging interpolation

**Functions:**
```python
class AquiferStratifiedKriging:
    """
    Kriging model that only interpolates within same aquifer.
    """

    def __init__(self):
        self.models = {}  # One kriging model per aquifer

    def fit(self, piezometers: pd.DataFrame, aquifer_col: str = 'aquifer_code'):
        """
        Fit separate variogram and kriging model for each aquifer.

        For each aquifer:
        1. Select piezometers in that aquifer
        2. Fit variogram (spherical or exponential)
        3. Create OrdinaryKriging model
        """
        for aquifer in piezometers[aquifer_col].unique():
            subset = piezometers[piezometers[aquifer_col] == aquifer]
            if len(subset) >= 3:  # Need minimum points for kriging
                self.models[aquifer] = OrdinaryKriging(
                    subset['lon'].values,
                    subset['lat'].values,
                    subset['water_level'].values,
                    variogram_model='spherical'
                )

    def predict(self, villages: pd.DataFrame) -> pd.DataFrame:
        """
        Predict water level for each village using its aquifer's model.

        Returns: DataFrame with [village_id, predicted_level, kriging_variance]
        """
        predictions = []
        for idx, village in villages.iterrows():
            aquifer = village['aquifer_code']
            if aquifer in self.models:
                level, variance = self.models[aquifer].execute(
                    'points',
                    np.array([village['lon']]),
                    np.array([village['lat']])
                )
                predictions.append({
                    'village_id': village['village_id'],
                    'spatial_prediction': level[0],
                    'spatial_variance': variance[0]
                })
            else:
                # Fallback for aquifers with insufficient data
                predictions.append({
                    'village_id': village['village_id'],
                    'spatial_prediction': np.nan,
                    'spatial_variance': np.nan
                })
        return pd.DataFrame(predictions)
```

**Output:** Spatial predictions with uncertainty

---

### Module 5: Feature Model (`backend/models/feature_model.py`)

**Purpose:** XGBoost model using all features

**Functions:**
```python
class FeatureBasedModel:
    """
    XGBoost model that predicts water level from features.
    """

    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.feature_columns = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train on piezometer data.

        X: feature matrix (aquifer, soil, rainfall, extraction, etc.)
        y: water level
        """
        self.feature_columns = X.columns.tolist()
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict water level from features."""
        return self.model.predict(X[self.feature_columns])

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance for explainability."""
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
```

**Output:** Feature-based predictions + importance

---

### Module 6: Temporal Model (`backend/models/temporal_model.py`)

**Purpose:** Decompose time series into trend + seasonal + anomaly

**Functions:**
```python
def decompose_time_series(series: pd.Series, period: int = 12) -> dict:
    """
    Decompose water level time series using STL.

    Returns:
    {
        'baseline': float,  # Long-term mean
        'seasonal': pd.Series,  # Monthly pattern
        'trend': pd.Series,  # Long-term trajectory
        'residual': pd.Series,  # Anomalies
        'trend_per_year': float,  # Slope of trend
        'seasonal_amplitude': float  # Range of seasonal
    }
    """
    from statsmodels.tsa.seasonal import STL

    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    return {
        'baseline': series.mean(),
        'seasonal': result.seasonal,
        'trend': result.trend,
        'residual': result.resid,
        'trend_per_year': calculate_trend_slope(result.trend),
        'seasonal_amplitude': result.seasonal.max() - result.seasonal.min()
    }

def calculate_current_anomaly(series: pd.Series, decomposition: dict) -> float:
    """
    Calculate how far current value is from expected.

    anomaly = actual - (trend + seasonal)
    normalized_anomaly = anomaly / historical_std
    """
    expected = decomposition['trend'].iloc[-1] + decomposition['seasonal'].iloc[-1]
    actual = series.iloc[-1]
    anomaly = actual - expected
    normalized = anomaly / decomposition['residual'].std()
    return normalized

def process_all_piezometers(water_levels: pd.DataFrame) -> pd.DataFrame:
    """
    Apply decomposition to all piezometers.

    Returns: DataFrame with temporal features per piezometer
    """
```

**Output:** Temporal decomposition for each piezometer

---

### Module 7: Ensemble Model (`backend/models/ensemble.py`)

**Purpose:** Combine models + apply GRACE constraint

**Functions:**
```python
class EnsembleModel:
    """
    Combines spatial, feature, and temporal models.
    Applies GRACE regional constraint.
    """

    def __init__(self, spatial_model, feature_model, grace_data):
        self.spatial_model = spatial_model
        self.feature_model = feature_model
        self.grace_data = grace_data
        self.weights = {'spatial': 0.4, 'feature': 0.6}  # Learned weights

    def predict(self, villages: pd.DataFrame, month: str) -> pd.DataFrame:
        """
        1. Get spatial predictions
        2. Get feature predictions
        3. Combine with weights
        4. Apply GRACE constraint
        """
        spatial_pred = self.spatial_model.predict(villages)
        feature_pred = self.feature_model.predict(villages)

        # Weighted combination
        combined = (
            self.weights['spatial'] * spatial_pred['spatial_prediction'] +
            self.weights['feature'] * feature_pred
        )

        # Apply GRACE constraint
        grace_value = self.get_grace_value(month)
        constrained = self.apply_grace_constraint(villages, combined, grace_value)

        return constrained

    def apply_grace_constraint(self, villages, predictions, grace_regional):
        """
        Adjust predictions so they sum to GRACE regional value.

        Simple approach: proportional scaling
        Advanced: optimization with uncertainty weighting
        """
        # Calculate current regional sum
        village_areas = villages['area_km2']
        current_sum = (predictions * village_areas).sum()

        # Scale factor
        scale = grace_regional / current_sum if current_sum != 0 else 1

        # Apply constraint (with damping to avoid extreme adjustments)
        adjusted = predictions * (1 + 0.5 * (scale - 1))

        return adjusted

    def calculate_uncertainty(self, spatial_variance, feature_std) -> np.ndarray:
        """
        Combine uncertainties from both models.
        """
        return np.sqrt(spatial_variance**2 + feature_std**2)
```

**Output:** Final predictions with GRACE constraint

---

### Module 8: Risk Classifier (`backend/models/risk_classifier.py`)

**Purpose:** Classify villages into risk tiers

**Functions:**
```python
class RiskClassifier:
    """
    Classify villages into CRITICAL/HIGH/MODERATE/LOW risk.
    """

    # Thresholds (can be calibrated)
    THRESHOLDS = {
        'trend_critical': -1.5,  # m/year
        'trend_high': -0.5,
        'anomaly_critical': -2.0,  # std deviations
        'anomaly_high': -1.0,
        'depth_critical': 0.8,  # fraction of well depth
        'depth_high': 0.6
    }

    def classify(self, village_data: pd.DataFrame) -> pd.DataFrame:
        """
        Assign risk tier based on multiple criteria.

        Input columns needed:
        - predicted_level
        - trend_per_year
        - anomaly_normalized
        - typical_well_depth

        Returns: DataFrame with risk_tier, risk_score, risk_factors
        """
        results = []

        for idx, v in village_data.iterrows():
            risk_factors = []
            risk_score = 0

            # Check trend
            if v['trend_per_year'] < self.THRESHOLDS['trend_critical']:
                risk_factors.append('Severe declining trend')
                risk_score += 3
            elif v['trend_per_year'] < self.THRESHOLDS['trend_high']:
                risk_factors.append('Declining trend')
                risk_score += 2

            # Check anomaly
            if v['anomaly_normalized'] < self.THRESHOLDS['anomaly_critical']:
                risk_factors.append('Significantly below expected')
                risk_score += 3
            elif v['anomaly_normalized'] < self.THRESHOLDS['anomaly_high']:
                risk_factors.append('Below expected')
                risk_score += 2

            # Check absolute depth
            depth_ratio = v['predicted_level'] / v['typical_well_depth']
            if depth_ratio > self.THRESHOLDS['depth_critical']:
                risk_factors.append('Approaching pump failure depth')
                risk_score += 4
            elif depth_ratio > self.THRESHOLDS['depth_high']:
                risk_factors.append('Deep water level')
                risk_score += 2

            # Assign tier
            if risk_score >= 6:
                tier = 'CRITICAL'
            elif risk_score >= 4:
                tier = 'HIGH'
            elif risk_score >= 2:
                tier = 'MODERATE'
            else:
                tier = 'LOW'

            results.append({
                'village_id': v['village_id'],
                'risk_tier': tier,
                'risk_score': risk_score,
                'risk_factors': risk_factors
            })

        return pd.DataFrame(results)

    def generate_alerts(self, village_data: pd.DataFrame) -> list:
        """Generate alert messages for high-risk villages."""
```

**Output:** Risk classification for all villages

---

### Module 9: Pipeline (`backend/pipeline.py`)

**Purpose:** End-to-end execution

**Functions:**
```python
def run_pipeline(target_date: str = None):
    """
    Execute full pipeline:

    1. Load all data
    2. Preprocess and join
    3. Engineer features
    4. Train models (or load pre-trained)
    5. Generate predictions
    6. Classify risks
    7. Save outputs
    """
    print("Step 1: Loading data...")
    data = load_all_data()

    print("Step 2: Preprocessing...")
    processed = preprocess_all(data)

    print("Step 3: Feature engineering...")
    features = create_all_features(processed)

    print("Step 4: Training models...")
    models = train_models(features)

    print("Step 5: Generating predictions...")
    predictions = generate_predictions(models, features, target_date)

    print("Step 6: Classifying risks...")
    risks = classify_risks(predictions)

    print("Step 7: Saving outputs...")
    save_outputs(predictions, risks)

    print("Pipeline complete!")
    return predictions, risks
```

---

### Module 10: Streamlit Dashboard (`frontend/app.py`)

**Structure:**
```python
# app.py - Main entry point
import streamlit as st

st.set_page_config(
    page_title="Smart Jal - Groundwater Intelligence",
    page_icon="💧",
    layout="wide"
)

# Load data
@st.cache_data
def load_predictions():
    return pd.read_csv('data/processed/predictions.csv')

@st.cache_data
def load_villages():
    return gpd.read_file('data/processed/villages_with_predictions.geojson')

# Sidebar
st.sidebar.title("Smart Jal")
st.sidebar.markdown("Groundwater Intelligence System")

page = st.sidebar.radio("Navigate", ["Dashboard", "Village Detail", "Scenarios"])

if page == "Dashboard":
    show_dashboard()
elif page == "Village Detail":
    show_village_detail()
else:
    show_scenarios()
```

**Dashboard View:**
```python
def show_dashboard():
    st.title("Krishna District - Groundwater Status")

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Critical", n_critical, delta=f"+{delta_critical}")
    with col2:
        st.metric("High Risk", n_high, delta=f"+{delta_high}")
    # ...

    # Map
    st.subheader("Village Risk Map")
    folium_map = create_risk_map(villages)
    st_folium(folium_map, width=800, height=500)

    # Risk table
    st.subheader("High Risk Villages")
    st.dataframe(high_risk_villages)
```

**Village Detail View:**
```python
def show_village_detail():
    # Village selector
    village = st.selectbox("Select Village", village_list)

    # Info cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Level", f"{level:.1f}m")
    with col2:
        st.metric("Risk Tier", risk_tier)
    with col3:
        st.metric("Trend", f"{trend:.2f} m/year")

    # Time series chart
    fig = create_timeseries_chart(village)
    st.plotly_chart(fig)

    # Influence factors
    st.subheader("Prediction Influences")
    for factor in influences:
        st.progress(factor['weight'], text=factor['description'])

    # Alerts
    if alerts:
        st.warning("⚠️ " + "\n".join(alerts))
```

**Scenario View:**
```python
def show_scenarios():
    st.title("What-If Scenario Analysis")

    scenario = st.selectbox("Select Scenario", [
        "Normal Conditions",
        "Drought (-30% rainfall)",
        "Reduced Extraction (-20%)",
        "Both Combined"
    ])

    # Run scenario
    scenario_results = run_scenario(scenario)

    # Show impact
    st.subheader("Impact Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Villages → Critical", f"+{delta_critical}")
    with col2:
        st.metric("Most Affected Mandal", worst_mandal)

    # Map comparison
    st.subheader("Before vs After")
    # Show side-by-side maps
```

---

## Implementation Order

### Day 1: Data Foundation
| # | Task | File | Time |
|---|------|------|------|
| 1 | Load all data sources | `load_data.py` | 1 hr |
| 2 | Preprocess water levels | `preprocess.py` | 1 hr |
| 3 | Spatial joins (village-aquifer-soil) | `preprocess.py` | 1 hr |
| 4 | Filter bore wells | `preprocess.py` | 0.5 hr |
| 5 | Calculate village extraction | `preprocess.py` | 0.5 hr |
| 6 | Extract raster features | `feature_engineering.py` | 1 hr |
| 7 | Calculate all features | `feature_engineering.py` | 1 hr |
| **Checkpoint** | `villages_with_features.csv` ready | | |

### Day 2: Core Models
| # | Task | File | Time |
|---|------|------|------|
| 8 | Implement kriging model | `spatial_model.py` | 2 hr |
| 9 | Implement XGBoost model | `feature_model.py` | 1.5 hr |
| 10 | Implement temporal decomposition | `temporal_model.py` | 1.5 hr |
| 11 | Cross-validation setup | `ensemble.py` | 1 hr |
| 12 | Train and evaluate models | `pipeline.py` | 1 hr |
| **Checkpoint** | Models trained, MAE < 2m | | |

### Day 3: Differentiation
| # | Task | File | Time |
|---|------|------|------|
| 13 | Implement ensemble + GRACE | `ensemble.py` | 2 hr |
| 14 | Implement risk classifier | `risk_classifier.py` | 1.5 hr |
| 15 | Generate predictions for all villages | `pipeline.py` | 1 hr |
| 16 | Create GeoJSON output | `pipeline.py` | 0.5 hr |
| 17 | Validate and tune | `evaluate_model.py` | 2 hr |
| **Checkpoint** | Full predictions + risks | | |

### Day 4: Visualization
| # | Task | File | Time |
|---|------|------|------|
| 18 | Streamlit app skeleton | `app.py` | 1 hr |
| 19 | Dashboard with summary cards | `dashboard.py` | 1.5 hr |
| 20 | Folium risk map | `map_view.py` | 2 hr |
| 21 | Village detail view | `village_detail.py` | 2 hr |
| 22 | Time series charts | `charts.py` | 1 hr |
| **Checkpoint** | Working dashboard | | |

### Day 5: Polish
| # | Task | File | Time |
|---|------|------|------|
| 23 | Scenario analysis | `scenarios.py` | 2 hr |
| 24 | Explainability display | `village_detail.py` | 1 hr |
| 25 | UI polish + styling | `style.css` | 1 hr |
| 26 | Testing + bug fixes | all | 2 hr |
| 27 | Documentation + presentation | docs | 1 hr |
| **Checkpoint** | Demo-ready | | |

---

## Dependencies (`requirements.txt`)

```
# Data processing
pandas>=2.0.0
geopandas>=0.14.0
rasterio>=1.3.0
shapely>=2.0.0
pyproj>=3.6.0
openpyxl>=3.1.0

# Machine learning
scikit-learn>=1.3.0
xgboost>=2.0.0
pykrige>=1.7.0
statsmodels>=0.14.0

# Visualization
streamlit>=1.29.0
streamlit-folium>=0.15.0
folium>=0.15.0
plotly>=5.18.0

# Utilities
numpy>=1.24.0
scipy>=1.11.0
```

---

## Validation Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| MAE | < 2.0 m | Leave-one-out CV on piezometers |
| RMSE | < 3.0 m | Same |
| R² | > 0.7 | Same |
| Risk accuracy | > 80% | Compare to known problem villages |
| Coverage | 100% | All 939 villages have predictions |

---

## Demo Script

```
1. Open dashboard → Show summary (47 critical, 156 high risk)
2. Click map → Zoom to critical cluster in Nandigama
3. Click village → Show detail (level, trend, influences)
4. Point out: "This village declined 1.2m/year for 5 years"
5. Show explainability: "Influenced by P-047 (same aquifer)"
6. Go to scenarios → Select "Drought"
7. Show impact: "81 more villages become critical"
8. Conclude: "This helps prioritize interventions"
```

---

*Document created: January 15, 2026*
*Ready for implementation*
