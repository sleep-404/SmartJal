# Smart Jal - AI-Powered Groundwater Prediction

**Hackathon Prototype** for village-level groundwater prediction in Krishna District, Andhra Pradesh.

## Key Innovation: Hierarchical Physics-Informed Prediction

```
┌─────────────────────────────────────────────────────────────┐
│                    GRACE SATELLITE                          │
│              (Regional Water Storage Constraint)            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               AQUIFER-LEVEL MODELS                          │
│         (8 aquifer zones, stratified kriging)               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              VILLAGE-LEVEL PREDICTIONS                      │
│   (939 villages × XGBoost + Kriging + Temporal)             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if needed)
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 2. Run the Prediction Pipeline

```bash
python backend/pipeline.py
```

This will:
- Load all 12 datasets (piezometers, villages, aquifers, bore wells, etc.)
- Preprocess and engineer features (rainfall, terrain, extraction)
- Fit spatial, temporal, and ML models
- Generate predictions for 939 villages
- Classify risk tiers (Critical/High/Moderate/Low)
- Save results to `outputs/` directory

### 3. Launch the Dashboard

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

## Differentiating Features

1. **Hierarchical Physics-Informed Model**
   - GRACE satellite data constrains regional predictions
   - Ensures village predictions sum to satellite observations

2. **Aquifer-Stratified Kriging**
   - No interpolation across aquifer boundaries
   - Respects hydrogeological heterogeneity

3. **Multi-Model Ensemble**
   - Spatial kriging (30% weight)
   - Feature-based ML (40% weight)
   - Temporal decomposition (30% weight)

4. **Risk Classification**
   - Multi-factor vulnerability assessment
   - Actionable alerts for decision-makers

5. **Scenario Analysis**
   - "What-if" drought simulation
   - Extraction impact assessment

## Project Structure

```
SmartJal/
├── backend/
│   ├── data_processing/
│   │   ├── load_data.py           # Load 12 datasets
│   │   ├── preprocess.py          # Clean, join, filter
│   │   └── feature_engineering.py # Build feature matrix
│   ├── models/
│   │   ├── spatial_model.py       # Aquifer-stratified kriging
│   │   ├── feature_model.py       # XGBoost/GBM model
│   │   ├── temporal_model.py      # Time series decomposition
│   │   ├── ensemble.py            # Combine models + GRACE
│   │   └── risk_classifier.py     # Risk tier assignment
│   └── pipeline.py                # End-to-end orchestration
├── frontend/
│   └── app.py                     # Streamlit dashboard
├── outputs/                       # Prediction results
└── downloaded_data/               # Input data files
```

## Data Sources

| Dataset | Records | Description |
|---------|---------|-------------|
| Water Levels | 138 piezometers × 28 years | Monthly groundwater levels |
| Villages | 939 polygons | Village boundaries |
| Aquifers | 8 zones | Aquifer type boundaries |
| Bore Wells | 88,988 records | Well locations and status |
| Soils | 151 polygons | Soil classification |
| Rainfall | 56 months | CHIRPS precipitation rasters |
| DEM | 30m resolution | SRTM elevation |
| GRACE | 120 months | Satellite water storage anomaly |

## Technical Approach

### Water Balance Equation
```
ΔStorage = Recharge - Extraction ± Lateral_Flow
```

### Temporal Decomposition
```
Water_Level = Baseline + Seasonal + Trend + Anomaly
```

### Model Outputs
- **Prediction**: Water level (m below ground)
- **Uncertainty**: Prediction confidence interval
- **Risk Tier**: Critical / High / Moderate / Low
- **Alerts**: Actionable recommendations

## Dashboard Features

1. **Executive Dashboard** - Key metrics and risk distribution
2. **Risk Map** - Interactive Folium map with village markers
3. **Predictions** - Searchable table with filters
4. **GRACE Satellite** - Regional water storage trends
5. **Alerts** - Prioritized list of villages needing attention
6. **Scenario Analysis** - What-if drought/extraction simulation

## License

Developed for the Smart Jal Hackathon Challenge (Challenge Code: 100008)
by the Andhra Pradesh Ground Water and Water Audit Department.
