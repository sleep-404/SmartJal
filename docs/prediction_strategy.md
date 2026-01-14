# SmartJal Prediction Strategy

**Goal**: Make the best possible water level predictions using available data.

---

## Available Data Assets

### 1. Piezometer Time Series
| Attribute | Value |
|-----------|-------|
| Stations | 138 |
| Time range | January 1997 - November 2025 |
| Frequency | Monthly |
| Total readings | ~46,000 data points |

**Untapped potential**:
- 28 years of patterns (seasonal, trends)
- Recovery rates after monsoon
- Long-term decline/improvement trends

### 2. Well Data
| Attribute | Value |
|-----------|-------|
| Total wells | 88,988 |
| Fields | Location, bore depth, pump capacity, irrigated area, crop type |

**Untapped potential**:
- Proxy for extraction pressure
- Bore depth indicates local water table depth
- Village-level aggregation gives demand estimate

### 3. Aquifer Boundaries
| Attribute | Value |
|-----------|-------|
| Types | 8 principal aquifers |
| Coverage | Full Krishna District |

**Key aquifer types**:
- Alluvium (shallow water, high yield)
- Sandstone (moderate depth)
- Limestone (variable, karst)
- Granite/Gneiss (deep water, low yield)
- Shale (poor aquifer)

### 4. Geomorphology
| Attribute | Value |
|-----------|-------|
| Features | 614 polygons |
| Classes | Multiple landform types |

**Recharge potential by landform**:
| Landform | Recharge |
|----------|----------|
| Flood plain | High |
| Valley fill | High |
| Alluvial plain | High |
| Pediment | Moderate |
| Buried pediment | Variable |
| Structural hills | Low |
| Denudational hills | Very low |

---

## Prediction Approaches (Levels)

### Level 1: Basic (Baseline)
**Aquifer-Constrained IDW**

```
For each village:
    1. Find aquifer type at village location
    2. Get piezometers in SAME aquifer only
    3. IDW interpolation using those piezometers
    4. Output: water level
```

**Pros**: Simple, respects aquifer boundaries
**Cons**: Ignores extraction, geomorphology, temporal patterns

---

### Level 2: Enhanced IDW
**IDW + Extraction Correction**

```
For each village:
    1. Basic IDW within aquifer
    2. Calculate extraction intensity from well data
    3. Adjust: high extraction → deeper prediction
    4. Output: corrected water level
```

**Features used**:
- Aquifer type
- Well density
- Average bore depth

**Pros**: Accounts for local extraction
**Cons**: Still linear interpolation

---

### Level 3: Machine Learning (Recommended)
**Random Forest / LightGBM with Full Features**

```
Training:
    - Use 138 piezometers as training samples
    - Extract features for each piezometer location
    - Train model: features → water level

Prediction:
    - Extract same features for each village
    - Apply trained model
    - Output: predicted water level + uncertainty
```

**Pros**: Uses all available information, handles non-linear relationships
**Cons**: More complex, needs validation

---

## Feature Engineering

### Features to Extract per Location

```
LOCATION FEATURES
├── latitude
├── longitude
├── distance_to_nearest_piezometer
└── distance_to_nearest_drainage

AQUIFER FEATURES
├── aquifer_type (one-hot encoded)
│   ├── is_alluvium
│   ├── is_sandstone
│   ├── is_limestone
│   ├── is_granite
│   └── is_shale
├── aquifer_area_sqkm
└── piezometer_count_in_aquifer

GEOMORPHOLOGY FEATURES
├── landform_class
├── recharge_potential_score (derived)
└── is_flood_plain (binary)

EXTRACTION FEATURES (from wells)
├── well_count (within 2km radius)
├── well_density (wells per sq km)
├── avg_bore_depth
├── max_bore_depth
├── std_bore_depth
├── total_pump_capacity
├── total_irrigated_area
└── dominant_crop_water_intensity

TEMPORAL FEATURES (from piezometers in same aquifer)
├── aquifer_mean_water_level
├── aquifer_median_water_level
├── aquifer_std_water_level
├── aquifer_trend_per_year (slope)
├── aquifer_seasonal_amplitude
└── aquifer_min_max_range
```

### Feature Rationale

| Feature | Why It Helps |
|---------|--------------|
| `aquifer_type` | Primary determinant of water level behavior |
| `well_density` | More wells = more extraction = deeper levels |
| `avg_bore_depth` | Farmers drill deeper where water is deep |
| `landform_class` | Affects recharge potential |
| `aquifer_mean_level` | Baseline for that geological unit |
| `aquifer_trend` | Captures long-term changes |

---

## Training Strategy

### Data Split
```
Piezometers: 138 total
├── Training: 110 (80%)
└── Validation: 28 (20%)

Cross-validation: 5-fold spatial CV
```

### Spatial Cross-Validation
Important: Don't use random split. Use spatial folds to avoid data leakage.

```
Fold 1: Train on North, Test on South
Fold 2: Train on East, Test on West
... etc
```

### Target Variable
```
For pre-monsoon: Use May readings (average of last 5 years)
For post-monsoon: Use November readings (average of last 5 years)
```

### Model Selection
| Model | Pros | Cons |
|-------|------|------|
| Random Forest | Robust, handles mixed features | Less accurate |
| LightGBM | Fast, accurate | Needs tuning |
| XGBoost | Very accurate | Slower |
| **Recommended**: LightGBM with default params |

---

## Validation Approach

### Step 1: Leave-One-Out on Piezometers
```
For each piezometer:
    1. Remove it from training
    2. Train model on remaining 137
    3. Predict for removed piezometer
    4. Compare: predicted vs actual
```

### Step 2: Metrics
| Metric | Target |
|--------|--------|
| RMSE | < 3 meters |
| MAE | < 2 meters |
| Category accuracy | > 70% |
| R² | > 0.7 |

### Step 3: Sanity Checks
- Alluvium predictions should be 2-8m on average
- Hard rock predictions should be 10-30m on average
- No negative predictions
- No predictions > 100m

---

## Implementation Plan

### Phase 1: Feature Extraction
```python
# Extract features for all piezometers
piezometer_features = extract_features(piezometers, wells, aquifers, geomorphology)

# Extract features for all villages
village_features = extract_features(villages, wells, aquifers, geomorphology)
```

### Phase 2: Model Training
```python
# Train model
X = piezometer_features
y = piezometer_water_levels  # Use recent May/November averages

model = LightGBMRegressor()
model.fit(X, y)

# Cross-validate
cv_scores = spatial_cross_validation(model, X, y, n_folds=5)
```

### Phase 3: Prediction
```python
# Predict for all villages
village_predictions = model.predict(village_features)

# Assign categories
categories = assign_category(village_predictions)
```

### Phase 4: Output
```python
# Generate final output
output = pd.DataFrame({
    'village': villages.name,
    'mandal': villages.mandal,
    'aquifer': villages.aquifer_type,
    'pre_monsoon_level': predictions_may,
    'post_monsoon_level': predictions_nov,
    'category': categories,
    'confidence': model.predict_uncertainty()
})
```

---

## Expected Improvement

| Approach | Expected RMSE | Category Accuracy |
|----------|---------------|-------------------|
| Simple IDW | ~5-6 meters | ~50% |
| Aquifer-constrained IDW | ~4 meters | ~60% |
| ML with all features | ~2-3 meters | ~75% |

---

## Why This Will Win

1. **Uses ALL available data** - Not just piezometers
2. **Respects aquifer boundaries** - Core requirement
3. **Accounts for extraction** - Via well data proxy
4. **Captures temporal patterns** - 28 years of data
5. **Considers recharge potential** - Via geomorphology
6. **Provides uncertainty** - ML models can estimate confidence

---

## Files to Create

| File | Purpose |
|------|---------|
| `feature_extractor.py` | Extract all features |
| `model_trainer.py` | Train and validate model |
| `predictor.py` | Generate village predictions |
| `output_generator.py` | Create final deliverables |

---

## Summary

**Minimum approach**: Aquifer-constrained IDW (Level 1)
**Recommended approach**: ML with full features (Level 3)
**Differentiator**: Using well data as extraction proxy + temporal patterns from 28 years of data
