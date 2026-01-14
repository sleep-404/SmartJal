# Smart Jal - Winning Architecture

**Goal:** A 10/10 idea that no other team will match.

---

## The Insight Most Teams Will Miss

Most teams will treat this as a **spatial interpolation problem**:
> "Given 138 points, predict 939 points"

This is wrong. Groundwater is a **physical system** with:
- Inputs (rainfall → recharge)
- Outputs (pumping → extraction)
- Storage (aquifer)
- Flow (hydraulic gradients)

**Our approach:** Model the PHYSICS, not just the PATTERN.

---

## The Winning Narrative

> "We don't just predict water levels - we understand WHY they change.
> Our system decomposes groundwater into physical components, uses satellite
> gravity data as a regional constraint, respects geological boundaries,
> quantifies uncertainty, and enables 'what-if' scenario planning for
> government interventions."

---

## Core Innovation: Hierarchical Physics-Informed Prediction

### The Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL CONSTRAINT SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LEVEL 1: REGIONAL (GRACE Satellite)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  GRACE TWS Anomaly (~50km resolution)                                │   │
│  │  Constraint: Sum of village predictions must match satellite total   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  LEVEL 2: AQUIFER (Geology-Aware)                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  8 Aquifer zones with distinct behavior                              │   │
│  │  Constraint: No interpolation across aquifer boundaries              │   │
│  │  Each aquifer has: storage coefficient, transmissivity, recharge %   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  LEVEL 3: VILLAGE (Water Balance)                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  For each village:                                                   │   │
│  │  ΔStorage = Recharge - Extraction ± Lateral_Flow                    │   │
│  │                                                                      │   │
│  │  Recharge = Rainfall × f(soil, slope, land_use)                     │   │
│  │  Extraction = Wells × Draft_per_well × Months_active                │   │
│  │  Lateral_Flow = f(hydraulic_gradient, aquifer_K)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  LEVEL 4: TEMPORAL (Decomposition)                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Water_Level(t) = Baseline + Seasonal(t) + Trend(t) + Anomaly(t)    │   │
│  │                                                                      │   │
│  │  Seasonal: Monsoon cycle (predictable, harmonic)                    │   │
│  │  Trend: Long-term trajectory (concerning if negative)               │   │
│  │  Anomaly: Deviations from expected (triggers alerts)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Wins

| Aspect | Other Teams | Our Approach |
|--------|-------------|--------------|
| Physics | Ignore | Water balance equation |
| Scale | Single level | Hierarchical (satellite → village) |
| Geology | Maybe stratify | Hard boundary constraint |
| Temporal | Ignore or simple | Decomposition (seasonal + trend + anomaly) |
| Uncertainty | None | Conformal prediction intervals |
| Actionability | Just predictions | Risk tiers + scenarios + recommendations |

---

## Six Differentiating Features

### 1. GRACE Satellite Integration (Nobody Else Will Do This)

**What:** Use NASA GRACE gravity satellite to constrain regional groundwater storage.

**Why it matters:**
- GRACE measures actual mass change (groundwater + soil water)
- Validated for India (Rodell et al., Nature 2009)
- Provides "ground truth" at 50km scale
- Our village predictions must SUM to satellite observation

**Implementation:**
```python
# Constraint during training/prediction
village_storage_sum = sum(village_predictions * village_areas)
grace_regional = get_grace_anomaly(krishna_bounds, month)

# Add as loss term or post-hoc calibration
constraint_loss = (village_storage_sum - grace_regional)²
```

**Differentiation:** ⭐⭐⭐⭐⭐ (Unique)

---

### 2. Physics-Informed Water Balance (Not Just Pattern Matching)

**What:** Model the actual water budget, not just spatial patterns.

**The Equation:**
```
ΔStorage = Recharge - Extraction ± Lateral_Flow

Where:
  Recharge = Rainfall × Recharge_Factor
  Recharge_Factor = f(soil_infiltration, slope, land_use, antecedent_moisture)

  Extraction = Σ(wells × draft_per_well × operating_hours)

  Lateral_Flow = K × i × A  (Darcy's Law)
  K = hydraulic conductivity (from aquifer type)
  i = hydraulic gradient (from neighboring water levels)
  A = cross-sectional area
```

**Why it matters:**
- Explainable: "Water level dropped because extraction exceeded recharge"
- Enables scenarios: "If rainfall is 20% below normal, expect X decline"
- Physically consistent: Predictions obey conservation laws

**Differentiation:** ⭐⭐⭐⭐⭐ (Rare in ML approaches)

---

### 3. Temporal Decomposition (28 Years of Signal)

**What:** Decompose water level time series into interpretable components.

```
Water_Level(t) = Baseline + Seasonal(t) + Trend(t) + Anomaly(t)
```

**Components:**

| Component | Meaning | Action |
|-----------|---------|--------|
| **Baseline** | Long-term equilibrium for location | Reference point |
| **Seasonal** | Monsoon cycle (predictable) | Plan around it |
| **Trend** | Secular change over years | ALERT if declining |
| **Anomaly** | Unexpected deviations | INVESTIGATE cause |

**Example Output:**
```
Village: Gudivada
├── Baseline: 12.5m (typical for alluvial aquifer)
├── Seasonal: +3.2m (Oct) to -4.1m (May) - normal monsoon cycle
├── Trend: -0.8m/year over 10 years ⚠️ DECLINING
└── Anomaly: -1.2m below expected this month ⚠️ ALERT
```

**Differentiation:** ⭐⭐⭐⭐ (Shows deep understanding)

---

### 4. Risk Classification + Early Warning System

**What:** Convert predictions to actionable risk categories.

**Risk Framework:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    RISK CLASSIFICATION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CRITICAL (Red) - Immediate Action Required                     │
│  ├── Water level approaching pump failure depth                 │
│  ├── Declining trend > 1.5m/year                               │
│  └── Anomaly > 2σ below expected                               │
│                                                                 │
│  HIGH (Orange) - Monitor Closely                                │
│  ├── Water level within 3m of critical                         │
│  ├── Declining trend 0.5-1.5m/year                             │
│  └── Below-normal monsoon recovery                              │
│                                                                 │
│  MODERATE (Yellow) - Watch                                      │
│  ├── Stable but high extraction pressure                        │
│  └── Seasonal stress during summer                              │
│                                                                 │
│  LOW (Green) - Stable                                           │
│  ├── Healthy recharge-extraction balance                        │
│  └── Normal seasonal patterns                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Early Warning Triggers:**
```python
def generate_alerts(village_prediction):
    alerts = []

    # Trend alert
    if village_prediction.trend_per_year < -1.0:
        alerts.append({
            'type': 'DECLINING_TREND',
            'severity': 'HIGH',
            'message': f'Water level declining at {abs(trend):.1f}m/year',
            'action': 'Consider recharge structures or extraction limits'
        })

    # Anomaly alert
    if village_prediction.anomaly < -2 * village_prediction.anomaly_std:
        alerts.append({
            'type': 'UNUSUAL_DECLINE',
            'severity': 'CRITICAL',
            'message': 'Water level significantly below expected',
            'action': 'Investigate cause - possible over-extraction or failed recharge'
        })

    # Threshold alert
    if village_prediction.level < village_prediction.critical_depth:
        alerts.append({
            'type': 'APPROACHING_FAILURE',
            'severity': 'CRITICAL',
            'message': 'Water level approaching pump failure depth',
            'action': 'Immediate intervention required'
        })

    return alerts
```

**Differentiation:** ⭐⭐⭐⭐ (Government actually needs this)

---

### 5. Scenario Planning ("What-If" Analysis)

**What:** Enable government to test intervention strategies.

**Scenarios:**

| Scenario | Inputs Modified | Question Answered |
|----------|-----------------|-------------------|
| Drought | Rainfall -30% | "What if monsoon fails?" |
| Extraction reduction | Pumping -20% | "Impact of usage restrictions?" |
| Recharge structures | Add recharge % to villages | "Where should we invest?" |
| Climate projection | Rainfall trend | "5-year outlook?" |

**Example Output:**
```
SCENARIO: Monsoon Failure (30% below normal)

Impact Assessment:
├── Villages moving to CRITICAL: 47 → 128 (+81)
├── Villages moving to HIGH: 156 → 234 (+78)
├── Average additional decline: 2.3m
└── Most vulnerable mandals: Nandigama, Mylavaram, Tiruvuru

Recommended Actions:
1. Pre-position tanker water for 128 critical villages
2. Enforce extraction limits in Nandigama mandal
3. Prioritize 23 villages for emergency recharge
```

**Differentiation:** ⭐⭐⭐⭐⭐ (Decision support, not just prediction)

---

### 6. Optimal Monitoring Network (Active Learning)

**What:** Tell the department WHERE to add new piezometers for maximum information gain.

**The Problem:**
- 138 piezometers for 939 villages (15% coverage)
- Adding more sensors is expensive
- Where should new sensors go?

**Our Solution:**
```python
def recommend_new_piezometer_locations(model, villages, n_new=10):
    """
    Recommend locations for new monitoring wells based on:
    1. Prediction uncertainty (high uncertainty = need data)
    2. Spatial coverage gaps
    3. Aquifer representation
    4. Population/importance weighting
    """
    scores = []
    for village in villages:
        # High uncertainty = valuable new data point
        uncertainty_score = model.predict_uncertainty(village)

        # Far from existing piezometers = coverage gap
        coverage_score = min_distance_to_piezometer(village)

        # Underrepresented aquifer = need more samples
        aquifer_score = 1 / aquifer_sample_count[village.aquifer]

        # Higher population = more important
        importance_score = village.population / max_population

        total = uncertainty_score * coverage_score * aquifer_score * importance_score
        scores.append((village, total))

    return sorted(scores, reverse=True)[:n_new]
```

**Example Output:**
```
RECOMMENDED NEW PIEZOMETER LOCATIONS (Top 10):

Rank  Village          Mandal       Aquifer    Uncertainty  Reason
1     Edupugallu       Nandigama    Granite    High         Coverage gap + high risk
2     Kanchikacherla   Penamaluru   Alluvium   High         Population center
3     Telaprolu        Gudivada     Shale      Very High    Underrepresented aquifer
...
```

**Differentiation:** ⭐⭐⭐⭐⭐ (Shows systems thinking, ongoing value)

---

## Technical Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RAW DATA                                                                   │
│  ├── Water Levels (138 piezometers × 28 years monthly)                     │
│  ├── Village Boundaries (939 polygons)                                      │
│  ├── Aquifer Boundaries (8 zones)                                          │
│  ├── Rainfall (CHIRPS 5km monthly)                                         │
│  ├── DEM (SRTM 30m)                                                        │
│  ├── Soil (151 polygons)                                                   │
│  ├── Bore Wells (88,988 points)                                            │
│  ├── Pumping Data (714 village records)                                    │
│  └── GRACE TWS (50km monthly) [TO DOWNLOAD]                                │
│                              ↓                                              │
│  FEATURE ENGINEERING                                                        │
│  ├── Per Village:                                                          │
│  │   ├── Aquifer type (categorical)                                        │
│  │   ├── Soil infiltration class                                           │
│  │   ├── Mean elevation, slope                                             │
│  │   ├── Monthly rainfall (lagged 0-6 months)                              │
│  │   ├── Number of wells, total extraction                                  │
│  │   ├── Land use fractions                                                │
│  │   └── Distance to nearest piezometer (by aquifer)                       │
│  │                                                                          │
│  ├── Per Piezometer:                                                       │
│  │   ├── All village features above                                        │
│  │   ├── Time series decomposition (seasonal, trend, anomaly)              │
│  │   └── Neighboring piezometer values (same aquifer)                      │
│  │                                                                          │
│  └── Regional:                                                             │
│      └── GRACE groundwater storage anomaly                                 │
│                              ↓                                              │
│  MODEL ENSEMBLE                                                            │
│  ├── Component 1: Aquifer-Stratified Kriging (spatial structure)           │
│  ├── Component 2: Gradient Boosting (feature relationships)                │
│  ├── Component 3: Temporal Model (ARIMA/Prophet per cluster)               │
│  └── Meta-Learner: Weighted combination + GRACE constraint                 │
│                              ↓                                              │
│  OUTPUT LAYERS                                                             │
│  ├── Point Prediction (water level in meters)                              │
│  ├── Uncertainty Interval (90% confidence bounds)                          │
│  ├── Temporal Decomposition (baseline, seasonal, trend, anomaly)           │
│  ├── Risk Classification (Critical/High/Moderate/Low)                      │
│  └── Alerts (if any thresholds crossed)                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL ENSEMBLE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │  SPATIAL COMPONENT  │  │  FEATURE COMPONENT  │  │ TEMPORAL COMPONENT  │ │
│  │                     │  │                     │  │                     │ │
│  │  Aquifer-Stratified │  │  XGBoost/LightGBM   │  │  STL Decomposition  │ │
│  │  Kriging            │  │                     │  │  + ARIMA            │ │
│  │                     │  │  Features:          │  │                     │ │
│  │  - Only interpolate │  │  - Rainfall (lags)  │  │  For each cluster:  │ │
│  │    within same      │  │  - Extraction       │  │  - Fit seasonal     │ │
│  │    aquifer          │  │  - Soil class       │  │  - Fit trend        │ │
│  │  - Learns spatial   │  │  - Elevation        │  │  - Identify anomaly │ │
│  │    correlation      │  │  - Distance to      │  │                     │ │
│  │    structure        │  │    water bodies     │  │                     │ │
│  │                     │  │  - LULC fractions   │  │                     │ │
│  └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘ │
│             │                        │                        │             │
│             └────────────────────────┼────────────────────────┘             │
│                                      ↓                                      │
│                        ┌─────────────────────────┐                          │
│                        │      META-LEARNER       │                          │
│                        │                         │                          │
│                        │  - Learns optimal       │                          │
│                        │    combination weights  │                          │
│                        │  - Applies GRACE        │                          │
│                        │    constraint           │                          │
│                        │  - Outputs uncertainty  │                          │
│                        └─────────────────────────┘                          │
│                                      ↓                                      │
│                        ┌─────────────────────────┐                          │
│                        │    FINAL PREDICTION     │                          │
│                        │                         │                          │
│                        │  - Water level (m)      │                          │
│                        │  - 90% CI bounds        │                          │
│                        │  - Risk class           │                          │
│                        │  - Decomposition        │                          │
│                        │  - Alerts               │                          │
│                        └─────────────────────────┘                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Visualization Dashboard

### Main Views

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SMART JAL - Groundwater Intelligence System                    [Jan 2026]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ SUMMARY CARDS ─────────────────────────────────────────────────────┐   │
│  │  [🔴 47]        [🟠 156]       [🟡 234]       [🟢 502]              │   │
│  │  Critical       High Risk      Moderate       Stable                │   │
│  │  ↑12 vs last    ↑23 vs last   ↓8 vs last    ↓27 vs last           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ MAP VIEW ──────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │     [Interactive Folium/Mapbox Map]                                 │   │
│  │                                                                      │   │
│  │     • Villages colored by risk level                                │   │
│  │     • Click village for details                                     │   │
│  │     • Toggle: Predictions | Trends | Anomalies | Aquifers          │   │
│  │     • Piezometer locations marked                                   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ SELECTED VILLAGE DETAIL ───────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  Village: Gudivada          Mandal: Gudivada      Aquifer: Alluvium │   │
│  │                                                                      │   │
│  │  Current Level: 14.2m       Predicted: 14.5m ± 1.2m                 │   │
│  │  Risk Level: 🟠 HIGH        Trend: -0.8m/year (declining)          │   │
│  │                                                                      │   │
│  │  [Time Series Chart: Historical + Forecast]                         │   │
│  │  ├── Actual measurements (where available)                          │   │
│  │  ├── Model predictions with confidence band                         │   │
│  │  ├── Seasonal component                                             │   │
│  │  └── Long-term trend line                                          │   │
│  │                                                                      │   │
│  │  Influencing Factors:                                               │   │
│  │  ├── Piezometer P-047 (3.2km, same aquifer): 45% weight            │   │
│  │  ├── Rainfall (last 3 months): 23% weight                          │   │
│  │  ├── Extraction intensity: 18% weight                               │   │
│  │  └── Regional GRACE anomaly: 14% weight                             │   │
│  │                                                                      │   │
│  │  ⚠️ ALERTS:                                                         │   │
│  │  • Declining trend detected (-0.8m/year for 5 years)               │   │
│  │  • Extraction exceeds estimated recharge by 20%                     │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ SCENARIO ANALYSIS ─────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  Scenario: [Dropdown: Normal | Drought | Reduced Extraction | ... ]│   │
│  │                                                                      │   │
│  │  Impact Preview:                                                    │   │
│  │  • Villages turning Critical: +81                                   │   │
│  │  • Worst affected mandals: Nandigama, Mylavaram                    │   │
│  │  • Recommended interventions: [List]                                │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Foundation (Day 1)
| Task | Time | Output |
|------|------|--------|
| Data preprocessing pipeline | 2 hrs | Clean, joined datasets |
| Feature engineering | 2 hrs | Village feature matrix |
| Temporal decomposition | 2 hrs | Seasonal/trend/anomaly per piezometer |
| **Checkpoint:** Basic features ready | | |

### Phase 2: Core Model (Day 2)
| Task | Time | Output |
|------|------|--------|
| Aquifer-stratified kriging | 2 hrs | Spatial predictions |
| XGBoost feature model | 2 hrs | Feature-based predictions |
| Ensemble + validation | 2 hrs | Combined predictions + metrics |
| **Checkpoint:** Working predictions | | |

### Phase 3: Differentiation (Day 3)
| Task | Time | Output |
|------|------|--------|
| GRACE data download + integration | 3 hrs | Regional constraint |
| Risk classification system | 2 hrs | Risk tiers + alerts |
| Uncertainty quantification | 1 hr | Confidence intervals |
| **Checkpoint:** Differentiated model | | |

### Phase 4: Visualization (Day 4)
| Task | Time | Output |
|------|------|--------|
| Streamlit dashboard | 3 hrs | Interactive UI |
| Folium map integration | 2 hrs | Spatial visualization |
| Village detail views | 2 hrs | Drilldown capability |
| **Checkpoint:** Demo-ready | | |

### Phase 5: Polish (Day 5)
| Task | Time | Output |
|------|------|--------|
| Scenario analysis | 2 hrs | What-if capability |
| Optimal sensor placement | 2 hrs | Recommendations |
| Documentation + presentation | 2 hrs | Pitch materials |
| **Checkpoint:** Competition-ready | | |

---

## Success Metrics

### Technical Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| MAE (vs held-out piezometers) | < 2.0m | Cross-validation |
| Uncertainty calibration | 90% CI covers 90% of actuals | Coverage test |
| Risk classification accuracy | > 85% | Confusion matrix |

### Business Metrics
| Metric | Target |
|--------|--------|
| Villages with predictions | 939/939 (100%) |
| Villages with risk classification | 939/939 (100%) |
| Scenarios supported | 4+ |
| Response time | < 3 seconds |

### Differentiation Checklist
- [ ] GRACE satellite integration (unique)
- [ ] Physics-informed water balance (rare)
- [ ] Temporal decomposition (uncommon)
- [ ] Risk classification + alerts (practical)
- [ ] Scenario planning (decision support)
- [ ] Optimal monitoring recommendations (value-add)

---

## Why This Wins

| Judging Criteria | Our Strength |
|------------------|--------------|
| **Technical Innovation** | GRACE integration, physics-informed approach, hierarchical modeling |
| **Practical Value** | Risk classification, alerts, scenario planning - immediately useful |
| **Domain Understanding** | Water balance equation, geology-aware, temporal decomposition |
| **Scalability** | Framework works for any district in India |
| **Explainability** | Can explain WHY each prediction, influence factors visible |
| **Completeness** | Not just predictions - full decision support system |

---

## Final Verdict

**This is a 10/10 idea because:**

1. **Technically novel** - GRACE + physics-informed + hierarchical is rare
2. **Practically useful** - Government can actually use risk tiers and scenarios
3. **Scientifically sound** - Based on actual hydrology, not just patterns
4. **Demonstrably better** - Can show uncertainty, explain predictions
5. **Forward-looking** - Optimal sensor placement shows ongoing value

**No other team will have:**
- Satellite gravity data constraining village predictions
- Physics-based water balance (not just ML)
- Scenario planning capability
- Recommendations for new monitoring locations

---

*Document created: January 15, 2026*
*Purpose: Winning architecture for Smart Jal hackathon*
