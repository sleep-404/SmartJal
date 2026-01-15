# Smart Jal - Experiment Log
## Groundwater Level Prediction for Krishna District

---

## Data Sources Available

### Provided by Hackathon
| Data | Description | Records |
|------|-------------|---------|
| WaterLevels_Krishna | 138 piezometers, monthly readings | 1997-2025 (347 months) |
| OKri_Vil | Village boundaries | 939 villages |
| OKri_Mdl | Mandal boundaries | 42 mandals |
| Aquifers_Krishna | Aquifer zones | 8 types |
| OKri_Soils | Soil polygons | 151 polygons |
| GM_Krishna | Geomorphology | 614 polygons |
| GTWells_Krishna | Bore well records | 88,988 wells |
| LULC_Krishna | Land use/land cover | 4,295 polygons |

### Downloaded by Us
| Data | Source | Coverage |
|------|--------|----------|
| DEM (Elevation) | SRTM | Static raster, 30m resolution |
| Rainfall | IMD Gridded | 2018-2022 (56 months) |
| MODIS ET | Google Earth Engine | 2018-2024 (84 months) |
| Sentinel NDVI | Google Earth Engine | 2018-2024 (77 months) |
| SMAP Soil Moisture | Google Earth Engine | 2018-2024 (84 months) |
| CGWB National Data | Public CGWB | 2013-2023 (550,850 records) |
| GRACE TWS | NASA | 2015-2024 (120 months) |

---

## Experiment History

### Experiment 1: Baseline XGBoost (Initial)
**Date:** Earlier session
**Data Used:**
- Hackathon 138 piezometers
- Basic features (lat, lon, elevation, rainfall)

**Algorithm:** XGBoost
**Train/Test Split:** Random 80/20
**Results:**
| Metric | Value |
|--------|-------|
| R² | 0.778 |
| RMSE | ~3.5m |

**Issue Identified:** Model may be memorizing locations (lat/lon had 87% feature importance)

---

### Experiment 2: GNN (Graph Neural Network)
**Date:** Earlier session
**Data Used:** Same as Experiment 1
**Algorithm:** Graph Attention Network (GAT)
**Results:**
| Metric | Value |
|--------|-------|
| R² | 0.446 |
| RMSE | ~5m |

**Finding:** GNN underperformed - only 138 nodes is too few for graph learning to work effectively.

---

### Experiment 3: Regional Model with CGWB Data
**Date:** Current session
**Data Used:**
- CGWB data from Krishna + 6 neighboring districts
- 5,344 records from 573 stations
- Elevation, rainfall, seasonal features

**Algorithm:** Gradient Boosting
**Train/Test Split:**
- Train: Other districts (4,232 samples)
- Test: Krishna district (656 samples)

**Results:**
| Metric | Value |
|--------|-------|
| Test R² | -0.082 |
| Test RMSE | ~6m |

**Finding:** Cross-district training FAILED. Model learned patterns from other districts that don't apply to Krishna (different geology).

---

### Experiment 4: Time-Based Split (Honest Evaluation)
**Date:** Current session
**Data Used:**
- CGWB Krishna + neighbors (5,344 records)
- Train: 2018-2020
- Test: 2021

**Algorithm:** Gradient Boosting
**Results:**
| Metric | Value |
|--------|-------|
| Test R² | 0.682 |
| Test RMSE | 2.60m |

**Finding:** Time-based split shows actual learning ability. Lower than random split but honest.

---

### Experiment 5: Without Location Features
**Date:** Current session
**Purpose:** Test if model learns physics or memorizes locations
**Data Used:**
- CGWB data (5,344 records)
- Features: elevation, rainfall, satellite (NO lat/lon)

**Algorithm:** Gradient Boosting
**Results:**
| Metric | Value |
|--------|-------|
| Test R² | -0.059 |
| Test RMSE | 5.98m |

**Finding:** Physical features alone have NO predictive power. Model needs location context.

---

### Experiment 6: With Location Features
**Date:** Current session
**Data Used:** Same + lat/lon added
**Results:**
| Metric | Value |
|--------|-------|
| Test R² | 0.384 |
| Test RMSE | 4.56m |

**Finding:** Location adds 0.44 R² - confirms model relies on knowing WHERE the well is.

---

### Experiment 7: Autoregressive Model
**Date:** Current session
**Purpose:** Use previous water level to predict next
**Data Used:**
- CGWB data
- Previous measurement (prev_level) as feature
- Station mean, station std

**Algorithm:** Gradient Boosting
**Results:**
| Metric | Value |
|--------|-------|
| Test R² | 0.687 |
| Test RMSE | 3.26m |
| Baseline (prev_level only) | R² = 0.452 |

**Finding:** Model improves 52% over naive baseline. Learning station patterns + trends.

---

### Experiment 8: Full Satellite Features
**Date:** Current session
**Data Used:**
- CGWB data (4,728 samples with history)
- Full satellite coverage 2018-2024:
  - MODIS ET (84 months)
  - Sentinel NDVI (77 months)
  - SMAP Soil Moisture (84 months)
- Rainfall (56 months)
- Elevation
- Seasonal encoding

**Algorithm:** Gradient Boosting (150 trees, depth=5)
**Train/Test:** 2018-2020 / 2021
**Results:**
| Metric | Value |
|--------|-------|
| Train R² | 0.960 |
| Test R² | 0.732 |
| Test RMSE | 3.03m |
| Improvement vs baseline | 61% |

**Top Features:**
1. station_mean (67%)
2. prev_level_2 (12%)
3. prev_level (4%)
4. days_since_prev (4%)

**Finding:** Satellite features contribute <2% importance. Station history dominates.

---

### Experiment 9: Static Pumping Proxies
**Date:** Current session
**Purpose:** Add extraction indicators from bore well data
**Data Used:**
- All from Experiment 8 +
- num_wells per village
- avg_well_depth
- total_pump_capacity
- total_irrigated_area

**Results:**
| Metric | Value |
|--------|-------|
| Test R² | 0.725 |
| Test RMSE | 3.07m |
| Pumping features contribution | 0.1% |

**Finding:** Static pumping data doesn't help - it doesn't vary over time.

---

### Experiment 10: Dynamic Proxies (GRACE + ET Gap) ⭐ BEST
**Date:** Current session
**Purpose:** Add time-varying extraction indicators
**Data Used:**
- All previous features +
- **GRACE TWS anomaly** (satellite groundwater storage)
- **GRACE trend** (long-term depletion)
- **ET-rainfall gap** (irrigation proxy)
- **Dry season NDVI** (irrigation activity)
- **SMAP anomaly** (soil moisture deviation)

**Algorithm:** Gradient Boosting (200 trees, depth=5, lr=0.08)
**Train/Test:** 2018-2020 / 2021
**Results:**
| Metric | Value |
|--------|-------|
| Train R² | 0.964 |
| **Test R²** | **0.854** |
| **Test RMSE** | **2.24m** |
| Test MAE | 1.14m |
| Improvement vs baseline | **87%** |

**Top Features:**
1. station_mean (67%)
2. prev_level_2 (12%)
3. station_range (4%)
4. prev_level (4%)
5. **grace_trend (3.1%)** ← NEW
6. station_std (3%)
7. **grace_tws (1.6%)** ← NEW

**Dynamic proxy contribution: 5.3%**

**Key Finding:** GRACE satellite data (groundwater storage anomaly) significantly improved predictions. It captures regional extraction patterns we couldn't observe directly.

---

## Summary: Accuracy Progression

| Experiment | Data Added | Test R² | Test RMSE | Change |
|------------|-----------|---------|-----------|--------|
| 1. Baseline | Basic features | 0.778* | ~3.5m | - |
| 2. GNN | Graph structure | 0.446 | ~5m | -0.33 |
| 3. Regional | Cross-district | -0.082 | ~6m | Failed |
| 4. Time-split | Honest eval | 0.682 | 2.60m | Baseline |
| 5. No location | Physics only | -0.059 | 5.98m | Failed |
| 6. With location | + lat/lon | 0.384 | 4.56m | -0.30 |
| 7. Autoregressive | + prev_level | 0.687 | 3.26m | +0.30 |
| 8. Satellite | + ET/NDVI/SMAP | 0.732 | 3.03m | +0.05 |
| 9. Static pumping | + well counts | 0.725 | 3.07m | -0.01 |
| **10. Dynamic proxies** | **+ GRACE** | **0.854** | **2.24m** | **+0.12** |

*Experiment 1 used random split, not time-based, so not directly comparable.

---

## Key Insights for Jury

### What Works:
1. **Station history** - Knowing a well's typical depth is the strongest predictor
2. **Autoregressive features** - Recent measurements predict near-future well
3. **GRACE satellite data** - Captures regional groundwater depletion trends
4. **Seasonal encoding** - Monsoon/post-monsoon patterns matter

### What Doesn't Work:
1. **Cross-district transfer** - Models don't generalize across geological boundaries
2. **Raw satellite features** - ET, NDVI, SMAP alone contribute <2%
3. **Static pumping data** - Well counts don't explain temporal changes
4. **Graph Neural Networks** - Need more nodes to learn graph structure

### Why 85% is the Ceiling:
- **Missing data:** Actual pumping rates (the #1 driver of water level changes)
- **Unpredictable factors:** Farmer decisions, electricity availability, crop choices

---

## Final Model Specification

**Algorithm:** Gradient Boosting Regressor
- n_estimators: 200
- max_depth: 5
- learning_rate: 0.08
- subsample: 0.8

**Features (30 total):**
- Station: station_mean, station_std, station_range
- Autoregressive: prev_level, prev_level_2, days_since_prev, prev_anomaly
- Spatial: elevation
- Rainfall: current + 3 lags + cumulative
- Satellite: modis_et, ndvi, smap
- **GRACE: tws_anomaly, trend** (key addition)
- Derived: et_rainfall_gap, dry_season_ndvi, smap_anomaly
- Seasonal: month_sin, month_cos, is_monsoon, is_post_monsoon, is_summer
- Temporal: year

**Performance:**
- Test R²: 0.854 (85.4% variance explained)
- Test RMSE: 2.24m
- Test MAE: 1.14m

**Model saved at:** `outputs/model_with_dynamic_proxies.joblib`

---

### Experiment 11: Hackathon Data (1997-2024) ⭐ FINAL
**Date:** Current session
**Purpose:** Train on actual hackathon-provided piezometer data
**Data Used:**
- 138 piezometers from hackathon (1997-November 2025)
- Train: 1997-2023 (20,198 samples)
- Test: 2024 (3,004 samples)
- Features: station stats, autoregressive (3 lags), GRACE, satellite, seasonal

**Algorithm:** Gradient Boosting (200 trees, depth=5, lr=0.08)
**Results:**
| Metric | Value |
|--------|-------|
| Train R² | 0.954 |
| **Test R² (2024)** | **0.913** |
| **Test RMSE** | **2.18m** |
| Test MAE | 0.95m |
| Baseline R² | 0.926 |
| vs Baseline | **-1.4%** |

**Top Features:**
1. prev_level (55.9%)
2. prev_level_2 (31.9%)
3. station_mean (5.1%)
4. prev_level_3 (3.1%)
5. station_range (1.4%)

**Critical Finding:**
The model performs **WORSE** than the simple baseline (just using previous month's level). This happens because:
1. Hackathon water level data is highly autocorrelated (levels change slowly month-to-month)
2. 88% of feature importance is on previous level features
3. Satellite features (GRACE, NDVI, ET, SMAP) contribute <1% combined
4. The "previous level" baseline is extremely strong for this dataset

**Why Different from CGWB Results?**
- CGWB data had more irregular measurement intervals (days_since_prev varied widely)
- Hackathon data has consistent monthly measurements, making autoregression dominant
- The model essentially learns: "predict same as last month" which IS the baseline

---

## Summary: Accuracy Progression (Updated)

| Experiment | Data Source | Test R² | Test RMSE | Notes |
|------------|-------------|---------|-----------|-------|
| 1. Baseline | Hackathon (random split) | 0.778* | ~3.5m | Memorizing locations |
| 2. GNN | Hackathon | 0.446 | ~5m | Too few nodes |
| 3. Regional | CGWB cross-district | -0.082 | ~6m | Failed |
| 4. Time-split | CGWB | 0.682 | 2.60m | Honest eval |
| 5. No location | CGWB | -0.059 | 5.98m | Failed |
| 6. With location | CGWB | 0.384 | 4.56m | Memorizing |
| 7. Autoregressive | CGWB | 0.687 | 3.26m | +52% vs baseline |
| 8. Satellite | CGWB | 0.732 | 3.03m | +61% vs baseline |
| 9. Static pumping | CGWB | 0.725 | 3.07m | No improvement |
| 10. Dynamic proxies | CGWB | 0.854 | 2.24m | +87% vs baseline |
| **11. Hackathon Final** | **Hackathon 138** | **0.913** | **2.18m** | **-1.4% vs baseline** |

*Random split results are not directly comparable to time-based splits.

---

## Key Insights for Jury (Updated)

### What Works:
1. **Autoregressive features** - Previous water levels are the strongest predictor
2. **Station history** - Knowing a well's historical mean and range helps
3. **91.3% accuracy** - High R² even if not beating baseline

### What This Means:
1. Groundwater levels are **highly persistent** - they don't change drastically month-to-month
2. A model that predicts "similar to last month" will naturally score high
3. The challenge isn't prediction accuracy, it's **predicting the CHANGE**

### Limitations:
1. **Cannot predict sudden changes** - If pumping increases dramatically, model will lag
2. **No real-time pumping data** - The biggest driver of level changes is unmeasured
3. **Satellite data limited** - GRACE, NDVI, ET don't capture local extraction

### Honest Assessment:
- **91.3% R²** sounds impressive, but baseline achieves 92.6%
- The model is essentially learning "predict same as last month"
- Real value would come from predicting **when** levels will drop significantly

---

## Final Model Specification (Updated)

**Algorithm:** Gradient Boosting Regressor
- n_estimators: 200
- max_depth: 5
- learning_rate: 0.08
- subsample: 0.8

**Features (19 total):**
- Station: station_mean, station_std, station_range, total_depth
- Autoregressive: prev_level, prev_level_2, prev_level_3, prev_anomaly
- GRACE: grace_tws, grace_trend, grace_x_summer
- Satellite: modis_et, ndvi, smap
- Seasonal: month_sin, month_cos, is_monsoon, is_summer, year

**Performance on 2024 Validation:**
- R²: 0.913 (91.3% variance explained)
- RMSE: 2.18m
- MAE: 0.95m

**Model saved at:** `outputs/hackathon_model_2024.joblib`

---

## Next Steps
1. Generate 2025 predictions using trained model
2. Create visualization dashboard
3. Consider predicting CHANGE in water level instead of absolute level
