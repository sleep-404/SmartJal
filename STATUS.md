# SmartJal Project Status

**Last Updated**: 2026-01-14 09:45 IST
**Challenge Code**: 100008 - AI-Based Mapping and Forecasting of Groundwater Levels

---

## Quick Links

| Document | Description |
|----------|-------------|
| [expectations.md](docs/expectations.md) | What the POC requires |
| [prediction_strategy.md](docs/prediction_strategy.md) | How to make best predictions |
| [challenge_requirements.md](docs/challenge_requirements.md) | Full hackathon requirements |
| [data_sources.md](docs/data_sources.md) | External data sources (optional) |

---

## Current Status: READY TO BUILD CORE ALGORITHM

### Summary
| Component | Status |
|-----------|--------|
| Backend API | ✅ Running (http://localhost:8000) |
| Frontend Dashboard | ✅ Running (http://localhost:3000) |
| Data Loading | ✅ All department data loads correctly |
| Feature Extraction | 🔄 TODO |
| ML Model Training | 🔄 TODO |
| Village Predictions | 🔄 TODO (currently mock) |

---

## The Goal

> Assign a water level (meters) to every village in Krishna District, respecting aquifer boundaries.

**Input**: 138 piezometers + 8 aquifer types
**Output**: ~800 villages with pre-monsoon & post-monsoon water levels

---

## Data Status

### Available (From Department) ✅
| Data | Records | Use |
|------|---------|-----|
| Piezometer readings | 138 × 28 years | Training data |
| Aquifer boundaries | 8 types | Constraint |
| Geomorphology | 614 features | Recharge potential |
| GT Wells | 88,988 | Extraction proxy + village locations |
| LULC | 2005 | Land use (optional) |

### External Data (NOT Required)
| Data | Status | Notes |
|------|--------|-------|
| DEM | Skip | Nice-to-have |
| Rainfall | Skip | Nice-to-have |
| Soil | Skip | Nice-to-have |

---

## Prediction Strategy

### Chosen Approach: ML with Full Features (Level 3)

```
Features:
├── Aquifer type (one-hot)
├── Well density (extraction proxy)
├── Average bore depth
├── Geomorphology class
├── Aquifer-level statistics (mean, trend)
└── Distance to nearest piezometer

Model: LightGBM
Training: 138 piezometers
Prediction: ~800 villages
```

### Why This Approach
1. Uses ALL available data (not just piezometers)
2. Respects aquifer boundaries
3. Accounts for extraction via well data
4. Captures 28 years of temporal patterns
5. Handles sparse piezometer coverage

---

## Implementation TODO

### Phase 1: Feature Extraction
- [ ] Extract features for 138 piezometers
- [ ] Extract features for ~800 villages
- [ ] Features: aquifer, wells, geomorphology, temporal stats

### Phase 2: Model Training
- [ ] Train LightGBM on piezometer data
- [ ] Spatial cross-validation
- [ ] Target: RMSE < 3m, Category accuracy > 70%

### Phase 3: Prediction
- [ ] Generate predictions for all villages
- [ ] Pre-monsoon (May) and Post-monsoon (November)
- [ ] Assign categories (Safe/Moderate/Stress/Critical)

### Phase 4: Integration
- [ ] Update API to serve real predictions
- [ ] Update dashboard to show real values
- [ ] Export predictions as CSV/GeoJSON

---

## Services Running

| Service | URL | Status |
|---------|-----|--------|
| Backend API | http://localhost:8000 | ✅ Running |
| Frontend | http://localhost:3000 | ✅ Running |
| API Docs | http://localhost:8000/docs | ✅ Running |
| Health Check | http://localhost:8000/health | ✅ Healthy |

---

## Validation Plan

1. **Leave-one-out** on 138 piezometers
2. **Metrics**: RMSE, MAE, Category accuracy
3. **Sanity checks**: Alluvium=shallow, Hard rock=deep
4. **Final test**: Department field verification

---

## Key Files

| Purpose | Location |
|---------|----------|
| Data Loader | `backend/ml/data/loader.py` |
| Piezometer Data | `WaterLevels_Krishna/master data_updated.xlsx` |
| Aquifers | `Aquifers_Krishna/Aquifers_Krishna.shp` |
| Wells | `GTWells_Krishna/GTWells/kris.csv` |
| Geomorphology | `GM_Krishna/GM_Krishna.shp` |
| API | `backend/app/main.py` |
| Frontend | `frontend/src/App.jsx` |

---

## Next Action

**Implement feature extraction and ML model training.**

This is the core missing piece that will turn mock predictions into real ones.

---

## Session History

| Date | Action |
|------|--------|
| 2026-01-13 | Built backend API + frontend dashboard |
| 2026-01-13 | Fixed import errors, NaN handling |
| 2026-01-14 | Read orientation transcript |
| 2026-01-14 | Created expectations.md |
| 2026-01-14 | Created prediction_strategy.md |
| 2026-01-14 | Focused approach: use only provided data |

---

## Commands

```bash
# Start backend
cd backend && source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Start frontend
cd frontend && npm run dev

# View status
cat STATUS.md
```
