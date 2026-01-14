# SmartJal POC Expectations

**Challenge Code**: 100008 - AI-Based Mapping and Forecasting of Groundwater Levels
**POC Area**: Krishna District, Andhra Pradesh
**Timeline**: 7-10 days from orientation

---

## Primary Deliverable

### What They Want
A **water level prediction (in meters below ground)** for every village in Krishna District.

| Output | Description |
|--------|-------------|
| Villages covered | ~800-900 villages in Krishna District |
| Values per village | Pre-monsoon + Post-monsoon water levels |
| Unit | Meters below ground level |

### Output Format Expected
```
Village | Mandal | Aquifer | Pre-Monsoon (m) | Post-Monsoon (m) | Category
--------|--------|---------|-----------------|------------------|----------
Avanigadda | Avanigadda | Alluvium | 3.2 | 2.1 | Safe
Gudiwada | Gudiwada | Alluvium | 4.8 | 3.5 | Moderate
Vijayawada | Vijayawada | Granite | 12.5 | 9.8 | Stress
...
```

### Water Level Categories
| Category | Depth Range | Department Action |
|----------|-------------|-------------------|
| **Safe** | 0-3 meters | No intervention needed |
| **Moderate** | 3-8 meters | Monitor regularly |
| **Stress** | 8-20 meters | Plan water conservation structures |
| **Critical** | >20 meters | Urgent intervention required |

---

## Why They Need This

### The Department's Goal
> "We are going for water conservation activity in a larger scale... we want to know the water level condition in a particular village so we can plan check dams, percolation tanks, recharge shafts."

### Current Problem
- Have **1,800 piezometers** across Andhra Pradesh
- Need to plan for **18,000+ villages**
- For POC: **138 piezometers** → **~800 villages** in Krishna District

### Why Simple Interpolation Fails
> "In alluvial area, water levels are very shallow. In crystalline rocks, water levels are very deep. If you interpolate, what we are getting is - in alluvial area we are getting deeper water levels, and in crystalline areas we are getting shallow water levels. That is not correct."

**Root cause**: Simple IDW/Kriging ignores aquifer boundaries and mixes readings from different geological formations.

---

## Technical Requirements

### Must Have
1. **Aquifer-aware predictions** - Do not interpolate across different aquifer types
2. **Pre-monsoon values** - Water levels during April-May (before rains)
3. **Post-monsoon values** - Water levels during October-November (after rains)
4. **Category assignment** - Classify each village as Safe/Moderate/Stress/Critical
5. **Visualization** - Map showing village-level predictions

### Nice to Have
1. Identify artificial recharge zones
2. Identify groundwater deficient zones (>20m)
3. Calculate water requirement to bring level to 8m
4. Recommendations for water conservation structures

---

## Validation Process

### How They Will Test

| Step | Description |
|------|-------------|
| 1 | Submit predictions for all ~800 villages |
| 2 | Department selects ~10% of villages randomly (~80 villages) |
| 3 | Field officers visit selected villages |
| 4 | Measure actual water levels in farmer bore wells |
| 5 | Compare predictions with actual measurements |

### Success Criteria
From Mr. Srinivas:
> "We will compare your data with the actual data from bore wells owned by farmers in the villages."

**Key**: Predictions should be **realistic for the aquifer type**:
- Alluvium villages → should predict shallow levels (2-8m)
- Hard rock villages → should predict deeper levels (10-30m)
- Should NOT assign shallow levels to hard rock areas or vice versa

---

## Demo Day Expectations

### What to Present

1. **Methodology**
   - How did you solve the aquifer boundary problem?
   - What approach/algorithm did you use?
   - What data did you use?

2. **Results**
   - Predictions for all villages
   - Distribution across categories
   - Map visualization

3. **Validation**
   - How accurate are your predictions?
   - Cross-validation results
   - Comparison with known piezometer values

### Questions They May Ask
- "How did you handle villages that span multiple aquifers?"
- "What happens when there are very few piezometers in an aquifer?"
- "How did you determine pre-monsoon vs post-monsoon values?"
- "Can you show predictions for a specific mandal?"

---

## Secondary Deliverables (If Time Permits)

### Recharge Zone Identification
- Identify areas suitable for artificial recharge
- Based on geomorphology, slope, soil type
- Output: Map of recharge potential zones

### Water Conservation Planning
For villages with deep water levels (>20m):
- Calculate water deficit
- Suggest suitable structures (check dam, percolation tank, etc.)
- Estimate water requirement to bring level to 8m

### Anomaly Detection
- Identify villages with unusual water level patterns
- Flag potential data quality issues
- Highlight areas needing special attention

---

## Data Provided for POC

| Dataset | Records | Purpose |
|---------|---------|---------|
| Piezometer Water Levels | 138 stations × 28 years | Training data |
| Aquifer Boundaries | 8 types | Constraint for interpolation |
| Geomorphology | 614 features | Recharge potential |
| GT Wells | 88,988 wells | Village locations + extraction proxy |
| LULC | 2005 shapefile | Land use patterns |

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Data Understanding | Day 1-2 | Explore data, understand patterns |
| Algorithm Development | Day 3-5 | Implement aquifer-constrained prediction |
| Validation | Day 6-7 | Cross-validate, fix issues |
| Demo Preparation | Day 8-10 | Visualization, presentation |

---

## Key Quotes from Orientation

### On the Problem
> "Simply if you interpolate the data points of 1,800, we are not getting the realistic picture in the villages. Because in the same village, the formation is alluvium or crystallines... different combinations are there."

### On the Solution
> "If simple interpolation is required, any of our department team can do that one. Every day, we are doing IDW interpolation. But we are not getting desired results."

### On Validation
> "We will visit that village and compare your output with the field data. We'll collect the data there itself in the field from farmers' bore wells."

### On the Goal
> "We want to assign a water level point data to every village, to all 18,000 villages, so that we will pick the villages where the water levels are deep and start water conservation activity in those villages."

---

## Summary

| Aspect | Requirement |
|--------|-------------|
| **Input** | 138 piezometers, 8 aquifer types (provided) |
| **Output** | Water level for ~800 villages |
| **Key constraint** | Must respect aquifer boundaries |
| **Validation** | Field verification by department |
| **Success** | Realistic predictions per aquifer type |
