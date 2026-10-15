# Data Dictionary — AI-Based Green Building Assistant

## Primary Dataset: UCI Energy Efficiency (ENB2012)

**File location (raw):** `dataset-sai/ENB2012_data.xlsx`
**Processed output:** `data/processed/energy_efficiency_clean.csv`
**Source:** [UCI ML Repository — Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
**Citation:** Tsanas, A. & Xifara, A. (2012). Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools. *Energy and Buildings*, 49, 560–567.

---

### Overview

| Attribute | Value |
|-----------|-------|
| Rows | 768 |
| Columns | 10 (8 features + 2 targets) |
| Missing values | 0 |
| Duplicate rows | 0 |
| Domain | Building energy simulation (Ecotect software) |
| Task | Multi-output regression — predict Y1 and Y2 |

---

### Feature Columns

| # | Raw Name | Renamed Column | Type | Unit / Range | Description |
|---|----------|----------------|------|-------------|-------------|
| 1 | X1 | `relative_compactness` | float64 | 0.62 – 0.98 | Ratio of the building's volume to its surface area; higher values indicate a more compact shape which typically reduces heat loss. |
| 2 | X2 | `surface_area` | float64 | 514.5 – 808.5 m² | Total external envelope area (floor + walls + roof). |
| 3 | X3 | `wall_area` | float64 | 245.0 – 416.5 m² | Total area of external walls. |
| 4 | X4 | `roof_area` | float64 | 110.25 – 220.5 m² | Total area of the roof. |
| 5 | X5 | `overall_height` | float64 | 3.5 or 7.0 m | Building height (2 discrete levels in the simulation). |
| 6 | X6 | `orientation` | int64 | {2, 3, 4, 5} | Cardinal facing of the building: 2=North, 3=East, 4=South, 5=West. |
| 7 | X7 | `glazing_area` | float64 | 0.0, 0.1, 0.25, 0.40 | Glazing area as a fraction of the floor area (0 = no glazing). |
| 8 | X8 | `glazing_area_distribution` | int64 | {0, 1, 2, 3, 4, 5} | Distribution of glazing across facades: 0=uniform, 1=55% N, 2=55% E, 3=55% S, 4=55% W, 5=skewed N. |

---

### Target Columns

| # | Raw Name | Renamed Column | Type | Unit | Description |
|---|----------|----------------|------|------|-------------|
| 9  | Y1 | `heating_load` | float64 | kWh/m² | Annual heating energy requirement per unit floor area. **Primary regression target for Phase 1.** |
| 10 | Y2 | `cooling_load` | float64 | kWh/m² | Annual cooling energy requirement per unit floor area. **Secondary regression target for Phase 1.** |

---

### Column Rename Map

```python
COLUMN_RENAME_MAP = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
    "Y1": "heating_load",
    "Y2": "cooling_load",
}
```

---

### Descriptive Statistics (raw dataset)

| Column | Min | Mean | Max | Std |
|--------|-----|------|-----|-----|
| relative_compactness | 0.62 | 0.764 | 0.98 | 0.106 |
| surface_area | 514.5 | 671.7 | 808.5 | 88.1 |
| wall_area | 245.0 | 318.5 | 416.5 | 43.6 |
| roof_area | 110.25 | 176.6 | 220.5 | 45.2 |
| overall_height | 3.5 | 5.25 | 7.0 | 1.75 |
| orientation | 2 | 3.5 | 5 | 1.12 |
| glazing_area | 0.0 | 0.234 | 0.4 | 0.134 |
| glazing_area_distribution | 0 | 2.81 | 5 | 1.55 |
| **heating_load** | 6.01 | 22.31 | 43.10 | 10.09 |
| **cooling_load** | 10.90 | 24.59 | 48.03 | 9.51 |

---

## Secondary Dataset: Appliances Energy Prediction

**File location (raw):** `dataset-sai/energydata_complete.csv`
(Identical copies: `energydata_complete 2.csv`, `energydata_complete 3.csv` — MD5 verified)

> ⚠️ **This dataset is NOT used in Phase 1** and must NOT be merged with ENB2012.
> It has a completely different schema (29 columns, IoT sensor readings) and a
> different prediction target (`Appliances` energy consumption in Wh, 10-minute intervals).

| Attribute | Value |
|-----------|-------|
| Rows | 19 735 |
| Columns | 29 |
| Time resolution | 10-minute intervals |
| Target | `Appliances` (Wh) — appliance energy consumption |
| Scope | Household IoT — temperatures, humidity, weather |

---

## Dataset Selection Rationale

The **ENB2012 dataset** is the recommended primary dataset for Phase 1 because:

1. **Target alignment** — Y1 (heating load) and Y2 (cooling load) are exactly the quantities the project objective specifies.
2. **Size** — 768 rows is sufficient for baseline sklearn models on an M2 MacBook with 8 GB RAM with negligible memory overhead.
3. **Cleanliness** — No missing values, no structural duplicates, all-numeric columns.
4. **Domain match** — Simulated residential buildings; directly relevant to green building energy assessment.
5. **Benchmark availability** — Widely cited in the literature, enabling comparison with published baselines.

The `energydata_complete.csv` dataset targets appliance-level consumption in a single household and is unsuitable for building-level heating/cooling load prediction.

---

## File Inventory

| File | Type | Rows | Cols | MD5 | Status |
|------|------|------|------|-----|--------|
| `dataset-sai/ENB2012_data.xlsx` | Excel | 768 | 10 | a708d1e1… | ✓ Primary — **use this** |
| `dataset-sai/ENB2012_data.csv` | CSV | 768 | 10 | a313d7d4… | ✓ Alternate format (1 row differs slightly due to float rounding) |
| `dataset-sai/energydata_complete.csv` | CSV | 19 735 | 29 | 69ef922b… | Different schema — Phase 2+ |
| `dataset-sai/energydata_complete 2.csv` | CSV | 19 735 | 29 | 69ef922b… | **Exact duplicate** of above |
| `dataset-sai/energydata_complete 3.csv` | CSV | 19 735 | 29 | 69ef922b… | **Exact duplicate** of above |
