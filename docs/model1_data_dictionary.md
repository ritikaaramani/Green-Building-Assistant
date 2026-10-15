# Model 1 — Data Dictionary

Source: U.S. EIA, 2018 CBECS public-use microdata (Dec 2022 release).
Dataset: data/processed/model1/model1_dataset.csv (Experiment 2 features, Option C target)

| Column | Role | Type | Meaning | Missing % |
|---|---|---|---|---|
| EFFICIENCY_BINARY | **Target (Option C)** | binary | 0 = High efficiency, 1 = Low efficiency; blank = Moderate (excluded) | 34.9 |
| EFFICIENCY_CLASS | Label source only (NOT a feature) | categorical | 0 = High, 1 = Moderate, 2 = Low (weighted EUI tertile within building type) | 0 |
| EUI | Label source only (NOT a feature) | numeric | MFBTU / SQFT, kBtu/sq ft/yr | 0 |
| FINALWT | Weight only (NOT a feature) | numeric | CBECS survey weight | 0 |
| PBA | Feature | categorical | Principal building activity | 0.0 |
| PUBCLIM | Feature | categorical | Third-party data: ASHRAE climate zone | 0.0 |
| REGION | Feature | categorical | Census region | 0.0 |
| CENDIV | Feature | categorical | Census division | 0.0 |
| WLCNS | Feature | categorical | Wall construction material | 0.0 |
| RFCNS | Feature | categorical | Roof construction material | 0.0 |
| BLDSHP | Feature | categorical | Building shape | 0.0 |
| OWNTYPE | Feature | categorical | Derived variable: Building owner | 0.0 |
| MAINHT | Feature | categorical | Derived variable: Main heating equipment | 5.1 |
| MAINCL | Feature | categorical | Derived variable: Main air-conditioning equipment | 5.6 |
| RENOV | Feature | categorical | Renovations | 6.3 |
| PBAPLUS | Feature | categorical | Derived variable: More specific building activity | 0.0 |
| NGUSED | Feature | categorical | Natural gas used | 0.0 |
| NGWATR | Feature | categorical | Natural gas used for water heating | 0.0 |
| EMCS | Feature | categorical | Building automation system (BAS) | 2.4 |
| MAINT | Feature | categorical | Regular HVAC maintenance | 2.4 |
| VAV | Feature | categorical | Variable air volume (VAV) system | 4.8 |
| SQFT | Feature | numeric | Square footage | 0.0 |
| NFLOOR | Feature | numeric | Number of floors | 0.0 |
| YRCONC | Feature | numeric | Year of construction category | 0.0 |
| WKHRS | Feature | numeric | Total hours open per week | 0.0 |
| NWKER | Feature | numeric | Number of employees | 0.0 |
| GLSSPC | Feature | numeric | Percent exterior glass | 0.0 |
| FLCEILHT | Feature | numeric | Floor to ceiling height | 0.0 |
| NELVTR | Feature | numeric | Number of elevators | 53.9 |
| BASEMNT | Feature | numeric | Number of underground floors | 39.7 |
| HEATP | Feature | numeric | Percent heated | 5.1 |
| COOLP | Feature | numeric | Percent cooled | 5.6 |
| LTOHRP | Feature | numeric | Percent lit when open | 0.5 |
| LTNHRP | Feature | numeric | Percent lit off hours | 3.2 |
| HDD65 | Feature | numeric | Third-party data: Heating degree days (base 65) | 0.0 |
| CDD65 | Feature | numeric | Third-party data: Cooling degree days (base 65) | 0.0 |
| PCTERMN | Feature | numeric | Number of desktop computers | 12.1 |
| LAPTPN | Feature | numeric | Number of laptops | 24.5 |
| RFGOPN | Feature | numeric | Number of open refrigerated cases | 82.4 |
| RFGWIN | Feature | numeric | Number of walk-in units | 62.3 |
| RFGVNN | Feature | numeric | Number of vending machines | 50.8 |
| FDPREPSF | Feature | numeric | Commercial kitchen square footage | 79.0 |
| LOG_SQFT | Feature | numeric | log10 of floor area (reduces skew) | 0.0 |
| WORKERS_PER_1000SQFT | Feature | numeric | employees on main shift per 1,000 sq ft (occupancy density) | 0.0 |
| OPEN_24_7 | Feature | numeric | 1 if open 168 hours/week, else 0 | 0.0 |
| PCS_PER_1000SQFT | Feature | numeric | desktop + laptop computers per 1,000 sq ft | 8.1 |
| KITCHEN_SHARE | Feature | numeric | commercial kitchen area / total floor area | 79.0 |
