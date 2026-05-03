# Feature Engineering Suggestions
Dataset: `D:\\Downloads\\Agents\\AI_Platform\\platform_projects\\chat_20260330t100543_20260330_153543\\shared\\datasets\\Dataset.csv`

**⚠️ Data‑quality quick‑scan**  
- ✅ No missing values; every column is fully populated.  
- 📏 The three hydrologic flux variables (`pr`, `mrros`, `evspsbl`) are **orders of magnitude smaller** than the target (`mrso`).  This can cause numerical instability for some models; consider scaling or log‑transforming them.  
- 📈 The range of `pr` spans from ~0 → 6.35e‑4 mm day⁻¹ (tiny).  A few near‑zero rows may dominate ratio‑based features – add a tiny constant (e.g. 1e‑8) before division or log.  
- 📊 With only **120 observations**, any engineered feature that dramatically inflates dimensionality (e.g. high‑order polynomials) risks severe over‑fitting. Prioritise parsimonious, domain‑driven features.  



---

## 📅 Datetime Features  *(if the row order reflects a regular time step – e.g. daily, monthly)*  

- **`time_idx`** – simple integer index (0,1,2,…).  
  - *Why*: Captures linear trend or drift in soil moisture over the observation period.  
  - *Cols*: none (derived from row order).  
  - *Impact*: **Medium**  

- **`rolling_mean_pr_3`**, **`rolling_mean_pr_7`** – 3‑ and 7‑step moving averages of precipitation.  
  - *Why*: Soil moisture reacts to accumulated rain over recent days/weeks rather than a single instant.  
  - *Cols*: `pr`.  
  - *Impact*: **High**  

- **`rolling_sum_evspsbl_5`** – 5‑step rolling sum of evaporation.  
  - *Why*: Captures short‑term water loss that can offset recent precipitation.  
  - *Cols*: `evspsbl`.  
  - *Impact*: **Medium**  

- **`diff_pr_1`**, **`diff_evspsbl_1`** – first‑order difference (current – previous).  
  - *Why*: Highlights abrupt changes (e.g., a rainstorm) that may drive spikes in `mrso`.  
  - *Cols*: `pr`, `evspsbl`.  
  - *Impact*: **Medium**  



---

## 🔗 Interaction Features  

- **`pr_x_evspsbl`** – product of precipitation and evaporation.  
  - *Why*: Represents simultaneous input and output of water; high values may indicate strong flux cycles affecting soil moisture.  
  - *Cols*: `pr`, `evspsbl`.  
  - *Impact*: **Medium**  

- **`pr_minus_evspsbl`** – simple water‑balance (net input).  
  - *Why*: Direct proxy for net water added to the column; correlates positively with `mrso`.  
  - *Cols*: `pr`, `evspsbl`.  
  - *Impact*: **High**  

- **`mrros_x_pr`** – interaction between runoff and precipitation.  
  - *Why*: When precipitation is high but runoff is low, more water stays in the soil → higher `mrso`.  
  - *Cols*: `mrros`, `pr`.  
  - *Impact*: **Medium**  

- **`mrros_x_evspsbl`** – product of runoff and evaporation.  
  - *Why*: Captures situations where both loss mechanisms are active; could explain residual variance.  
  - *Cols*: `mrros`, `evspsbl`.  
  - *Impact*: **Low**  



---

## 📊 Aggregation Features  

- **`cumulative_pr`** – cumulative sum of precipitation up to current row.  
  - *Why*: Soil moisture integrates precipitation over time; cumulative amount is a strong predictor.  
  - *Cols*: `pr`.  
  - *Impact*: **High**  

- **`cumulative_evspsbl`** – cumulative evaporation.  
  - *Why*: Cumulative loss counteracts cumulative gain; the net balance (cumulative_pr – cumulative_evspsbl) is informative.  
  - *Cols*: `evspsbl`.  
  - *Impact*: **Medium**  

- **`window_std_pr_5`** – rolling standard deviation of precipitation over 5 steps.  
  - *Why*: High variability in recent rain may destabilize soil moisture, affecting `mrso`.  
  - *Cols*: `pr`.  
  - *Impact*: **Low**  

- **`window_mean_mrros_3`** – 3‑step moving average of runoff.  
  - *Why*: Smooths noisy runoff signal, better aligns with slower soil‑moisture response.  
  - *Cols*: `mrros`.  
  - *Impact*: **Medium**  



---

## 🧩 Encoding / Binning Features  

- **`pr_bin`** – discretize precipitation into quantile bins (e.g., 4‑quartile categories).  
  - *Why*: Non‑linear response of soil moisture to low vs. high rain events; tree‑based models benefit from categorical splits.  
  - *Cols*: `pr`.  
  - *Impact*: **Medium**  

- **`evspsbl_bin`** – similar binning for evaporation.  
  - *Why*: Allows model to capture thresholds where evaporation starts dominating.  
  - *Cols*: `evspsbl`.  
  - *Impact*: **Low**  

- **`mrros_bin`** – runoff intensity bins.  
  - *Why*: Runoff may have a threshold effect on soil water retention.  
  - *Cols*: `mrros`.  
  - *Impact*: **Low**  



---

## ⚖️ Ratio / Normalisation Features  

- **`pr_over_evspsbl`** – precipitation divided by evaporation.  
  - *Why*: Directly measures net water gain; values >1 imply wet conditions, <1 imply drying.  
  - *Cols*: `pr`, `evspsbl`.  
  - *Impact*: **High**  

- **`mrros_over_pr`** – runoff per unit precipitation.  
  - *Why*: Proxy for drainage efficiency; high ratio means water leaves the system quickly, limiting soil moisture.  
  - *Cols*: `mrros`, `pr`.  
  - *Impact*: **Medium**  

- **`pr_over_mrros`** – inverse of above (precipitation per unit runoff).  
  - *Why*: Highlights periods when most rain infiltrates rather than runs off.  
  - *Cols*: `pr`, `mrros`.  
  - *Impact*: **Medium**  

- **`evspsbl_over_mrros`** – evaporation relative to runoff.  
  - *Why*: If evaporation dominates runoff, soil moisture may decline faster.  
  - *Cols*: `evspsbl`, `mrros`.  
  - *Impact*: **Low**  

- **`log_pr`**, **`log_evspsbl`**, **`log_mrros`** – natural‑log transform (add 1e‑8 to avoid log(0)).  
  - *Why*: Handles the extreme skewness of these fluxes, making linear relationships more linear and stabilising variance.  
  - *Cols*: `pr`, `evspsbl`, `mrros`.  
  - *Impact*: **Medium**  



---

## 🌍 Domain‑Specific Hydrological Features  

- **`water_balance`** = `pr` – `evspsbl` – `mrros`  
  - *Why*: Classic water‑budget equation; net input to the soil column after accounting for loss pathways. Strongly tied to `mrso`.  
  - *Cols*: `pr`, `evspsbl`, `mrros`.  
  - *Impact*: **High**  

- **`potential_soil_moisture_gain`** = cumulative(`water_balance`)  
  - *Why*: Integrates net water input over time, approximating the theoretical soil moisture trajectory.  
  - *Cols*: `pr`, `evspsbl`, `mrros`.  
  - *Impact*: **High**  

- **`runoff_coefficient`** = `mrros` / (`pr` + ε)  
  - *Why*: Fraction of precipitation that becomes runoff; high values suggest low infiltration → lower `mrso`.  
  - *Cols*: `mrros`, `pr`.  
  - *Impact*: **Medium**  

- **`evaporation_efficiency`** = `evspsbl` / (`pr` + ε)  
  - *Why*: When evaporation consumes a large share of incoming rain, soil moisture is depleted faster.  
  - *Cols*: `evspsbl`, `pr`.  
  - *Impact*: **Medium**  

- **`soil_moisture_deficit`** = `mrso` – (`pr` – `evspsbl`) (used as a *target‑derived* feature for residual modeling)  
  - *Why*: Helps a second‑stage model capture the portion of `mrso` not explained by simple flux balance.  
  - *Cols*: `mrso`, `pr`, `evspsbl`.  
  - *Impact*: **Low** (mainly for stacked/ensemble approaches)  



---

### Quick implementation tip
```python
import numpy as np
import pandas as pd

df = df.copy()
eps = 1e-8

# basic ratios
df['pr_over_evspsbl'] = df['pr'] / (df['evspsbl'] + eps)
df['mrros_over_pr']   = df['mrros'] / (df['pr'] + eps)

# water‑balance
df['water_balance'] = df['pr'] - df['evspsbl'] - df['mrros']

# cumulative net input
df['cumulative_balance'] = df['water_balance'].cumsum()

# rolling aggregates (assuming daily frequency)
df['pr_rollmean_3'] = df['pr'].rolling(3, min_periods=1).mean()
df['pr_rollsum_7']  = df['pr'].rolling(7, min_periods=1).sum()
df['evspsbl_rollsum_5'] = df['evspsbl'].rolling(5, min_periods=1).sum()
```

These engineered variables should give your model richer hydrological context while keeping the feature space modest—critical for a 120‑row data set. Good luck!