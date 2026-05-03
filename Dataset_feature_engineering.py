"""Feature engineering code for D:\\Downloads\\Agents\\AI_Platform\\platform_projects\\chat_20260330t100543_20260330_153543\\shared\\datasets\\Dataset.csv — verified by execution agent"""

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"D:\\Downloads\\Agents\\AI_Platform\\platform_projects\\chat_20260330t100543_20260330_153543\\shared\\datasets\\Dataset.csv")
df_copy = df.copy()


# ── General ──
# ------------------------------------------------------------------
# Simple integer time index reflecting row order (captures linear trend)
df_copy['time_idx'] = np.arange(len(df_copy))

# ------------------------------------------------------------------
# 3‑step rolling mean of precipitation (captures short‑term accumulation)
df_copy['rolling_mean_pr_3'] = df_copy['pr'].rolling(window=3, min_periods=1).mean()

# 7‑step rolling mean of precipitation (captures longer‑term accumulation)
df_copy['rolling_mean_pr_7'] = df_copy['pr'].rolling(window=7, min_periods=1).mean()

# ------------------------------------------------------------------
# 5‑step rolling sum of evaporation (short‑term loss integration)
df_copy['rolling_sum_evspsbl_5'] = df_copy['evspsbl'].rolling(window=5, min_periods=1).sum()

# ------------------------------------------------------------------
# First‑order difference of precipitation (detects abrupt changes)
df_copy['diff_pr_1'] = df_copy['pr'].diff().fillna(0)

# First‑order difference of evaporation (detects abrupt changes)
df_copy['diff_evspsbl_1'] = df_copy['evspsbl'].diff().fillna(0)

# ------------------------------------------------------------------
# Interaction: product of precipitation and evaporation
df_copy['pr_x_evspsbl'] = df_copy['pr'] * df_copy['evspsbl']

# Interaction: net water input (precipitation minus evaporation)
df_copy['pr_minus_evspsbl'] = df_copy['pr'] - df_copy['evspsbl']

# Interaction: runoff × precipitation (captures infiltration vs runoff trade‑off)
df_copy['mrros_x_pr'] = df_copy['mrros'] * df_copy['pr']

# Interaction: runoff × evaporation (both loss mechanisms active)
df_copy['mrros_x_evspsbl'] = df_copy['mrros'] * df_copy['evspsbl']

# ------------------------------------------------------------------
# Cumulative precipitation up to the current row (integrated input)
df_copy['cumulative_pr'] = df_copy['pr'].cumsum()

# Cumulative evaporation up to the current row (integrated loss)
df_copy['cumulative_evspsbl'] = df_copy['evspsbl'].cumsum()

# ------------------------------------------------------------------
# 5‑step rolling standard deviation of precipitation (recent variability)
df_copy['window_std_pr_5'] = df_copy['pr'].rolling(window=5, min_periods=1).std().fillna(0)

# 3‑step rolling mean of runoff (smooths noisy runoff signal)
df_copy['window_mean_mrros_3'] = df_copy['mrros'].rolling(window=3, min_periods=1).mean()

# ------------------------------------------------------------------
# Quantile‑based binning of precipitation into 4 categories (captures non‑linear response)
df_copy['pr_bin'] = pd.qcut(df_copy['pr'], q=4, labels=False, duplicates='drop')

# Quantile‑based binning of evaporation into 4 categories
df_copy['evspsbl_bin'] = pd.qcut(df_copy['evspsbl'], q=4, labels=False, duplicates='drop')

# Quantile‑based binning of runoff into 4 categories
df_copy['mrros_bin'] = pd.qcut(df_copy['mrros'], q=4, labels=False, duplicates='drop')

# ------------------------------------------------------------------
# Small constant to avoid division by zero or log of zero
_eps = 1e-8

# Ratio: precipitation over evaporation (net wetness indicator)
df_copy['pr_over_evspsbl'] = df_copy['pr'] / (df_copy['evspsbl'] + _eps)

# Ratio: runoff per unit precipitation (drainage efficiency)
df_copy['mrros_over_pr'] = df_copy['mrros'] / (df_copy['pr'] + _eps)

# Ratio: precipitation per unit runoff (infiltration efficiency)
df_copy['pr_over_mrros'] = df_copy['pr'] / (df_copy['mrros'] + _eps)

# Ratio: evaporation per unit runoff (evaporation dominance)
df_copy['evspsbl_over_mrros'] = df_copy['evspsbl'] / (df_copy['mrros'] + _eps)

# ------------------------------------------------------------------
# Log‑transformations (handle zeros with epsilon)
df_copy['log_pr'] = np.log(df_copy['pr'] + _eps)
df_copy['log_evspsbl'] = np.log(df_copy['evspsbl'] + _eps)
df_copy['log_mrros'] = np.log(df_copy['mrros'] + _eps)

# ------------------------------------------------------------------
# Classic water‑budget balance: precipitation – evaporation – runoff
df_copy['water_balance'] = df_copy['pr'] - df_copy['evspsbl'] - df_copy['mrros']

# Cumulative water balance (theoretical soil‑moisture gain trajectory)
df_copy['potential_soil_moisture_gain'] = df_copy['water_balance'].cumsum()

# Runoff coefficient: fraction of precipitation that becomes runoff
df_copy['runoff_coefficient'] = df_copy['mrros'] / (df_copy['pr'] + _eps)

# Evaporation efficiency: fraction of precipitation lost to evaporation
df_copy['evaporation_efficiency'] = df_copy['evspsbl'] / (df_copy['pr'] + _eps)

# ------------------------------------------------------------------
# Target‑derived residual feature (useful for stacked models)
df_copy['soil_moisture_deficit'] = df_copy['mrso'] - (df_copy['pr'] - df_copy['evspsbl'])

# ── Quick implementation tip ──
# small epsilon to avoid division‑by‑zero errors
eps = 1e-8

# Ratio of precipitation to evaporation (moisture availability indicator)
df_copy['pr_over_evspsbl'] = df_copy['pr'] / (df_copy['evspsbl'] + eps)

# Ratio of runoff to precipitation (runoff efficiency)
df_copy['mrros_over_pr'] = df_copy['mrros'] / (df_copy['pr'] + eps)

# Net water balance: precipitation minus evaporation minus runoff
df_copy['water_balance'] = df_copy['pr'] - df_copy['evspsbl'] - df_copy['mrros']

# Cumulative net water input over time
df_copy['cumulative_balance'] = df_copy['water_balance'].cumsum()

# 3‑day rolling mean of precipitation (short‑term smoothing)
df_copy['pr_rollmean_3'] = df_copy['pr'].rolling(3, min_periods=1).mean()

# 7‑day rolling sum of precipitation (weekly accumulation)
df_copy['pr_rollsum_7'] = df_copy['pr'].rolling(7, min_periods=1).sum()

# 5‑day rolling sum of evaporation (short‑term evapotranspiration total)
df_copy['evspsbl_rollsum_5'] = df_copy['evspsbl'].rolling(5, min_periods=1).sum()

# ── Auto-injected by agent: save engineered DataFrame ──
_new_cols = [c for c in df_copy.columns if c not in df.columns]
print()
print(f"✓ {len(_new_cols)} new features engineered:")
for _col in _new_cols:
    print(f"  • {_col:35s}  dtype={df_copy[_col].dtype}")
print()
print(f"Original shape : {df.shape}")
print(f"Engineered shape: {df_copy.shape}")
df_copy.to_csv("Dataset_engineered.csv", index=False)
print(f"Engineered dataset saved → Dataset_engineered.csv")
