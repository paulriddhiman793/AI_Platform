"""Feature engineering code for D:\\Downloads\\Agents\\AI_Platform\\platform_projects\\chat_20260316t004703_20260316_061703\\shared\\datasets\\Housing.csv — verified by execution agent"""

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"D:\\Downloads\\Agents\\AI_Platform\\platform_projects\\chat_20260316t004703_20260316_061703\\shared\\datasets\\Housing.csv")
df_copy = df.copy()


# ── General ──
# Log‑transform the target price to reduce skewness
df_copy['log_price'] = np.log1p(df_copy['price'])

# Log‑transform the area (sqft) to bring it onto a comparable scale with price
df_copy['log_area'] = np.log1p(df_copy['area'])

# Detect impossible combinations: 0 stories but at least one bedroom
df_copy['invalid_story_bedrooms'] = ((df_copy['stories'] == 0) & (df_copy['bedrooms'] > 0)).astype(int)

# Detect impossible combinations: 0 stories but any parking spaces
df_copy['invalid_story_parking'] = ((df_copy['stories'] == 0) & (df_copy['parking'] > 0)).astype(int)

# ── 📅 Datetime Features ──
try:
    if 'year_built' in df_copy.columns:
        # Use pandas Timestamp to avoid missing datetime import
        current_year = pd.Timestamp.now().year
        # Compute age and ensure non‑negative values
        df_copy['property_age'] = (current_year - df_copy['year_built']).clip(lower=0)
    else:
        # Column missing → fill with NaN so downstream logic can handle it
        df_copy['property_age'] = np.nan
except Exception:
    df_copy['property_age'] = np.nan


try:
    def _age_to_category(age):
        """Map numeric age to a categorical bucket."""
        if pd.isna(age):
            return np.nan
        if age <= 5:
            return 'new'          # ≤ 5 years old
        elif age <= 20:
            return 'mid-age'      # 6‑20 years old
        else:
            return 'old'          # > 20 years old

    df_copy['age_category'] = df_copy['property_age'].apply(_age_to_category)
except Exception:
    df_copy['age_category'] = np.nan

# ── 🤝 Interaction Features ──
# Ratio of bedrooms to bathrooms (high values may indicate under‑serviced units)
df_copy['bed_bath_ratio'] = df_copy['bedrooms'] / df_copy['bathrooms']

# Average area per bedroom (larger per‑bedroom area often drives up price)
df_copy['area_per_bedroom'] = df_copy['area'] / df_copy['bedrooms']

# Average area per bathroom (captures space relative to costlier bathroom count)
df_copy['area_per_bathroom'] = df_copy['area'] / df_copy['bathrooms']

# Combined count of bedrooms, bathrooms, and stories (overall livable volume)
df_copy['total_rooms'] = df_copy['bedrooms'] + df_copy['bathrooms'] + df_copy['stories']

# ── 📊 Aggregation Features ──
# 📊 area_per_story: Normalises total footprint by vertical density (area divided by number of stories)
df_copy['area_per_story'] = df_copy['area'] / df_copy['stories']

# 📊 parking_per_story: Estimates parking provision per floor (parking spaces divided by number of stories) – only if the column exists
if 'parking' in df_copy.columns:
    df_copy['parking_per_story'] = df_copy['parking'] / df_copy['stories']

# 📊 total_amenities: Aggregates binary amenity flags (yes=1, no=0) into a single richness score
_binary_possible = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'prefarea']
_binary_cols = [col for col in _binary_possible if col in df_copy.columns]

if _binary_cols:
    df_copy['total_amenities'] = (
        df_copy[_binary_cols]
        .apply(lambda s: s.str.strip().str.lower().eq('yes').astype(int))
        .sum(axis=1)
    )
else:
    df_copy['total_amenities'] = 0

# ── 🔤 Encoding Features ──
# Map binary “yes”/“no” columns to integer flags (1/0)
binary_map = {"yes": 1, "no": 0}
binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_cols:
    if col in df_copy.columns:
        df_copy[f"{col}_int"] = df_copy[col].map(binary_map)

# One‑hot encode the categorical “furnishingstatus” column (creates binary columns)
if "furnishingstatus" in df_copy.columns:
    _furnish_dummies = pd.get_dummies(df_copy["furnishingstatus"], prefix="furnish_onehot")
    df_copy = pd.concat([df_copy, _furnish_dummies], axis=1)

# Bucket the ordinal “total_amenities” into low (0‑2), medium (3‑4), high (5‑6) groups
def _amenity_bucket(val):
    if val <= 2:
        return 0          # low
    elif val <= 4:
        return 1          # medium
    else:
        return 2          # high

if "total_amenities" in df_copy.columns:
    df_copy["high_amenity_score"] = df_copy["total_amenities"].apply(_amenity_bucket)

# ── ⚖️ Ratio / Normalisation Features ──
df_copy['log_price'] = np.log(df_copy['price'])
df_copy['log_area'] = np.log(df_copy['area'])
df_copy['price_per_sqm'] = df_copy['price'] / (df_copy['area'] + 1)

# ── 🏗️ Domain‑Specific Features ──
# ------------------------------------------------------------------
# Encode all Yes/No string columns to binary 0‑1 values (in‑place)
# ------------------------------------------------------------------
yes_no_cols = ["mainroad", "guestroom", "basement"]
for col in yes_no_cols:
    if col in df_copy.columns:
        df_copy[col] = df_copy[col].map({"yes": 1, "no": 0}).fillna(0).astype(int)

# ------------------------------------------------------------------
# has_basement_and_guestroom: flag = 1 if both basement and guestroom are “yes”
# ------------------------------------------------------------------
if {"basement", "guestroom"}.issubset(df_copy.columns):
    df_copy["has_basement_and_guestroom"] = (
        (df_copy["basement"] == 1) & (df_copy["guestroom"] == 1)
    ).astype(int)

# ------------------------------------------------------------------
# luxury_indicator: flag = 1 when area > 9000, furnishingstatus == 'furnished',
# and the property is on the main road (mainroad == 1)
# ------------------------------------------------------------------
if {"area", "mainroad", "furnishingstatus"}.issubset(df_copy.columns):
    df_copy["luxury_indicator"] = (
        (df_copy["area"] > 9000) &
        (df_copy["furnishingstatus"] == "furnished") &
        (df_copy["mainroad"] == 1)
    ).astype(int)

# ------------------------------------------------------------------
# story_density: ratio of number of stories to total area (stories / area)
# ------------------------------------------------------------------
if {"stories", "area"}.issubset(df_copy.columns):
    df_copy["story_density"] = df_copy["stories"] / (df_copy["area"] + 1)

# ------------------------------------------------------------------
# amenity_score_weighted: weighted sum of binary amenity flags.
# ------------------------------------------------------------------
amenity_weights = {
    "airconditioning": 2.0,
    "hotwaterheating": 2.0,
    "basement": 1.5,
    "guestroom": 1.5,
    "mainroad": 1.0,
    "prefarea": 1.0
}
# Ensure all amenity columns are numeric (0/1)
for col in amenity_weights.keys():
    if col in df_copy.columns and df_copy[col].dtype == object:
        df_copy[col] = df_copy[col].map({"yes": 1, "no": 0}).fillna(0).astype(int)

valid_weights = {
    col: w for col, w in amenity_weights.items() if col in df_copy.columns
}
if valid_weights:
    df_copy["amenity_score_weighted"] = sum(
        df_copy[col] * weight for col, weight in valid_weights.items()
    )
else:
    df_copy["amenity_score_weighted"] = 0.0

# ── Auto-injected by agent: save engineered DataFrame ──
_new_cols = [c for c in df_copy.columns if c not in df.columns]
print()
print(f"✓ {len(_new_cols)} new features engineered:")
for _col in _new_cols:
    print(f"  • {_col:35s}  dtype={df_copy[_col].dtype}")
print()
print(f"Original shape : {df.shape}")
print(f"Engineered shape: {df_copy.shape}")
df_copy.to_csv("Housing_engineered.csv", index=False)
print(f"Engineered dataset saved → Housing_engineered.csv")
