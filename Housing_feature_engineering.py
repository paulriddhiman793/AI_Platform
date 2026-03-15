"""Feature engineering code for D:\\Downloads\\Agents\\AI_Platform\\platform_projects\\chat_20260315t202954_20260316_015954\\shared\\datasets\\Housing.csv — verified by execution agent"""

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"D:\\Downloads\\Agents\\AI_Platform\\platform_projects\\chat_20260315t202954_20260316_015954\\shared\\datasets\\Housing.csv")
df_copy = df.copy()


# ── General ──
# Clean binary string columns: lower‑case and strip whitespace
binary_cols = ['mainroad', 'guestroom', 'basement',
               'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    if col in df_copy.columns:
        df_copy[col] = df_copy[col].astype(str).str.lower().str.strip()

# Interaction Features
df_copy['total_rooms'] = df_copy['bedrooms'] + df_copy['bathrooms']

df_copy['area_per_bedroom'] = df_copy['area'] / df_copy['bedrooms'].replace(0, np.nan)
df_copy['area_per_bathroom'] = df_copy['area'] / df_copy['bathrooms'].replace(0, np.nan)

df_copy['bedrooms_x_bathrooms'] = df_copy['bedrooms'] * df_copy['bathrooms']

if 'parking' in df_copy.columns:
    df_copy['stories_x_parking'] = df_copy['stories'] * df_copy['parking']
else:
    df_copy['stories_x_parking'] = np.nan

if set(['mainroad', 'prefarea']).issubset(df_copy.columns):
    df_copy['mainroad_x_prefarea'] = (
        (df_copy['mainroad'] == 'yes').astype(int) *
        (df_copy['prefarea'] == 'yes').astype(int)
    )
else:
    df_copy['mainroad_x_prefarea'] = np.nan

# Aggregation Features
amenity_flags = ['mainroad', 'guestroom', 'basement',
                 'hotwaterheating', 'airconditioning', 'prefarea']
existing_amenities = [col for col in amenity_flags if col in df_copy.columns]
df_copy['amenities_count'] = sum(df_copy[col] == 'yes' for col in existing_amenities).astype(int)

# weighted luxury score (custom weights)
furnish_weights = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
if 'furnishingstatus' in df_copy.columns:
    df_copy['furnishing_numeric'] = df_copy['furnishingstatus'].map(furnish_weights).fillna(0)
else:
    df_copy['furnishing_numeric'] = 0

df_copy['luxury_score'] = (
    (df_copy['mainroad'] == 'yes').astype(int) * 1 +
    (df_copy['airconditioning'] == 'yes').astype(int) * 2 +
    (df_copy['basement'] == 'yes').astype(int) * 2 +
    df_copy['furnishing_numeric'] * 3
)

df_copy['room_density'] = df_copy['total_rooms'] / df_copy['area'].replace(0, np.nan)

# Encoding Features
for col in existing_amenities:
    df_copy[f'{col}_bin'] = (df_copy[col] == 'yes').astype(int)

if 'furnishingstatus' in df_copy.columns:
    df_copy['furnishing_ordinal'] = df_copy['furnishingstatus'].map({
        'unfurnished': 0,
        'semi-furnished': 1,
        'furnished': 2
    }).astype(int)
else:
    df_copy['furnishing_ordinal'] = 0

# target‑mean encoding for low‑cardinality flags (mainroad & prefarea)
try:
    if 'price' in df_copy.columns and 'mainroad' in df_copy.columns:
        mainroad_means = df_copy.groupby('mainroad')['price'].mean()
        df_copy['mainroad_target_mean'] = df_copy['mainroad'].map(mainroad_means)
    else:
        df_copy['mainroad_target_mean'] = np.nan

    if 'price' in df_copy.columns and 'prefarea' in df_copy.columns:
        prefarea_means = df_copy.groupby('prefarea')['price'].mean()
        df_copy['prefarea_target_mean'] = df_copy['prefarea'].map(prefarea_means)
    else:
        df_copy['prefarea_target_mean'] = np.nan
except Exception:
    df_copy['mainroad_target_mean'] = np.nan
    df_copy['prefarea_target_mean'] = np.nan

# Ratio / Normalisation Features
if 'price' in df_copy.columns:
    df_copy['price_per_area'] = df_copy['price'] / df_copy['area'].replace(0, np.nan)
else:
    df_copy['price_per_area'] = np.nan

df_copy['area_scaled_log'] = np.log1p(df_copy['area'])

if 'parking' in df_copy.columns:
    df_copy['parking_ratio'] = df_copy['parking'] / df_copy['stories'].replace(0, np.nan)
else:
    df_copy['parking_ratio'] = np.nan

# Domain‑Specific Features
if 'parking' in df_copy.columns:
    df_copy['has_garage'] = (df_copy['parking'] >= 1).astype(int)
else:
    df_copy['has_garage'] = 0

df_copy['is_multi_story'] = (df_copy['stories'] > 1).astype(int)

df_copy['is_spacious'] = (df_copy['area'] > df_copy['area'].quantile(0.75)).astype(int)

df_copy['has_full_basement'] = (
    (df_copy['basement'] == 'yes') &
    (df_copy['stories'] == 1)
).astype(int)

# ── Quick implementation checklist ──
# ------------------------------------------------------------------
# 1. Clean binary string columns: lower‑case, strip, map "yes"/"no" → 1/0
# ------------------------------------------------------------------
binary_cols = ['mainroad', 'guestroom', 'basement',
               'hotwaterheating', 'airconditioning', 'prefarea']

for col in binary_cols:
    if col in df_copy.columns:
        # ensure strings are lower‑cased and whitespace‑free
        df_copy[col] = df_copy[col].astype(str).str.lower().str.strip()
        # map yes → 1, no → 0 (any unexpected value becomes NaN, then fill with 0)
        df_copy[col] = df_copy[col].map({'yes': 1, 'no': 0}).fillna(0).astype('int8')
    else:
        # create missing binary column with default 0
        df_copy[col] = 0

# ------------------------------------------------------------------
# 2. Create interaction / aggregation features
# ------------------------------------------------------------------
# total number of rooms (bedrooms + bathrooms)
df_copy['total_rooms'] = df_copy['bedrooms'] + df_copy['bathrooms']

# area per bedroom (size efficiency per bedroom)
df_copy['area_per_bedroom'] = df_copy['area'] / df_copy['bedrooms']

# area per bathroom (size efficiency per bathroom)
df_copy['area_per_bathroom'] = df_copy['area'] / df_copy['bathrooms']

# rooms per story (how many rooms are packed per floor)
df_copy['rooms_per_story'] = df_copy['total_rooms'] / df_copy['stories']

# parking spots per total rooms (parking adequacy)
if 'parking' in df_copy.columns:
    df_copy['parking_per_room'] = df_copy['parking'] / (df_copy['total_rooms'] + 1)
else:
    df_copy['parking_per_room'] = 0.0

# count of positive amenities (binary features summed)
amenity_cols = ['mainroad', 'guestroom', 'basement',
                'hotwaterheating', 'airconditioning', 'prefarea']
df_copy['amenities_count'] = df_copy[amenity_cols].sum(axis=1)

# ------------------------------------------------------------------
# 3. Log‑transform size‑related numeric columns
# ------------------------------------------------------------------
# log‑scaled area (helps linear models handle skewness)
df_copy['log_area'] = np.log1p(df_copy['area'])

# optional: log‑scaled target – useful for downstream modeling pipelines
df_copy['log_price'] = np.log1p(df_copy['price'])

# ------------------------------------------------------------------
# 4. One‑hot encode remaining categorical variable(s)
# ------------------------------------------------------------------
# furnishingstatus has three categories → create dummy columns
if 'furnishingstatus' in df_copy.columns:
    furnish_dummies = pd.get_dummies(df_copy['furnishingstatus'],
                                     prefix='furnish',
                                     drop_first=True)
    df_copy = pd.concat([df_copy, furnish_dummies], axis=1)

# ------------------------------------------------------------------
# 5. Scale numeric features (standard‑score scaling)
# ------------------------------------------------------------------
# list of numeric columns to scale (exclude target and already‑scaled ones)
numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories',
                'parking', 'total_rooms', 'area_per_bedroom',
                'area_per_bathroom', 'rooms_per_story',
                'parking_per_room', 'amenities_count', 'log_area']

for col in numeric_cols:
    if col in df_copy.columns:
        mean_val = df_copy[col].mean()
        std_val = df_copy[col].std(ddof=0)
        # avoid division by zero
        if std_val != 0:
            df_copy[f'{col}_scaled'] = (df_copy[col] - mean_val) / std_val
        else:
            df_copy[f'{col}_scaled'] = 0.0

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
