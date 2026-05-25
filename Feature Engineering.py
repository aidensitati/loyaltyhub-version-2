# =========================================================
# ECA STAGE: FEATURE ENGINEERING EXECUTION
# =========================================================
# Purpose:
# This stage operationalizes the previously defined feature
# engineering plan into a fully constructed modeling-ready
# dataset.
#
# The goal is to convert raw churn dataset variables into
# validated, non-redundant, behaviorally meaningful features
# that maximize separability between churned and non-churned
# customers while strictly adhering to transformation
# constraints (bias correction, distribution correction,
# and structural extraction).
#
# This stage does not explore or hypothesize. It executes
# only validated transformations and produces a stable
# feature matrix for downstream modeling validation.
#
# Dataset/Input:
# C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub V2\LoyaltyHub Gold Silver.csv
#
# Output:
# Engineered churn-ready dataset (model matrix)
# Feature transformation log
# Feature mapping table
# Dropped feature registry
# =========================================================

# ---------------------------------------------------------
# STAGE OBJECTIVES
# ---------------------------------------------------------
# 1. Core Behavioral Rate Features
#    - Compute points_per_day
#    - Compute value_density
#    - Compute engagement_efficiency
#
# 2. Distribution and Skew Corrections
#    - Compute log_avg_txn
#    - Compute log_points
#    - Apply mild clipping if required
#
# 3. Temporal Engagement Signals
#    - Compute recency_score
#    - Compute logins_per_day
#    - Compute time_per_day
#
# 4. Encode Structural Features and Assemble Final Output
#    - membership_category -> is_gold
#    - Encode region_category
#    - Exclude complaint_status
#    - Remove raw categorical columns
#
# 5. Interaction Feature Construction
#    - region × avg_transaction_value
#    - points_per_day × time_per_day
#
# 6. Feature Assembly + Logging
#    - Final dataset
#    - Transformation log
#    - Mapping table
#    - Dropped feature registry

# ---------------------------------------------------------
# LOAD DEPENDENCIES
# ---------------------------------------------------------
import os
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub V2\LoyaltyHub Gold Silver.csv"

df = pd.read_csv(file_path)
engineered_df = df.copy()

# Standard smoothing constant
epsilon = 1e-6

# Transformation artifacts
transformation_log = []
feature_mapping = []
dropped_features = []

# Ensure numeric conversions where needed
numeric_columns = [
    "points_in_wallet",
    "avg_transaction_value",
    "avg_time_spent",
    "avg_frequency_login_days",
    "days_since_last_login"
]

for col in numeric_columns:
    engineered_df[col] = pd.to_numeric(engineered_df[col], errors="coerce")

# Fill numeric nulls conservatively
engineered_df[numeric_columns] = engineered_df[numeric_columns].fillna(0)

# Derive tenure_days from joining_date if available
if "joining_date" in engineered_df.columns:
    engineered_df["joining_date"] = pd.to_datetime(
        engineered_df["joining_date"],
        errors="coerce"
    )
    reference_date = engineered_df["joining_date"].max()
    engineered_df["tenure_days"] = (
        reference_date - engineered_df["joining_date"]
    ).dt.days
    engineered_df["tenure_days"] = engineered_df["tenure_days"].fillna(1)
else:
    engineered_df["tenure_days"] = 1

engineered_df["tenure_days"] = engineered_df["tenure_days"].clip(lower=1)


# =========================================================
# OBJECTIVE 1: CORE BEHAVIORAL RATE FEATURES
# =========================================================

print("Executing Objective 1: Core Behavioral Rate Features...")

engineered_df["points_per_day"] = (
    engineered_df["points_in_wallet"] /
    (engineered_df["tenure_days"] + epsilon)
)

engineered_df["value_density"] = (
    engineered_df["avg_transaction_value"] /
    (engineered_df["avg_time_spent"] + epsilon)
)

engineered_df["engagement_efficiency"] = (
    engineered_df["points_in_wallet"] /
    (engineered_df["avg_time_spent"] + epsilon)
)

transformation_log.extend([
    {
        "original_feature": "points_in_wallet",
        "engineered_feature": "points_per_day",
        "transformation": "rate_conversion",
        "trigger": "TENURE_CORRELATED"
    },
    {
        "original_feature": "avg_transaction_value + avg_time_spent",
        "engineered_feature": "value_density",
        "transformation": "ratio_metric",
        "trigger": "Behavioral Intensity"
    },
    {
        "original_feature": "points_in_wallet + avg_time_spent",
        "engineered_feature": "engagement_efficiency",
        "transformation": "ratio_metric",
        "trigger": "Behavioral Efficiency"
    }
])


# =========================================================
# OBJECTIVE 2: DISTRIBUTION AND SKEW CORRECTIONS
# =========================================================

print("Executing Objective 2: Distribution and Skew Corrections...")

# Mild clipping at 99th percentile
txn_cap = engineered_df["avg_transaction_value"].quantile(0.99)
points_cap = engineered_df["points_per_day"].quantile(0.99)

engineered_df["avg_transaction_value_clipped"] = (
    engineered_df["avg_transaction_value"].clip(upper=txn_cap)
)

engineered_df["points_per_day_clipped"] = (
    engineered_df["points_per_day"].clip(upper=points_cap)
)

engineered_df["log_avg_txn"] = np.log1p(
    engineered_df["avg_transaction_value_clipped"]
)

engineered_df["log_points"] = np.log1p(
    engineered_df["points_per_day_clipped"]
)

transformation_log.extend([
    {
        "original_feature": "avg_transaction_value",
        "engineered_feature": "log_avg_txn",
        "transformation": "clip + log1p",
        "trigger": "HIGH_SKEW"
    },
    {
        "original_feature": "points_per_day",
        "engineered_feature": "log_points",
        "transformation": "clip + log1p",
        "trigger": "HIGH_SKEW"
    }
])


# =========================================================
# OBJECTIVE 3: TEMPORAL ENGAGEMENT SIGNALS
# =========================================================

print("Executing Objective 3: Temporal Engagement Signals...")

engineered_df["days_since_last_login"] = (
    engineered_df["days_since_last_login"].clip(lower=0)
)

engineered_df["recency_score"] = (
    1 / (1 + engineered_df["days_since_last_login"])
)

engineered_df["avg_frequency_login_days"] = (
    engineered_df["avg_frequency_login_days"].replace(0, np.nan)
)

engineered_df["avg_frequency_login_days"] = (
    engineered_df["avg_frequency_login_days"].fillna(1)
)

engineered_df["logins_per_day"] = (
    1 / engineered_df["avg_frequency_login_days"]
)

engineered_df["time_per_day"] = (
    engineered_df["avg_time_spent"] /
    (engineered_df["tenure_days"] + epsilon)
)

transformation_log.extend([
    {
        "original_feature": "days_since_last_login",
        "engineered_feature": "recency_score",
        "transformation": "inverse_decay",
        "trigger": "Recency Signal"
    },
    {
        "original_feature": "avg_frequency_login_days",
        "engineered_feature": "logins_per_day",
        "transformation": "inverse_frequency",
        "trigger": "Activity Rate"
    },
    {
        "original_feature": "avg_time_spent",
        "engineered_feature": "time_per_day",
        "transformation": "rate_conversion",
        "trigger": "TENURE_CORRELATED"
    }
])

# =========================================================
# OBJECTIVE 4: ENCODE STRUCTURAL FEATURES AND ASSEMBLE FINAL OUTPUT
# =========================================================

print("Executing Objective 4: Encode Structural Features and Assemble Final Output...")

engineered_df["membership_category"] = (
    engineered_df["membership_category"]
    .astype(str)
    .str.strip()
    .str.lower()
)

region_dummies = pd.get_dummies(
    engineered_df["region_category"],
    prefix="region",
    dtype=int
)

engineered_df = pd.concat(
    [engineered_df, region_dummies],
    axis=1
)

# Remove excluded raw fields
columns_to_drop = [
    "complaint_status",
    "region_category",
    "feedback"
]

for col in columns_to_drop:
    if col in engineered_df.columns:
        dropped_features.append(col)

engineered_df = engineered_df.drop(
    columns=columns_to_drop,
    errors="ignore"
)

# =========================================================
# OBJECTIVE 5: INTERACTION FEATURE CONSTRUCTION
# =========================================================

print("Executing Objective 5: Interaction Feature Construction...")

# points_per_day × time_per_day
engineered_df["points_time_interaction"] = (
    engineered_df["points_per_day"] *
    engineered_df["time_per_day"]
)

# region × avg_transaction_value
region_cols = [
    col for col in engineered_df.columns
    if col.startswith("region_")
]

for col in region_cols:
    new_col = f"{col}_x_value"
    engineered_df[new_col] = (
        engineered_df[col] *
        engineered_df["avg_transaction_value"]
    )

transformation_log.append({
    "original_feature": "points_per_day + time_per_day",
    "engineered_feature": "points_time_interaction",
    "transformation": "interaction",
    "trigger": "Signal Amplification"
})


# =========================================================
# OBJECTIVE 6: FEATURE ASSEMBLY + LOGGING
# =========================================================

print("Executing Objective 6: Feature Assembly + Logging...")

for col in engineered_df.columns:
    feature_mapping.append({
        "final_feature": col,
        "source": "engineered_or_retained"
    })

# Remove helper clipped columns
helper_drop = [
    "avg_transaction_value_clipped",
    "points_per_day_clipped"
]

for col in helper_drop:
    if col in engineered_df.columns:
        engineered_df.drop(columns=col, inplace=True)
        dropped_features.append(col)


# ---------------------------------------------------------
# STAGE OUTPUT
# ---------------------------------------------------------
output_dir = os.path.dirname(file_path)

dataset_output = os.path.join(
    output_dir,
    "Engineered_Churn_Dataset.csv"
)

log_output = os.path.join(
    output_dir,
    "Feature_Transformation_Log.csv"
)

mapping_output = os.path.join(
    output_dir,
    "Feature_Mapping_Table.csv"
)

dropped_output = os.path.join(
    output_dir,
    "Dropped_Feature_Registry.csv"
)

engineered_df.to_csv(dataset_output, index=False)
pd.DataFrame(transformation_log).to_csv(log_output, index=False)
pd.DataFrame(feature_mapping).to_csv(mapping_output, index=False)
pd.DataFrame({"dropped_feature": dropped_features}).to_csv(
    dropped_output,
    index=False
)

print("Stage outputs successfully saved.")


# =========================================================
# FEATURE ENGINEERING EXECUTION STAGE COMPLETE
# =========================================================

print("Feature Engineering Execution stage execution complete.")