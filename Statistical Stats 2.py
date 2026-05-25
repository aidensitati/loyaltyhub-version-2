# =========================================================
# ECA STAGE: Statistical Stats 2
# =========================================================
# Purpose:
# This stage rigorously validates whether engineered features
# (log transforms, interaction terms, and structural encodings)
# have successfully improved churn signal quality without
# introducing statistical distortion.
#
# Dataset/Input:
# LoyaltyHub engineered churn dataset (post feature engineering stage)
# C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub V2\Engineered_Churn_Dataset.csv
#
# Output:
# Statistical Validation Report for Engineered Feature Set
# =========================================================

# ---------------------------------------------------------
# STAGE OBJECTIVES
# ---------------------------------------------------------
# 1. Transformation Integrity Audit
# 2. Pre vs Post Distribution Pathology Assessment
# 3. Churn Signal Strength Validation
# 4. Interaction Feature Signal & Redundancy Evaluation
# 5. Membership Structure Impact Validation

# ---------------------------------------------------------
# LOAD DEPENDENCIES
# ---------------------------------------------------------
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, pointbiserialr

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub V2\Engineered_Churn_Dataset.csv"
df = pd.read_csv(file_path)

output_dir = os.path.dirname(file_path)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def detect_target_column(dataframe):
    possible_targets = [
        "churn", "is_churned", "churn_flag",
        "churn_status", "target", "label"
    ]
    lower_map = {col.lower(): col for col in dataframe.columns}

    for col in possible_targets:
        if col in lower_map:
            return lower_map[col]

    binary_candidates = []
    for col in dataframe.columns:
        unique_vals = dataframe[col].dropna().unique()
        if len(unique_vals) == 2:
            binary_candidates.append(col)

    if binary_candidates:
        return binary_candidates[0]

    raise ValueError("No churn target column detected.")

def cohen_d(group1, group2):
    group1 = pd.Series(group1).dropna()
    group2 = pd.Series(group2).dropna()

    if len(group1) < 2 or len(group2) < 2:
        return np.nan

    mean1 = group1.mean()
    mean2 = group2.mean()

    var1 = group1.var(ddof=1)
    var2 = group2.var(ddof=1)

    pooled_std = np.sqrt(((len(group1) - 1) * var1 + (len(group2) - 1) * var2) /
                         (len(group1) + len(group2) - 2))

    if pooled_std == 0:
        return np.nan

    return (mean1 - mean2) / pooled_std

def safe_corr(series_x, series_y):
    valid = pd.concat([series_x, series_y], axis=1).dropna()
    if valid.shape[0] < 3:
        return np.nan

    if valid.iloc[:, 0].nunique() <= 1:
        return np.nan

    if valid.iloc[:, 1].nunique() <= 1:
        return np.nan

    return valid.iloc[:, 0].corr(valid.iloc[:, 1])

def coefficient_variation(series):
    series = series.dropna()
    mean_val = series.mean()
    std_val = series.std()

    if mean_val == 0:
        return np.nan

    return std_val / mean_val

# ---------------------------------------------------------
# FEATURE DISCOVERY
# ---------------------------------------------------------
target_col = detect_target_column(df)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

log_features = [col for col in df.columns if col.lower().startswith("log_")]

interaction_features = []
for col in df.columns:
    lower_col = col.lower()
    if (
        ("interaction" in lower_col) or
        ("_x_" in lower_col) or
        ("*_" in lower_col) or
        ("_mul_" in lower_col)
    ):
        interaction_features.append(col)

membership_col = None
for col in df.columns:
    if "membership" in col.lower():
        membership_col = col
        break

# ---------------------------------------------------------
# OBJECTIVE 1: Transformation Integrity Audit
# =========================================================
print("Executing Objective 1: Transformation Integrity Audit...")

integrity_results = []

derived_features = list(set(log_features + interaction_features))

for feature in derived_features:
    if feature not in df.columns:
        continue

    series = df[feature]

    nan_count = series.isna().sum()
    inf_count = np.isinf(series).sum() if pd.api.types.is_numeric_dtype(series) else 0

    anomaly = "none"
    status = "PASS"

    if nan_count > 0:
        anomaly = "NaN"
        status = "FAIL"

    if inf_count > 0:
        anomaly = "Inf"
        status = "FAIL"

    if feature in log_features:
        if (series.dropna() < 0).any():
            anomaly = "distortion"
            status = "FAIL"

    integrity_results.append({
        "feature_name": feature,
        "validity_status": status,
        "anomaly_type": anomaly
    })

integrity_df = pd.DataFrame(integrity_results)

# ---------------------------------------------------------
# OBJECTIVE 2: Pre vs Post Distribution Pathology Assessment
# =========================================================
print("Executing Objective 2: Pre vs Post Distribution Pathology Assessment...")

pathology_results = []

for feature in log_features:
    parent_feature = feature.replace("log_", "")

    if parent_feature in df.columns and feature in df.columns:
        pre = df[parent_feature].dropna()
        post = df[feature].dropna()

        if len(pre) > 3 and len(post) > 3:
            pre_skew = skew(pre)
            post_skew = skew(post)

            pre_std = pre.std()
            post_std = post.std()

            skew_delta = abs(post_skew) - abs(pre_skew)
            variance_delta = post_std - pre_std

            if skew_delta < 0:
                correction = "improved"
            elif skew_delta > 0:
                correction = "worsened"
            else:
                correction = "unchanged"

            pathology_results.append({
                "feature_pair": f"{parent_feature} -> {feature}",
                "skew_delta": skew_delta,
                "variance_delta": variance_delta,
                "correction_status": correction
            })

pathology_df = pd.DataFrame(pathology_results)

# ---------------------------------------------------------
# OBJECTIVE 3: Churn Signal Strength Validation
# =========================================================
print("Executing Objective 3: Churn Signal Strength Validation...")

signal_results = []

for feature in numeric_cols:
    series = df[feature]

    churn_group = df[df[target_col] == 1][feature]
    retain_group = df[df[target_col] == 0][feature]

    effect = cohen_d(churn_group, retain_group)
    corr_val = safe_corr(series, df[target_col])

    if pd.isna(effect):
        net_gain = np.nan
    else:
        net_gain = abs(effect) + (abs(corr_val) if pd.notna(corr_val) else 0)

    signal_results.append({
        "feature": feature,
        "effect_size": effect,
        "churn_correlation": corr_val,
        "net_signal_gain": net_gain
    })

signal_df = pd.DataFrame(signal_results)
signal_df = signal_df.sort_values(by="net_signal_gain", ascending=False)

# ---------------------------------------------------------
# OBJECTIVE 4: Interaction Feature Signal & Redundancy Evaluation
# =========================================================
print("Executing Objective 4: Interaction Feature Signal & Redundancy Evaluation...")

interaction_results = []

for feature in interaction_features:
    if feature not in df.columns:
        continue

    feature_series = df[feature]
    churn_corr = safe_corr(feature_series, df[target_col])

    parent_candidates = []
    split_tokens = re.split(r"_x_|_mul_|interaction", feature.lower())

    for token in split_tokens:
        token = token.strip("_ ")
        for col in numeric_cols:
            if token and token in col.lower() and col != feature:
                parent_candidates.append(col)

    parent_candidates = list(set(parent_candidates))

    parent_corrs = []
    for parent in parent_candidates:
        corr_parent = safe_corr(feature_series, df[parent])
        if pd.notna(corr_parent):
            parent_corrs.append(abs(corr_parent))

    redundancy_score = max(parent_corrs) if parent_corrs else np.nan

    strongest_parent_signal = 0
    for parent in parent_candidates:
        parent_to_target = safe_corr(df[parent], df[target_col])
        if pd.notna(parent_to_target):
            strongest_parent_signal = max(strongest_parent_signal, abs(parent_to_target))

    marginal_gain = abs(churn_corr) - strongest_parent_signal if pd.notna(churn_corr) else np.nan

    if pd.notna(marginal_gain):
        if marginal_gain > 0.02 and (pd.isna(redundancy_score) or redundancy_score < 0.95):
            classification = "useful"
        elif pd.notna(redundancy_score) and redundancy_score >= 0.95:
            classification = "redundant"
        else:
            classification = "harmful"
    else:
        classification = "harmful"

    interaction_results.append({
        "interaction_feature": feature,
        "redundancy_score": redundancy_score,
        "marginal_gain": marginal_gain,
        "classification": classification
    })

interaction_df = pd.DataFrame(interaction_results)

# ---------------------------------------------------------
# OBJECTIVE 5: Membership Structure Impact Validation
# =========================================================
print("Executing Objective 5: Membership Structure Impact Validation...")

membership_results = []

if membership_col is not None:
    membership_values = df[membership_col].dropna().unique()

    for segment in membership_values:
        segment_df = df[df[membership_col] == segment]

        churn_rate = segment_df[target_col].mean()

        segment_effects = []
        for feature in log_features:
            if feature in segment_df.columns:
                churn_group = segment_df[segment_df[target_col] == 1][feature]
                retain_group = segment_df[segment_df[target_col] == 0][feature]
                d_val = cohen_d(churn_group, retain_group)

                if pd.notna(d_val):
                    segment_effects.append(abs(d_val))

        separation_strength = np.mean(segment_effects) if segment_effects else np.nan

        if pd.notna(separation_strength):
            if separation_strength >= 0.50:
                role = "strong structural stratifier"
            elif separation_strength >= 0.20:
                role = "weak/noisy segmentation variable"
            else:
                role = "misleading proxy"
        else:
            role = "misleading proxy"

        membership_results.append({
            "segment": segment,
            "churn_rate": churn_rate,
            "separation_strength": separation_strength,
            "structural_role": role
        })

membership_df = pd.DataFrame(membership_results)

# ---------------------------------------------------------
# STAGE OUTPUT
# ---------------------------------------------------------
integrity_df.to_csv(os.path.join(output_dir, "Feature_Integrity_Table.csv"), index=False)
pathology_df.to_csv(os.path.join(output_dir, "Pathology_Correction_Summary.csv"), index=False)
signal_df.to_csv(os.path.join(output_dir, "Churn_Signal_Table.csv"), index=False)
interaction_df.to_csv(os.path.join(output_dir, "Interaction_Evaluation_Table.csv"), index=False)
membership_df.to_csv(os.path.join(output_dir, "Membership_Impact_Summary.csv"), index=False)

print("Stage outputs successfully saved.")

# =========================================================
# Statistical Stats 2 STAGE COMPLETE
# =========================================================
print("Statistical Stats 2 stage execution complete.")