# =========================================================  
# ECA STAGE: Statistical Stats (Dataset-Adaptive Diagnostic Layer)  
# =========================================================  
# Purpose:  
# To perform a full statistical audit of all features in the loyalty hub dataset, 
# translating raw distributions into structured, bracket-aware behavioral insights.  
#  
# Dataset/Input:  
# LoyaltyHub Updated.csv  
#  
# Output:  
# Feature-level statistical diagnostics with bracket behavior and engineering recommendations  
# =========================================================


# ---------------------------------------------------------  
# STAGE OBJECTIVES  
# ---------------------------------------------------------
# 1. Classify all features into precise analytical types and assign them to risk-aware processing groups.
# 2. Compute full statistical profiles for each feature, incorporating exposure-aware and skewness-aware diagnostics.
# 3. Construct bracketed representations (Low / Medium / High) using adaptive binning aligned with distribution shape.
# 4. Diagnose structural behavior and churn interaction patterns, then assign feature engineering transformations.


# ---------------------------------------------------------  
# LOAD DEPENDENCIES  
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import KBinsDiscretizer


# ---------------------------------------------------------  
# LOAD DATA  
# ---------------------------------------------------------
file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub V2\LoyaltyHub Updated.csv"
df = pd.read_csv(file_path)

# Ensure churn is numeric
df['churn'] = df['churn'].astype(int)


# =========================================================  
# OBJECTIVE 1: Feature Classification & Risk Tagging  
# =========================================================  

print("Executing Objective 1: Feature Classification & Risk Tagging...")

# Feature groups
structural_features = ['age', 'gender', 'region_category', 'membership_category']
exposure_features = ['avg_time_spent', 'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet']
epistemic_features = ['feedback', 'complaint_status']

# Detect numerical vs categorical
feature_types = {}
for col in df.columns:
    if col == 'churn':
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        feature_types[col] = 'Numerical'
    else:
        feature_types[col] = 'Categorical'

# Assign roles
feature_roles = {}
for col in df.columns:
    if col in structural_features:
        feature_roles[col] = 'STRUCTURAL'
    elif col in exposure_features:
        feature_roles[col] = 'EXPOSURE_SENSITIVE'
    elif col in epistemic_features:
        feature_roles[col] = 'EPISTEMIC'
    else:
        feature_roles[col] = 'LOW_SIGNAL'

# Priority weights
feature_priority = {}
for col in df.columns:
    if col == 'membership_category':
        feature_priority[col] = 'HIGH'
    elif col in exposure_features:
        feature_priority[col] = 'MEDIUM'
    elif col in epistemic_features:
        feature_priority[col] = 'CONTROLLED'
    else:
        feature_priority[col] = 'LOW'


# =========================================================  
# OBJECTIVE 2: Statistical Computation Layer  
# =========================================================  

print("Executing Objective 2: Statistical Computation Layer...")

feature_diagnostics = {}

for col in df.columns:
    if col == 'churn':
        continue

    diagnostics = {}

    if feature_types[col] == 'Numerical':
        series = df[col].dropna()

        diagnostics['mean'] = series.mean()
        diagnostics['median'] = series.median()
        diagnostics['std'] = series.std()
        diagnostics['skewness'] = skew(series)
        diagnostics['kurtosis'] = kurtosis(series)
        diagnostics['cv'] = diagnostics['std'] / diagnostics['mean'] if diagnostics['mean'] != 0 else np.nan

        # Ensure numeric-only correlation for churn
        if pd.api.types.is_numeric_dtype(df[col]):
            diagnostics['corr_churn'] = df[[col, 'churn']].apply(pd.to_numeric, errors='coerce').corr().iloc[0,1]
        else:
            diagnostics['corr_churn'] = np.nan

        # Exposure proxy correlation (SAFE NUMERIC FILTER)
        if col != 'avg_frequency_login_days' and pd.api.types.is_numeric_dtype(df[col]):
            temp_df = df[[col, 'avg_frequency_login_days']].apply(pd.to_numeric, errors='coerce')
            diagnostics['corr_exposure_proxy'] = temp_df.corr().iloc[0,1]
        else:
            diagnostics['corr_exposure_proxy'] = np.nan

    else:
        counts = df[col].value_counts()
        proportions = df[col].value_counts(normalize=True)
        churn_rate = df.groupby(col)['churn'].mean()

        diagnostics['counts'] = counts.to_dict()
        diagnostics['proportions'] = proportions.to_dict()
        diagnostics['churn_rate'] = churn_rate.to_dict()

    feature_diagnostics[col] = diagnostics


# =========================================================  
# OBJECTIVE 3: Adaptive Bracketing System  
# =========================================================  

print("Executing Objective 3: Adaptive Bracketing System...")

bracket_results = {}

for col in df.columns:
    if col == 'churn':
        continue

    if feature_types[col] == 'Numerical':
        series = df[col]

        # Determine binning strategy
        use_quantile = False
        if col in exposure_features:
            use_quantile = True
        else:
            if abs(feature_diagnostics[col]['skewness']) > 1:
                use_quantile = True

        # Winsorization for extreme values
        if col == 'avg_transaction_value':
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            series = series.clip(lower, upper)

        # Apply binning
        if use_quantile:
            df[f"{col}_bracket"] = pd.qcut(series, q=3, labels=['LOW', 'MEDIUM', 'HIGH'], duplicates='drop')
        else:
            df[f"{col}_bracket"] = pd.cut(series, bins=3, labels=['LOW', 'MEDIUM', 'HIGH'])

        bracket_stats = {}

        for bracket in ['LOW', 'MEDIUM', 'HIGH']:
            subset = df[df[f"{col}_bracket"] == bracket]

            if len(subset) == 0:
                continue

            bracket_stats[bracket] = {
                'count': len(subset),
                'proportion': len(subset) / len(df),
                'mean': subset[col].mean(),
                'variance': subset[col].var(),
                'churn_rate': subset['churn'].mean()
            }

        bracket_results[col] = bracket_stats

    else:
        # Categorical grouping based on churn rate
        churn_rates = df.groupby(col)['churn'].mean().sort_values()

        categories = churn_rates.index.tolist()
        n = len(categories)

        low = categories[:n//3]
        medium = categories[n//3:2*n//3]
        high = categories[2*n//3:]

        grouping = {}
        for c in categories:
            if c in low:
                grouping[c] = 'LOW'
            elif c in medium:
                grouping[c] = 'MEDIUM'
            else:
                grouping[c] = 'HIGH'

        df[f"{col}_bracket"] = df[col].map(grouping)

        bracket_stats = {}
        for bracket in ['LOW', 'MEDIUM', 'HIGH']:
            subset = df[df[f"{col}_bracket"] == bracket]

            if len(subset) == 0:
                continue

            bracket_stats[bracket] = {
                'count': len(subset),
                'proportion': len(subset) / len(df),
                'churn_rate': subset['churn'].mean()
            }

        bracket_results[col] = bracket_stats


# =========================================================  
# OBJECTIVE 4: Structural Diagnosis & Feature Engineering Assignment  
# =========================================================  

print("Executing Objective 4: Structural Diagnosis & Feature Engineering Assignment...")

final_feature_report = []

# Restrict to original features only (exclude derived *_bracket columns)
original_features = [col for col in df.columns if not col.endswith('_bracket') and col != 'churn']

for col in original_features:

    flags = []
    patterns = "UNKNOWN"

    if feature_types[col] == 'Numerical':
        skewness = feature_diagnostics[col]['skewness']
        if abs(skewness) > 1:
            flags.append('HIGH_SKEW')

        if feature_roles[col] == 'EXPOSURE_SENSITIVE':
            flags.append('ACCUMULATIVE_PATTERN')

    if feature_roles[col] == 'EPISTEMIC':
        flags.append('POST_OUTCOME_LEAKAGE_RISK')

    if feature_priority[col] == 'HIGH':
        flags.append('STRONG_CHURN_SEPARATION')
    elif feature_priority[col] == 'LOW':
        flags.append('WEAK_CHURN_SIGNAL')

    # Detect pattern (simple monotonic check)
    brackets = bracket_results.get(col, {})
    churn_rates = [brackets[b]['churn_rate'] for b in brackets if b in brackets]

    if len(churn_rates) == 3:
        if churn_rates[0] < churn_rates[1] < churn_rates[2] or churn_rates[0] > churn_rates[1] > churn_rates[2]:
            patterns = 'MONOTONIC'
        elif max(churn_rates) == churn_rates[-1]:
            patterns = 'THRESHOLD'
        else:
            patterns = 'NON_MONOTONIC'
    else:
        patterns = 'FLAT'

    # Feature engineering suggestions
    fe_actions = {
        'LOW': [],
        'MEDIUM': [],
        'HIGH': [],
        'CROSS': []
    }

    if feature_roles[col] == 'STRUCTURAL':
        fe_actions['CROSS'].append('segmentation anchor')

    elif feature_roles[col] == 'EXPOSURE_SENSITIVE':
        fe_actions['CROSS'].extend(['rate conversion', 'log transform if skewed'])

    elif feature_roles[col] == 'EPISTEMIC':
        fe_actions['CROSS'].append('isolate or restrict usage')

    elif feature_roles[col] == 'LOW_SIGNAL':
        fe_actions['CROSS'].append('candidate for elimination')

    final_feature_report.append({
        'feature': col,
        'type': feature_types[col],
        'flags': flags,
        'pattern': patterns,
        'diagnostics': feature_diagnostics[col],
        'brackets': bracket_results.get(col, {}),
        'feature_engineering': fe_actions
    })


# ---------------------------------------------------------  
# STAGE OUTPUT  
# ---------------------------------------------------------

import os

output_df = pd.DataFrame(final_feature_report)
output_path = r"C:\Users\hp\Exploratory Data Analysis\Outputs\statistical_stats_stage_output.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

output_df.to_csv(output_path, index=False)

print("Stage outputs successfully saved.")


# =========================================================  
# Statistical Stats (Dataset-Adaptive Diagnostic Layer) STAGE COMPLETE  
# =========================================================  

print("Statistical Stats stage execution complete.")