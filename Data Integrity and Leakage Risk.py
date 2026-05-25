# =========================================================  
# ECA STAGE: Data Integrity & Leakage Risk Assessment  
# =========================================================  
# Purpose:  
# To verify that the dataset’s representation of churn is temporally valid and free from outcome-conditioned artifacts within a behavioral-dominant, subscription-based digital service context. This stage ensures that observed behavioral signals reflect information that could have existed prior to churn.  
#  
# Dataset/Input:  
# C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\LoyaltyHub.csv  
#  
# Output:  
# Epistemic feature map, leakage diagnostics, exposure sensitivity analysis, missingness report, and dominance mapping  
# =========================================================

# ---------------------------------------------------------  
# STAGE OBJECTIVES  
# ---------------------------------------------------------
# Objective 1: Classify all features based on temporal observability and epistemic legitimacy relative to churn.
# Objective 2: Identify and document variables that exhibit leakage risk, including lifecycle-terminal, retrospective, or administratively deterministic features.
# Objective 3: Assess exposure sensitivity of behavioral and cumulative features to detect lifecycle-length-driven signal inflation.
# Objective 4: Evaluate measurement consistency and missingness patterns across churn outcomes to detect outcome-conditioned observation bias.
# Objective 5: Construct an epistemic classification map assigning each feature to a defined knowledge category.
# Objective 6: Perform epistemic dominance mapping to identify which features disproportionately drive churn signal and assess their temporal validity.

# ---------------------------------------------------------  
# LOAD DEPENDENCIES  
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

# ---------------------------------------------------------  
# LOAD DATA  
# ---------------------------------------------------------
file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub\LoyaltyHub.csv"
df = pd.read_csv(file_path)

# Drop index column (assumed first column)
df = df.iloc[:, 1:]

# Rename target variable
df = df.rename(columns={"churn_risk_score": "churn"})

# =========================================================  
# OBJECTIVE 1: Temporal Observability Classification  
# =========================================================  
print("Executing Objective 1: Temporal Observability Classification...")

feature_types = {}

for col in df.columns:
    if col in ["age", "gender", "region_category"]:
        feature_types[col] = "static"
    elif col in ["membership_category", "joined_through_referral", "preferred_offer_types", "medium_of_operation", "internet_option"]:
        feature_types[col] = "structural"
    elif col in ["avg_time_spent", "avg_transaction_value", "avg_frequency_login_days", "points_in_wallet"]:
        feature_types[col] = "behavioral"
    elif col in ["last_visit_time", "days_since_last_login"]:
        feature_types[col] = "recency_based"
    elif col in ["complaint_status", "feedback", "churn"]:
        feature_types[col] = "terminal_or_outcome"
    else:
        feature_types[col] = "unknown"

feature_types_df = pd.DataFrame.from_dict(feature_types, orient='index', columns=['feature_type'])

# =========================================================  
# OBJECTIVE 2: Leakage Risk Identification  
# =========================================================  
print("Executing Objective 2: Leakage Risk Identification...")

leakage_risk = {}

for col in df.columns:
    if col in ["complaint_status", "feedback", "churn"]:
        leakage_risk[col] = "high_risk_terminal"
    elif col in ["avg_time_spent", "avg_transaction_value", "avg_frequency_login_days", "points_in_wallet"]:
        leakage_risk[col] = "exposure_sensitive"
    elif col in ["last_visit_time", "days_since_last_login"]:
        leakage_risk[col] = "temporal_proximity"
    else:
        leakage_risk[col] = "low_risk"

leakage_df = pd.DataFrame.from_dict(leakage_risk, orient='index', columns=['leakage_risk'])

# =========================================================  
# OBJECTIVE 3: Exposure Sensitivity Assessment  
# =========================================================  
print("Executing Objective 3: Exposure Sensitivity Assessment...")

exposure_features = ["avg_time_spent", "avg_transaction_value", "avg_frequency_login_days", "points_in_wallet"]

exposure_analysis = []

for col in exposure_features:
    if col in df.columns:
        # Convert column to numeric to avoid string-related errors
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        
        churned_mean = numeric_col[df["churn"] == 1].mean()
        retained_mean = numeric_col[df["churn"] == 0].mean()
        diff = churned_mean - retained_mean
        
        exposure_analysis.append({
            "feature": col,
            "churned_mean": churned_mean,
            "retained_mean": retained_mean,
            "difference": diff
        })

exposure_df = pd.DataFrame(exposure_analysis)

# =========================================================  
# OBJECTIVE 4: Missingness & Measurement Bias  
# =========================================================  
print("Executing Objective 4: Missingness & Measurement Bias...")

missingness_analysis = []

for col in df.columns:
    churn_missing = df[df["churn"] == 1][col].isna().mean()
    retained_missing = df[df["churn"] == 0][col].isna().mean()
    
    missingness_analysis.append({
        "feature": col,
        "churn_missing_rate": churn_missing,
        "retained_missing_rate": retained_missing,
        "difference": churn_missing - retained_missing
    })

missingness_df = pd.DataFrame(missingness_analysis)

# =========================================================  
# OBJECTIVE 5: Epistemic Classification Map  
# =========================================================  
print("Executing Objective 5: Epistemic Classification Map...")

epistemic_classification = {}

for col in df.columns:
    if col in ["age", "gender", "region_category", "membership_category"]:
        epistemic_classification[col] = "pre_outcome_observable"
    elif col in ["avg_time_spent", "avg_transaction_value", "avg_frequency_login_days", "points_in_wallet"]:
        epistemic_classification[col] = "exposure_sensitive"
    elif col in ["complaint_status", "feedback"]:
        epistemic_classification[col] = "lifecycle_terminal"
    elif col in ["last_visit_time", "days_since_last_login"]:
        epistemic_classification[col] = "retrospective_proxy"
    elif col == "churn":
        epistemic_classification[col] = "target"
    else:
        epistemic_classification[col] = "administrative_or_unknown"

epistemic_df = pd.DataFrame.from_dict(epistemic_classification, orient='index', columns=['epistemic_class'])

# =========================================================  
# OBJECTIVE 6: Epistemic Dominance Mapping  
# =========================================================  
print("Executing Objective 6: Epistemic Dominance Mapping...")

# Reduce high-cardinality columns BEFORE encoding to prevent memory explosion
high_cardinality_cols = [col for col in df.columns if df[col].nunique() > 50 and col != "churn"]

# Drop high-cardinality columns (e.g., IDs, free-text, timestamps)
df_reduced = df.drop(columns=high_cardinality_cols)

# Prepare data (encode categorical variables)
df_encoded = pd.get_dummies(df_reduced.drop(columns=["churn"]), drop_first=True)
target = df_reduced["churn"]

# Mutual Information
mi_scores = mutual_info_classif(df_encoded.fillna(0), target, discrete_features='auto')

mi_df = pd.DataFrame({
    "feature": df_encoded.columns,
    "mutual_info_score": mi_scores
}).sort_values(by="mutual_info_score", ascending=False)

# Map back epistemic classification
mi_df["epistemic_class"] = mi_df["feature"].apply(
    lambda x: epistemic_classification.get(x.split("_")[0], "derived")
)
# ---------------------------------------------------------  
# STAGE OUTPUT  
# ---------------------------------------------------------

# Define explicit output directory
output_dir = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub V2"

# Save all outputs to the specified directory
feature_types_df.to_csv(f"{output_dir}\\feature_types.csv", index=True)
leakage_df.to_csv(f"{output_dir}\\leakage_risk.csv", index=True)
exposure_df.to_csv(f"{output_dir}\\exposure_analysis.csv", index=False)
missingness_df.to_csv(f"{output_dir}\\missingness_analysis.csv", index=False)
epistemic_df.to_csv(f"{output_dir}\\epistemic_classification.csv", index=True)
mi_df.to_csv(f"{output_dir}\\dominance_mapping.csv", index=False)

print("Stage outputs successfully saved.")

# =========================================================  
# Data Integrity & Leakage Risk Assessment STAGE COMPLETE  
# =========================================================  

print("Data Integrity & Leakage Risk Assessment stage execution complete.")