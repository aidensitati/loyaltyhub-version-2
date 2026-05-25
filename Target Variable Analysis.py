# =========================================================
# ECA STAGE: TARGET VARIABLE ANALYSIS
# =========================================================
# Purpose:
# This stage quantifies how churn behaves as a structural
# outcome across lifecycle time, customer segments, and
# system states within the LoyaltyHub dataset. It maps where
# churn concentrates, where it becomes difficult to
# distinguish from retention, and where it creates usable
# analytical constraints for future feature diagnostics.
#
# Dataset/Input:
# C:\Users\hp\Exploratory Data Analysis\CSV files\
# LoyaltyHub V2\Engineered_Churn_Dataset.csv
#
# Output:
# Structural Churn Geometry Report + supporting CSV tables
# =========================================================

# ---------------------------------------------------------
# STAGE OBJECTIVES
# ---------------------------------------------------------
# Objective 1: Compute global churn balance and churn
# concentration across major customer segments.
#
# Objective 2: Quantify churn geometry across lifecycle
# tenure phases (Early / Mid / Late) using data-driven
# tenure bins.
#
# Objective 3: Measure churn concentration across behavioral
# state bands using high-signal variables (transaction value,
# wallet points, login frequency).
#
# Objective 4: Establish baseline structural difficulty
# using naive churn rules across full data, tenure bins,
# and membership segments.
#
# Objective 5: Generate churn-conditioned feature relevance
# constraints for downstream diagnostic exploration.

# ---------------------------------------------------------
# LOAD DEPENDENCIES
# ---------------------------------------------------------
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub V2\Engineered_Churn_Dataset.csv"
df = pd.read_csv(file_path)

# Standardize column names
df.columns = df.columns.str.strip()

# Standardize churn label to numeric 0/1
if df["churn"].dtype == object:
    df["churn"] = (
        df["churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    )

df["churn"] = pd.to_numeric(df["churn"], errors="coerce")
df = df.dropna(subset=["churn"]).copy()
df["churn"] = df["churn"].astype(int)

# Output directory
output_dir = os.path.dirname(file_path)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def safe_qcut(series, labels):
    valid = series.dropna()
    if valid.nunique() < len(labels):
        return pd.Series(np.nan, index=series.index)
    try:
        binned = pd.qcut(series, q=len(labels), labels=labels, duplicates="drop")
        return pd.Series(binned, index=series.index)
    except Exception:
        return pd.Series(np.nan, index=series.index)

def monotonicity(values):
    vals = [v for v in values if pd.notna(v)]
    if len(vals) < 3:
        return "unstable"
    if vals[0] <= vals[1] <= vals[2]:
        return "increasing"
    if vals[0] >= vals[1] >= vals[2]:
        return "decreasing"
    return "unstable"

def evaluate_rule(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "false_negative_rate": (
            ((y_true == 1) & (y_pred == 0)).sum() / max((y_true == 1).sum(), 1)
        ),
    }

# =========================================================
# OBJECTIVE 1: Compute global churn balance and churn
# concentration across major customer segments
# =========================================================
print("Executing Objective 1: Compute global churn balance and churn concentration across major customer segments...")

overall_summary = pd.DataFrame({
    "total_customers": [len(df)],
    "churn_count": [df["churn"].sum()],
    "churn_rate": [df["churn"].mean()]
})

segment_results = []

# Membership segmentation
if "membership_category" in df.columns:
    temp = (
        df.groupby("membership_category")["churn"]
        .agg(["count", "mean"])
        .reset_index()
    )
    for _, row in temp.iterrows():
        segment_results.append({
            "segment_type": "membership_category",
            "segment_name": row["membership_category"],
            "customer_count": row["count"],
            "churn_rate": row["mean"]
        })

# Region dummies
region_cols = ["region_City", "region_Town", "region_Village"]
for col in region_cols:
    if col in df.columns:
        subset = df[df[col] == 1]
        if len(subset) > 0:
            segment_results.append({
                "segment_type": "region",
                "segment_name": col.replace("region_", ""),
                "customer_count": len(subset),
                "churn_rate": subset["churn"].mean()
            })

# Behavioral terciles
for feature in ["avg_transaction_value", "points_in_wallet"]:
    if feature in df.columns:
        bands = safe_qcut(df[feature], ["Low", "Mid", "High"])
        temp_df = pd.DataFrame({"band": bands, "churn": df["churn"]})
        temp = temp_df.groupby("band")["churn"].agg(["count", "mean"]).reset_index()
        for _, row in temp.iterrows():
            segment_results.append({
                "segment_type": feature,
                "segment_name": row["band"],
                "customer_count": row["count"],
                "churn_rate": row["mean"]
            })

segment_churn_table = pd.DataFrame(segment_results)

if not segment_churn_table.empty:
    max_rate = segment_churn_table["churn_rate"].max()
    min_rate = segment_churn_table["churn_rate"].min()
    imbalance_ratio = max_rate / min_rate if min_rate != 0 else np.nan
    churn_variance = segment_churn_table["churn_rate"].var()
else:
    max_rate = np.nan
    min_rate = np.nan
    imbalance_ratio = np.nan
    churn_variance = np.nan

imbalance_summary = pd.DataFrame({
    "max_segment_churn_rate": [max_rate],
    "min_segment_churn_rate": [min_rate],
    "imbalance_ratio": [imbalance_ratio],
    "segment_churn_variance": [churn_variance]
})

# =========================================================
# OBJECTIVE 2: Quantify churn geometry across lifecycle
# tenure phases
# =========================================================
print("Executing Objective 2: Quantify churn geometry across lifecycle tenure phases...")

tenure_geometry = pd.DataFrame()

if "tenure_days" in df.columns:
    df["tenure_phase"] = safe_qcut(df["tenure_days"], ["Early", "Mid", "Late"])

    tenure_geometry = (
        df.groupby("tenure_phase")
        .agg(
            count=("churn", "size"),
            churn_rate=("churn", "mean"),
            mean_transaction_value=("avg_transaction_value", "mean"),
            mean_login_frequency=("avg_frequency_login_days", "mean")
        )
        .reset_index()
    )

    tenure_geometry["non_churn_rate"] = 1 - tenure_geometry["churn_rate"]
    tenure_geometry["churn_nonchurn_ratio"] = (
        tenure_geometry["churn_rate"] /
        tenure_geometry["non_churn_rate"].replace(0, np.nan)
    )

    overall_churn_rate = df["churn"].mean()
    tenure_geometry["concentration_index"] = (
        tenure_geometry["churn_rate"] / overall_churn_rate
    )

# =========================================================
# OBJECTIVE 3: Measure churn concentration across
# behavioral state bands
# =========================================================
print("Executing Objective 3: Measure churn concentration across behavioral state bands...")

behavior_features = [
    "avg_transaction_value",
    "points_in_wallet",
    "avg_frequency_login_days",
    "value_density"
]

behavior_rows = []

for feature in behavior_features:
    if feature in df.columns:
        bands = safe_qcut(df[feature], ["Low", "Mid", "High"])
        temp_df = pd.DataFrame({"band": bands, "churn": df["churn"]})
        grouped = temp_df.groupby("band")["churn"].mean()

        low_val = grouped.get("Low", np.nan)
        mid_val = grouped.get("Mid", np.nan)
        high_val = grouped.get("High", np.nan)

        spread = np.nanmax([low_val, mid_val, high_val]) - np.nanmin([low_val, mid_val, high_val])

        behavior_rows.append({
            "feature": feature,
            "low_band_churn": low_val,
            "mid_band_churn": mid_val,
            "high_band_churn": high_val,
            "monotonicity": monotonicity([low_val, mid_val, high_val]),
            "churn_spread": spread
        })

behavioral_separation = pd.DataFrame(behavior_rows)
behavioral_separation = behavioral_separation.sort_values(
    by="churn_spread", ascending=False
)

# =========================================================
# OBJECTIVE 4: Establish baseline structural difficulty
# =========================================================
print("Executing Objective 4: Establish baseline structural difficulty...")

baseline_rows = []

y_true = df["churn"]

# Rule 1: predict non-churn
pred_non_churn = np.zeros(len(df), dtype=int)
metrics = evaluate_rule(y_true, pred_non_churn)
baseline_rows.append({
    "rule_name": "predict_non_churn",
    "scope": "full_dataset",
    **metrics
})

# Threshold rules
rules = {
    "low_transaction_value": "avg_transaction_value",
    "low_points_in_wallet": "points_in_wallet",
    "low_login_frequency": "avg_frequency_login_days"
}

for rule_name, feature in rules.items():
    if feature in df.columns:
        threshold = df[feature].quantile(0.33)
        pred = (df[feature] <= threshold).astype(int)
        metrics = evaluate_rule(y_true, pred)
        baseline_rows.append({
            "rule_name": rule_name,
            "scope": "full_dataset",
            **metrics
        })

# Membership scope
if "membership_category" in df.columns:
    for segment in df["membership_category"].dropna().unique():
        subset = df[df["membership_category"] == segment]
        if len(subset) > 0:
            pred = np.zeros(len(subset), dtype=int)
            metrics = evaluate_rule(subset["churn"], pred)
            baseline_rows.append({
                "rule_name": "predict_non_churn",
                "scope": str(segment),
                **metrics
            })

baseline_difficulty = pd.DataFrame(baseline_rows)

# =========================================================
# OBJECTIVE 5: Generate churn-conditioned feature relevance
# constraints
# =========================================================
print("Executing Objective 5: Generate churn-conditioned feature relevance constraints...")

relevance_rows = []

phase_priority_map = {
    "Early": "activation / onboarding",
    "Mid": "engagement / usage",
    "Late": "fatigue / value decay"
}

if "tenure_phase" in df.columns and not tenure_geometry.empty:
    for _, row in tenure_geometry.iterrows():
        phase = row["tenure_phase"]
        relevance_rows.append({
            "lifecycle_phase": phase,
            "strongest_behavior_signal": "Insufficient stage-local ranking",
            "weakest_signal_zone": "Insufficient stage-local evidence",
            "engineering_priority": phase_priority_map.get(str(phase), "review")
        })

feature_sensitivity_map = pd.DataFrame(relevance_rows)

# ---------------------------------------------------------
# STAGE OUTPUT
# ---------------------------------------------------------
overall_summary.to_csv(
    os.path.join(output_dir, "TVA_Overall_Churn_Summary.csv"),
    index=False
)

segment_churn_table.to_csv(
    os.path.join(output_dir, "TVA_Segment_Churn_Table.csv"),
    index=False
)

imbalance_summary.to_csv(
    os.path.join(output_dir, "TVA_Imbalance_Summary.csv"),
    index=False
)

tenure_geometry.to_csv(
    os.path.join(output_dir, "TVA_Tenure_Geometry.csv"),
    index=False
)

behavioral_separation.to_csv(
    os.path.join(output_dir, "TVA_Behavioral_Separation.csv"),
    index=False
)

baseline_difficulty.to_csv(
    os.path.join(output_dir, "TVA_Baseline_Difficulty.csv"),
    index=False
)

feature_sensitivity_map.to_csv(
    os.path.join(output_dir, "TVA_Feature_Sensitivity_Map.csv"),
    index=False
)

print("Stage outputs successfully saved.")

# =========================================================
# TARGET VARIABLE ANALYSIS STAGE COMPLETE
# =========================================================
print("Target Variable Analysis stage execution complete.")