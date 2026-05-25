# =========================================================
# ECA STAGE: FEATURE TRIAGE
# =========================================================
# Purpose:
# To classify all available features and dataset views into
# analytically valid pathways before any target-based
# exploration begins. This stage preserves legitimate churn
# signals while preventing leakage, post-outcome
# contamination, structurally broken variables, and low-value
# noise from entering downstream analysis.
#
# Dataset/Input:
# Engineered churn dataset (gold and silver members only):
# C:\Users\hp\Exploratory Data Analysis\CSV files\
# LoyaltyHub V2\Engineered_Churn_Dataset.csv
#
# Output:
# Master Feature Triage Register, feature pathway lists,
# tenure validation summary, churn/non-churn subsets.
# =========================================================

# ---------------------------------------------------------
# STAGE OBJECTIVES
# ---------------------------------------------------------
# Objective 1: Build a full feature inventory and classify
# each variable by structural type.
#
# Objective 2: Apply temporal admissibility screening to
# every feature and assign eligibility status.
#
# Objective 3: Apply analytical value triage using
# previous-stage evidence, feature integrity, and business
# relevance.
#
# Objective 4: Route all features into final usage pathways
# and generate clean downstream feature lists.
#
# Objective 5: Validate and classify dataset views requiring
# separate handling, including tenure and churn subsets.
#
# Objective 6: Create structurally aligned population splits
# for future within-class analysis.

# ---------------------------------------------------------
# LOAD DEPENDENCIES
# ---------------------------------------------------------
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub V2\Engineered_Churn_Dataset.csv"
df = pd.read_csv(file_path)

# Standardize column names
df.columns = df.columns.str.strip()

# Output directory
output_dir = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub V2\Feature_Triage_Outputs"
os.makedirs(output_dir, exist_ok=True)

# Target column
target_col = "churn"

# Feature columns
feature_columns = [col for col in df.columns if col != target_col]

# Initialize triage register
triage_register = pd.DataFrame({"feature": feature_columns})


# =========================================================
# OBJECTIVE 1: BUILD FEATURE INVENTORY AND CLASSIFY TYPES
# =========================================================
print("Executing Objective 1: Build feature inventory and classify types...")

time_keywords = ["date", "time", "last_visit", "joining"]
rate_keywords = ["per_day", "density", "frequency", "efficiency", "score", "interaction"]

def classify_structural_type(series, column_name):
    col_lower = column_name.lower()

    if any(word in col_lower for word in time_keywords):
        return "Time-based"

    if any(word in col_lower for word in rate_keywords):
        return "Engineered rate / metric"

    if pd.api.types.is_numeric_dtype(series):
        unique_vals = series.nunique(dropna=True)

        if unique_vals == 2:
            return "Binary indicator"

        if pd.api.types.is_integer_dtype(series):
            if unique_vals <= 20:
                return "Discrete numeric"
            return "Continuous numeric"

        return "Continuous numeric"

    return "Categorical"

triage_register["structural_class"] = [
    classify_structural_type(df[col], col) for col in triage_register["feature"]
]


# =========================================================
# OBJECTIVE 2: APPLY TEMPORAL ADMISSIBILITY SCREENING
# =========================================================
print("Executing Objective 2: Apply temporal admissibility screening...")

ambiguous_features = {
    "avg_frequency_login_days",
    "recency_score",
    "days_since_last_login",
    "log_avg_txn",
    "points_time_interaction",
    "joining_date",
    "last_visit_time"
}

diagnostic_only_features = {
    "days_since_last_login",
    "recency_score",
    "last_visit_time"
}

def classify_admissibility(feature_name):
    if feature_name in diagnostic_only_features:
        return "Outcome-relative (diagnostic-only)"

    if feature_name in ambiguous_features:
        return "Ambiguous"

    return "Pre-decision admissible"

triage_register["admissibility_label"] = triage_register["feature"].apply(classify_admissibility)


# =========================================================
# OBJECTIVE 3: APPLY ANALYTICAL VALUE TRIAGE
# =========================================================
print("Executing Objective 3: Apply analytical value triage...")

high_priority = {
    "points_in_wallet",
    "avg_transaction_value",
    "value_density"
}

secondary_priority = {
    "membership_category",
    "region_City",
    "region_Town",
    "region_Village",
    "avg_frequency_login_days",
    "tenure_days"
}

discard_priority = {
    "log_points",
    "region_City_x_value",
    "region_Town_x_value",
    "region_Village_x_value"
}

def assign_priority(feature_name):
    if feature_name in high_priority:
        return "High-priority"

    if feature_name in secondary_priority:
        return "Secondary"

    if feature_name in discard_priority:
        return "Discard"

    return "Secondary"

triage_register["priority_tier"] = triage_register["feature"].apply(assign_priority)


# =========================================================
# OBJECTIVE 4: ROUTE FEATURES INTO FINAL PATHWAYS
# =========================================================
print("Executing Objective 4: Route features into final pathways...")

exclusion_features = {
    "log_points",
    "region_City_x_value",
    "region_Town_x_value",
    "region_Village_x_value",
    "security_no",
    "referral_id"
}

def assign_pathway(row):
    feature = row["feature"]
    admissibility = row["admissibility_label"]
    priority = row["priority_tier"]

    if feature in exclusion_features or priority == "Discard":
        return "Exclusion Candidates"

    if admissibility == "Outcome-relative (diagnostic-only)":
        return "Diagnostic-Only Variables"

    if admissibility == "Ambiguous":
        return "Conditional Candidates"

    if admissibility == "Pre-decision admissible" and priority == "High-priority":
        return "Approved Predictive Candidates"

    if admissibility == "Pre-decision admissible":
        return "Approved Predictive Candidates"

    return "Leakage / Post-Outcome Risks"

triage_register["final_pathway_assignment"] = triage_register.apply(assign_pathway, axis=1)

# Notes / risks column
def build_notes(row):
    feature = row["feature"]

    if feature == "points_in_wallet":
        return "Strong prior churn separation; validate timing."
    if feature == "avg_frequency_login_days":
        return "Strong behavior signal; semantic timing unclear."
    if feature == "log_points":
        return "Previously failed integrity checks."
    if feature == "tenure_days":
        return "Requires datatype and distribution validation."
    if row["admissibility_label"] == "Ambiguous":
        return "Needs business definition review."
    if row["final_pathway_assignment"] == "Diagnostic-Only Variables":
        return "Use only for diagnostic exploration."

    return ""

triage_register["notes_risks"] = triage_register.apply(build_notes, axis=1)

# Generate feature lists
approved_features = triage_register.loc[
    triage_register["final_pathway_assignment"] == "Approved Predictive Candidates",
    "feature"
].tolist()

conditional_features = triage_register.loc[
    triage_register["final_pathway_assignment"] == "Conditional Candidates",
    "feature"
].tolist()

diagnostic_features = triage_register.loc[
    triage_register["final_pathway_assignment"] == "Diagnostic-Only Variables",
    "feature"
].tolist()

leakage_features = triage_register.loc[
    triage_register["final_pathway_assignment"] == "Leakage / Post-Outcome Risks",
    "feature"
].tolist()

exclusion_list = triage_register.loc[
    triage_register["final_pathway_assignment"] == "Exclusion Candidates",
    "feature"
].tolist()


# =========================================================
# OBJECTIVE 5: VALIDATE TENURE AND DATASET VIEWS
# =========================================================
print("Executing Objective 5: Validate tenure and classify dataset views...")

tenure_summary = {}

if "tenure_days" in df.columns:
    tenure_series = pd.to_numeric(df["tenure_days"], errors="coerce")

    tenure_summary = {
        "feature": "tenure_days",
        "non_null_count": int(tenure_series.notna().sum()),
        "null_count": int(tenure_series.isna().sum()),
        "unique_values": int(tenure_series.nunique(dropna=True)),
        "min_value": float(tenure_series.min()) if tenure_series.notna().any() else np.nan,
        "max_value": float(tenure_series.max()) if tenure_series.notna().any() else np.nan,
        "negative_count": int((tenure_series < 0).sum())
    }

tenure_validation_df = pd.DataFrame([tenure_summary])


# =========================================================
# OBJECTIVE 6: CREATE STRUCTURALLY ALIGNED SPLITS
# =========================================================
print("Executing Objective 6: Create structurally aligned population splits...")

churned_df = df[df[target_col] == 1].copy()
non_churned_df = df[df[target_col] == 0].copy()

predictive_ready_columns = approved_features + [target_col]
predictive_ready_columns = [col for col in predictive_ready_columns if col in df.columns]

predictive_ready_df = df[predictive_ready_columns].copy()


# ---------------------------------------------------------
# STAGE OUTPUT
# ---------------------------------------------------------
triage_register.to_csv(
    os.path.join(output_dir, "Master_Feature_Triage_Register.csv"),
    index=False
)

pd.DataFrame({"Approved_Predictive_Features": pd.Series(approved_features)}).to_csv(
    os.path.join(output_dir, "Approved_Predictive_Features.csv"),
    index=False
)

pd.DataFrame({"Conditional_Features": pd.Series(conditional_features)}).to_csv(
    os.path.join(output_dir, "Conditional_Features.csv"),
    index=False
)

pd.DataFrame({"Diagnostic_Only_Features": pd.Series(diagnostic_features)}).to_csv(
    os.path.join(output_dir, "Diagnostic_Only_Features.csv"),
    index=False
)

pd.DataFrame({"Leakage_Risk_Features": pd.Series(leakage_features)}).to_csv(
    os.path.join(output_dir, "Leakage_Risk_Features.csv"),
    index=False
)

pd.DataFrame({"Exclusion_Features": pd.Series(exclusion_list)}).to_csv(
    os.path.join(output_dir, "Exclusion_Features.csv"),
    index=False
)

tenure_validation_df.to_csv(
    os.path.join(output_dir, "Tenure_Validation_Summary.csv"),
    index=False
)

churned_df.to_csv(
    os.path.join(output_dir, "Churned_Subset.csv"),
    index=False
)

non_churned_df.to_csv(
    os.path.join(output_dir, "Non_Churned_Subset.csv"),
    index=False
)

predictive_ready_df.to_csv(
    os.path.join(output_dir, "Predictive_Ready_Dataset.csv"),
    index=False
)

print("Stage outputs successfully saved.")


# =========================================================
# FEATURE TRIAGE STAGE COMPLETE
# =========================================================
print("Feature Triage stage execution complete.")