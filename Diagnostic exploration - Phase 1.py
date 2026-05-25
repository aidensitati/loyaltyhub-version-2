# =========================================================
# ECA STAGE: Diagnostic Exploration — Behavioral State,
# Interaction & Transition Mapping
# PHASE 1: State Construction
# =========================================================
# Stage Purpose:
# To convert validated high-signal behavioral variables
# into structured state systems, quantify how churn risk
# evolves across those states, and isolate the directional
# transitions and compound behavioral conditions that
# precede churn.
#
# Phase Purpose:
# Construct behavioral state boundaries for high-signal
# features using distribution-aware thresholding logic.
#
# Inputs from Previous Phases:
# None
#
# Outputs Generated:
# - Distribution summary statistics
# - Feature skewness diagnostics
# - Behavioral state thresholds
# - State-mapped categorical features
# - Distribution validation plots
# =========================================================


# ---------------------------------------------------------
# PHASE OBJECTIVES
# ---------------------------------------------------------
# Objective 1:
# Derive distribution ranges and define behavioral state
# thresholds.
#
# Procedures:
# - Compute min, max, Q1, median, Q3, skewness
# - Define thresholds based on distribution structure
# - Assign behavioral state categories
# - Persist mapping logic
# - Generate histogram/KDE validation plots


# ---------------------------------------------------------
# LOAD DEPENDENCIES
# ---------------------------------------------------------

import os
import json

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew


# ---------------------------------------------------------
# LOAD / ACCESS DATA
# ---------------------------------------------------------

data_path = (
    r"C:\Users\hp\Exploratory Data Analysis\CSV files"
    r"\LoyaltyHub V2\Feature_Triage_Outputs"
    r"\Churned_Subset.csv"
)

df = pd.read_csv(data_path)

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

# ---------------------------------------------------------
# CREATE OUTPUT DIRECTORIES (MUST BE FIRST)
# ---------------------------------------------------------

base_output_directory = (
    r"C:\Users\hp\Exploratory Data Analysis\CSV files"
    r"\LoyaltyHub V2\Outputs"
)

plot_output_dir = os.path.join(
    base_output_directory,
    "phase_1_distribution_plots"
)

output_directory = os.path.join(
    base_output_directory,
    "phase_1_outputs"
)

os.makedirs(plot_output_dir, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)

print(f"Plot output directory created: {plot_output_dir}")
print(f"Phase output directory created: {output_directory}")
# =========================================================
# OBJECTIVE 1: Derive Distribution Ranges and Define
# Behavioral State Thresholds
# =========================================================

print(
    "Executing Objective 1: "
    "Derive Distribution Ranges and Define "
    "Behavioral State Thresholds..."
)

# ---------------------------------------------------------
# PRIMARY FEATURES IN SCOPE
# ---------------------------------------------------------

primary_features = [
    "points_in_wallet",
    "avg_transaction_value",
    "avg_frequency_login_days",
    "value_density"
]

# ---------------------------------------------------------
# OUTPUT CONTAINERS
# ---------------------------------------------------------

distribution_summary_records = []
behavioral_state_thresholds = {}
state_mapping_logic = {}

# ---------------------------------------------------------
# CREATE OUTPUT DIRECTORY
# ---------------------------------------------------------

plot_output_dir = "phase_1_distribution_plots"
os.makedirs(plot_output_dir, exist_ok=True)

print(f"Plot output directory created: {plot_output_dir}")

# ---------------------------------------------------------
# FEATURE PROCESSING LOOP
# ---------------------------------------------------------

for feature in primary_features:

    print(f"\nProcessing feature: {feature}")

    # -----------------------------------------------------
    # CLEAN FEATURE SERIES
    # -----------------------------------------------------

    feature_series = df[feature].dropna()

    # -----------------------------------------------------
    # COMPUTE DISTRIBUTION STATISTICS
    # -----------------------------------------------------

    feature_min = feature_series.min()
    feature_max = feature_series.max()

    q1 = feature_series.quantile(0.25)
    median = feature_series.quantile(0.50)
    q3 = feature_series.quantile(0.75)

    feature_skewness = skew(feature_series)

    # -----------------------------------------------------
    # DETERMINE DISTRIBUTION TYPE
    # -----------------------------------------------------

    if abs(feature_skewness) < 1:
        distribution_type = "symmetric"
    else:
        distribution_type = "skewed"

    # -----------------------------------------------------
    # DEFINE STATE THRESHOLDS
    # -----------------------------------------------------

    if distribution_type == "symmetric":

        low_threshold = q1
        high_threshold = q3

    else:

        # Dense-region emphasis for skewed distributions
        low_threshold = feature_series.quantile(0.40)
        high_threshold = feature_series.quantile(0.80)

    # -----------------------------------------------------
    # STORE THRESHOLD LOGIC
    # -----------------------------------------------------

    behavioral_state_thresholds[feature] = {
        "distribution_type": distribution_type,
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold)
    }

    # -----------------------------------------------------
    # DEFINE STATE LABELS
    # -----------------------------------------------------

    if feature == "avg_frequency_login_days":

        state_labels = [
            "inactive",
            "irregular",
            "consistent"
        ]

    elif feature == "value_density":

        state_labels = [
            "low",
            "moderate",
            "high"
        ]

    else:

        state_labels = [
            "low",
            "mid",
            "high"
        ]

    # -----------------------------------------------------
    # STATE MAPPING FUNCTION
    # -----------------------------------------------------

    def map_behavioral_state(
        value,
        lower_bound=low_threshold,
        upper_bound=high_threshold,
        labels=state_labels
    ):

        if value <= lower_bound:
            return labels[0]

        elif value <= upper_bound:
            return labels[1]

        else:
            return labels[2]

    # -----------------------------------------------------
    # APPLY STATE MAPPING
    # -----------------------------------------------------

    state_column_name = f"{feature}_state"

    df[state_column_name] = df[feature].apply(
        map_behavioral_state
    )

   # -----------------------------------------------------
   # STORE MAPPING LOGIC
   # -----------------------------------------------------

    state_mapping_logic[feature] = {
    "state_column": state_column_name,
    "labels": state_labels,
    "logic": {
        f"<= {low_threshold}": state_labels[0],
        f"> {low_threshold} and <= {high_threshold}": state_labels[1],
        f"> {high_threshold}": state_labels[2]
    }
}
    # -----------------------------------------------------
    # STORE DISTRIBUTION SUMMARY
    # -----------------------------------------------------

    distribution_summary_records.append({
        "feature": feature,
        "min": feature_min,
        "max": feature_max,
        "q1": q1,
        "median": median,
        "q3": q3,
        "skewness": feature_skewness,
        "distribution_type": distribution_type,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold
    })

    # -----------------------------------------------------
    # DISTRIBUTION VALIDATION PLOT
    # -----------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.histplot(
        feature_series,
        kde=True,
        bins=30
    )

    plt.axvline(
        low_threshold,
        linestyle="--",
        label="Low Threshold"
    )

    plt.axvline(
        high_threshold,
        linestyle="--",
        label="High Threshold"
    )

    plt.title(
        f"Distribution Validation Plot — {feature}"
    )

    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.legend()

    plt.tight_layout()

    plot_file_path = os.path.join(
        plot_output_dir,
        f"{feature}_distribution_plot.png"
    )

    plt.savefig(plot_file_path)

    plt.close()

    print(
        f"Distribution plot saved: {plot_file_path}"
    )


# ---------------------------------------------------------
# CREATE SUMMARY DATAFRAME
# ---------------------------------------------------------

distribution_summary_df = pd.DataFrame(
    distribution_summary_records
)

print("\nDistribution summary generation complete.")


# ---------------------------------------------------------
# PREVIEW GENERATED STATES
# ---------------------------------------------------------

generated_state_columns = [
    f"{feature}_state"
    for feature in primary_features
]

state_preview_columns = (
    primary_features +
    generated_state_columns
)

print("\nBehavioral state preview:")

print(
    df[state_preview_columns]
    .head()
)



# ---------------------------------------------------------
# SAVE DISTRIBUTION SUMMARY
# ---------------------------------------------------------

distribution_summary_output_path = os.path.join(
    output_directory,
    "distribution_summary.csv"
)

distribution_summary_df.to_csv(
    distribution_summary_output_path,
    index=False
)

# ---------------------------------------------------------
# SAVE STATE-MAPPED DATASET
# ---------------------------------------------------------

state_dataset_output_path = os.path.join(
    output_directory,
    "state_constructed_dataset.csv"
)

df.to_csv(
    state_dataset_output_path,
    index=False
)

# ---------------------------------------------------------
# SAVE THRESHOLD LOGIC
# ---------------------------------------------------------

threshold_output_path = os.path.join(
    output_directory,
    "behavioral_state_thresholds.json"
)

with open(threshold_output_path, "w") as json_file:
    json.dump(
        behavioral_state_thresholds,
        json_file,
        indent=4
    )

# ---------------------------------------------------------
# SAVE STATE MAPPING LOGIC
# ---------------------------------------------------------

mapping_logic_output_path = os.path.join(
    output_directory,
    "state_mapping_logic.json"
)

with open(mapping_logic_output_path, "w") as json_file:
    json.dump(
        state_mapping_logic,
        json_file,
        indent=4
    )

print("\nPhase outputs generated successfully.")


# =========================================================
# PHASE 1 COMPLETE
# =========================================================

print("Phase 1 execution complete.")