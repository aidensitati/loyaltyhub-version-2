# =========================================================
# ECA STAGE: Diagnostic Exploration
# PHASE 2.5: Behavioral Archetype Assignment
# =========================================================
# Stage Purpose:
# To convert validated high-signal behavioral variables
# into structured behavioral systems and identify
# interpretable churn archetypes.
#
# Phase Purpose:
# To assign churned users into psychologically and
# behaviorally interpretable archetypes based on
# dominant state structures, behavioral imbalance
# patterns, and compound interaction zones identified
# in Phase 2.
#
# Inputs from Previous Phases:
# - state_constructed_dataset.csv
# - *_state behavioral columns
# - Phase 2 interaction interpretations
#
# Outputs Generated:
# - Behavioral archetype assignments
# - Archetype reasoning labels
# - Archetype priority mapping
# - Archetype-enriched dataset
# =========================================================


# ---------------------------------------------------------
# PHASE OBJECTIVES
# ---------------------------------------------------------
# Objective 1:
# Assign users into dominant behavioral archetypes
#
# Objective 2:
# Preserve behavioral contradiction structures
#
# Objective 3:
# Generate archetype-enriched dataset for
# downstream transition modeling


# ---------------------------------------------------------
# LOAD DEPENDENCIES
# ---------------------------------------------------------

import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# LOAD / ACCESS DATA
# ---------------------------------------------------------

data_path = (
    r"C:\Users\hp\Exploratory Data Analysis\CSV files"
    r"\LoyaltyHub V2\Outputs\phase_1_outputs"
    r"\state_constructed_dataset.csv"
)

df = pd.read_csv(data_path)

print("State-constructed dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# ---------------------------------------------------------
# DEFINE OUTPUT DIRECTORY
# ---------------------------------------------------------

output_directory = (
    r"C:\Users\hp\Exploratory Data Analysis\CSV files"
    r"\LoyaltyHub V2\Outputs"
    r"\phase_2_5_outputs"
)

os.makedirs(output_directory, exist_ok=True)

print(f"Output directory created: {output_directory}")


# =========================================================
# OBJECTIVE 1: ASSIGN USERS INTO DOMINANT
# BEHAVIORAL ARCHETYPES
# =========================================================

print(
    "\nExecuting Objective 1: "
    "Assign Users Into Dominant Behavioral Archetypes..."
)


# ---------------------------------------------------------
# INITIALIZE ARCHETYPE COLUMNS
# ---------------------------------------------------------

df["behavioral_archetype"] = "unclassified"

df["archetype_priority"] = np.nan

df["archetype_reason"] = "no_rule_triggered"


# ---------------------------------------------------------
# ARCHETYPE 1:
# ECONOMICALLY DISCONNECTED USERS
# PRIORITY = 1
# ---------------------------------------------------------

economically_disconnected_mask = (

    (
        (df["avg_transaction_value_state"] == "low")
        &
        (df["value_density_state"] == "high")
    )

    |

    (
        (df["avg_transaction_value_state"] == "high")
        &
        (df["value_density_state"] == "low")
    )
)

df.loc[
    economically_disconnected_mask,
    "behavioral_archetype"
] = "economically_disconnected_user"

df.loc[
    economically_disconnected_mask,
    "archetype_priority"
] = 1

df.loc[
    economically_disconnected_mask,
    "archetype_reason"
] = (
    "transaction_value_and_perceived_value_misalignment"
)


# ---------------------------------------------------------
# ARCHETYPE 2:
# DORMANT FORMER-VALUE USERS
# PRIORITY = 2
# ---------------------------------------------------------

dormant_former_value_mask = (

    (df["behavioral_archetype"] == "unclassified")

    &

    (df["value_density_state"] == "high")

    &

    (df["avg_frequency_login_days_state"] == "inactive")
)

df.loc[
    dormant_former_value_mask,
    "behavioral_archetype"
] = "dormant_former_value_user"

df.loc[
    dormant_former_value_mask,
    "archetype_priority"
] = 2

df.loc[
    dormant_former_value_mask,
    "archetype_reason"
] = (
    "high_perceived_value_with_behavioral_collapse"
)


# ---------------------------------------------------------
# ARCHETYPE 3:
# ACTIVE HIGH-VALUE CHURNERS
# PRIORITY = 3
# ---------------------------------------------------------

active_high_value_mask = (

    (df["behavioral_archetype"] == "unclassified")

    &

    (df["value_density_state"] == "high")

    &

    (
        (df["avg_frequency_login_days_state"]
         == "consistent")

        |

        (df["avg_transaction_value_state"]
         == "high")
    )
)

df.loc[
    active_high_value_mask,
    "behavioral_archetype"
] = "active_high_value_churner"

df.loc[
    active_high_value_mask,
    "archetype_priority"
] = 3

df.loc[
    active_high_value_mask,
    "archetype_reason"
] = (
    "high_value_extraction_despite_eventual_churn"
)


# ---------------------------------------------------------
# ARCHETYPE 4:
# PARTIAL ADOPTION STALLERS
# PRIORITY = 4
# ---------------------------------------------------------

partial_adoption_mask = (

    (df["behavioral_archetype"] == "unclassified")

    &

    (df["avg_transaction_value_state"] == "low")

    &

    (df["value_density_state"] == "moderate")
)

df.loc[
    partial_adoption_mask,
    "behavioral_archetype"
] = "partial_adoption_staller"

df.loc[
    partial_adoption_mask,
    "archetype_priority"
] = 4

df.loc[
    partial_adoption_mask,
    "archetype_reason"
] = (
    "moderate_value_realization_without_behavioral_expansion"
)


# ---------------------------------------------------------
# ARCHETYPE 5:
# STABLE MID-LEVEL DRIFTERS
# PRIORITY = 5
# ---------------------------------------------------------

mid_level_drifter_mask = (

    (df["behavioral_archetype"] == "unclassified")

    &

    (df["points_in_wallet_state"] == "mid")

    &

    (df["avg_transaction_value_state"] == "mid")

    &

    (df["avg_frequency_login_days_state"] == "irregular")

    &

    (
        (df["value_density_state"] == "low")

        |

        (df["value_density_state"] == "moderate")
    )
)

df.loc[
    mid_level_drifter_mask,
    "behavioral_archetype"
] = "stable_mid_level_drifter"

df.loc[
    mid_level_drifter_mask,
    "archetype_priority"
] = 5

df.loc[
    mid_level_drifter_mask,
    "archetype_reason"
] = (
    "behavioral_instability_without_extreme_failure"
)


# =========================================================
# OBJECTIVE 2: PRESERVE BEHAVIORAL
# CONTRADICTION STRUCTURES
# =========================================================

print(
    "\nExecuting Objective 2: "
    "Preserve Behavioral Contradiction Structures..."
)


# ---------------------------------------------------------
# GENERATE ARCHETYPE DISTRIBUTION SUMMARY
# ---------------------------------------------------------

archetype_summary_df = (

    df["behavioral_archetype"]
    .value_counts(dropna=False)
    .reset_index()
)

archetype_summary_df.columns = [
    "behavioral_archetype",
    "frequency"
]

archetype_summary_df[
    "prevalence_percentage"
] = (

    archetype_summary_df["frequency"]
    / len(df)

) * 100


print("\nBehavioral archetype summary generated.")


# =========================================================
# OBJECTIVE 3: GENERATE ARCHETYPE-ENRICHED
# DATASET FOR DOWNSTREAM TRANSITION MODELING
# =========================================================

print(
    "\nExecuting Objective 3: "
    "Generate Archetype-Enriched Dataset..."
)


# ---------------------------------------------------------
# SORT DATASET BY ARCHETYPE PRIORITY
# ---------------------------------------------------------

df = df.sort_values(
    by=[
        "archetype_priority",
        "behavioral_archetype"
    ],
    na_position="last"
)


# ---------------------------------------------------------
# RESET INDEX
# ---------------------------------------------------------

df = df.reset_index(drop=True)


# ---------------------------------------------------------
# DISPLAY ARCHETYPE COUNTS
# ---------------------------------------------------------

print("\nBehavioral Archetype Distribution:")

print(archetype_summary_df)


# ---------------------------------------------------------
# IDENTIFY UNCLASSIFIED USERS
# ---------------------------------------------------------

unclassified_count = (

    df[
        df["behavioral_archetype"]
        == "unclassified"
    ]
    .shape[0]
)

print(f"\nUnclassified users: {unclassified_count}")


# ---------------------------------------------------------
# SAVE ARCHETYPE-ENRICHED DATASET
# ---------------------------------------------------------

archetype_dataset_output_path = os.path.join(
    output_directory,
    "behavioral_archetype_dataset.csv"
)

df.to_csv(
    archetype_dataset_output_path,
    index=False
)

print(
    "\nArchetype-enriched dataset saved successfully."
)


# ---------------------------------------------------------
# SAVE ARCHETYPE SUMMARY
# ---------------------------------------------------------

summary_output_path = os.path.join(
    output_directory,
    "behavioral_archetype_summary.csv"
)

archetype_summary_df.to_csv(
    summary_output_path,
    index=False
)

print(
    "Behavioral archetype summary saved successfully."
)


# ---------------------------------------------------------
# PHASE OUTPUT
# ---------------------------------------------------------

print("\nPhase outputs generated successfully.")


# =========================================================
# PHASE 2.5 COMPLETE
# =========================================================

print("Phase 2.5 execution complete.")