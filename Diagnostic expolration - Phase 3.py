# =========================================================
# ECA STAGE: DIAGNOSTIC EXPLORATION
# PHASE 3: BEHAVIORAL PROGRESSION & DECAY TOPOLOGY
# =========================================================

# Stage Purpose:
# To reconstruct the latent behavioral progression structure embedded
# within the churned population by analyzing lifecycle maturity,
# behavioral decay, engagement instability, and archetype positioning
# using cross-sectional temporal residue variables.

# Phase Purpose:
# To infer probable behavioral deterioration structures, lifecycle
# positioning, instability concentration regions, and latent churn
# progression tendencies across churn archetypes without imposing
# deterministic behavioral paths.

# Inputs from Previous Phases:
# - Behavioral state system from Phase 1
# - Derived behavioral metrics from Phase 1
# - Behavioral topology outputs from Phase 2
# - Archetype assignments from Phase 2.5

# Outputs Generated:
# - Archetype lifecycle positioning profiles
# - Behavioral decay signature profiles
# - Structural instability profiles
# - Latent progression topology mappings
# - Visualization plots and topology heatmaps

# =========================================================


# ---------------------------------------------------------
# PHASE OBJECTIVES
# ---------------------------------------------------------

# Objective 1:
# Map Archetype Lifecycle Positioning

# Objective 2:
# Map Behavioral Decay Signatures

# Objective 3:
# Measure Structural Behavioral Instability

# Objective 4:
# Construct Latent Progression Topology


# ---------------------------------------------------------
# LOAD DEPENDENCIES
# ---------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------------------------------------------------------
# LOAD / ACCESS DATA
# ---------------------------------------------------------

dataset_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub V2\Outputs\phase_2_5_outputs\behavioral_archetype_dataset.csv"

df = pd.read_csv(dataset_path)

base_output_path = os.path.join(
    os.path.dirname(dataset_path),
    "ECA_PHASE_3_BEHAVIORAL_PROGRESSION_AND_DECAY_TOPOLOGY_outputs"
)

os.makedirs(base_output_path, exist_ok=True)

plots_path = os.path.join(base_output_path, "plots")
os.makedirs(plots_path, exist_ok=True)


# =========================================================
# OBJECTIVE 1: MAP ARCHETYPE LIFECYCLE POSITIONING
# =========================================================

print("Executing Objective 1: Map Archetype Lifecycle Positioning...")

lifecycle_positioning = (
    df.groupby("behavioral_archetype")
    .agg(
        median_tenure_days=("tenure_days", "median"),
        mean_tenure_days=("tenure_days", "mean"),
        tenure_std=("tenure_days", "std"),
        median_points_per_day=("points_per_day", "median"),
        median_engagement_efficiency=("engagement_efficiency", "median"),
        median_transaction_value=("avg_transaction_value", "median"),
        archetype_population=("behavioral_archetype", "count")
    )
    .reset_index()
)

lifecycle_positioning["lifecycle_maturity_rank"] = (
    lifecycle_positioning["median_tenure_days"]
    .rank(method="dense", ascending=True)
)

lifecycle_positioning = lifecycle_positioning.sort_values(
    by="median_tenure_days",
    ascending=True
)

lifecycle_positioning_output_path = os.path.join(
    base_output_path,
    "archetype_lifecycle_positioning.csv"
)

lifecycle_positioning.to_csv(
    lifecycle_positioning_output_path,
    index=False
)


# ---------------------------------------------------------
# OBJECTIVE 1 PLOTS
# ---------------------------------------------------------

plt.figure(figsize=(14, 7))

sns.boxplot(
    data=df,
    x="behavioral_archetype",
    y="tenure_days"
)

plt.xticks(rotation=20)
plt.title("Archetype Lifecycle Distribution")
plt.xlabel("Behavioral Archetype")
plt.ylabel("Tenure Days")

plt.tight_layout()

plt.savefig(
    os.path.join(
        plots_path,
        "archetype_lifecycle_distribution_boxplot.png"
    )
)

plt.close()


plt.figure(figsize=(14, 7))

sns.violinplot(
    data=df,
    x="behavioral_archetype",
    y="tenure_days"
)

plt.xticks(rotation=20)
plt.title("Archetype Lifecycle Density Distribution")
plt.xlabel("Behavioral Archetype")
plt.ylabel("Tenure Days")

plt.tight_layout()

plt.savefig(
    os.path.join(
        plots_path,
        "archetype_lifecycle_distribution_violinplot.png"
    )
)

plt.close()


# =========================================================
# OBJECTIVE 2: MAP BEHAVIORAL DECAY SIGNATURES
# =========================================================

print("Executing Objective 2: Map Behavioral Decay Signatures...")

decay_signatures = (
    df.groupby("behavioral_archetype")
    .agg(
        median_days_since_last_login=("days_since_last_login", "median"),
        median_recency_score=("recency_score", "median"),
        median_engagement_efficiency=("engagement_efficiency", "median"),
        median_time_per_day=("time_per_day", "median"),
        inactivity_std=("days_since_last_login", "std")
    )
    .reset_index()
)

state_decay_distribution = (
    df.groupby(
        [
            "behavioral_archetype",
            "avg_frequency_login_days_state"
        ]
    )
    .size()
    .reset_index(name="frequency")
)

state_decay_distribution["state_percentage"] = (
    state_decay_distribution.groupby("behavioral_archetype")["frequency"]
    .transform(lambda x: (x / x.sum()) * 100)
)

decay_signatures_output_path = os.path.join(
    base_output_path,
    "behavioral_decay_signatures.csv"
)

decay_signatures.to_csv(
    decay_signatures_output_path,
    index=False
)


# ---------------------------------------------------------
# OBJECTIVE 2 PLOTS
# ---------------------------------------------------------

decay_heatmap_data = decay_signatures.set_index(
    "behavioral_archetype"
)

plt.figure(figsize=(12, 7))

sns.heatmap(
    decay_heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="Reds"
)

plt.title("Behavioral Decay Signature Heatmap")
plt.xlabel("Decay Metrics")
plt.ylabel("Behavioral Archetype")

plt.tight_layout()

plt.savefig(
    os.path.join(
        plots_path,
        "behavioral_decay_heatmap.png"
    )
)

plt.close()


plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=df,
    x="recency_score",
    y="engagement_efficiency",
    hue="behavioral_archetype",
    alpha=0.7
)

plt.title("Recency vs Engagement Efficiency")
plt.xlabel("Recency Score")
plt.ylabel("Engagement Efficiency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        plots_path,
        "recency_vs_engagement_scatterplot.png"
    )
)

plt.close()


# =========================================================
# OBJECTIVE 3: MEASURE STRUCTURAL BEHAVIORAL INSTABILITY
# =========================================================

print("Executing Objective 3: Measure Structural Behavioral Instability...")

df["high_spend_low_value_flag"] = np.where(
    (
        (df["avg_transaction_value_state"] == "high") &
        (df["value_density_state"] == "low")
    ),
    1,
    0
)

df["low_spend_high_value_flag"] = np.where(
    (
        (df["avg_transaction_value_state"] == "low") &
        (df["value_density_state"] == "high")
    ),
    1,
    0
)

df["inactive_high_value_flag"] = np.where(
    (
        (df["avg_frequency_login_days_state"] == "inactive") &
        (df["value_density_state"] == "high")
    ),
    1,
    0
)

df["high_engagement_low_monetization_flag"] = np.where(
    (
        (df["avg_frequency_login_days_state"] == "consistent") &
        (df["avg_transaction_value_state"] == "low")
    ),
    1,
    0
)

df["irregular_mid_behavior_flag"] = np.where(
    (
        (df["avg_frequency_login_days_state"] == "irregular") &
        (df["avg_transaction_value_state"] == "mid")
    ),
    1,
    0
)

instability_columns = [
    "high_spend_low_value_flag",
    "low_spend_high_value_flag",
    "inactive_high_value_flag",
    "high_engagement_low_monetization_flag",
    "irregular_mid_behavior_flag"
]

structural_instability_profiles = (
    df.groupby("behavioral_archetype")[instability_columns]
    .mean()
    .reset_index()
)

structural_instability_profiles[instability_columns] = (
    structural_instability_profiles[instability_columns] * 100
)

structural_instability_profiles["overall_instability_score"] = (
    structural_instability_profiles[instability_columns]
    .mean(axis=1)
)

instability_output_path = os.path.join(
    base_output_path,
    "structural_instability_profiles.csv"
)

structural_instability_profiles.to_csv(
    instability_output_path,
    index=False
)


# ---------------------------------------------------------
# OBJECTIVE 3 PLOTS
# ---------------------------------------------------------

instability_heatmap_data = (
    structural_instability_profiles
    .set_index("behavioral_archetype")[instability_columns]
)

plt.figure(figsize=(12, 7))

sns.heatmap(
    instability_heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="Reds"
)

plt.title("Structural Behavioral Instability Heatmap")
plt.xlabel("Behavioral Contradiction Indicators")
plt.ylabel("Behavioral Archetype")

plt.tight_layout()

plt.savefig(
    os.path.join(
        plots_path,
        "structural_instability_heatmap.png"
    )
)

plt.close()


plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=df,
    x="avg_transaction_value",
    y="value_density",
    hue="behavioral_archetype",
    alpha=0.7
)

plt.title("Transaction Value vs Value Density")
plt.xlabel("Average Transaction Value")
plt.ylabel("Value Density")

plt.tight_layout()

plt.savefig(
    os.path.join(
        plots_path,
        "value_transaction_instability_scatterplot.png"
    )
)

plt.close()


# =========================================================
# OBJECTIVE 4: CONSTRUCT LATENT PROGRESSION TOPOLOGY
# =========================================================

print("Executing Objective 4: Construct Latent Progression Topology...")

topology_features = [
    "median_tenure_days",
    "median_points_per_day",
    "median_transaction_value"
]

topology_decay_features = [
    "median_days_since_last_login",
    "median_recency_score",
    "median_engagement_efficiency",
    "median_time_per_day"
]

topology_instability_features = [
    "overall_instability_score"
]

topology_dataframe = lifecycle_positioning.merge(
    decay_signatures,
    on="behavioral_archetype",
    how="left",
    suffixes=("_lifecycle", "_decay")
)

topology_dataframe = topology_dataframe.merge(
    structural_instability_profiles[
        [
            "behavioral_archetype",
            "overall_instability_score"
        ]
    ],
    on="behavioral_archetype",
    how="left"
)

topology_analysis_columns = (
    topology_features +
    topology_decay_features +
    topology_instability_features
)

# FIX: ensure no missing columns break scaling step
missing_cols = [col for col in topology_analysis_columns if col not in topology_dataframe.columns]

if len(missing_cols) > 0:
    print(f"Warning: Missing columns removed from topology matrix: {missing_cols}")
    topology_analysis_columns = [col for col in topology_analysis_columns if col in topology_dataframe.columns]

scaler = StandardScaler()

scaled_topology_matrix = scaler.fit_transform(
    topology_dataframe[topology_analysis_columns]
)

pca_model = PCA(n_components=2)

topology_coordinates = pca_model.fit_transform(
    scaled_topology_matrix
)

topology_dataframe["topology_dimension_1"] = topology_coordinates[:, 0]
topology_dataframe["topology_dimension_2"] = topology_coordinates[:, 1]

topology_dataframe["relative_progression_position"] = (
    topology_dataframe["topology_dimension_1"]
    .rank(method="dense")
)

latent_progression_output_path = os.path.join(
    base_output_path,
    "latent_progression_topology.csv"
)

topology_dataframe.to_csv(
    latent_progression_output_path,
    index=False
)

# ---------------------------------------------------------
# OBJECTIVE 4 PLOTS
# ---------------------------------------------------------

plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=topology_dataframe,
    x="topology_dimension_1",
    y="topology_dimension_2",
    hue="behavioral_archetype",
    s=200
)

for _, row in topology_dataframe.iterrows():

    plt.text(
        row["topology_dimension_1"],
        row["topology_dimension_2"],
        row["behavioral_archetype"],
        fontsize=9
    )

plt.title("Archetype Progression Topology Map")
plt.xlabel("Topology Dimension 1")
plt.ylabel("Topology Dimension 2")

plt.tight_layout()

plt.savefig(
    os.path.join(
        plots_path,
        "archetype_progression_topology_map.png"
    )
)

plt.close()


topology_heatmap_data = topology_dataframe.set_index(
    "behavioral_archetype"
)[topology_analysis_columns]

plt.figure(figsize=(14, 8))

sns.heatmap(
    topology_heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Latent Behavioral Progression Topology")
plt.xlabel("Behavioral Progression Metrics")
plt.ylabel("Behavioral Archetype")

plt.tight_layout()

plt.savefig(
    os.path.join(
        plots_path,
        "latent_progression_topology_heatmap.png"
    )
)

plt.close()


# ---------------------------------------------------------
# PHASE OUTPUT
# ---------------------------------------------------------

progression_summary_dictionary = (
    topology_dataframe[
        [
            "behavioral_archetype",
            "relative_progression_position",
            "overall_instability_score"
        ]
    ]
    .to_dict(orient="records")
)

archetype_positioning_dictionary = (
    lifecycle_positioning
    .to_dict(orient="records")
)

instability_mapping_dictionary = (
    structural_instability_profiles
    .to_dict(orient="records")
)

with open(
    os.path.join(
        base_output_path,
        "progression_summary_dictionary.json"
    ),
    "w"
) as file:

    json.dump(
        progression_summary_dictionary,
        file,
        indent=4
    )

with open(
    os.path.join(
        base_output_path,
        "archetype_positioning_dictionary.json"
    ),
    "w"
) as file:

    json.dump(
        archetype_positioning_dictionary,
        file,
        indent=4
    )

with open(
    os.path.join(
        base_output_path,
        "instability_mapping_dictionary.json"
    ),
    "w"
) as file:

    json.dump(
        instability_mapping_dictionary,
        file,
        indent=4
    )

print("Phase outputs generated successfully.")


# =========================================================
# PHASE 3 COMPLETE
# =========================================================

print("Phase 3 execution complete.")