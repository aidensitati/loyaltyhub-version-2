# =========================================================
# ECA STAGE: Diagnostic Exploration
# PHASE 2: Behavioral State & Compound Topology Mapping
# =========================================================
# Stage Purpose:
# To convert validated high-signal behavioral variables
# into structured state systems, quantify how churn risk
# evolves across those states, and isolate the directional
# transitions and compound behavioral conditions that
# precede churn.
#
# Phase Purpose:
# To map the structural organization of churned users
# across both individual behavioral states and compound
# state combinations in order to identify dominant churn
# archetypes, concentrated behavioral regions, and
# high-density interaction zones.
#
# Inputs from Previous Phases:
# - state_constructed_dataset.csv
# - *_state columns from Phase 1
# - Behavioral state thresholds
# - State mapping logic
#
# Outputs Generated:
# - State prevalence profiling dataset
# - Compound interaction matrices
# - Dominant state rankings
# - Behavioral concentration metrics
# - Archetype dictionaries
# - High-density interaction mappings
# - Visualization plots
# =========================================================


# ---------------------------------------------------------
# PHASE OBJECTIVES
# ---------------------------------------------------------
# Objective 1:
# Map prevalence distribution across all behavioral states
#
# Objective 2:
# Measure behavioral concentration and state dominance
#
# Objective 3:
# Construct pairwise compound state interaction grids
#
# Objective 4:
# Measure compound state prevalence and co-occurrence density
#
# Objective 5:
# Identify dominant churn archetypes and concentrated
# behavioral zones


# ---------------------------------------------------------
# LOAD DEPENDENCIES
# ---------------------------------------------------------

import os
import json
from itertools import combinations

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# LOAD / ACCESS DATA
# ---------------------------------------------------------

data_path = (
    r"C:\Users\hp\Exploratory Data Analysis\CSV files"
    r"\LoyaltyHub V2\Outputs\phase_1_outputs"
    r"\state_constructed_dataset.csv"
)

df = pd.read_csv(data_path)

print("Phase 1 dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# ---------------------------------------------------------
# DEFINE OUTPUT DIRECTORIES
# ---------------------------------------------------------

dataset_directory = os.path.dirname(data_path)

phase_output_directory = os.path.join(
    dataset_directory,
    "ECA PHASE 2 — BEHAVIORAL STATE & COMPOUND TOPOLOGY MAPPING outputs"
)

state_plot_directory = os.path.join(
    phase_output_directory,
    "state_prevalence_plots"
)

interaction_plot_directory = os.path.join(
    phase_output_directory,
    "interaction_heatmaps"
)

os.makedirs(phase_output_directory, exist_ok=True)
os.makedirs(state_plot_directory, exist_ok=True)
os.makedirs(interaction_plot_directory, exist_ok=True)

print(f"Phase output directory created: {phase_output_directory}")


# ---------------------------------------------------------
# IDENTIFY STATE FEATURES
# ---------------------------------------------------------

state_features = [
    column
    for column in df.columns
    if column.endswith("_state")
]

print("\nDetected state features:")

for feature in state_features:
    print(f"- {feature}")


# ---------------------------------------------------------
# INITIALIZE GLOBAL CONTAINERS
# ---------------------------------------------------------

total_observations = len(df)

state_prevalence_records = []

dominant_state_mapping = {}

behavioral_concentration_metrics = {}

compound_interaction_records = []

high_density_interaction_zones = {}

archetype_mapping = {}


# =========================================================
# OBJECTIVE 1: Map Prevalence Distribution Across
# All Behavioral States
# =========================================================

print(
    "\nExecuting Objective 1: "
    "Map Prevalence Distribution Across All Behavioral States..."
)

# ---------------------------------------------------------
# STATE PREVALENCE PROFILING
# ---------------------------------------------------------

for state_feature in state_features:

    print(f"\nProcessing state feature: {state_feature}")

    state_counts = (
        df[state_feature]
        .value_counts(dropna=False)
        .reset_index()
    )

    state_counts.columns = [
        "state",
        "frequency"
    ]

    state_counts["feature"] = state_feature

    state_counts["prevalence_percentage"] = (
        state_counts["frequency"]
        / total_observations
    ) * 100

    state_counts = state_counts.sort_values(
        by="prevalence_percentage",
        ascending=False
    )

    state_counts[
        "cumulative_prevalence_contribution"
    ] = (
        state_counts["prevalence_percentage"]
        .cumsum()
    )

    for _, row in state_counts.iterrows():

        state_prevalence_records.append({

            "feature": row["feature"],
            "state": row["state"],
            "frequency": int(row["frequency"]),
            "prevalence_percentage": (
                row["prevalence_percentage"]
            ),
            "cumulative_prevalence_contribution": (
                row[
                    "cumulative_prevalence_contribution"
                ]
            )
        })

print("\nState prevalence profiling completed.")


# ---------------------------------------------------------
# CREATE STATE PREVALENCE DATAFRAME
# ---------------------------------------------------------

state_prevalence_summary_df = pd.DataFrame(
    state_prevalence_records
)

print("\nState prevalence summary table created.")


# =========================================================
# OBJECTIVE 2: Measure Behavioral Concentration
# And State Dominance
# =========================================================

print(
    "\nExecuting Objective 2: "
    "Measure Behavioral Concentration And State Dominance..."
)

# ---------------------------------------------------------
# COMPUTE CONCENTRATION METRICS
# ---------------------------------------------------------

for state_feature in state_features:

    feature_subset = (
        state_prevalence_summary_df[
            state_prevalence_summary_df["feature"]
            == state_feature
        ]
        .copy()
    )

    feature_subset = feature_subset.sort_values(
        by="prevalence_percentage",
        ascending=False
    )

    prevalence_variance = (
        feature_subset[
            "prevalence_percentage"
        ]
        .var()
    )

    concentration_spread = (
        feature_subset[
            "prevalence_percentage"
        ].max()
        -
        feature_subset[
            "prevalence_percentage"
        ].min()
    )

    prevalence_distribution = (
        feature_subset[
            "prevalence_percentage"
        ] / 100
    )

    entropy = -np.sum(
        prevalence_distribution
        * np.log2(prevalence_distribution)
    )

    dominant_threshold = (
        feature_subset[
            "prevalence_percentage"
        ].quantile(0.66)
    )

    sparse_threshold = (
        feature_subset[
            "prevalence_percentage"
        ].quantile(0.33)
    )

    def classify_state(prevalence):

        if prevalence >= dominant_threshold:
            return "dominant_state"

        elif prevalence <= sparse_threshold:
            return "sparse_state"

        else:
            return "moderate_state"

    feature_subset[
        "dominance_classification"
    ] = (
        feature_subset[
            "prevalence_percentage"
        ]
        .apply(classify_state)
    )

    for _, row in feature_subset.iterrows():

        row_mask = (
            (
                state_prevalence_summary_df["feature"]
                == row["feature"]
            )
            &
            (
                state_prevalence_summary_df["state"]
                == row["state"]
            )
        )

        state_prevalence_summary_df.loc[
            row_mask,
            "dominance_classification"
        ] = row[
            "dominance_classification"
        ]

    dominant_states = (
        feature_subset[
            feature_subset[
                "dominance_classification"
            ]
            == "dominant_state"
        ]["state"]
        .tolist()
    )

    dominant_state_mapping[
        state_feature
    ] = dominant_states

    behavioral_concentration_metrics[
        state_feature
    ] = {

        "prevalence_variance": float(
            prevalence_variance
        ),

        "concentration_spread": float(
            concentration_spread
        ),

        "entropy": float(entropy)
    }

print("\nBehavioral concentration metrics computed.")


# =========================================================
# OBJECTIVE 3: Construct Pairwise Compound
# State Interaction Grids
# =========================================================

print(
    "\nExecuting Objective 3: "
    "Construct Pairwise Compound State Interaction Grids..."
)

# ---------------------------------------------------------
# GENERATE FEATURE PAIRS
# ---------------------------------------------------------

feature_pairs = list(
    combinations(state_features, 2)
)

# ---------------------------------------------------------
# BUILD INTERACTION MATRICES
# ---------------------------------------------------------

for feature_a, feature_b in feature_pairs:

    print(
        f"\nProcessing interaction pair: "
        f"{feature_a} × {feature_b}"
    )

    interaction_summary = (
        df
        .groupby([feature_a, feature_b])
        .size()
        .reset_index(name="joint_frequency")
    )

    interaction_summary[
        "joint_prevalence_percentage"
    ] = (
        interaction_summary[
            "joint_frequency"
        ]
        / total_observations
    ) * 100

    interaction_summary[
        "feature_pair"
    ] = f"{feature_a}__{feature_b}"

    for _, row in interaction_summary.iterrows():

        compound_interaction_records.append({

            "feature_pair": row["feature_pair"],

            "feature_a": feature_a,
            "feature_b": feature_b,

            "state_a": row[feature_a],
            "state_b": row[feature_b],

            "joint_frequency": int(
                row["joint_frequency"]
            ),

            "joint_prevalence_percentage": (
                row[
                    "joint_prevalence_percentage"
                ]
            )
        })

print("\nCompound interaction grids constructed.")


# ---------------------------------------------------------
# CREATE INTERACTION DATAFRAME
# ---------------------------------------------------------

compound_interaction_df = pd.DataFrame(
    compound_interaction_records
)

print("\nCompound interaction dataframe created.")


# =========================================================
# OBJECTIVE 4: Measure Compound State Prevalence
# And Co-Occurrence Density
# =========================================================

print(
    "\nExecuting Objective 4: "
    "Measure Compound State Prevalence "
    "And Co-Occurrence Density..."
)

# ---------------------------------------------------------
# COMPUTE CO-OCCURRENCE DENSITY
# ---------------------------------------------------------

for feature_a, feature_b in feature_pairs:

    feature_pair_name = (
        f"{feature_a}__{feature_b}"
    )

    interaction_subset = (
        compound_interaction_df[
            compound_interaction_df[
                "feature_pair"
            ]
            == feature_pair_name
        ]
        .copy()
    )

    interaction_subset[
        "marginal_prevalence_a"
    ] = interaction_subset[
        "state_a"
    ].map(

        state_prevalence_summary_df[
            state_prevalence_summary_df[
                "feature"
            ] == feature_a
        ]
        .set_index("state")[
            "prevalence_percentage"
        ]
    )

    interaction_subset[
        "marginal_prevalence_b"
    ] = interaction_subset[
        "state_b"
    ].map(

        state_prevalence_summary_df[
            state_prevalence_summary_df[
                "feature"
            ] == feature_b
        ]
        .set_index("state")[
            "prevalence_percentage"
        ]
    )

    interaction_subset[
        "expected_joint_prevalence"
    ] = (
        interaction_subset[
            "marginal_prevalence_a"
        ]
        *
        interaction_subset[
            "marginal_prevalence_b"
        ]
    ) / 100

    interaction_subset[
        "co_occurrence_density_ratio"
    ] = (
        interaction_subset[
            "joint_prevalence_percentage"
        ]
        /
        interaction_subset[
            "expected_joint_prevalence"
        ]
    )

    density_threshold = (
        interaction_subset[
            "co_occurrence_density_ratio"
        ]
        .quantile(0.75)
    )

    sparse_threshold = (
        interaction_subset[
            "co_occurrence_density_ratio"
        ]
        .quantile(0.25)
    )

    def classify_density(ratio):

        if ratio >= density_threshold:
            return "overrepresented_zone"

        elif ratio <= sparse_threshold:
            return "sparse_zone"

        else:
            return "moderate_zone"

    interaction_subset[
        "density_classification"
    ] = (
        interaction_subset[
            "co_occurrence_density_ratio"
        ]
        .apply(classify_density)
    )

    for _, row in interaction_subset.iterrows():

        row_mask = (

            (
                compound_interaction_df[
                    "feature_pair"
                ]
                == row["feature_pair"]
            )

            &

            (
                compound_interaction_df[
                    "state_a"
                ]
                == row["state_a"]
            )

            &

            (
                compound_interaction_df[
                    "state_b"
                ]
                == row["state_b"]
            )
        )

        compound_interaction_df.loc[
            row_mask,
            "marginal_prevalence_a"
        ] = row[
            "marginal_prevalence_a"
        ]

        compound_interaction_df.loc[
            row_mask,
            "marginal_prevalence_b"
        ] = row[
            "marginal_prevalence_b"
        ]

        compound_interaction_df.loc[
            row_mask,
            "expected_joint_prevalence"
        ] = row[
            "expected_joint_prevalence"
        ]

        compound_interaction_df.loc[
            row_mask,
            "co_occurrence_density_ratio"
        ] = row[
            "co_occurrence_density_ratio"
        ]

        compound_interaction_df.loc[
            row_mask,
            "density_classification"
        ] = row[
            "density_classification"
        ]

    high_density_subset = (
        interaction_subset[
            interaction_subset[
                "density_classification"
            ]
            == "overrepresented_zone"
        ]
    )

    high_density_interaction_zones[
        feature_pair_name
    ] = (

        high_density_subset[
            ["state_a", "state_b"]
        ]
        .to_dict(orient="records")
    )

print("\nCompound state density metrics computed.")


# =========================================================
# OBJECTIVE 5: Identify Dominant Churn Archetypes
# And Concentrated Behavioral Zones
# =========================================================

print(
    "\nExecuting Objective 5: "
    "Identify Dominant Churn Archetypes "
    "And Concentrated Behavioral Zones..."
)

# ---------------------------------------------------------
# BUILD ARCHETYPE MAPPING
# ---------------------------------------------------------

for feature_pair in high_density_interaction_zones:

    dense_regions = (
        high_density_interaction_zones[
            feature_pair
        ]
    )

    archetype_mapping[
        feature_pair
    ] = {

        "dominant_compound_states": (
            dense_regions
        ),

        "interaction_density_count": (
            len(dense_regions)
        )
    }

print("\nBehavioral archetype mapping completed.")


# ---------------------------------------------------------
# GENERATE STATE PREVALENCE PLOTS
# ---------------------------------------------------------

print("\nGenerating state prevalence plots...")

for state_feature in state_features:

    plot_subset = (
        state_prevalence_summary_df[
            state_prevalence_summary_df[
                "feature"
            ]
            == state_feature
        ]
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=plot_subset,
        x="state",
        y="prevalence_percentage"
    )

    plt.title(
        f"Behavioral State Prevalence — "
        f"{state_feature}"
    )

    plt.xlabel("Behavioral State")

    plt.ylabel(
        "Prevalence Percentage"
    )

    plt.tight_layout()

    plot_output_path = os.path.join(
        state_plot_directory,
        f"{state_feature}_prevalence_plot.png"
    )

    plt.savefig(plot_output_path)

    plt.close()

    print(f"Saved plot: {plot_output_path}")


# ---------------------------------------------------------
# GENERATE INTERACTION HEATMAPS
# ---------------------------------------------------------

print("\nGenerating interaction heatmaps...")

for feature_a, feature_b in feature_pairs:

    feature_pair_name = (
        f"{feature_a}__{feature_b}"
    )

    interaction_subset = (
        compound_interaction_df[
            compound_interaction_df[
                "feature_pair"
            ]
            == feature_pair_name
        ]
    )

    heatmap_matrix = (
        interaction_subset
        .pivot(
            index="state_a",
            columns="state_b",
            values="joint_prevalence_percentage"
        )
    )

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".2f"
    )

    plt.title(
        f"Compound State Prevalence Heatmap\n"
        f"{feature_a} × {feature_b}"
    )

    plt.xlabel(feature_b)

    plt.ylabel(feature_a)

    plt.tight_layout()

    heatmap_output_path = os.path.join(
        interaction_plot_directory,
        f"{feature_pair_name}_heatmap.png"
    )

    plt.savefig(heatmap_output_path)

    plt.close()

    print(
        f"Saved heatmap: "
        f"{heatmap_output_path}"
    )


# ---------------------------------------------------------
# PHASE OUTPUT
# ---------------------------------------------------------

# ---------------------------------------------------------
# SAVE STATE PREVALENCE SUMMARY
# ---------------------------------------------------------

state_prevalence_output_path = os.path.join(
    phase_output_directory,
    "state_prevalence_summary.csv"
)

state_prevalence_summary_df.to_csv(
    state_prevalence_output_path,
    index=False
)

# ---------------------------------------------------------
# SAVE COMPOUND INTERACTION MATRIX
# ---------------------------------------------------------

compound_interaction_output_path = os.path.join(
    phase_output_directory,
    "compound_interaction_matrix.csv"
)

compound_interaction_df.to_csv(
    compound_interaction_output_path,
    index=False
)

# ---------------------------------------------------------
# SAVE DOMINANT STATE MAPPING
# ---------------------------------------------------------

dominant_state_output_path = os.path.join(
    phase_output_directory,
    "dominant_state_mapping.json"
)

with open(
    dominant_state_output_path,
    "w"
) as json_file:

    json.dump(
        dominant_state_mapping,
        json_file,
        indent=4
    )

# ---------------------------------------------------------
# SAVE BEHAVIORAL CONCENTRATION METRICS
# ---------------------------------------------------------

concentration_metrics_output_path = os.path.join(
    phase_output_directory,
    "behavioral_concentration_metrics.json"
)

with open(
    concentration_metrics_output_path,
    "w"
) as json_file:

    json.dump(
        behavioral_concentration_metrics,
        json_file,
        indent=4
    )

# ---------------------------------------------------------
# SAVE HIGH-DENSITY INTERACTION ZONES
# ---------------------------------------------------------

high_density_output_path = os.path.join(
    phase_output_directory,
    "high_density_interaction_zones.json"
)

with open(
    high_density_output_path,
    "w"
) as json_file:

    json.dump(
        high_density_interaction_zones,
        json_file,
        indent=4
    )

# ---------------------------------------------------------
# SAVE ARCHETYPE MAPPING
# ---------------------------------------------------------

archetype_output_path = os.path.join(
    phase_output_directory,
    "behavioral_archetype_mapping.json"
)

with open(
    archetype_output_path,
    "w"
) as json_file:

    json.dump(
        archetype_mapping,
        json_file,
        indent=4
    )

print("\nPhase outputs generated successfully.")


# =========================================================
# PHASE 2 COMPLETE
# =========================================================

print("Phase 2 execution complete.")