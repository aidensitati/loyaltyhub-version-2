# =========================================================
# ECA STAGE: INFERENTIAL STRUCTURAL ALIGNMENT & BEHAVIORAL RESIDUE VALIDATION
# =========================================================
# Purpose:
# To numerically reconstruct behavioral structures discovered
# during Diagnostic Exploration and determine whether those
# structures persist, weaken, reverse, or disappear across
# other churned and retained membership populations.
#
# Dataset/Input:
# LoyaltyHub Updated.csv
#
# Output:
# Comprehensive inferential structural comparison outputs,
# similarity matrices, topology metrics, instability metrics,
# contradiction preservation metrics, and structural summaries.
# =========================================================

# ---------------------------------------------------------
# STAGE OBJECTIVES
# ---------------------------------------------------------

# Objective 1:
# Numerically reconstruct diagnostic behavioral structures
# into measurable statistical reference signatures.

# Objective 2:
# Construct equivalent behavioral residue representations
# for all remaining population groups.

# Objective 3:
# Measure structural alignment, divergence, preservation,
# and breakdown across populations.

# Objective 4:
# Quantify behavioral instability, contradiction concentration,
# topology displacement, and archetype persistence.

# Objective 5:
# Classify the structural nature of each diagnostic mechanism.

# Objective 6:
# Generate a consolidated inferential comparison summary table.

# ---------------------------------------------------------
# LOAD DEPENDENCIES
# ---------------------------------------------------------

import os
import json
import warnings

import numpy as np
import pandas as pd

from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine
from scipy.spatial.distance import jensenshannon
from scipy.spatial.distance import mahalanobis

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.covariance import EmpiricalCovariance

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

dataset_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\LoyaltyHub V2\LoyaltyHub Updated.csv"

df = pd.read_csv(dataset_path)

base_output_path = os.path.join(
    os.path.dirname(dataset_path),
    "ECA_INFERENTIAL_STRUCTURAL_ALIGNMENT_AND_BEHAVIORAL_RESIDUE_VALIDATION_outputs"
)

os.makedirs(base_output_path, exist_ok=True)

# ---------------------------------------------------------
# ENGINEER REQUIRED FEATURES
# ---------------------------------------------------------

print("Engineering required features...")

df["joining_date"] = pd.to_datetime(
    df["joining_date"],
    errors="coerce"
)

reference_date = df["joining_date"].max()

df["tenure_days"] = (
    reference_date - df["joining_date"]
).dt.days

df["tenure_days"] = df["tenure_days"].fillna(1)
df["tenure_days"] = df["tenure_days"].replace(0, 1)

df["points_in_wallet"] = pd.to_numeric(
    df["points_in_wallet"],
    errors="coerce"
).fillna(0)

df["avg_transaction_value"] = pd.to_numeric(
    df["avg_transaction_value"],
    errors="coerce"
).fillna(0)

df["avg_time_spent"] = pd.to_numeric(
    df["avg_time_spent"],
    errors="coerce"
).fillna(0)

df["days_since_last_login"] = pd.to_numeric(
    df["days_since_last_login"],
    errors="coerce"
).fillna(0)

df["avg_frequency_login_days"] = pd.to_numeric(
    df["avg_frequency_login_days"],
    errors="coerce"
).fillna(1)

df["points_per_day"] = (
    df["points_in_wallet"] /
    (df["tenure_days"] + 1)
)

df["value_density"] = (
    df["avg_transaction_value"] /
    (df["avg_time_spent"] + 1)
)

df["engagement_efficiency"] = (
    df["points_per_day"] /
    (df["avg_time_spent"] + 1)
)

df["recency_score"] = (
    1 / (df["days_since_last_login"] + 1)
)

df["logins_per_day"] = (
    1 / (df["avg_frequency_login_days"] + 1)
)

df["time_per_day"] = (
    df["avg_time_spent"] /
    (df["tenure_days"] + 1)
)

df["wallet_balance"] = df["points_in_wallet"]

# ---------------------------------------------------------
# BUILD STATE VARIABLES
# ---------------------------------------------------------

print("Constructing behavioral state system...")

behavioral_features = [
    "avg_transaction_value",
    "value_density",
    "engagement_efficiency",
    "points_per_day",
    "wallet_balance",
    "logins_per_day"
]

for feature in behavioral_features:

    try:

        df[f"{feature}_state"] = pd.qcut(
            df[feature],
            q=3,
            labels=["low", "mid", "high"],
            duplicates="drop"
        )

    except Exception:

        df[f"{feature}_state"] = "mid"

df["avg_frequency_login_days_state"] = pd.cut(
    df["avg_frequency_login_days"],
    bins=[-np.inf, 5, 15, np.inf],
    labels=["consistent", "irregular", "inactive"]
)

# ---------------------------------------------------------
# BUILD INSTABILITY FLAGS
# ---------------------------------------------------------

print("Constructing instability indicators...")

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

# ---------------------------------------------------------
# BUILD POPULATION SEGMENTS (FIXED NORMALIZATION LAYER)
# ---------------------------------------------------------

print("Constructing population segments...")

# -----------------------------
# NORMALIZE TEXT COLUMNS
# -----------------------------
df["membership_category"] = (
    df["membership_category"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# convert "gold membership" → "gold"
df["membership_category"] = df["membership_category"].str.replace(
    " membership", "",
    regex=False
).str.strip()

# -----------------------------
# NORMALIZE CHURN (1/0 → yes/no)
# -----------------------------
df["churn"] = pd.to_numeric(df["churn"], errors="coerce")

df["churn"] = df["churn"].map({
    1: "yes",
    0: "no"
})

# drop invalid churn rows
df = df[df["churn"].isin(["yes", "no"])].copy()

# -----------------------------
# POPULATION BUILDING
# -----------------------------
population_groups = {}

membership_groups = [
    "gold",
    "silver",
    "premium",
    "platinum",
    "basic",
    "no"
]

churn_map = {
    "churned": "yes",
    "retained": "no"
}

for membership in membership_groups:
    for churn_label, churn_value in churn_map.items():

        population_name = f"{membership}_{churn_label}"

        subset = df[
            (df["membership_category"] == membership) &
            (df["churn"] == churn_value)
        ].copy()

        population_groups[population_name] = subset

# DEBUG CHECK
for k, v in population_groups.items():
    print(k, len(v))

# =========================================================
# OBJECTIVE 1: RECONSTRUCT DIAGNOSTIC STRUCTURES
# =========================================================

print("Executing Objective 1: Reconstruct Diagnostic Structures...")

reference_population = pd.concat(
    [
        population_groups["gold_churned"],
        population_groups["silver_churned"]
    ]
)

state_columns = [
    "avg_transaction_value_state",
    "value_density_state",
    "engagement_efficiency_state",
    "points_per_day_state",
    "wallet_balance_state",
    "avg_frequency_login_days_state"
]

reference_statistics = {}

for column in state_columns:

    state_distribution = (
        reference_population[column]
        .value_counts(normalize=True)
    )

    reference_statistics[f"{column}_entropy"] = entropy(
        state_distribution
    )

    reference_statistics[f"{column}_gini"] = (
        1 - np.sum(np.square(state_distribution))
    )

    reference_statistics[f"{column}_dispersion"] = (
        state_distribution.std()
    )

contradiction_metrics = {}

for flag in instability_columns:

    contradiction_metrics[f"{flag}_prevalence"] = (
        reference_population[flag].mean()
    )

reference_statistics.update(contradiction_metrics)

reference_behavioral_columns = [
    "points_per_day",
    "engagement_efficiency",
    "avg_transaction_value",
    "value_density",
    "recency_score",
    "time_per_day"
]

reference_population_clean = (
    reference_population[
        reference_behavioral_columns
    ]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

reference_behavioral_vector = (
    reference_population_clean
    .median()
)

reference_covariance = None

reference_covariance_matrix = np.eye(
    len(reference_behavioral_columns)
)

if len(reference_population_clean) > 0:

    reference_covariance = EmpiricalCovariance()

    reference_covariance.fit(
        reference_population_clean
    )

    reference_covariance_matrix = (
        reference_covariance.covariance_
    )

else:

    print(
        "Warning: Reference population behavioral matrix is empty. "
        "Using identity covariance matrix fallback."
    )

    reference_covariance_matrix = np.eye(
        len(reference_behavioral_columns)
    )
reference_statistics_dataframe = pd.DataFrame(
    [reference_statistics]
)

reference_statistics_dataframe.to_csv(
    os.path.join(
        base_output_path,
        "reference_behavioral_statistics.csv"
    ),
    index=False
)

# =========================================================
# OBJECTIVE 2: CONSTRUCT POPULATION STRUCTURES (FIXED)
# =========================================================

print("Executing Objective 2: Construct Population Structures...")

population_profile_results = []
population_signature_vectors = {}

for population_name, population_df in population_groups.items():

    # STRICT SAFETY: skip truly empty groups (prevents downstream NaN collapse)
    if population_df is None or len(population_df) == 0:
        continue

    # Keep minimal viable signal requirement (prevents noisy micro-populations)
    if len(population_df) < 3:
        continue

    profile_dictionary = {}

    profile_dictionary["population"] = population_name
    profile_dictionary["population_size"] = len(population_df)

    # -----------------------------
    # STATE DISTRIBUTIONS (SAFE)
    # -----------------------------
    for column in state_columns:

        if column not in population_df.columns:
            continue

        distribution = population_df[column].value_counts(normalize=True)

        # Guard against empty distributions (prevents entropy NaN collapse)
        if len(distribution) == 0:
            profile_dictionary[f"{column}_entropy"] = 0
            profile_dictionary[f"{column}_gini"] = 0
            profile_dictionary[f"{column}_dispersion"] = 0
            continue

        profile_dictionary[f"{column}_entropy"] = entropy(distribution)
        profile_dictionary[f"{column}_gini"] = 1 - np.sum(np.square(distribution))
        profile_dictionary[f"{column}_dispersion"] = distribution.std()

    # -----------------------------
    # INSTABILITY FLAGS (SAFE)
    # -----------------------------
    for flag in instability_columns:

        if flag not in population_df.columns:
            profile_dictionary[f"{flag}_prevalence"] = 0
        else:
            profile_dictionary[f"{flag}_prevalence"] = population_df[flag].mean()

    # -----------------------------
    # BEHAVIORAL VECTOR (CRITICAL FIX)
    # -----------------------------
    behavioral_subset = population_df[
        [
            "points_per_day",
            "engagement_efficiency",
            "avg_transaction_value",
            "value_density",
            "recency_score",
            "time_per_day"
        ]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    # fallback if sparse population
    if len(behavioral_subset) == 0:
        behavioral_vector = np.zeros(6)
    else:
        behavioral_vector = behavioral_subset.median().values

    population_signature_vectors[population_name] = behavioral_vector

    profile_dictionary["behavioral_vector_norm"] = np.linalg.norm(behavioral_vector)

    profile_dictionary["instability_index"] = (
        population_df[instability_columns].mean(axis=1).mean()
        if len(population_df) > 0 else 0
    )

    population_profile_results.append(profile_dictionary)

population_profiles_dataframe = pd.DataFrame(population_profile_results)

population_profiles_dataframe.to_csv(
    os.path.join(base_output_path, "population_structural_profiles.csv"),
    index=False
)

# =========================================================
# OBJECTIVE 3: MEASURE STRUCTURAL ALIGNMENT
# =========================================================

print("Executing Objective 3: Measure Structural Alignment...")

reference_vector = reference_behavioral_vector.values

alignment_results = []

inverse_covariance_matrix = np.linalg.pinv(
    reference_covariance_matrix
)

for population_name, vector in population_signature_vectors.items():

    cosine_distance = cosine(
        reference_vector,
        vector
    )

    euclidean_distance = np.linalg.norm(
        reference_vector - vector
    )

    try:

        mahalanobis_distance = mahalanobis(
            reference_vector,
            vector,
            inverse_covariance_matrix
        )

    except Exception:

        mahalanobis_distance = np.nan

    try:

        js_divergence = jensenshannon(
            np.abs(reference_vector) + 1e-10,
            np.abs(vector) + 1e-10
        )

    except Exception:

        js_divergence = np.nan

    wasserstein_metric = wasserstein_distance(
        reference_vector,
        vector
    )

    alignment_results.append(
        {
            "population": population_name,
            "cosine_distance": cosine_distance,
            "euclidean_distance": euclidean_distance,
            "mahalanobis_distance": mahalanobis_distance,
            "jensen_shannon_divergence": js_divergence,
            "wasserstein_distance": wasserstein_metric
        }
    )

alignment_dataframe = pd.DataFrame(
    alignment_results
)

alignment_dataframe.to_csv(
    os.path.join(
        base_output_path,
        "population_alignment_metrics.csv"
    ),
    index=False
)

# =========================================================
# OBJECTIVE 4: TOPOLOGY & INSTABILITY ANALYSIS
# =========================================================

print("Executing Objective 4: Topology & Instability Analysis...")

topology_features = [
    "points_per_day",
    "engagement_efficiency",
    "avg_transaction_value",
    "value_density",
    "recency_score",
    "time_per_day"
]

topology_dataset = []

for population_name, population_df in population_groups.items():

    if len(population_df) < 5:
        continue

    temporary_dataframe = population_df.copy()
    temporary_dataframe["population"] = population_name

    topology_dataset.append(
        temporary_dataframe[topology_features + ["population"]]
    )

# ---------------------------------------------------------
# FIX: SAFE CONCAT WITH GUARDED EMPTY HANDLING
# ---------------------------------------------------------

if len(topology_dataset) == 0:

    print("Warning: No valid populations for topology analysis. Skipping PCA pipeline.")

    topology_dataframe = pd.DataFrame(columns=topology_features + ["population"])
    topology_summary = pd.DataFrame(
        columns=["population", "centroid_x", "centroid_y", "dispersion_x", "dispersion_y"]
    )
    silhouette_dataframe = pd.DataFrame(columns=["silhouette_score"])

else:

    topology_dataframe = pd.concat(topology_dataset, axis=0, ignore_index=True)

    scaler = StandardScaler()
    scaled_topology_matrix = scaler.fit_transform(
        topology_dataframe[topology_features]
    )

    pca_model = PCA(n_components=2)
    topology_coordinates = pca_model.fit_transform(scaled_topology_matrix)

    topology_dataframe["topology_dimension_1"] = topology_coordinates[:, 0]
    topology_dataframe["topology_dimension_2"] = topology_coordinates[:, 1]

    topology_summary = (
        topology_dataframe.groupby("population")
        .agg(
            centroid_x=("topology_dimension_1", "mean"),
            centroid_y=("topology_dimension_2", "mean"),
            dispersion_x=("topology_dimension_1", "std"),
            dispersion_y=("topology_dimension_2", "std")
        )
        .reset_index()
    )

    silhouette_dataframe = pd.DataFrame()

    if topology_dataframe["population"].nunique() > 1:

        silhouette_value = silhouette_score(
            scaled_topology_matrix,
            topology_dataframe["population"]
        )

        silhouette_dataframe = pd.DataFrame(
            [{"silhouette_score": silhouette_value}]
        )

# ---------------------------------------------------------
# SAFE OUTPUT WRITING
# ---------------------------------------------------------

topology_summary.to_csv(
    os.path.join(base_output_path, "topology_summary.csv"),
    index=False
)

silhouette_dataframe.to_csv(
    os.path.join(base_output_path, "silhouette_metrics.csv"),
    index=False
)

# =========================================================
# OBJECTIVE 5: CLASSIFY STRUCTURAL MECHANISMS
# =========================================================

print("Executing Objective 5: Classify Structural Mechanisms...")

classification_results = []

for _, row in alignment_dataframe.iterrows():

    population_name = row["population"]

    instability_score = (
        population_profiles_dataframe[
            population_profiles_dataframe["population"] == population_name
        ]["instability_index"]
        .values[0]
    )

    cosine_distance_metric = row["cosine_distance"]

    if cosine_distance_metric < 0.10:

        structure_classification = (
            "generalized_churn_structure"
        )

    elif cosine_distance_metric < 0.25:

        structure_classification = (
            "membership_conditioned_structure"
        )

    elif instability_score < 0.05:

        structure_classification = (
            "retained_suppression_structure"
        )

    else:

        structure_classification = (
            "diagnostic_only_structure"
        )

    classification_results.append(
        {
            "population": population_name,
            "instability_index": instability_score,
            "cosine_distance": cosine_distance_metric,
            "structural_classification": structure_classification
        }
    )

classification_dataframe = pd.DataFrame(
    classification_results
)

classification_dataframe.to_csv(
    os.path.join(
        base_output_path,
        "structural_classifications.csv"
    ),
    index=False
)


# ---------------------------------------------------------
# STAGE OUTPUT
# ---------------------------------------------------------

print("Stage outputs successfully saved.")

# =======================# =========================================================
# OBJECTIVE 6: GENERATE INFERENTIAL SUMMARY
# =========================================================

print("Executing Objective 6: Generate Inferential Summary...")

# ---------------------------------------------------------
# SAFETY: ENSURE REQUIRED KEY EXISTS ACROSS ALL TABLES
# ---------------------------------------------------------

required_key = "population"

if population_profiles_dataframe.empty:
    population_profiles_dataframe = pd.DataFrame(columns=[required_key])

if alignment_dataframe.empty:
    alignment_dataframe = pd.DataFrame(columns=[required_key])

if classification_dataframe.empty:
    classification_dataframe = pd.DataFrame(columns=[required_key])

# ---------------------------------------------------------
# SAFE MERGE (GUARDED AGAINST EMPTY INPUTS)
# ---------------------------------------------------------

inferential_summary_dataframe = population_profiles_dataframe.merge(
    alignment_dataframe,
    on="population",
    how="left"
)

inferential_summary_dataframe = inferential_summary_dataframe.merge(
    classification_dataframe,
    on="population",
    how="left"
)

inferential_summary_dataframe = inferential_summary_dataframe.merge(
    topology_summary,
    on="population",
    how="left"
)

# ---------------------------------------------------------
# OUTPUT WRITING
# ---------------------------------------------------------

inferential_summary_output_path = os.path.join(
    base_output_path,
    "inferential_structural_comparison_summary.csv"
)

inferential_summary_dataframe.to_csv(
    inferential_summary_output_path,
    index=False
)

inferential_summary_dictionary = inferential_summary_dataframe.to_dict(orient="records")

with open(
    os.path.join(
        base_output_path,
        "inferential_structural_comparison_summary.json"
    ),
    "w"
) as file:

    json.dump(
        inferential_summary_dictionary,
        file,
        indent=4
    )
# ==================================
# INFERENTIAL STRUCTURAL ALIGNMENT & BEHAVIORAL RESIDUE VALIDATION STAGE COMPLETE
# =========================================================

print(
    "Inferential Structural Alignment & Behavioral Residue Validation stage execution complete."
)

# =========================================================
# OBJECTIVE 7: INFERENTIAL STRUCTURE SYNTHESIS LAYER (FIXED)
# =========================================================

print("Executing Objective 7: Inferential Structure Synthesis Layer...")

# -----------------------------
# STEP 1: NORMALIZATION (ROBUST FIX)
# -----------------------------

final_table = inferential_summary_dataframe.copy()

# core metrics actually used downstream
metrics_to_normalize = [
    "cosine_distance",
    "euclidean_distance",
    "mahalanobis_distance",
    "jensen_shannon_divergence",
    "wasserstein_distance",
    "instability_index"
]

# ensure numeric safety
for col in metrics_to_normalize:
    if col in final_table.columns:
        final_table[col] = pd.to_numeric(final_table[col], errors="coerce").fillna(0)

# normalize safely (no missing columns allowed in scoring stage)
for col in metrics_to_normalize:
    if col not in final_table.columns:
        final_table[col] = 0  # hard fallback

    min_val = final_table[col].min()
    max_val = final_table[col].max()

    if max_val - min_val == 0:
        final_table[f"{col}_norm"] = 0
    else:
        final_table[f"{col}_norm"] = (
            (final_table[col] - min_val) / (max_val - min_val)
        )

# -----------------------------
# STEP 2: STRUCTURAL INTEGRITY SCORE
# -----------------------------

final_table["structural_integrity_score"] = (
    (1 - final_table["cosine_distance_norm"]) * 0.30 +
    (1 - final_table["euclidean_distance_norm"]) * 0.15 +
    (1 - final_table["mahalanobis_distance_norm"]) * 0.15 +
    (1 - final_table["instability_index_norm"]) * 0.25 +
    (1 - final_table["jensen_shannon_divergence_norm"]) * 0.10 +
    (1 - final_table["wasserstein_distance_norm"]) * 0.05
)

# -----------------------------
# STEP 3: RANKING
# -----------------------------

final_table["structural_rank"] = final_table[
    "structural_integrity_score"
].rank(ascending=False)

final_table = final_table.sort_values(
    "structural_integrity_score",
    ascending=False
)

# -----------------------------
# STEP 4: ARCHETYPE CLASSIFICATION (UNCHANGED LOGIC)
# -----------------------------

conditions = [
    (final_table["structural_integrity_score"] > 0.75),
    (final_table["cosine_distance"] < 0.10) & (final_table["instability_index"] > 0.08),
    (final_table["instability_index"] < 0.05) & (final_table["cosine_distance"] > 0.25),
    (final_table["dispersion_x"].fillna(0) + final_table["dispersion_y"].fillna(0) > 3.0),
    (final_table["cosine_distance"] > 0.20) & (final_table["instability_index"] > 0.08)
]

labels = [
    "stable_retention_core",
    "latent_churn_equivalent",
    "behaviorally_inconsistent_retention",
    "structurally_fragmented_group",
    "high_instability_low_divergence"
]

final_table["final_archetype"] = np.select(
    conditions,
    labels,
    default="diagnostic_noise_group"
)

# -----------------------------
# OUTPUTS
# -----------------------------

final_output_path = os.path.join(
    base_output_path,
    "final_inferential_structure_synthesis.csv"
)

final_table.to_csv(final_output_path, index=False)

print("Objective 7 complete. Final synthesis layer generated.")