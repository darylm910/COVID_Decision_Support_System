import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PROJECT_DIR = PROJECT_ROOT / "SOURCE_PROJECT"

DATA_DIR = SOURCE_PROJECT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORT_DIR = SOURCE_PROJECT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

MODEL_DIR = SOURCE_PROJECT_DIR / "models"

DATA_FILE = PROCESSED_DATA_DIR / "covid_analysis_dataset.csv"


# ---------------------------------------
# Prediction constants
# ---------------------------------------

PREDICTION_FEATURES = [
    "median_age",
    "life_expectancy",
    "total_cases_per_million",
    "gdp_per_capita",
    "hospital_beds_per_thousand",
    "people_fully_vaccinated_per_hundred",
    "stringency_index",
]

DATASET_DISPLAY_COLUMNS = [
    "country",
    "continent",
    "total_cases_per_million",
    "total_deaths_per_million",
    "people_fully_vaccinated_per_hundred",
    "median_age",
    "life_expectancy",
    "hospital_beds_per_thousand",
    "gdp_per_capita",
    "stringency_index",
]

# ---------------------------------------
# Display labels
# ---------------------------------------

FEATURE_LABELS = {
    "country": "Country",
    "code": "Country Code",
    "continent": "Continent",

    "total_cases_per_million": "Cases per Million",
    "total_deaths_per_million": "Deaths per Million",
    "people_fully_vaccinated_per_hundred": "Fully Vaccinated (%)",
    "stringency_index": "Government Stringency Index",

    "median_age": "Median Age",
    "life_expectancy": "Life Expectancy (Years)",
    "hospital_beds_per_thousand": "Hospital Beds per 1,000",
    "gdp_per_capita": "GDP per Capita (USD)",
    "population_density": "Population Density (people/km²)",

    "continent_Asia": "Asia",
    "continent_Europe": "Europe",
    "continent_North America": "North America",
    "continent_Oceania": "Oceania",
    "continent_South America": "South America",

    "Cluster": "Country Cluster",
}

FEATURE_STEPS = {
    "median_age": 1.0,
    "life_expectancy": 1.0,
    "gdp_per_capita": 1000.0,
    "hospital_beds_per_thousand": 0.5,
    "total_cases_per_million": 10000.0,
    "people_fully_vaccinated_per_hundred": 1.0,
    "stringency_index": 5.0,
}


FEATURE_FORMATS = {
    "median_age": "%.1f",
    "life_expectancy": "%.1f",
    "gdp_per_capita": "%.0f",
    "hospital_beds_per_thousand": "%.1f",
    "total_cases_per_million": "%.0f",
    "people_fully_vaccinated_per_hundred": "%.1f",
    "stringency_index": "%.1f",
}

FEATURE_DISPLAY_FORMATS = {
    "median_age": "{:,.1f}",
    "life_expectancy": "{:,.1f}",
    "gdp_per_capita": "{:,.0f}",
    "hospital_beds_per_thousand": "{:,.1f}",
    "total_cases_per_million": "{:,.0f}",
    "total_deaths_per_million": "{:,.0f}",
    "people_fully_vaccinated_per_hundred": "{:,.1f}%",
    "stringency_index": "{:,.1f}",
}

FEATURE_LIMITS = {
    "median_age": (10.0, 70.0),
    "life_expectancy": (30.0, 100.0),
    "gdp_per_capita": (0.0, 150000.0),
    "hospital_beds_per_thousand": (0.0, 20.0),
    "total_cases_per_million": (0.0, 1000000.0),
    "people_fully_vaccinated_per_hundred": (0.0, 100.0),
    "stringency_index": (0.0, 100.0),
}

# ---------------------------------------
# Cluster Profiles
# ---------------------------------------

CLUSTER_LABELS = {
    0: "Cluster 0: Younger, Lower-Income",
    1: "Cluster 1: Older, High-Income",
    2: "Cluster 2: Dense Urban Jurisdictions",
    3: "Cluster 3: Mixed Middle-Income",
}

REPORT_CLUSTER_ORDER = list(CLUSTER_LABELS.values())

CLUSTER_PROFILES = {
    0: {
        "short_label": "Younger, Lower-Income Countries",
        "description": """
Countries in this cluster generally have the youngest populations,
lowest GDP per capita, shortest life expectancy, fewest hospital beds,
and the lowest vaccination rates. They also experienced the lowest
reported COVID-19 case and mortality rates.
"""
    },
    1: {
        "short_label": "Older, High-Income Countries",
        "description": """
Countries in this cluster have older populations, higher income levels,
longer life expectancy, and more developed healthcare infrastructure.
They reported the highest COVID-19 case rates and mortality, reflecting
both older populations and extensive disease surveillance.
"""
    },
    2: {
        "short_label": "Dense Urban Jurisdictions",
        "description": """
This small cluster consists of three highly urbanized, high-income
jurisdictions characterized by exceptionally high population density,
very high vaccination coverage, and the highest reported case rates.
Despite widespread transmission, mortality remained somewhat lower than
in Cluster 1.
"""
    },
    3: {
        "short_label": "Mixed Middle-Income Countries",
        "description": """
Countries in this cluster exhibit intermediate demographic, economic,
and healthcare characteristics. COVID-19 case and mortality rates
generally fell between those observed in the younger, lower-income
countries and the older, wealthier countries.
"""
    },
}

def get_cluster_display(cluster):
    """Return a formatted cluster label for display."""
    cluster = int(cluster)
    return f"Cluster {cluster} – {CLUSTER_PROFILES[cluster]['short_label']}"

# ---------------------------------------
# Model loaders
# ---------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "random_forest_model.joblib")


@st.cache_resource
def load_scaler():
    return joblib.load(MODEL_DIR / "standard_scaler.joblib")


@st.cache_resource
def load_feature_names():
    return joblib.load(MODEL_DIR / "feature_names.joblib")


@st.cache_resource
def load_prediction_assets():
    return load_model(), load_scaler(), load_feature_names()

MODEL_COMPARISON_FILE = (
    TABLE_DIR / "Table2_Model_Comparison.csv"
)


@st.cache_data
def load_model_comparison():
    return pd.read_csv(MODEL_COMPARISON_FILE)

# ---------------------------------------
# Data loaders
# ---------------------------------------

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find data file: {DATA_FILE}")

    return pd.read_csv(DATA_FILE)


def get_country_list(df):
    return sorted(df["country"].dropna().unique())

@st.cache_data
def load_cluster_assignments():
    """Load country cluster assignments."""
    path = PROCESSED_DATA_DIR / "covid_cluster_assignments.csv"
    return pd.read_csv(path)
  
def get_country_profile(df, country):
    return df[df["country"] == country].iloc[0]


def add_log_mortality(df):
    df = df.copy()
    df["log_deaths_per_million"] = np.log10(
        df["total_deaths_per_million"] + 1
    )
    return df
  
def format_feature_value(feature, value):
    """
    Format a feature value for display in the dashboard.
    """
    if feature in FEATURE_DISPLAY_FORMATS:
        return FEATURE_DISPLAY_FORMATS[feature].format(value)

    return value



# ---------------------------------------
# Create Cluster Summary
# ---------------------------------------

@st.cache_data
def create_cluster_summary(df):
    """
    Create summary statistics for each K-Means cluster.

    Returns a DataFrame containing the median value of each key feature
    used to characterize the clusters.
    """

    return (
        df.groupby("Cluster")
        .agg(
            Countries=("country", "count"),
            median_age=("median_age", "median"),
            gdp_per_capita=("gdp_per_capita", "median"),
            life_expectancy=("life_expectancy", "median"),
            hospital_beds_per_thousand=(
                "hospital_beds_per_thousand",
                "median",
            ),
            people_fully_vaccinated_per_hundred=(
                "people_fully_vaccinated_per_hundred",
                "median",
            ),
            total_cases_per_million=(
                "total_cases_per_million",
                "median",
            ),
            total_deaths_per_million=(
                "total_deaths_per_million",
                "median",
            ),
            population_density=(
                "population_density",
                "median",
            ),
        )
        .round(1)
        .reset_index()
    )
    
# ---------------------------------------
# Prediction preprocessing
# ---------------------------------------

def prepare_prediction_input(
    input_values,
    country_profile,
    scaler,
    feature_names,
):
    """
    Prepare a single observation for prediction using the
    same preprocessing steps used during model training.
    """

    # Start with the numeric user inputs
    model_input = input_values.copy()

    # Add continent dummy variables
    for feature in feature_names:
        if feature.startswith("continent_"):
            continent_name = feature.replace("continent_", "")
            model_input[feature] = (
                1.0
                if country_profile["continent"] == continent_name
                else 0.0
            )

    # Create dataframe in model feature order
    input_df = pd.DataFrame([model_input])
    input_df = input_df[feature_names]

    # Scale only the numeric features
    scaled_numeric = scaler.transform(
        input_df[PREDICTION_FEATURES]
    )

    scaled_df = pd.DataFrame(
        scaled_numeric,
        columns=PREDICTION_FEATURES,
        index=input_df.index,
    )

    # Add continent dummy variables back
    for feature in feature_names:
        if feature.startswith("continent_"):
            scaled_df[feature] = input_df[feature]

    # Ensure final column order matches training
    scaled_df = scaled_df[feature_names]

    return scaled_df
  


# ---------------------------------------
# Filter Utility Function
# ---------------------------------------

def apply_filter(df, column, operator, value):
    if column == "None" or value == "":
        return df

    if operator == "=":
        return df[df[column] == float(value)]

    if operator == "!=":
        return df[df[column] != float(value)]

    if operator == ">":
        return df[df[column] > float(value)]

    if operator == ">=":
        return df[df[column] >= float(value)]

    if operator == "<":
        return df[df[column] < float(value)]

    if operator == "<=":
        return df[df[column] <= float(value)]

    if operator == "Contains":
        return df[
            df[column]
            .astype(str)
            .str.contains(value, case=False, na=False)
        ]

    if operator == "Starts with":
        return df[
            df[column]
            .astype(str)
            .str.startswith(value, na=False)
        ]

    if operator == "Ends with":
        return df[
            df[column]
            .astype(str)
            .str.endswith(value, na=False)
        ]

    if operator == "Equals":
        return df[
            df[column]
            .astype(str)
            .str.lower() == value.lower()
        ]

    return df
