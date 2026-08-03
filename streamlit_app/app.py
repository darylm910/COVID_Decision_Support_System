# ---------------------------------------
# Imports
# ---------------------------------------

import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

from dashboard_utils import (
    FIGURE_DIR,
    PREDICTION_FEATURES,
    DATASET_DISPLAY_COLUMNS,
    FEATURE_LABELS,
    FEATURE_STEPS,
    FEATURE_FORMATS,  
    FEATURE_LIMITS,
    CLUSTER_PROFILES,
    CLUSTER_LABELS, 
    REPORT_CLUSTER_ORDER,
    load_data,
    load_cluster_assignments,
    get_cluster_display,
    get_country_list,
    get_country_profile,
    create_cluster_summary,
    add_log_mortality,
    load_prediction_assets,
    prepare_prediction_input,
    load_model_comparison,
    apply_filter,
    prepare_dataset_for_model,
    format_feature_value
)

from shap_utils import (
    create_shap_explainer,
    compute_global_shap_values,
    compute_local_shap_values,
)

# ---------------------------------------
# Paths
# ---------------------------------------

APP_DIR = Path(__file__).resolve().parent
STYLE_PATH = APP_DIR / "assets" / "style.css"

# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="Explainable Machine Learning for COVID-19 Mortality",
    page_icon="🌍",
    layout="wide",
)

SHAP_DISPLAY_THRESHOLD = 10.0
SHAP_SUMMARY_THRESHOLD = 2.0

# ---------------------------------------
# Load Custom CSS
# ---------------------------------------

with STYLE_PATH.open(encoding="utf-8") as css:
    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True,
    )
# ---------------------------------------
# Load Data
# ---------------------------------------

try:
    df = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading data:\n{e}")
    st.stop()

model_df = load_data()
clusters_df = load_cluster_assignments()

df = model_df.merge(
    clusters_df[["country", "Cluster"]],
    on="country",
    how="left",
)

#st.write(df.columns.tolist())

model, scaler, feature_names = load_prediction_assets()

explainer = create_shap_explainer(model)

# ---------------------------------------
# Global SHAP
# ---------------------------------------

X_global = prepare_dataset_for_model(
    model_df,
    scaler,
    feature_names,
)

global_shap_values = compute_global_shap_values(
    explainer,
    X_global,
)

# Create a display version with friendly feature names
global_shap_display = shap.Explanation(
    values=global_shap_values.values,
    base_values=global_shap_values.base_values,
    data=global_shap_values.data,
    feature_names=[
        FEATURE_LABELS.get(feature, feature)
        for feature in feature_names
    ],
)

model_comparison = load_model_comparison()


# ---------------------------------------
# Sidebar Controls
# ---------------------------------------

#st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("## ⚙️ Dashboard Controls")

# Select the country displayed in the Country Explorer and Prediction tabs
country = st.sidebar.selectbox(
    "Country",
    get_country_list(df),
)

# Variables available in the Global Map selector
map_options = [
    "Cluster",  
    "total_deaths_per_million",
    "total_cases_per_million",
    "people_fully_vaccinated_per_hundred",
    "median_age",
    "life_expectancy",
    "hospital_beds_per_thousand",
    "gdp_per_capita",
    "stringency_index",
]

map_metric = st.sidebar.selectbox(
    "Map Variable",
    map_options,
    format_func=lambda x: FEATURE_LABELS.get(x, x),
)

# Apply a log transformation when displaying mortality values
use_log = st.sidebar.checkbox(
    "Use Log Scale for Mortality",
    value=True,
)

# ---------------------------------------
# Title
# ---------------------------------------

st.title("🌍 Explainable AI for Global COVID-19 Mortality")

st.markdown(
    """
Explore how demographic, healthcare, economic, vaccination, and public health factors influence **COVID-19 mortality across 239 countries** using machine learning, country clustering, and explainable AI.

### This application allows you to:

- **Explore** country profiles and global trends
- **Visualize** countries on an interactive world map
- **Compare** countries using K-Means clustering
- **Understand** model behavior with SHAP explainability
- **Predict** mortality under user-defined scenarios
- **Evaluate** Random Forest model performance
- **Browse** the complete analysis dataset
"""
)

st.divider()


# ---------------------------------------
# Tabs
# ---------------------------------------
TAB_NAMES = [
    "Country Explorer",
    "Global Map",
    "Clusters",
    "Explainability",
    "Model Performance",
    "Prediction",
    "Dataset",
]

(
    country_tab,
    map_tab,
    clusters_tab,
    explainability_tab,
    performance_tab,
    prediction_tab,
    dataset_tab,
) = st.tabs(TAB_NAMES)


# ---------------------------------------
# Country Detail Tab
# ---------------------------------------

with country_tab:
    # Get the selected country's row from the full dataset
    selected = get_country_profile(df, country)

    st.header(f"Country Profile: {country}")

    st.write(
        "View key demographic, healthcare, and COVID-19 metrics for the "
        "selected country."
    )

    # Show continent and cluster
    st.code(
        f"""Continent: {selected['continent']}
Country Cluster: {get_cluster_display(selected['Cluster'])}"""
    )

    # ---------------------------------------
    # Key summary metrics
    # ---------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Deaths per Million",
        format_feature_value(
            "total_deaths_per_million",
            selected["total_deaths_per_million"],
        ),
    )

    col2.metric(
        "Median Age",
        format_feature_value(
            "median_age",
            selected["median_age"],
        ),
    )

    col3.metric(
        "Fully Vaccinated",
        format_feature_value(
            "people_fully_vaccinated_per_hundred",
            selected["people_fully_vaccinated_per_hundred"],
        ),
    )

    st.divider()

    # ---------------------------------------
    # Country Detail Tables
    # ---------------------------------------

    left_col, right_col = st.columns(2)

    with left_col:

        st.subheader("Demographics and Healthcare")

        demographics_table = pd.DataFrame(
            {
                "Metric": [
                    "Median Age",
                    "Life Expectancy (Years)",
                    "GDP per Capita (USD)",
                    "Hospital Beds per 1,000",
                ],
                "Value": [
                    format_feature_value(
                        "median_age",
                        selected["median_age"],
                    ),
                    format_feature_value(
                        "life_expectancy",
                        selected["life_expectancy"],
                    ),
                    format_feature_value(
                        "gdp_per_capita",
                        selected["gdp_per_capita"],
                    ),
                    format_feature_value(
                        "hospital_beds_per_thousand",
                        selected["hospital_beds_per_thousand"],
                    ),
                ],
            }
        )

        st.dataframe(
            demographics_table,
            hide_index=True,
            width="stretch",
        )

    with right_col:

        st.subheader("COVID-19 Metrics")

        covid_table = pd.DataFrame(
            {
                "Metric": [
                    "Cases per Million",
                    "Deaths per Million",
                    "Fully Vaccinated (%)",
                    "Government Stringency Index",
                ],
                "Value": [
                    format_feature_value(
                        "total_cases_per_million",
                        selected["total_cases_per_million"],
                    ),
                    format_feature_value(
                        "total_deaths_per_million",
                        selected["total_deaths_per_million"],
                    ),
                    format_feature_value(
                        "people_fully_vaccinated_per_hundred",
                        selected["people_fully_vaccinated_per_hundred"],
                    ),
                    format_feature_value(
                        "stringency_index",
                        selected["stringency_index"],
                    ),
                ],
            }
        )

        st.dataframe(
            covid_table,
            hide_index=True,
            width="stretch",
        )
        
        
# ---------------------------------------
# Global Map Tab
# ---------------------------------------

with map_tab:

    st.header("Global Map")

    if map_metric == "Cluster":

        st.markdown(
            """
            This map displays the geographic distribution of the four country
            clusters identified using K-Means clustering. Countries within the
            same cluster share similar demographic, healthcare, economic,
            vaccination, policy, and pandemic characteristics, although they
            are not necessarily geographically close.
            """
        )

        cluster_labels = {
            0: "Cluster 0 – Younger, Lower-Income Countries",
            1: "Cluster 1 – Older, High-Income Countries",
            2: "Cluster 2 – Dense Urban Jurisdictions",
            3: "Cluster 3 – Mixed Middle-Income Countries",
        }

        map_df = (
          df.dropna(subset=["iso_code", "Cluster"])
          .copy()
        )
        
        map_df["Cluster Label"] = (
          map_df["Cluster"]
          .astype(int)
          .map(CLUSTER_LABELS)
        )
        
        fig = px.choropleth(
          map_df,
          locations="iso_code",
          color="Cluster Label",
          hover_name="country",
          hover_data={
            "continent": True,
            "Cluster Label": False,
            "iso_code": False,
          },
          projection="natural earth",
          title="Geographic Distribution of Country Clusters",
          color_discrete_sequence=px.colors.qualitative.Set2,
            category_orders={"Cluster Label": REPORT_CLUSTER_ORDER},
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            legend_title="Country Cluster",
        )

        st.plotly_chart(
            fig,
            #use_container_width=True,
            width="stretch",
        )

    else:

        map_title = FEATURE_LABELS.get(map_metric, map_metric)

        map_data = df.copy()

        color_metric = map_metric

        if (
            map_metric == "total_deaths_per_million"
            and use_log
        ):
            map_data = add_log_mortality(map_data)
            color_metric = "log_deaths_per_million"
            map_title += " (Log Scale)"

        fig = px.choropleth(
            map_data,
            locations="iso_code",
            color=color_metric,
            hover_name="country",
            projection="natural earth",
            color_continuous_scale="Viridis",
            title=map_title,
            labels={
                color_metric: map_title,
            },
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
        )

        st.plotly_chart(
            fig,
            #use_container_width=True,
            width="stretch",
        )


# ---------------------------------------
# Clusters Tab
# ---------------------------------------

with clusters_tab:
    st.header("Country Cluster Analysis")

    st.write(
        "Countries were grouped using K-Means clustering based on demographic, healthcare, economic, "
        "and vaccination characteristics. COVID-19 outcome variables were intentionally excluded "
        "so that differences in cases and mortality could be evaluated after clustering.")

    st.write(
        "The resulting clusters represent countries with similar underlying profiles rather than "
        "geographic proximity. Select a cluster below to explore its defining characteristics, "
        "representative metrics, and how COVID-19 outcomes differ across the resulting groups.")

    st.info(
        "Select **Country Cluster** in the **Global Map** tab to view the "
        "geographic distribution of the four clusters."
    )

    # ---------------------------------------
    # Cluster Selection
    # ---------------------------------------

    cluster_options = sorted(CLUSTER_PROFILES.keys())

    selected_cluster = st.selectbox(
        "Select a Country Cluster",
        options=cluster_options,
        format_func=lambda cluster: (
            f"Cluster {cluster} – "
            f"{CLUSTER_PROFILES[cluster]['short_label']}"
        ),
        key="cluster_selector",
    )

    selected_profile = CLUSTER_PROFILES[selected_cluster]

    selected_cluster_df = (
        df[df["Cluster"] == selected_cluster]
        .copy()
    )

    cluster_summary = create_cluster_summary(df)

    selected_summary = (
        cluster_summary.loc[
            cluster_summary["Cluster"] == selected_cluster
        ]
        .iloc[0]
    )

    # ---------------------------------------
    # Cluster Description
    # ---------------------------------------

    st.subheader(
        f"Cluster {selected_cluster} – "
        f"{selected_profile['short_label']}"
    )

    st.write(selected_profile["description"])

    st.divider()

    # ---------------------------------------
    # Cluster Profile
    # ---------------------------------------

    st.subheader("Cluster Profile")

    st.caption(
        "Values represent cluster medians and summarize the typical "
        "demographic, healthcare, economic, vaccination, and COVID-19 "
        "characteristics of countries in the selected cluster."
    )

    profile_col1, profile_col2, profile_col3 = st.columns(3)

    profile_col1.metric(
        "Countries",
        f"{int(selected_summary['Countries']):,}",
    )

    profile_col2.metric(
        FEATURE_LABELS["median_age"],
        format_feature_value(
            "median_age",
            selected_summary["median_age"],
        ),
    )

    profile_col3.metric(
        FEATURE_LABELS["gdp_per_capita"],
        format_feature_value(
            "gdp_per_capita",
            selected_summary["gdp_per_capita"],
        ),
    )

    profile_col4, profile_col5, profile_col6 = st.columns(3)

    profile_col4.metric(
        FEATURE_LABELS["life_expectancy"],
        format_feature_value(
            "life_expectancy",
            selected_summary["life_expectancy"],
        ),
    )

    profile_col5.metric(
        FEATURE_LABELS["hospital_beds_per_thousand"],
        format_feature_value(
            "hospital_beds_per_thousand",
            selected_summary["hospital_beds_per_thousand"],
        ),
    )

    profile_col6.metric(
        FEATURE_LABELS["people_fully_vaccinated_per_hundred"],
        format_feature_value(
            "people_fully_vaccinated_per_hundred",
            selected_summary[
                "people_fully_vaccinated_per_hundred"
            ],
        ),
    )

    profile_col7, profile_col8, profile_col9 = st.columns(3)

    profile_col7.metric(
        FEATURE_LABELS["population_density"],
        format_feature_value(
            "population_density",
            selected_summary["population_density"],
        ),
    )

    profile_col8.metric(
        FEATURE_LABELS["total_cases_per_million"],
        format_feature_value(
            "total_cases_per_million",
            selected_summary["total_cases_per_million"],
        ),
    )

    profile_col9.metric(
        FEATURE_LABELS["total_deaths_per_million"],
        format_feature_value(
            "total_deaths_per_million",
            selected_summary["total_deaths_per_million"],
        ),
    )

    st.divider()

    # ---------------------------------------
    # Mortality Distribution by Cluster
    # ---------------------------------------

    st.subheader("Mortality Distribution by Cluster")

    st.write(
        "The boxplot below compares the distribution of reported COVID-19 "
        "deaths per million across the four country clusters."
    )

    st.image(
        FIGURE_DIR / "Figure6_COVID_Mortality_by_Cluster.png",
        width="stretch",
    )    

    st.caption(
        "COVID-19 mortality differed substantially across the four clusters. "
        "Older, high-income countries generally experienced higher mortality, "
        "while younger, lower-income countries tended to have lower reported "
        "mortality. The spread within each cluster also highlights the "
        "considerable variability among countries with otherwise similar "
        "demographic and healthcare characteristics."
    )

    st.divider()

    # ---------------------------------------
    # Countries in Selected Cluster
    # ---------------------------------------

    st.subheader("Countries in This Cluster")

    cluster_countries = sorted(
        selected_cluster_df["country"]
        .dropna()
        .unique()
        .tolist()
    )

    st.write(", ".join(cluster_countries))

    st.divider()

    # ---------------------------------------
    # Cluster Data
    # ---------------------------------------

    with st.expander("View Cluster Data"):

        cluster_display_columns = [
            "country",
            "continent",
            "median_age",
            "life_expectancy",
            "population_density",
            "gdp_per_capita",
            "hospital_beds_per_thousand",
            "people_fully_vaccinated_per_hundred",
            "total_cases_per_million",
            "total_deaths_per_million",
            "stringency_index",
        ]

        available_columns = [
            column
            for column in cluster_display_columns
            if column in selected_cluster_df.columns
        ]

        cluster_display_df = (
            selected_cluster_df[available_columns]
            .sort_values("country")
            .rename(columns=FEATURE_LABELS)
        )

        st.dataframe(
            cluster_display_df,
            hide_index=True,
            width="stretch",
        )


# ---------------------------------------
#  Explainability Tab
# ---------------------------------------

with explainability_tab:
    st.header("Model Explainability")
    
    #st.code(str(type(global_shap_values)))
    #st.write("SHAP values shape:", global_shap_values.values.shape)    
    
    if False:
      st.write(
        "The Random Forest model ranks each predictor according to its "
        "contribution to predicting country-level COVID-19 mortality. "
        "Variables with higher importance values had a greater influence on "
        "the model's predictions."
      )
      
      # Display the Random Forest feature importance plot
      st.image(
        FIGURE_DIR / "Figure3_RF_Feature_Importance.png",
        width="stretch",
      )
      
      # Summarize the primary predictors identified by the model
      st.caption(
        "Median age, life expectancy, and total reported COVID-19 cases "
        "were the strongest predictors in the Random Forest model."
      )
      
      st.divider()

    st.subheader("Global SHAP Feature Importance")

    st.write(
        "The SHAP bar plot summarizes the average absolute contribution "
        "of each feature across all country-level predictions. Features "
        "with larger values had a greater overall effect on the model's "
        "predictions, regardless of whether that effect increased or "
        "decreased predicted mortality."
    )

    shap.plots.bar(
      global_shap_display,
      max_display=12,
      show=False,
    )

    shap_fig = plt.gcf()
    plt.tight_layout()

    st.pyplot(
        shap_fig,
        clear_figure=True,
    )
    
    st.divider()
    st.subheader("Global SHAP Beeswarm Plot")
    
    st.write(
      "The beeswarm plot shows both the importance and direction of each "
      "feature's effect on predicted mortality. Each point represents one "
      "country. Color indicates whether the feature value is relatively high "
      "(red) or low (blue), while horizontal position shows whether that "
      "feature increased or decreased the prediction."
    )
    
    plt.figure(figsize=(10, 7))
    
    shap.plots.beeswarm(
      global_shap_display,
      max_display=12,
      show=False,
    )
    
    beeswarm_fig = plt.gcf()
    plt.title(
      "How Features Influence Model Predictions",
      fontsize=16,
      pad=15,
    )
    plt.xlabel(
      "SHAP Value (Impact on Predicted COVID-19 Mortality)",
      fontsize=12,
    )
    
    plt.ylabel("")

    plt.tight_layout()
    
    st.pyplot(
      beeswarm_fig,
      clear_figure=True,
    )

# ---------------------------------------
# Model Performance Tab
# ---------------------------------------

with performance_tab:
    st.header("Model Performance")

    st.write(
        "Three machine learning models were evaluated to predict COVID-19 "
        "mortality. The Random Forest model was selected because it achieved "
        "the best overall performance on the independent test set."
    )

    # Rename columns for display
    display_comparison = model_comparison.rename(
        columns={"R2": "R²"}
    )

    # Display key performance metrics for the selected model
    best_model = model_comparison.loc[
        model_comparison["Model"] == "Random Forest"
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "RMSE",
        f"{best_model['RMSE']:,.1f}",
    )

    col2.metric(
        "MAE",
        f"{best_model['MAE']:,.1f}",
    )

    col3.metric(
        "R²",
        f"{best_model['R2']:.3f}",
    )

    st.divider()

    # Compare model performance across the three evaluation metrics
    st.subheader("Performance Metric Comparison")

    st.write(
        "The three evaluation metrics below compare model performance. Lower "
        "RMSE and MAE indicate lower prediction error, while higher R² "
        "indicates better overall model fit."
    )

    chart_df = model_comparison.copy()

    model_labels = {
        "Random Forest": "RF",
        "Gradient Boosting": "GB",
        "Linear Regression": "LR",
    }

    chart_df["Model_Label"] = chart_df["Model"].map(model_labels)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    metrics = ["RMSE", "MAE", "R2"]
    titles = [
        "RMSE\n(Lower is Better)",
        "MAE\n(Lower is Better)",
        "R²\n(Higher is Better)",
    ]

    # Create one comparison chart for each performance metric
    for ax, metric, title in zip(axes, metrics, titles):

        colors = [
            "tab:green" if model == "Random Forest"
            else "lightgray"
            for model in chart_df["Model"]
        ]

        ax.barh(
            chart_df["Model_Label"],
            chart_df[metric],
            color=colors,
        )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(metric)

        # Display the best-performing model at the top
        ax.invert_yaxis()

        # Simplify the chart appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Label each bar with its corresponding metric value
        max_value = chart_df[metric].max()

        for y, value in enumerate(chart_df[metric]):
            if metric == "R2":
                label = f"{value:.3f}"
            else:
                label = f"{value:,.1f}"

            ax.text(
                value + max_value * 0.02,
                y,
                label,
                va="center",
                fontsize=9,
            )

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    # Compare observed and predicted mortality for the Random Forest model
    st.subheader("Actual vs. Predicted Values")

    st.write(
        "The scatterplot below compares the Random Forest model's predicted "
        "mortality with the observed mortality for the test dataset. Points "
        "closer to the diagonal reference line indicate more accurate "
        "predictions."
    )

    st.image(
        FIGURE_DIR / "Figure4_Actual_vs_Predicted.png",
        width="stretch",
    )

    st.divider()

    # Summarize important considerations when interpreting the model
    st.subheader("Model Limitations")

    st.markdown(
        """
        - Predictions are based on historical country-level COVID-19 data.
        - The model is predictive rather than causal and should not be used to infer cause-and-effect relationships.
        - Results summarize national averages and do not capture within-country variation.
        - Predictions should be interpreted as estimates rather than exact forecasts.
        """
    )



# ---------------------------------------
# Prediction Tab
# ---------------------------------------

with prediction_tab:
    st.header("Prediction")

    st.write(
        "This tool uses the trained Random Forest model to estimate COVID-19 "
        "deaths per million. Modify one or more predictors to explore how the "
        "model's prediction changes."
    )

    # ---------------------------------------
    # Selected country and baseline values
    # ---------------------------------------

    country_profile = get_country_profile(df, country)

    st.subheader(f"Country: {country}")
    st.write(f"**Continent:** {country_profile['continent']}")

    original_input_values = {}
    imputed_features = []

    for feature in PREDICTION_FEATURES:
        value = country_profile[feature]

        # Replace missing country values with the dataset median.
        if pd.isna(value):
            value = df[feature].median(skipna=True)
            imputed_features.append(feature)

        if pd.isna(value):
            raise ValueError(
                f"No usable value is available for '{feature}'. "
                "The country value and dataset median are both missing."
            )

        original_input_values[feature] = float(value)

    # ---------------------------------------
    # Reset inputs when country changes
    # ---------------------------------------

    previous_country = st.session_state.get("prediction_country")
    
    if previous_country != country:
      st.session_state.prediction_country = country
      
      for feature in PREDICTION_FEATURES:
        st.session_state[f"prediction_{feature}"] = (
            original_input_values[feature]
        )
    else:
      for feature in PREDICTION_FEATURES:
        st.session_state.setdefault(
            f"prediction_{feature}",
            original_input_values[feature],
        )

    if imputed_features:
        imputed_labels = [
            FEATURE_LABELS.get(feature, feature)
            for feature in imputed_features
        ]

        st.caption(
            "Missing values replaced with dataset medians: "
            + ", ".join(imputed_labels)
        )

    # ---------------------------------------
    # Model inputs
    # ---------------------------------------

    st.subheader("Model Inputs")

    if st.button(
        "Restore Original Values",
        key="restore_prediction_values",
    ):
        for feature in PREDICTION_FEATURES:
            st.session_state[f"prediction_{feature}"] = (
                original_input_values[feature]
            )

        st.rerun()

    input_values = {}

    col1, col2 = st.columns(2)

    with col1:
        for feature in [
            "median_age",
            "life_expectancy",
            "gdp_per_capita",
            "hospital_beds_per_thousand",
        ]:
            min_value, max_value = FEATURE_LIMITS[feature]

            input_values[feature] = st.number_input(
              FEATURE_LABELS[feature],
              min_value=min_value,
              max_value=max_value,
              step=FEATURE_STEPS[feature],
              format=FEATURE_FORMATS[feature],
              key=f"prediction_{feature}",
            )

    with col2:
        for feature in [
            "total_cases_per_million",
            "people_fully_vaccinated_per_hundred",
            "stringency_index",
        ]:
            min_value, max_value = FEATURE_LIMITS[feature]

            input_values[feature] = st.number_input(
                FEATURE_LABELS[feature],
                min_value=min_value,
                max_value=max_value,
                step=FEATURE_STEPS[feature],
                format=FEATURE_FORMATS[feature],
                key=f"prediction_{feature}",
            )

    # Compare values safely, avoiding floating-point comparison issues.
    def input_has_changed(feature):
        return not np.isclose(
            float(input_values[feature]),
            float(original_input_values[feature]),
            rtol=0.0,
            atol=1e-9,
            equal_nan=True,
        )

    # ---------------------------------------
    # Generate predictions
    # ---------------------------------------

    actual_value = float(
        country_profile["total_deaths_per_million"]
    )

    original_prediction_input = prepare_prediction_input(
        input_values=original_input_values,
        country_profile=country_profile,
        scaler=scaler,
        feature_names=feature_names,
    )

    original_prediction = float(
        model.predict(original_prediction_input)[0]
    )

    prediction_input = prepare_prediction_input(
        input_values=input_values,
        country_profile=country_profile,
        scaler=scaler,
        feature_names=feature_names,
    )

    modified_prediction = float(
        model.predict(prediction_input)[0]
    )

    # ---------------------------------------
    # Compute local SHAP explanations
    # ---------------------------------------

    original_local_shap_values = compute_local_shap_values(
        explainer,
        original_prediction_input,
    )

    local_shap_values = compute_local_shap_values(
        explainer,
        prediction_input,
    )

    display_values = []

    for feature in feature_names:
        if feature in input_values:
            display_values.append(input_values[feature])
        else:
            display_values.append(
                prediction_input.iloc[0][feature]
            )

    feature_labels = [
        FEATURE_LABELS.get(feature, feature)
        for feature in feature_names
    ]

    original_shap_values = (
        original_local_shap_values.values[0].copy()
    )

    shap_values = local_shap_values.values[0].copy()

    # ---------------------------------------
    # Combine one-hot continent variables
    # ---------------------------------------

    continent_features = [
        "continent_Asia",
        "continent_Europe",
        "continent_North America",
        "continent_Oceania",
        "continent_South America",
    ]

    available_continent_features = [
        feature
        for feature in continent_features
        if feature in feature_names
    ]

    continent_indices = [
        feature_names.index(feature)
        for feature in available_continent_features
    ]

    original_continent_effect = (
        original_shap_values[continent_indices].sum()
        if continent_indices
        else 0.0
    )

    continent_effect = (
        shap_values[continent_indices].sum()
        if continent_indices
        else 0.0
    )

    keep_indices = [
        index
        for index in range(len(feature_names))
        if index not in continent_indices
    ]

    original_shap_values = original_shap_values[
        keep_indices
    ]

    shap_values = shap_values[keep_indices]

    feature_labels = [
        feature_labels[index]
        for index in keep_indices
    ]

    display_values = [
        display_values[index]
        for index in keep_indices
    ]

    if continent_indices:
        original_shap_values = np.append(
            original_shap_values,
            original_continent_effect,
        )

        shap_values = np.append(
            shap_values,
            continent_effect,
        )

        feature_labels.append("Continent")
        display_values.append(
            country_profile["continent"]
        )

    local_shap_display = shap.Explanation(
        values=shap_values,
        base_values=local_shap_values.base_values[0],
        data=display_values,
        feature_names=feature_labels,
    )

    # ---------------------------------------
    # Determine whether inputs changed
    # ---------------------------------------

    inputs_modified = any(
        input_has_changed(feature)
        for feature in PREDICTION_FEATURES
    )

    # ---------------------------------------
    # Model results
    # ---------------------------------------

    st.subheader("Model Results")

    if inputs_modified:
        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Actual Deaths per Million",
            f"{actual_value:,.1f}",
        )

        result_col2.metric(
            "Original Prediction",
            f"{original_prediction:,.1f}",
            delta=(
                f"{original_prediction - actual_value:,.1f}"
            ),
        )

        prediction_change = (
            modified_prediction - original_prediction
        )

        result_col3.metric(
            "Modified Prediction",
            f"{modified_prediction:,.1f}",
            delta=f"{prediction_change:,.1f}",
        )

        percent_change = (
            prediction_change / original_prediction * 100
            if original_prediction != 0
            else 0.0
        )

        st.info(
            f"Estimated effect of your changes: "
            f"{prediction_change:+,.1f} deaths per million "
            f"({percent_change:+.1f}%)."
        )

        # ---------------------------------------
        # Changed inputs
        # ---------------------------------------

        changed_inputs = []

        for feature in PREDICTION_FEATURES:
            if not input_has_changed(feature):
                continue

            original_value = original_input_values[feature]
            modified_value = input_values[feature]
            change = modified_value - original_value

            changed_inputs.append(
                {
                    "Input": FEATURE_LABELS[feature],
                    "Original": format_feature_value(
                        feature,
                        original_value,
                    ),
                    "Modified": (
                        f"{format_feature_value(feature, modified_value)} "
                        f"({change:+,.1f})"
                    ),
                }
            )

      #  st.subheader("Changed Inputs")
        st.subheader(f"Changed Inputs (Baseline: {country})")

        st.dataframe(
            pd.DataFrame(changed_inputs),
            hide_index=True,
            width="stretch",
        )

        # ---------------------------------------
        # What drove the prediction change?
        # ---------------------------------------

        st.subheader("What Drove the Prediction Change?")

        st.write(
            "Changing one input can also change the influence of other "
            "features because the Random Forest model captures interactions "
            "between predictors."
        )

        scenario_comparison = pd.DataFrame(
            {
                "Feature": feature_labels,
                "Original": original_shap_values,
                "Modified": shap_values,
            }
        )

        scenario_comparison["Effect"] = (
            scenario_comparison["Modified"]
            - scenario_comparison["Original"]
        )

        edited_features = {
            FEATURE_LABELS[feature]
            for feature in PREDICTION_FEATURES
            if input_has_changed(feature)
        }

        scenario_comparison["Edited"] = (
            scenario_comparison["Feature"].isin(
                edited_features
            )
        )

        # Separate major and minor effects.
        major_effects = (
            scenario_comparison[
                scenario_comparison["Effect"].abs()
                >= SHAP_DISPLAY_THRESHOLD
            ]
            .sort_values(
                "Effect",
                key=lambda values: values.abs(),
                ascending=False,
            )
        )

        minor_effects = scenario_comparison[
            scenario_comparison["Effect"].abs()
            < SHAP_DISPLAY_THRESHOLD
        ]

        minor_effect_sum = minor_effects["Effect"].sum()
        minor_effect_count = len(minor_effects)

        # ---------------------------------------
        # Direct effects
        # ---------------------------------------

        direct_effects = major_effects[
            major_effects["Edited"]
        ]

        if not direct_effects.empty:
            st.markdown("##### Direct Effects")

            for _, row in direct_effects.iterrows():
                arrow = "▲" if row["Effect"] > 0 else "▼"

                st.write(
                    f"{arrow} **{row['Feature']}** "
                    f"({row['Effect']:+,.1f} deaths per million)"
                )

        # ---------------------------------------
        # Indirect model interactions
        # ---------------------------------------
        
        interaction_effects = major_effects[
          ~major_effects["Edited"]
        ]
        
        if not interaction_effects.empty:
          st.markdown("##### Model Interactions")
          
          interaction_strings = []
          
          for _, row in interaction_effects.iterrows():
            arrow = "▲" if row["Effect"] > 0 else "▼"
            
            interaction_strings.append(
              f"{arrow} **{row['Feature']}** "
              f"({row['Effect']:+,.1f})"
            )
            
          st.markdown(", ".join(interaction_strings))
    
        # ---------------------------------------
        # Remaining smaller interactions
        # ---------------------------------------

        if (
            minor_effect_count > 0
            and abs(minor_effect_sum)
            >= SHAP_SUMMARY_THRESHOLD
        ):
            st.markdown("##### Other Interactions")

            feature_word = (
                "feature"
                if minor_effect_count == 1
                else "features"
            )

            st.write(
                f"The remaining {minor_effect_count} "
                f"{feature_word} had a combined effect of "
                f"**{minor_effect_sum:+,.1f} deaths per million**."
            )

        # ---------------------------------------
        # Fallback
        # ---------------------------------------

        if (
            direct_effects.empty
            and interaction_effects.empty
            and abs(minor_effect_sum)
            < SHAP_SUMMARY_THRESHOLD
        ):
            st.info(
                "No individual feature had a substantial effect on "
                "the change in prediction."
            )

    else:
        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Actual Deaths per Million",
            f"{actual_value:,.1f}",
        )

        result_col2.metric(
            "Predicted Deaths per Million",
            f"{original_prediction:,.1f}",
            delta=(
                f"{original_prediction - actual_value:,.1f}"
            ),
        )

    # ---------------------------------------
    # Prediction explanation
    # ---------------------------------------

    st.subheader("Prediction Explanation")

    feature_effects = pd.DataFrame(
        {
            "Feature": feature_labels,
            "Value": display_values,
            "SHAP": shap_values,
        }
    )

    positive_effects = (
        feature_effects[
            feature_effects["SHAP"] > 0
        ]
        .sort_values(
            "SHAP",
            ascending=False,
        )
        .head(3)
    )

    negative_effects = (
        feature_effects[
            feature_effects["SHAP"] < 0
        ]
        .sort_values("SHAP")
        .head(3)
    )

    # ---------------------------------------
    # Compact feature contribution summary
    # ---------------------------------------

    st.write(
        "The chart below explains how each feature influenced the predicted "
        "COVID-19 mortality. Positive values increase the prediction, while "
        "negative values decrease it."
    )

    if positive_effects.empty:
        st.markdown(
            "**▲ Increased Prediction:** "
            "No features increased the prediction."
        )
    else:
        increased_items = [
            f"**{row['Feature']}** (+{row['SHAP']:.0f})"
            for _, row in positive_effects.iterrows()
        ]

        st.markdown(
            "**▲ Increased Prediction:** "
            + ", ".join(increased_items)
        )

    if negative_effects.empty:
        st.markdown(
            "**▼ Decreased Prediction:** "
            "No features decreased the prediction."
        )
    else:
        decreased_items = [
            f"**{row['Feature']}** ({row['SHAP']:.0f})"
            for _, row in negative_effects.iterrows()
        ]

        st.markdown(
            "**▼ Decreased Prediction:** "
            + ", ".join(decreased_items)
        )
        
    # ---------------------------------------
    # SHAP waterfall plot
    # ---------------------------------------

    plt.figure(figsize=(9, 4))

    shap.plots.waterfall(
        local_shap_display,
        max_display=12,
        show=False,
    )

    plt.title(
        "Feature Contributions to the Prediction",
        fontsize=12,
        pad=4,
        loc="left",
    )
    
    plt.subplots_adjust(left=0.18)
    
    plt.tight_layout(pad=0.5)

    st.pyplot(
        plt.gcf(),
        clear_figure=True,
    )


   
# ---------------------------------------
# Dataset Tab
# ---------------------------------------

with dataset_tab:
    st.header("Dataset")

    st.write(
        "This tab provides access to the reduced country-level analysis dataset "
        "used throughout the dashboard."
    )

    # Create a working copy of the reduced analysis dataset
    dataset_df = df[DATASET_DISPLAY_COLUMNS].copy()

    # Display the overall size of the dataset
    row_count, column_count = dataset_df.shape

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Countries",
        f"{row_count:,}",
    )

    col2.metric(
        "Variables",
        f"{column_count:,}",
    )

    st.divider()

    # Filter the dataset by continent and additional user-defined criteria
    st.subheader("Filters")

    selected_continent = st.selectbox(
        "Continent",
        ["All"] + sorted(dataset_df["continent"].dropna().unique()),
        key="dataset_continent_filter",
    )

    filtered_df = dataset_df.copy()

    if selected_continent != "All":
        filtered_df = filtered_df[
            filtered_df["continent"] == selected_continent
        ]

    st.markdown("**Additional Filters**")

    # Allow users to define up to three additional filters
    for i in range(1, 4):
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            filter_column = st.selectbox(
                f"Filter {i} Column",
                ["None"] + DATASET_DISPLAY_COLUMNS,
                format_func=lambda x: FEATURE_LABELS.get(x, x),
                key=f"dataset_filter_{i}_column",
            )

        if (
            filter_column != "None"
            and pd.api.types.is_numeric_dtype(dataset_df[filter_column])
        ):
            operators = ["=", "!=", ">", ">=", "<", "<="]
        else:
            operators = ["Contains", "Starts with", "Ends with", "Equals"]

        with filter_col2:
            filter_operator = st.selectbox(
                f"Filter {i} Operator",
                operators,
                key=f"dataset_filter_{i}_operator",
            )

        with filter_col3:
            filter_value = st.text_input(
                f"Filter {i} Value",
                key=f"dataset_filter_{i}_value",
            )

        filtered_df = apply_filter(
            filtered_df,
            filter_column,
            filter_operator,
            filter_value,
        )

    st.divider()

    # Sort the filtered dataset using one or two variables
    st.subheader("Sorting")

    sort_col1, sort_col2, sort_col3, sort_col4 = st.columns(4)

    with sort_col1:
        primary_sort = st.selectbox(
            "Primary Sort",
            DATASET_DISPLAY_COLUMNS,
            format_func=lambda x: FEATURE_LABELS.get(x, x),
            index=DATASET_DISPLAY_COLUMNS.index("country"),
            key="dataset_primary_sort",
        )

    with sort_col2:
        primary_direction = st.radio(
            "Primary Direction",
            ["Ascending", "Descending"],
            horizontal=True,
            key="dataset_primary_direction",
        )

    with sort_col3:
        secondary_sort = st.selectbox(
            "Secondary Sort",
            ["None"] + DATASET_DISPLAY_COLUMNS,
            format_func=lambda x: FEATURE_LABELS.get(x, x),
            key="dataset_secondary_sort",
        )

    with sort_col4:
        secondary_direction = st.radio(
            "Secondary Direction",
            ["Ascending", "Descending"],
            horizontal=True,
            key="dataset_secondary_direction",
        )

    sort_columns = [primary_sort]
    sort_ascending = [primary_direction == "Ascending"]

    if secondary_sort != "None":
        sort_columns.append(secondary_sort)
        sort_ascending.append(secondary_direction == "Ascending")

    # Apply the selected sort order
    filtered_df = filtered_df.sort_values(
        by=sort_columns,
        ascending=sort_ascending,
    )

    st.divider()

    # Display the number of countries remaining after filtering
    st.metric(
        "Countries Displayed",
        f"{len(filtered_df):,}",
    )

    # Optionally display descriptive statistics for the filtered dataset
    show_summary = st.checkbox(
        "Show summary statistics",
        key="dataset_show_summary",
    )

    if show_summary:
        st.subheader("Summary Statistics")

        summary_df = filtered_df.describe().T.rename(index=FEATURE_LABELS)

        st.dataframe(
            summary_df,
            width="stretch",
        )

    # Rename columns for display without modifying the underlying dataset
    display_df = filtered_df.rename(columns=FEATURE_LABELS)

    st.subheader("Data Table")

    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
    )


st.divider()

st.caption(
    "Data sources: Our World in Data, World Bank, and Oxford COVID-19 "
    "Government Response Tracker. Developed as part of Springboard Capstone 3."
)
