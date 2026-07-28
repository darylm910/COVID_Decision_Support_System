# ---------------------------------------
# Imports
# ---------------------------------------

import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt


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
    format_feature_value
)

# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="Explainable Machine Learning for COVID-19 Mortality",
    page_icon="🌍",
    layout="wide",
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

model_comparison = load_model_comparison()


# ---------------------------------------
# Sidebar Controls
# ---------------------------------------

st.sidebar.title("Controls")

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

st.title("🌍 Explainable Machine Learning for COVID-19 Mortality")

st.markdown(
    """
    This interactive application explores country-level COVID-19 mortality
    using unsupervised clustering and Random Forest machine learning.

    The project combines two complementary analytical approaches:

    - **Country clustering** groups countries with similar demographic,
      healthcare, economic, vaccination, and policy characteristics.
      COVID-19 outcome variables were intentionally excluded from the
      clustering process so that differences in cases and mortality could
      be evaluated across the resulting groups.
    - **Random Forest regression** predicts country-level COVID-19 deaths
      per million using demographic, healthcare, economic, vaccination,
      policy, and pandemic-related variables. Feature importance analysis
      identifies the variables that most strongly influence the model's
      predictions.

    Use the tabs below to:

    - Explore country-level characteristics
    - Visualize global patterns
    - Examine country clusters
    - Understand the model's most influential predictors
    - Evaluate predictive performance
    - Experiment with interactive what-if scenarios
    - Browse the underlying analysis dataset
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

    # Show continent as metadata instead of mixing text into numeric tables
    #st.caption(f"Continent: {selected['continent']}")
    st.code(
      f"""Continent: {selected['continent']}
Country Cluster: {get_cluster_display(selected['Cluster'])}"""
    )

    # Key summary metrics
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

    # Demographic and healthcare variables
    st.subheader("Demographics and Healthcare")

    demographic_features = [
        "median_age",
        "life_expectancy",
        "gdp_per_capita",
        "hospital_beds_per_thousand",
    ]

    demographic_df = pd.DataFrame(
        {
            "Value": [
                format_feature_value(feature, selected[feature])
                for feature in demographic_features
            ]
        },
        index=[
            FEATURE_LABELS[feature]
            for feature in demographic_features
        ],
    )

    st.dataframe(
        demographic_df,
        width="stretch",
    )

    # COVID-specific variables
    st.subheader("COVID-19 Metrics")

    covid_features = [
        "total_cases_per_million",
        "total_deaths_per_million",
        "people_fully_vaccinated_per_hundred",
        "stringency_index",
    ]

    covid_df = pd.DataFrame(
        {
            "Value": [
                format_feature_value(feature, selected[feature])
                for feature in covid_features
            ]
        },
        index=[
            FEATURE_LABELS[feature]
            for feature in covid_features
        ],
    )

    st.dataframe(
        covid_df,
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

    # Retrieve the selected country's data
    country_profile = get_country_profile(df, country)

    st.caption(
        f"Selected Country: {country} | "
        f"Continent: {country_profile['continent']}"
    )

    # Store the original model inputs for comparison and reset
    original_input_values = {
        feature: float(country_profile[feature])
        for feature in PREDICTION_FEATURES
    }

    input_values = {}

    st.subheader("Model Inputs")

    # Restore all inputs to the selected country's original values
    if st.button("Restore Original Values"):
        for feature in PREDICTION_FEATURES:
            st.session_state[f"prediction_{feature}"] = (
                original_input_values[feature]
            )
        st.rerun()

    col1, col2 = st.columns(2)

    # Collect demographic and healthcare inputs
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
                value=original_input_values[feature],
                step=FEATURE_STEPS[feature],
                format=FEATURE_FORMATS[feature],
                key=f"prediction_{feature}",
            )

    # Collect COVID-19 related inputs
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
                value=original_input_values[feature],
                step=FEATURE_STEPS[feature],
                format=FEATURE_FORMATS[feature],
                key=f"prediction_{feature}",
            )

    # Generate baseline and modified model predictions
    actual_value = float(country_profile["total_deaths_per_million"])

    original_prediction_input = prepare_prediction_input(
        input_values=original_input_values,
        country_profile=country_profile,
        scaler=scaler,
        feature_names=feature_names,
    )

    original_prediction = model.predict(original_prediction_input)[0]

    prediction_input = prepare_prediction_input(
        input_values=input_values,
        country_profile=country_profile,
        scaler=scaler,
        feature_names=feature_names,
    )

    modified_prediction = model.predict(prediction_input)[0]

    # Determine whether any model inputs have changed
    inputs_modified = any(
        input_values[feature] != original_input_values[feature]
        for feature in PREDICTION_FEATURES
    )

    st.subheader("Model Results")

    # Compare the baseline and modified predictions when inputs change
    if inputs_modified:
        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Actual Deaths per Million",
            f"{actual_value:,.1f}",
        )

        result_col2.metric(
            "Original Prediction",
            f"{original_prediction:,.1f}",
            delta=f"{original_prediction - actual_value:,.1f}",
        )

        result_col3.metric(
            "Modified Prediction",
            f"{modified_prediction:,.1f}",
            delta=f"{modified_prediction - original_prediction:,.1f}",
        )

        prediction_change = modified_prediction - original_prediction
        percent_change = (
            prediction_change / original_prediction * 100
            if original_prediction != 0
            else 0
        )

        # Summarize the estimated effect of the user's changes
        st.info(
            f"Estimated effect of your changes: "
            f"{prediction_change:+,.1f} deaths per million "
            f"({percent_change:+.1f}%)."
        )

        # Display the inputs that were modified
        changed_inputs = []

        for feature in PREDICTION_FEATURES:
            original_value = original_input_values[feature]
            modified_value = input_values[feature]

            if modified_value != original_value:
                change = modified_value - original_value

                changed_inputs.append({
                  "Input": FEATURE_LABELS[feature],
                  "Original": format_feature_value(
                    feature,
                    original_value,
                ),
                "Modified": (
                  f"{format_feature_value(feature, modified_value)} "
                  f"({change:+,.1f})"
                ),
              })

        st.subheader("Changed Inputs")

        st.dataframe(
            pd.DataFrame(changed_inputs),
            hide_index=True,
            width="stretch",
        )

    # Display the baseline prediction when no inputs have changed
    else:
        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Actual Deaths per Million",
            f"{actual_value:,.1f}",
        )

        result_col2.metric(
            "Predicted Deaths per Million",
            f"{original_prediction:,.1f}",
            delta=f"{original_prediction - actual_value:,.1f}",
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
