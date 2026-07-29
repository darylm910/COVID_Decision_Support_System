# 🌍 Explainable AI for COVID-19 Mortality Prediction

> An interactive machine learning dashboard for predicting country-level COVID-19 mortality and explaining model predictions using SHAP.

---

## Dashboard Preview

*(Add overview screenshot here)*

---

## Overview

This project combines machine learning, explainable AI, unsupervised learning, and interactive visualization to explore the factors associated with COVID-19 mortality across countries.

Beginning with publicly available global health, demographic, economic, and policy data, a Random Forest regression model was developed to predict COVID-19 deaths per million. The predictive model is deployed through an interactive Streamlit dashboard that enables users to:

- Explore country-level characteristics
- Compare countries around the world
- Modify risk factors through scenario analysis
- Predict changes in mortality
- Understand model predictions using SHAP explainability
- Discover groups of similar countries through unsupervised clustering

Rather than functioning as a static predictive model, the application serves as an interactive decision-support tool for exploring how demographic, healthcare, and policy variables influence predicted mortality.

---

## Key Features

### 🤖 Machine Learning

- Random Forest regression model
- Feature engineering and preprocessing pipeline
- Model comparison across multiple algorithms
- Cross-validation and performance evaluation

### 🔍 Explainable AI

- Global SHAP feature importance
- Local SHAP waterfall plots
- Interactive explanation of prediction changes
- Scenario-based model interpretation

### 🌍 Interactive Dashboard

- Country Explorer
- Global mortality map
- Prediction simulator
- Feature importance visualization
- Model performance summary
- Dataset explorer with filtering and sorting

### 📊 Unsupervised Learning

- Country clustering using K-Means
- Cluster visualization
- Cluster comparison
- Cluster interpretation

---

# Dashboard

## Country Explorer

Explore demographic, healthcare, economic, and COVID-19 characteristics for each country.

*(Add Country Explorer screenshot here)*

---

## Prediction

Modify country characteristics to explore hypothetical scenarios and generate updated mortality predictions.

Features include:

- Automatic population of country values
- Editable model inputs
- Original vs. modified prediction comparison
- Estimated impact of user changes
- SHAP explanation of prediction changes
- Local SHAP waterfall visualization

*(Add Prediction screenshot here)*

---

## Explainable AI

The dashboard uses SHAP (SHapley Additive exPlanations) to explain how each feature contributes to individual predictions.

Users can explore:

- Global feature importance
- Local feature contributions
- Prediction changes after editing inputs
- Direct feature effects
- Model interactions

*(Add SHAP waterfall screenshot here)*

---

## Country Clustering

Countries are grouped using K-Means clustering based on demographic, healthcare, and COVID-19 characteristics.

The clustering analysis helps identify countries with similar profiles and compare mortality patterns across groups.

*(Add Cluster screenshot here)*

---

# Machine Learning Workflow

```
Public Datasets
        │
        ▼
Data Cleaning & Feature Engineering
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Model Development
(Random Forest)
        │
        ▼
Model Evaluation
        │
        ▼
SHAP Explainability
        │
        ▼
Interactive Streamlit Dashboard
```

---

# Repository Structure

```
.
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── streamlit_app/
│
├── README.md
└── requirements.txt
```

---

# Technologies

- Python
- pandas
- NumPy
- scikit-learn
- SHAP
- Streamlit
- Plotly
- Matplotlib
- Joblib

---

# Installation

Clone the repository

```bash
git clone https://github.com/darylm910/Capstone3.git
```

Navigate to the project directory

```bash
cd Capstone3
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Dashboard

```bash
streamlit run streamlit_app/app.py
```

---

# Future Enhancements

Potential future improvements include:

- Additional machine learning models
- Expanded scenario analysis
- Confidence intervals for predictions
- Enhanced clustering exploration
- Additional explainable AI visualizations

---

# Author

**Daryl Morris**

MS Biostatistics

Senior Data Scientist | Machine Learning | Predictive Modeling | Decision Support Systems | Healthcare Analytics

---

*Developed as part of the Springboard Data Science Career Track Capstone Project.*
