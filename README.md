# Reigniting Growth — OptimaLife Churn Prediction & Portfolio Analysis

**GENBUS 895 Master's Capstone | Group 2**

A Streamlit application deployed on Snowflake that provides interactive tools for customer churn prediction and portfolio strategy optimization for OptimaLife's subscription business.

## Overview

OptimaLife grew from $5.60M to $50.35M ARR (2019–2022) but is now experiencing growth deceleration with net retention dropping below 100%. This app delivers a data-driven retention-first strategy powered by a Gradient Boosting churn model (AUC: 0.8661).

## App Structure (5 Tabs)

| Tab | Content |
|-----|---------|
| **Strategic Context** | ARR trends, growth deceleration metrics, $9.42M revenue leakage diagnosis |
| **Data Modeling** | Model selection (LR/RF/GB comparison), feature importance, AUC performance |
| **Churn Prediction** | Interactive scorer + cohort results (605,831 customers, 35× risk spread) |
| **Portfolio Analysis** | Product classification (Growth Engine / Retention Anchors / Contraction Drag) |
| **Business Impact** | Strategic priority matrix, phased roadmap, revenue recovery simulator |

## Key Findings

- **Top churn driver:** Current subscription amount (1.6× more important than next feature)
- **Best acquisition target

## Run And Deploy

For the local Codespaces preview, the app is served on port 8507 and the workspace config already forwards that port publicly. Open the forwarded Streamlit preview after the container starts.

For Snowflake, the repo only defines the app in `snowflake.yml`; it still has to be deployed from the Snowflake side before there is a usable public app URL. If you see a 404 in Snowflake, verify that the app has been deployed in the right account/schema and opened from the Snowflake app page rather than a guessed URL.
