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
