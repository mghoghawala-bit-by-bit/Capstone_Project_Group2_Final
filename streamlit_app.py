# OptimaLife Churn Prediction & Portfolio Analysis — 5-tab app matching presentation
# Co-authored with CoCo

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import importlib
import sys
import io
import pickle

import altair as alt

# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    "MODEL_PATH": os.environ.get("MODEL_PATH", "churn_model_final_project.pkl"),
    "ENCODER_PATH": os.environ.get("ENCODER_PATH", "churn_encoder_final_project.pkl"),
}

st.set_page_config(page_title="OptimaLife — Reigniting Growth", page_icon="📊", layout="wide")

# =============================================================================
# Sklearn compatibility patch for older .pkl models
# =============================================================================

class _SklearnCompatFinder:
    def find_module(self, fullname, path=None):
        if fullname == '_loss' or fullname.startswith('_loss.'):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        real = 'sklearn.' + fullname
        mod = importlib.import_module(real)
        sys.modules[fullname] = mod
        return mod


if not any(isinstance(f, _SklearnCompatFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _SklearnCompatFinder())

try:
    _sklearn_loss = importlib.import_module('sklearn._loss')
    _loss_classes = {}
    for _submod in ('sklearn._loss.loss', 'sklearn._loss._loss', 'sklearn._loss'):
        try:
            _mod = importlib.import_module(_submod)
            for _attr in dir(_mod):
                if 'Loss' in _attr or 'Error' in _attr:
                    _loss_classes[_attr] = getattr(_mod, _attr)
        except ImportError:
            continue

    _CY_TO_PY_RENAMES = {
        'CyHalfBinomialLoss': 'HalfBinomialLoss',
        'CyHalfSquaredError': 'HalfSquaredError',
        'CyAbsoluteError': 'AbsoluteError',
        'CyHalfPoissonLoss': 'HalfPoissonLoss',
        'CyHalfGammaLoss': 'HalfGammaLoss',
        'CyHalfTweedieLoss': 'HalfTweedieLoss',
        'CyHalfMultinomialLoss': 'HalfMultinomialLoss',
        'CyHalfTweedieLossIdentity': 'HalfTweedieLossIdentity',
        'CyPinballLoss': 'PinballLoss',
        'CyHuberLoss': 'HuberLoss',
    }
    for cy_name, py_name in _CY_TO_PY_RENAMES.items():
        if not hasattr(_sklearn_loss, cy_name):
            if py_name in _loss_classes:
                setattr(_sklearn_loss, cy_name, _loss_classes[py_name])
            elif hasattr(_sklearn_loss, py_name):
                setattr(_sklearn_loss, cy_name, getattr(_sklearn_loss, py_name))
except ImportError:
    pass

# =============================================================================
# Load model artifacts
# =============================================================================
@st.cache_resource
def load_artifacts():
    import joblib

    def _load_with_compat_unpickler(filepath):
        _sklearn_loss = importlib.import_module('sklearn._loss')
        available_classes = {}
        for submod_name in ('sklearn._loss.loss', 'sklearn._loss._loss', 'sklearn._loss'):
            try:
                submod = importlib.import_module(submod_name)
                for attr in dir(submod):
                    obj = getattr(submod, attr, None)
                    if isinstance(obj, type):
                        available_classes[attr] = obj
            except ImportError:
                continue

        cy_renames = {
            'CyHalfBinomialLoss': 'HalfBinomialLoss',
            'CyHalfSquaredError': 'HalfSquaredError',
            'CyAbsoluteError': 'AbsoluteError',
            'CyHalfPoissonLoss': 'HalfPoissonLoss',
            'CyHalfGammaLoss': 'HalfGammaLoss',
            'CyHalfTweedieLoss': 'HalfTweedieLoss',
            'CyHalfMultinomialLoss': 'HalfMultinomialLoss',
            'CyHalfTweedieLossIdentity': 'HalfTweedieLossIdentity',
            'CyPinballLoss': 'PinballLoss',
            'CyHuberLoss': 'HuberLoss',
        }

        class CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == '_loss' or module.startswith('_loss.'):
                    module = 'sklearn.' + module
                if name in cy_renames:
                    new_name = cy_renames[name]
                    if new_name in available_classes:
                        return available_classes[new_name]
                try:
                    return super().find_class(module, name)
                except AttributeError:
                    if name in available_classes:
                        return available_classes[name]
                    if name.startswith('Cy'):
                        base_name = name[2:]
                        if base_name in available_classes:
                            return available_classes[base_name]
                    raise

        with open(filepath, 'rb') as f:
            data = f.read()

        buf = io.BytesIO(data)
        try:
            buf.seek(0)
            return joblib.load(buf)
        except (AttributeError, ModuleNotFoundError):
            buf.seek(0)
            return CompatUnpickler(buf).load()

    try:
        model = _load_with_compat_unpickler(CONFIG["MODEL_PATH"])
        encoder = _load_with_compat_unpickler(CONFIG["ENCODER_PATH"])
        return model, encoder, None
    except FileNotFoundError:
        return None, None, "Model artifacts not found. Place .pkl files next to streamlit_app.py."
    except Exception as e:
        import traceback as tb
        return None, None, f"Failed to load model: {e}\n\n{tb.format_exc()}"


model, encoder, _load_err = load_artifacts()

categorical_cols = ['PRODUCT', 'INCOME_LEVEL', 'EDUCATION', 'DEVICE_TYPE']
if encoder is not None:
    category_map = {col: list(cats) for col, cats in zip(categorical_cols, encoder.categories_)}
else:
    category_map = {col: [] for col in categorical_cols}

# =============================================================================
# Load precomputed data
# =============================================================================
@st.cache_data
def load_portfolio_data():
    df = pd.read_csv("product_retention_summary.csv")
    df.columns = df.columns.str.lower()
    df["year_starts"] = pd.to_datetime(df["year_starts"])
    return df


@st.cache_data
def load_segment_data():
    df = pd.read_csv("acquisition_segments.csv")
    df.columns = df.columns.str.lower()
    return df


# =============================================================================
# App Header & Tabs
# =============================================================================
st.title("Reigniting Growth — OptimaLife")
st.caption("A data-driven expansion & retention strategy | GENBUS 895 Master's Capstone")

tab_ctx, tab_model, tab_churn, tab_portfolio, tab_impact = st.tabs([
    "📋 Strategic Context",
    "🧪 Data Modeling",
    "🔮 Churn Prediction",
    "📈 Portfolio Analysis",
    "💰 Business Impact",
])

# =============================================================================
# TAB 1 — STRATEGIC CONTEXT (Slides 2–3)
# =============================================================================
with tab_ctx:
    st.header("Strategic Context")
    st.subheader("From Hyper-Growth to Growth Deceleration")

    # Key metrics matching slide 2
    m1, m2, m3 = st.columns(3)
    m1.metric("ARR Growth (2019–2022)", "$5.60M → $50.35M", delta="Hyper-Growth Phase")
    m2.metric("2024 YoY ARR Growth", "$1.69M", delta="down from $15.99M peak in 2021")
    m3.metric("Net Retention Rate", "96%", delta="Below 100% for first time (end of 2023)")

    # Interactive ARR trend explorer
    st.markdown("---")
    st.markdown("##### ARR Trends — December 2019–2024")

    arr_data = pd.DataFrame({
        "Period": pd.to_datetime(["2019-12-01", "2020-12-01", "2021-12-01",
                                  "2022-12-01", "2023-12-01", "2024-12-01"]),
        "BEGINNING_ARR": [0, 5.60, 21.59, 37.58, 50.35, 52.04],
        "NEW_ARR": [5.60, 15.99, 15.99, 12.77, 6.84, 5.50],
        "ENDING_ARR": [5.60, 21.59, 37.58, 50.35, 52.04, 53.73],
        "CHURN_ARR": [0, -0.80, -1.50, -2.50, -4.15, -4.15],
        "CONTRACTION_ARR": [0, -0.50, -1.00, -2.00, -5.27, -5.27],
    })

    arr_component = st.selectbox(
        "Explore ARR component",
        ["ENDING_ARR", "NEW_ARR", "BEGINNING_ARR", "CHURN_ARR", "CONTRACTION_ARR"],
        key="arr_component"
    )

    arr_chart = alt.Chart(arr_data).mark_line(point=True).encode(
        x=alt.X("Period:T", title="Period"),
        y=alt.Y(f"{arr_component}:Q", title=f"{arr_component.replace('_', ' ')} ($M)"),
    ).properties(height=300, title=f"{arr_component.replace('_', ' ')} Over Time")
    st.altair_chart(arr_chart, use_container_width=True)

    # Diagnosis to Action framework (slide 3)
    st.markdown("---")
    st.subheader("From Diagnosis to Action")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("### 1. Identify Revenue Drag")
        st.metric("Annual Leakage", "$9.42M")
        st.caption("Churn ($4.15M) + Contraction ($5.27M)")
    with d2:
        st.markdown("### 2. Maximize Capital Efficiency")
        st.markdown("High customer acquisition costs (CAC) make **replacing** lost revenue "
                    "costlier than **retaining** it.")
    with d3:
        st.markdown("### 3. Predict & Prevent")
        st.markdown("Behavioral churn model flags at-risk accounts early — enabling "
                    "proactive intervention before renewal.")

    # Leakage breakdown tool
    with st.expander("Explore leakage breakdown"):
        leakage = pd.DataFrame([
            {"Type": "Contraction (downgrades)", "Amount ($M)": 5.27,
             "Insight": "Users stepping down tiers — outpaces full cancellations"},
            {"Type": "Churn (cancellations)", "Amount ($M)": 4.15,
             "Insight": "Full account losses"},
        ])
        st.dataframe(leakage, use_container_width=True, hide_index=True)
        st.info(
            "Account downgrades (-$5.27M) outpace full cancellations (-$4.15M). "
            "Users are **stepping down** more than leaving."
        )


# =============================================================================
# TAB 2 — DATA MODELING (Slides 4–5)
# =============================================================================
with tab_model:
    st.header("Data Modeling")
    st.subheader("Model Selection Process & Feature Engineering")

    # Model comparison (slide 4)
    st.markdown("##### Model Comparison (AUC on held-out test set)")
    mod1, mod2, mod3 = st.columns(3)
    mod1.metric("Logistic Regression", "0.8427")
    mod2.metric("Random Forest", "0.8639")
    with mod3:
        st.metric("Gradient Boosting", "0.8661")
        st.caption("SELECTED")

    why_col, feat_col = st.columns(2)
    with why_col:
        st.markdown("##### Why Gradient Boosting Won")
        st.markdown(
            "- Captures **non-linear price × engagement** interactions a linear model misses\n"
            "- Matches ensemble-level accuracy with fast, stable in-database inference"
        )
    with feat_col:
        st.markdown("##### Key Engineered Features")
        st.markdown(
            "- **Subscription Price:** dominant churn driver\n"
            "- **Session Intensity:** engagement trend signal\n"
            "- **Tech Comfort & Device:** usage-pattern risk\n"
            "- **Activity Recency:** flags dormant accounts"
        )

    # Feature importance (slide 5)
    st.markdown("---")
    st.subheader("Model Performance & Feature Importance")
    st.metric("AUC", "0.8661",
              delta="Separates renewing from churning well above random chance")

    feature_importance = pd.DataFrame([
        {"Feature": "CURRENT_AMOUNT", "Importance (AUC drop)": 0.072},
        {"Feature": "SESSION_INTENSITY", "Importance (AUC drop)": 0.052},
        {"Feature": "TECH_COMFORT_SCORE", "Importance (AUC drop)": 0.045},
        {"Feature": "PRODUCT_Wellness Tracker", "Importance (AUC drop)": 0.038},
        {"Feature": "PRODUCT_Mindful Living", "Importance (AUC drop)": 0.036},
        {"Feature": "PRODUCT_Premium Health", "Importance (AUC drop)": 0.035},
        {"Feature": "GROSS_SESSION_LENGTH", "Importance (AUC drop)": 0.034},
        {"Feature": "DEVICE_TYPE_Multi-device", "Importance (AUC drop)": 0.022},
        {"Feature": "TENURE_DAYS", "Importance (AUC drop)": 0.015},
        {"Feature": "INCOME_LEVEL_Low", "Importance (AUC drop)": 0.012},
        {"Feature": "EDUCATION_High School", "Importance (AUC drop)": 0.011},
        {"Feature": "TOTAL_NUM_SESSIONS", "Importance (AUC drop)": 0.010},
        {"Feature": "INCOME_LEVEL_Medium", "Importance (AUC drop)": 0.009},
        {"Feature": "INCOME_LEVEL_Very High", "Importance (AUC drop)": 0.008},
        {"Feature": "EDUCATION_Other", "Importance (AUC drop)": 0.005},
    ])

    fi_chart = alt.Chart(feature_importance).mark_bar().encode(
        x=alt.X("Importance (AUC drop):Q", title="Permutation Importance (AUC drop)"),
        y=alt.Y("Feature:N", sort="-x", title=""),
        color=alt.condition(
            alt.datum.Feature == "CURRENT_AMOUNT",
            alt.value("#8B1A4A"),
            alt.value("#4A90A4"),
        ),
    ).properties(height=400, title="Top Predictors of Renewal — Gradient Boosting")
    st.altair_chart(fi_chart, use_container_width=True)

    st.caption(
        "Subscription price and session intensity are the two strongest churn drivers. "
        "Churn is driven by price + engagement — not demographics."
    )


# =============================================================================
# TAB 3 — CHURN PREDICTION (Slide 6)
# =============================================================================
with tab_churn:
    st.header("Churn Prediction")
    st.subheader("Interactive Scorer")
    st.markdown(
        "Score a customer profile using the deployed Gradient Boosting model. "
        "Inputs match the top drivers: **subscription amount, session intensity, "
        "tech comfort, session length, product, and device type.**"
    )

    if model is None or encoder is None:
        st.error(_load_err or "Model artifacts could not be loaded.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Subscription**")
            current_amount = st.number_input(
                "Current Monthly Amount ($)", min_value=5.0, max_value=500.0,
                value=75.0, step=5.0, help="Top driver — 1.6× more important than next"
            )
            product = st.selectbox("Product", category_map['PRODUCT'])
            income_level = st.selectbox("Income Level", category_map['INCOME_LEVEL'])
        with col2:
            st.markdown("**Engagement**")
            session_intensity = st.slider(
                "Session Intensity (0–1)", 0.0, 1.0, 0.5, 0.05,
                help="Ratio of active sessions to available days"
            )
            gross_session_length = st.number_input(
                "Gross Session Length (min)", min_value=0, max_value=5000,
                value=500, step=50
            )
            active_days = st.slider("Active Days (past year)", 0, 365, 150)
        with col3:
            st.markdown("**Profile**")
            tech_comfort = st.slider("Tech Comfort Score (1–10)", 1, 10, 5)
            device_type = st.selectbox("Device Type", category_map['DEVICE_TYPE'])
            prior_renewals = st.number_input(
                "Prior Renewals", min_value=0, max_value=20, value=1
            )

        if st.button("Predict Churn Probability", type="primary"):
            try:
                t_start = time.perf_counter()
                active_quarters = min(4, max(1, active_days // 90))
                tenure_days = max(365, prior_renewals * 365)
                total_sessions = int(session_intensity * active_days * 0.8)
                avg_sessions_per_q = total_sessions / max(active_quarters, 1)

                raw = pd.DataFrame([{
                    'PRODUCT': product, 'INCOME_LEVEL': income_level,
                    'EDUCATION': category_map['EDUCATION'][0],
                    'DEVICE_TYPE': device_type,
                }])
                encoded = encoder.transform(raw)
                encoded_df = pd.DataFrame(
                    encoded, columns=encoder.get_feature_names_out(categorical_cols)
                )
                numeric_df = pd.DataFrame([{
                    'AGE': 35, 'TECH_COMFORT_SCORE': float(tech_comfort),
                    'TOTAL_NUM_SESSIONS': np.log1p(total_sessions),
                    'GROSS_SESSION_LENGTH': np.log1p(gross_session_length),
                    'ACTIVE_DAYS': active_days, 'ACTIVE_QUARTERS': active_quarters,
                    'AVG_SESSIONS_PER_ACTIVE_QUARTER': avg_sessions_per_q,
                    'SESSION_INTENSITY': session_intensity, 'SESSION_TREND': 0.0,
                    'TENURE_DAYS': tenure_days, 'PRIOR_RENEWALS': prior_renewals,
                    'DAYS_SINCE_LAST_ACTIVITY': max(1, 365 - active_days) // 4,
                    'CURRENT_AMOUNT': current_amount,
                    'AVG_PRIOR_AMOUNT': current_amount * 0.95,
                    'AMOUNT_CHANGE': current_amount * 0.05,
                }])

                input_df = pd.concat([numeric_df, encoded_df], axis=1)
                input_df = input_df[model.feature_names_in_]
                renewal_prob = model.predict_proba(input_df)[0][1]
                churn_prob = 1 - renewal_prob
                latency_ms = (time.perf_counter() - t_start) * 1000
                risk = "Low" if churn_prob < 0.05 else "Medium" if churn_prob < 0.15 else "High"

                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Churn Probability", f"{churn_prob:.1%}")
                c2.metric("Renewal Probability", f"{renewal_prob:.1%}")
                with c3:
                    if risk == "Low":
                        st.success(f"Risk: {risk}")
                    elif risk == "Medium":
                        st.warning(f"Risk: {risk}")
                    else:
                        st.error(f"Risk: {risk}")
                st.caption(f"Inference: {latency_ms:.0f} ms")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        # Model Results & Cohort Scoring (slide 6 content)
        st.divider()
        st.subheader("Model Results & Cohort Scoring")

        drv_col, qrt_col = st.columns(2)
        with drv_col:
            st.markdown("##### TOP CHURN DRIVERS")
            st.markdown("### #1 Subscription Amount\n*1.6× the next-closest driver*")
            st.markdown(
                "**Also significant:** Session Intensity, Tech Comfort, "
                "Session Length, Product, Device Type"
            )
            st.caption("Churn is driven by price + engagement — not demographics.")

        with qrt_col:
            st.markdown("##### CHURN RISK BY QUARTILE")
            g1, g2 = st.columns(2)
            g1.metric("Q1 (Lowest Risk)", "1.2%")
            g2.metric("Q4 (Highest Risk)", "41.7%")
            st.markdown("**35× spread** in churn risk across quartiles")
            st.caption(
                "CLTR: $948 (Q1) → $780 (Q4) — gap reflects subscription size, "
                "not retention (99.67–99.73% flat)"
            )

        st.markdown("##### INCOME & CHURN")
        inc1, inc2 = st.columns(2)
        with inc1:
            st.metric("High Income — Churn", "6.4%")
            st.markdown("$1,007 CLTR · **Best acquisition target**")
        with inc2:
            st.metric("Very High Income — Churn", "30.5%")
            st.markdown("*Counterintuitively risky* — nearly as high as Low income")

        st.error(
            "**KEY TAKEAWAY:** Prioritize retention over acquisition — focus on "
            "Premium Health across all income tiers (4 of top 6 at-risk segments)."
        )

        with st.expander("Quartile Detail"):
            cohort_q = pd.DataFrame([
                {"Quartile": "Q1 (Lowest Risk)", "Avg Churn": 0.012, "Customers": 151458,
                 "Avg CLTR ($)": 948, "Gross Retention": 0.988},
                {"Quartile": "Q2", "Avg Churn": 0.042, "Customers": 151458,
                 "Avg CLTR ($)": 893, "Gross Retention": 0.958},
                {"Quartile": "Q3", "Avg Churn": 0.091, "Customers": 151458,
                 "Avg CLTR ($)": 840, "Gross Retention": 0.909},
                {"Quartile": "Q4 (Highest Risk)", "Avg Churn": 0.417, "Customers": 151457,
                 "Avg CLTR ($)": 780, "Gross Retention": 0.583},
            ])
            st.dataframe(cohort_q.style.format({
                "Avg Churn": "{:.1%}", "Gross Retention": "{:.1%}",
                "Avg CLTR ($)": "${:,.0f}", "Customers": "{:,}",
            }), use_container_width=True, hide_index=True)

        with st.expander("Income Segment Detail"):
            inc_seg = pd.DataFrame([
                {"Income Level": "High", "Avg CLTR": 1006.70, "Gross Retention": 0.9475,
                 "Avg Churn": 0.0639, "Customers": 261155},
                {"Income Level": "Medium", "Avg CLTR": 520.84, "Gross Retention": 0.8313,
                 "Avg Churn": 0.1814, "Customers": 232103},
                {"Income Level": "Very High", "Avg CLTR": 239.18, "Gross Retention": 0.6919,
                 "Avg Churn": 0.3047, "Customers": 71219},
                {"Income Level": "Low", "Avg CLTR": 136.04, "Gross Retention": 0.5617,
                 "Avg Churn": 0.3880, "Customers": 41354},
            ])
            st.dataframe(inc_seg.style.format({
                "Avg CLTR": "${:,.2f}", "Gross Retention": "{:.2%}",
                "Avg Churn": "{:.2%}", "Customers": "{:,}",
            }), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 4 — PORTFOLIO ANALYSIS (Slide 7)
# =============================================================================
with tab_portfolio:
    st.header("Portfolio Analysis")
    st.subheader("Product Optimization & Revenue Alignment")

    # Product classification — slide 7 layout
    col_ge, col_ra, col_cd = st.columns(3)
    with col_ge:
        st.markdown("#### 🟢 GROWTH ENGINE")
        st.markdown("### Premium Health")
        st.metric("Churn Risk", "22.0%")
        st.caption("$18.39M ARR · 78.2% renewal (+5.0pt YoY)")
        st.markdown(
            "*Highest in the portfolio, despite improving retention.* "
            "Maintain investment as the top-of-funnel anchor, but monitor churn closely."
        )
    with col_ra:
        st.markdown("#### 🟡 RETENTION ANCHORS")
        st.markdown("### Daily Fitness")
        st.metric("Churn Risk — Lowest", "12.7%")
        st.caption("$8.81M ARR | 88.0% renewal")
        st.markdown("### Healthy Meals")
        st.metric("Churn Risk", "11.9%")
        st.caption("$11.80M ARR | 86.2% renewal")
        st.markdown("*Export Daily Fitness's onboarding playbook portfolio-wide.*")
    with col_cd:
        st.markdown("#### 🔴 CONTRACTION DRAG")
        st.markdown("### Mindful Living")
        st.metric("Churn Risk", "16.3%")
        st.caption("$13.16M ARR (from $15.52M) | 75.0% renewal")
        st.markdown("### Wellness Tracker")
        st.metric("Churn Risk", "15.5%")
        st.caption("$5.95M ARR (from $6.41M) | 77.7% renewal")
        st.markdown("*Bundle into Premium Health tiers as retention-protected add-ons.*")

    # Retention trend explorer
    st.divider()
    st.subheader("Retention Trend Explorer")
    try:
        pf = load_portfolio_data()
        products_available = [p for p in pf["product"].unique() if p != "ALL (Aggregate)"]
        selected_products = st.multiselect(
            "Compare products", products_available, default=products_available,
            key="portfolio_compare"
        )
        show_agg = st.checkbox("Show portfolio aggregate", value=True, key="show_agg")
        filter_list = selected_products.copy()
        if show_agg:
            filter_list.append("ALL (Aggregate)")
        pf_filtered = pf[pf["product"].isin(filter_list)]

        chart = alt.Chart(pf_filtered).mark_line(point=True).encode(
            x=alt.X("year_starts:T", title="Period"),
            y=alt.Y("gross_retention_rate:Q", title="Gross Retention Rate",
                    axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0.7, 1.0])),
            color=alt.Color("product:N", title="Product"),
            strokeWidth=alt.condition(
                alt.datum.product == "ALL (Aggregate)",
                alt.value(3), alt.value(1.5)
            ),
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
    except FileNotFoundError:
        st.info("Retention data not available.")

    # Portfolio Classifier tool
    st.divider()
    st.subheader("Portfolio Classifier Tool")
    st.markdown("Select a product to see why it receives its classification.")

    PRODUCT_METRICS = {
        "Premium Health": {"arr_current": 18.39, "arr_prior": 13.50, "churn_risk": 0.220,
                           "retention": 0.782, "retention_prior": 0.732},
        "Daily Fitness": {"arr_current": 8.81, "arr_prior": 7.94, "churn_risk": 0.127,
                          "retention": 0.878, "retention_prior": 0.862},
        "Healthy Meals": {"arr_current": 11.80, "arr_prior": 10.70, "churn_risk": 0.119,
                          "retention": 0.863, "retention_prior": 0.883},
        "Mindful Living": {"arr_current": 13.16, "arr_prior": 15.52, "churn_risk": 0.163,
                           "retention": 0.752, "retention_prior": 0.789},
        "Wellness Tracker": {"arr_current": 5.95, "arr_prior": 6.41, "churn_risk": 0.155,
                             "retention": 0.776, "retention_prior": 0.784},
    }

    classify_product = st.selectbox("Analyze product", list(PRODUCT_METRICS.keys()),
                                    key="classifier")
    pm = PRODUCT_METRICS[classify_product]
    arr_growth = (pm["arr_current"] - pm["arr_prior"]) / pm["arr_prior"]
    ret_change = pm["retention"] - pm["retention_prior"]

    if arr_growth > 0.10 and pm["retention"] > 0.70:
        category, color, action = "Growth Engine", "🟢", "Continue investing — monitor churn."
    elif pm["churn_risk"] < 0.13 and pm["retention"] > 0.85:
        category, color, action = "Retention Anchor", "🟡", "Export onboarding playbook."
    else:
        category, color, action = "Contraction Drag", "🔴", "Bundle into Premium Health tiers."

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("ARR", f"${pm['arr_current']:.2f}M", delta=f"{arr_growth:+.1%} YoY")
    pc2.metric("Churn Risk", f"{pm['churn_risk']:.1%}")
    pc3.metric("Retention", f"{pm['retention']:.1%}", delta=f"{ret_change:+.1%} YoY")
    pc4.metric("Classification", f"{color} {category}")
    st.info(f"**Action:** {action}")


# =============================================================================
# TAB 5 — BUSINESS IMPACT (Slides 8–10)
# =============================================================================
with tab_impact:
    st.header("Business Impact")

    # Strategic Priority Matrix (slide 8)
    st.subheader("Strategic Priority Matrix")
    st.markdown("**Business Impact** (vertical) × **Ease of Execution** (horizontal)")

    hi_left, hi_right = st.columns(2)
    lo_left, lo_right = st.columns(2)
    with hi_left:
        st.markdown(
            "##### Repackage Contraction Products\n"
            "### $5.27M\n"
            "*contraction stopped | strategic value driver*"
        )
    with hi_right:
        st.markdown(
            "##### Protect At-Risk Retention\n"
            "### $248.97M\n"
            "*CLTR at risk | over 25× acquisition opportunity*"
        )
    with lo_left:
        st.markdown(
            "##### Standardize Onboarding\n"
            "### 88%\n"
            "*renewal target | scale Daily Fitness's playbook*"
        )
    with lo_right:
        st.markdown(
            "##### Target High-Income Acquisition\n"
            "### $1,007\n"
            "*avg CLTV | immediate win*"
        )

    # Phased Implementation Roadmap (slide 9)
    st.divider()
    st.subheader("Phased Implementation Roadmap")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("### Phase 1\n**Months 1–3: Stem the Leakage**")
        st.markdown(
            "Redirect acquisition spend to high-income audiences and deploy "
            "churn-risk triggers on flagged accounts."
        )
    with p2:
        st.markdown("### Phase 2\n**Months 3–6: Restructure the Tiers**")
        st.markdown(
            "Bundle Mindful Living and Wellness Tracker into Premium Health tiers; "
            "launch renewal-protection marketing 60 days pre-renewal."
        )
    with p3:
        st.markdown("### Phase 3\n**Months 6–12: Scale What Works**")
        st.markdown(
            "Scale Daily Fitness's onboarding playbook portfolio-wide to attempt "
            "to replicate their 88% renewal standard."
        )

    # Retention improvement simulator (tool)
    st.divider()
    st.subheader("Revenue Recovery Simulator")
    st.markdown("Model the impact of retention improvements on ARR leakage.")

    PRODUCT_ARR = {"Premium Health": 18.39, "Mindful Living": 13.16,
                   "Healthy Meals": 11.80, "Daily Fitness": 8.81, "Wellness Tracker": 5.95}
    PRODUCT_RET = {"Premium Health": 0.782, "Mindful Living": 0.752,
                   "Healthy Meals": 0.863, "Daily Fitness": 0.878, "Wellness Tracker": 0.776}

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_product = st.selectbox("Product", list(PRODUCT_ARR.keys()), index=1, key="sim_p")
        current_ret = PRODUCT_RET[sim_product]
        current_arr = PRODUCT_ARR[sim_product]
        st.caption(f"Current: {current_ret:.1%} retention, ${current_arr:.2f}M ARR")
    with sim_col2:
        default_t = 85 if sim_product == "Mindful Living" else min(int(current_ret * 100) + 5, 95)
        target_ret = st.slider("Target retention", int(current_ret * 100), 95,
                               default_t, format="%d%%", key="sim_s")

    improvement = (target_ret / 100) - current_ret
    saved_arr = current_arr * (improvement / (1 - current_ret))
    s1, s2, s3 = st.columns(3)
    s1.metric("Retention Gain", f"+{improvement:.1%}")
    s2.metric("ARR Preserved", f"${saved_arr:.2f}M")
    s3.metric("Fewer Churns", f"{improvement / (1 - current_ret):.0%}")

    if saved_arr > 0.5:
        st.success(
            f"Improving {sim_product} to {target_ret}% retention preserves "
            f"**${saved_arr:.2f}M** in ARR annually."
        )

    # Executive Summary (slide 10)
    st.divider()
    st.subheader("Executive Summary — Diagnosis to Action")

    diag, solution, impact = st.columns(3)
    with diag:
        st.markdown("##### THE BUSINESS DIAGNOSIS")
        st.markdown(
            "**Revenue Drag:** Growth flattened at $58.1M ARR "
            "due to $9.42M in total annual leakage\n\n"
            "**Contraction Threat:** Account downgrades (-$5.27M) outpace "
            "full cancellations (-$4.15M)\n\n"
            "**Portfolio Divergence:** Premium Health leads at $18.39M ARR "
            "while Mindful Living fell 15% from peak"
        )
    with solution:
        st.markdown("##### THE ANALYTICAL SOLUTION")
        st.markdown(
            "**Snowpark ML Pipeline:** Deploy a 0.8661 AUC model directly "
            "inside Snowflake to segment subscribers into risk quartiles\n\n"
            "**High-CLTV Acquisition:** Reallocate marketing toward high-income "
            "audiences ($1,007 avg CLTR)\n\n"
            "**Portfolio Restructuring:** Stop selling Mindful Living and "
            "Wellness Tracker standalone; bundle into Premium Health"
        )
    with impact:
        st.markdown("##### EXPECTED BUSINESS IMPACT")
        st.markdown(
            "**Retention Opportunity:** $248.97M in at-risk lifetime revenue "
            "now identifiable — 25–30× the acquisition opportunity\n\n"
            "**Revenue Protection:** Recover up to $9.42M in annual leakage\n\n"
            "**Capital Efficiency:** Protect margin by retaining subscribers "
            "at a fraction of new acquisition costs"
        )
