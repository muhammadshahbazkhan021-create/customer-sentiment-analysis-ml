import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "customer_sentiment_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

st.set_page_config(page_title="Customer Sentiment Analytics", page_icon="🛒", layout="wide")

# ---------------------------------------------------------------
# Load data & models
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    sentiment_model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.pkl"))
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    complaint_model = joblib.load(os.path.join(MODELS_DIR, "complaint_model.pkl"))
    complaint_le = joblib.load(os.path.join(MODELS_DIR, "complaint_label_encoder.pkl"))
    resolved_model = joblib.load(os.path.join(MODELS_DIR, "resolved_model.pkl"))
    resolved_le = joblib.load(os.path.join(MODELS_DIR, "resolved_label_encoder.pkl"))
    with open(os.path.join(MODELS_DIR, "model_summary.json")) as f:
        summary = json.load(f)
    return sentiment_model, tfidf, complaint_model, complaint_le, resolved_model, resolved_le, summary

df = load_data()
(sentiment_model, tfidf, complaint_model, complaint_le,
 resolved_model, resolved_le, summary) = load_models()

cat_vals = summary["categorical_values"]

st.title("🛒 Customer Sentiment & Support Analytics")
st.caption("25,000 online orders — sentiment, complaints, and issue resolution, explored and predicted.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Dataset Insights", "🏆 Algorithm Comparison"])

# ---------------------------------------------------------------
# TAB 1 — PREDICT
# ---------------------------------------------------------------
with tab1:
    st.subheader("Try the models yourself")

    sub1, sub2, sub3 = st.tabs(["Sentiment from a review", "Will this be a complaint?", "Will the issue get resolved?"])

    with sub1:
        st.markdown("Type (or paste) a customer review and see what the model predicts.")
        review = st.text_area(
            "Review text",
            value="fast delivery and great packaging.",
            height=100,
        )
        if st.button("Predict sentiment", key="sent_btn"):
            X = tfidf.transform([review])
            pred = sentiment_model.predict(X)[0]
            emoji = {"positive": "😊", "neutral": "😐", "negative": "😠"}.get(pred, "")
            st.success(f"Predicted sentiment: **{pred.upper()}** {emoji}")

    with sub2:
        st.markdown("Fill in order details to check complaint likelihood.")
        c1, c2, c3 = st.columns(3)
        with c1:
            rating_c = st.slider("Customer rating", 1, 5, 3, key="rating_c")
            gender_c = st.selectbox("Gender", cat_vals["gender"], key="gender_c")
        with c2:
            age_c = st.selectbox("Age group", cat_vals["age_group"], key="age_c")
            region_c = st.selectbox("Region", cat_vals["region"], key="region_c")
        with c3:
            product_c = st.selectbox("Product category", cat_vals["product_category"], key="product_c")
            platform_c = st.selectbox("Platform", cat_vals["platform"], key="platform_c")
        response_c = st.slider("Support response time (hours)", 1, 71, 24, key="response_c")

        if st.button("Predict complaint likelihood", key="complaint_btn"):
            row = pd.DataFrame([{
                "customer_rating": rating_c,
                "response_time_hours": response_c,
                "gender": gender_c,
                "age_group": age_c,
                "region": region_c,
                "product_category": product_c,
                "platform": platform_c,
            }])
            pred_enc = complaint_model.predict(row)[0]
            proba = complaint_model.predict_proba(row)[0]
            pred_label = complaint_le.inverse_transform([pred_enc])[0]
            yes_idx = list(complaint_le.classes_).index("yes")
            if pred_label == "yes":
                st.error(f"⚠️ Likely to register a complaint (confidence: {proba[yes_idx]:.1%})")
            else:
                st.success(f"✅ Unlikely to complain (confidence: {1 - proba[yes_idx]:.1%})")

    with sub3:
        st.markdown("Check whether an issue is likely to get resolved, based on response time and context.")
        c1, c2, c3 = st.columns(3)
        with c1:
            rating_r = st.slider("Customer rating", 1, 5, 3, key="rating_r")
            gender_r = st.selectbox("Gender", cat_vals["gender"], key="gender_r")
        with c2:
            age_r = st.selectbox("Age group", cat_vals["age_group"], key="age_r")
            region_r = st.selectbox("Region", cat_vals["region"], key="region_r")
        with c3:
            product_r = st.selectbox("Product category", cat_vals["product_category"], key="product_r")
            platform_r = st.selectbox("Platform", cat_vals["platform"], key="platform_r")
        response_r = st.slider("Support response time (hours)", 1, 71, 24, key="response_r")
        complaint_r = st.selectbox("Complaint registered?", ["no", "yes"], key="complaint_r")

        if st.button("Predict resolution outcome", key="resolved_btn"):
            row = pd.DataFrame([{
                "customer_rating": rating_r,
                "response_time_hours": response_r,
                "gender": gender_r,
                "age_group": age_r,
                "region": region_r,
                "product_category": product_r,
                "platform": platform_r,
                "complaint_registered": complaint_r,
            }])
            pred_enc = resolved_model.predict(row)[0]
            proba = resolved_model.predict_proba(row)[0]
            pred_label = resolved_le.inverse_transform([pred_enc])[0]
            yes_idx = list(resolved_le.classes_).index("yes")
            if pred_label == "yes":
                st.success(f"✅ Likely to be resolved (confidence: {proba[yes_idx]:.1%})")
            else:
                st.error(f"⚠️ Likely to stay unresolved (confidence: {1 - proba[yes_idx]:.1%})")
            if response_r >= 48:
                st.info("Response time is at or above the ~48 hour mark, which is where resolutions drop off sharply in this data.")

# ---------------------------------------------------------------
# TAB 2 — DATASET INSIGHTS
# ---------------------------------------------------------------
with tab2:
    st.subheader("Explore the data")

    with st.expander("Filters", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sel_gender = st.multiselect("Gender", cat_vals["gender"], default=cat_vals["gender"])
        with f2:
            sel_region = st.multiselect("Region", cat_vals["region"], default=cat_vals["region"])
        with f3:
            sel_product = st.multiselect("Product category", cat_vals["product_category"], default=cat_vals["product_category"])
        with f4:
            sel_platform = st.multiselect("Platform", cat_vals["platform"], default=cat_vals["platform"])

    fdf = df[
        df["gender"].isin(sel_gender)
        & df["region"].isin(sel_region)
        & df["product_category"].isin(sel_product)
        & df["platform"].isin(sel_platform)
    ]

    st.caption(f"Showing {len(fdf):,} of {len(df):,} orders")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg. rating", f"{fdf['customer_rating'].mean():.2f}")
    m2.metric("Complaint rate", f"{(fdf['complaint_registered'] == 'yes').mean():.1%}")
    m3.metric("Resolution rate", f"{(fdf['issue_resolved'] == 'yes').mean():.1%}")
    m4.metric("Avg. response time", f"{fdf['response_time_hours'].mean():.1f}h")

    g1, g2 = st.columns(2)
    with g1:
        fig = px.histogram(fdf, x="sentiment", color="sentiment",
                            color_discrete_map={"positive": "#2ecc71", "neutral": "#f1c40f", "negative": "#e74c3c"},
                            title="Sentiment distribution")
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig = px.histogram(fdf, x="customer_rating", color="sentiment",
                            color_discrete_map={"positive": "#2ecc71", "neutral": "#f1c40f", "negative": "#e74c3c"},
                            barmode="stack", title="Rating vs sentiment")
        st.plotly_chart(fig, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        fig = px.box(fdf, x="issue_resolved", y="response_time_hours", color="issue_resolved",
                     title="Response time by resolution outcome")
        fig.add_hline(y=48, line_dash="dash", line_color="red", annotation_text="~48h cutoff")
        st.plotly_chart(fig, use_container_width=True)
    with g4:
        cat_rate = fdf.groupby("product_category")["complaint_registered"].apply(lambda s: (s == "yes").mean()).sort_values()
        fig = px.bar(cat_rate, orientation="h", title="Complaint rate by product category",
                     labels={"value": "Complaint rate", "product_category": "Category"})
        st.plotly_chart(fig, use_container_width=True)

    g5, g6 = st.columns(2)
    with g5:
        plat_rate = fdf.groupby("platform")["issue_resolved"].apply(lambda s: (s == "yes").mean()).sort_values()
        fig = px.bar(plat_rate, orientation="h", title="Resolution rate by platform")
        st.plotly_chart(fig, use_container_width=True)
    with g6:
        fig = px.sunburst(fdf, path=["region", "product_category"], title="Orders by region → product category")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**Key pattern:** `sentiment` and `complaint_registered` are both direct restatements of `customer_rating`, "
        "while `issue_resolved` is driven almost entirely by `response_time_hours` — resolutions fall off sharply "
        "once response time crosses ~48 hours."
    )

# ---------------------------------------------------------------
# TAB 3 — ALGORITHM COMPARISON
# ---------------------------------------------------------------
with tab3:
    st.subheader("Model comparison")

    task = st.radio("Choose a task", ["Sentiment (text)", "Complaint registered", "Issue resolved"], horizontal=True)

    if task == "Sentiment (text)":
        comp = pd.read_csv(os.path.join(MODELS_DIR, "sentiment_model_comparison.csv"))
        st.dataframe(comp.style.highlight_max(subset=["Accuracy", "F1 Score"], color="lightgreen"), use_container_width=True)
        fig = px.bar(comp, x="Model", y="F1 Score", title="F1 Score by model — Sentiment (from review text)")
        st.plotly_chart(fig, use_container_width=True)

    elif task == "Complaint registered":
        comp = pd.read_csv(os.path.join(MODELS_DIR, "complaint_model_comparison.csv"))
        st.dataframe(comp.style.highlight_max(subset=["Accuracy", "F1 Score", "ROC-AUC"], color="lightgreen"), use_container_width=True)
        fig = px.bar(comp, x="Model", y="ROC-AUC", title="ROC-AUC by model — Complaint Registered")
        st.plotly_chart(fig, use_container_width=True)

    else:
        comp = pd.read_csv(os.path.join(MODELS_DIR, "resolved_model_comparison.csv"))
        st.dataframe(comp.style.highlight_max(subset=["Accuracy", "F1 Score", "ROC-AUC"], color="lightgreen"), use_container_width=True)
        fig = px.bar(comp, x="Model", y="ROC-AUC", title="ROC-AUC by model — Issue Resolved")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "Most algorithms score near-perfectly on `complaint_registered` and `issue_resolved` because both targets "
        "have a clean, deterministic driver in this dataset (rating and response time respectively) — see the "
        "EDA notebook for the full explanation."
    )
