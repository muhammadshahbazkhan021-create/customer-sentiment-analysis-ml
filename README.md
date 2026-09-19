# Customer Sentiment & Support Analytics

An end-to-end data science project on a 25,000-row customer feedback dataset — covering exploratory data analysis, three machine learning classification tasks, and an interactive Streamlit app.

## 📁 Project structure

```
├── data/
│   ├── customer_sentiment.csv          # raw data
│   └── customer_sentiment_clean.csv    # cleaned (purchase_channel dropped)
├── notebooks/
│   ├── eda.ipynb                       # narrative exploratory analysis
│   └── model_training.ipynb            # 3 ML tasks, 6-algorithm comparison
├── models/                             # saved best models + comparison tables
├── app/
│   ├── app.py                          # Streamlit app
│   └── requirements.txt
└── README.md
```

## 🔍 Key EDA findings

- `sentiment` is a perfect, deterministic function of `customer_rating` (1–2 → negative, 3 → neutral, 4–5 → positive)
- `complaint_registered` is also fully determined by rating (rating ≤ 2 ⇒ complaint, always)
- `review_text` is templated — only 15 distinct sentences exist across all 25,000 rows
- `issue_resolved` is the one genuinely useful target: resolved whenever support response time is under ~48 hours, unresolved otherwise
- Demographics (gender, age, region) and product/platform carry no independent signal on any of the above

## 🤖 Modeling approach

| Task | Input features | Approach |
|---|---|---|
| **Sentiment** | `review_text` only (NLP) | TF-IDF + Naive Bayes / Logistic Regression / Linear SVM / Random Forest |
| **Complaint registered** | rating, response time, demographics, platform | Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| **Issue resolved** | rating, response time, demographics, platform, complaint flag | Same 6-algorithm comparison |

Sentiment and complaint models score near-perfectly because those targets are direct restatements of the rating; the issue-resolved model demonstrates a model correctly learning the ~48-hour response-time threshold.

## 🖥️ Running the app

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

The app has three tabs:
- **Predict** — try all three models with your own inputs
- **Dataset Insights** — interactive, filterable Plotly charts
- **Algorithm Comparison** — metrics tables and ROC curves for every algorithm tried

## 🚀 Deployment

Ready for Streamlit Community Cloud: point it at `app/app.py` with `app/requirements.txt`. Model paths are resolved relative to the script's own directory, so it works regardless of the working directory the app is launched from.
