# Customer Sentiment & Support Analytics

An end-to-end **Data Science and Machine Learning project** built to analyze customer feedback, identify sentiment and complaint patterns, predict support outcomes, and present insights through an interactive Streamlit application.

The project covers the complete data science workflow:

**Data Cleaning → EDA → Feature Analysis → Machine Learning → Model Comparison → Model Saving → Streamlit Deployment**

---

## 👨‍💻 Author

**Muhammad Shahbaz Khan**

**Data Analyst | Aspiring Data Scientist**

### Education

* BS Information Technology (BSIT)
* Government Emerson College, Multan — Intermediate
* Govt. Model High-School Kabirwala — Matriculation

### Areas of Interest

* Data Analytics
* Data Science
* Machine Learning
* Exploratory Data Analysis
* Business Intelligence
* Natural Language Processing
* Data Visualization

---

## 📌 Project Overview

This project analyzes a **25,000-row customer feedback dataset** and builds machine learning solutions for three classification problems:

1. **Customer Sentiment Prediction**
2. **Complaint Registration Prediction**
3. **Issue Resolution Prediction**

The project also includes an interactive **Streamlit dashboard** where users can enter customer information, generate predictions, explore dataset insights, and compare machine learning algorithms.

---

## 🎯 Project Objectives

* Perform exploratory data analysis on customer feedback.
* Understand relationships between ratings, complaints, response time, and issue resolution.
* Clean and prepare the dataset for machine learning.
* Apply NLP techniques for sentiment classification.
* Train and compare multiple machine learning algorithms.
* Save trained models for application use.
* Build an interactive Streamlit application.
* Present model performance and analytical insights in an easy-to-use interface.

---

## 📊 Dataset

The dataset contains **25,000 customer feedback records**.

### Main Variables

* Customer rating
* Review text
* Complaint registration
* Issue resolution
* Support response time
* Gender
* Age
* Region
* Product
* Platform
* Other customer/support attributes

### Dataset Files

```text
data/
├── customer_sentiment.csv
└── customer_sentiment_clean.csv
```

---

## 🔍 Exploratory Data Analysis

The EDA investigated:

* Dataset structure
* Missing values
* Duplicate records
* Numerical distributions
* Categorical distributions
* Customer ratings
* Sentiment distribution
* Complaint patterns
* Issue resolution patterns
* Response-time relationships
* Review-text patterns
* Demographic relationships
* Product and platform relationships

### Key EDA Findings

* `sentiment` is a deterministic function of `customer_rating`:

  * **1–2 → Negative**
  * **3 → Neutral**
  * **4–5 → Positive**

* `complaint_registered` is also determined by rating:

  * **Rating ≤ 2 → Complaint**

* `review_text` is highly templated, with only **15 distinct sentences** across the 25,000 records.

* `issue_resolved` is strongly associated with support response time, with resolution changing around the **~48-hour threshold** in this dataset.

* Demographic and product/platform variables provide little independent signal for the targets investigated.

---

## 🤖 Machine Learning

Three classification tasks were developed.

### 1. Customer Sentiment Prediction

**Input:**

```text
review_text
```

**Technique:**

```text
TF-IDF Vectorization
```

**Algorithms evaluated:**

* Naive Bayes
* Logistic Regression
* Linear SVM
* Random Forest

---

### 2. Complaint Registration Prediction

**Input features include:**

* Customer rating
* Response time
* Demographics
* Platform
* Other relevant features

**Algorithms evaluated:**

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

---

### 3. Issue Resolution Prediction

**Input features include:**

* Customer rating
* Response time
* Demographics
* Platform
* Complaint status
* Other relevant features

**Algorithms evaluated:**

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

---

## 📈 Model Comparison

Multiple classification algorithms were trained and evaluated using appropriate classification metrics.

The project stores model comparison results in the `models/` directory.

```text
models/
├── complaint_model.pkl
├── complaint_label_encoder.pkl
├── complaint_model_comparison.csv
├── sentiment_model_comparison.csv
├── resolved_model_comparison.csv
└── model_summary.json
```

The comparisons help identify how different algorithms perform on each classification task.

---

## 🧠 Important Modeling Insight

The extremely high performance of the sentiment and complaint models should **not automatically be interpreted as evidence that the models have discovered complex customer behavior**.

The EDA shows that these targets are directly determined by customer rating in this dataset.

Therefore:

```text
Rating → Sentiment
Rating → Complaint
```

This creates a very strong predictive relationship.

The issue-resolution task provides a more meaningful example of the model learning a relationship involving **support response time**.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application.

### Application Features

### 🔮 Predict

Users can enter customer information and generate predictions using the trained models.

### 📊 Dataset Insights

Interactive Plotly visualizations allow users to explore:

* Customer ratings
* Sentiment
* Complaints
* Issue resolution
* Response time
* Other dataset patterns

### 🤖 Algorithm Comparison

Displays:

* Model performance metrics
* Algorithm comparisons
* ROC curves
* Classification results

---

## 📁 Project Structure

```text
customer-sentiment-analysis-ml/
│
├── data/
│   ├── customer_sentiment.csv
│   └── customer_sentiment_clean.csv
│
├── notebooks/
│   ├── eda.ipynb
│   └── model_training.ipynb
│
├── models/
│   ├── complaint_label_encoder.pkl
│   ├── complaint_model.pkl
│   ├── complaint_model_comparison.csv
│   ├── sentiment_model_comparison.csv
│   ├── resolved_model_comparison.csv
│   └── model_summary.json
│
├── app/
│   ├── app.py
│   └── requirements.txt
│
├── README.md
└── ...
```

---

## 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn
* Plotly

### Machine Learning

* Scikit-learn
* XGBoost

### Natural Language Processing

* TF-IDF
* Text preprocessing
* Naive Bayes
* Logistic Regression
* Linear SVM

### Application

* Streamlit

### Development Tools

* Jupyter Notebook
* Git
* GitHub

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/muhammadshahbazkhan021-create/customer-sentiment-analysis-ml.git
```

Move into the project directory:

```bash
cd customer-sentiment-analysis-ml
```

Install the required packages:

```bash
cd app
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit App

From the `app` directory:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📓 Notebooks

### EDA Notebook

```text
notebooks/eda.ipynb
```

Contains the exploratory analysis, visualizations, distributions, relationships, and key dataset findings.

### Model Training Notebook

```text
notebooks/model_training.ipynb
```

Contains:

* Data preparation
* Feature engineering
* Model training
* Algorithm comparison
* Model evaluation
* Model saving

---

## 🚀 Deployment

The application is structured for deployment using **Streamlit Community Cloud**.

The application uses paths relative to the application's location, allowing the project to work independently of the current working directory.

---

## 💡 Key Skills Demonstrated

This project demonstrates practical experience with:

* Data Cleaning
* Exploratory Data Analysis
* Statistical Analysis
* Feature Engineering
* Classification
* Natural Language Processing
* TF-IDF
* Model Evaluation
* Algorithm Comparison
* Model Persistence
* Data Visualization
* Streamlit
* Git & GitHub
* End-to-End Machine Learning Workflow

---

## 📚 Learning Outcomes

Through this project, I practiced taking a real-world style dataset from raw data to a working machine learning application.

The project helped strengthen my understanding of:

```text
Raw Data
   ↓
Data Cleaning
   ↓
EDA
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Model Saving
   ↓
Streamlit Application
   ↓
Deployment
```

---

## 👤 About Me

I am **Muhammad Shahbaz Khan**, a BSIT graduate focused on developing my skills in **Data Analytics, Data Science, and Machine Learning**.

My current focus is on building practical projects that combine:

**Python + SQL + Power BI + Statistics + Machine Learning + Data Visualization**

I am interested in opportunities where I can apply data-driven problem solving and continue developing professionally in the data field.

---

## 📫 Connect With Me

**GitHub:**
https://github.com/muhammadshahbazkhan021-create

**Repository:**
https://github.com/muhammadshahbazkhan021-create/customer-sentiment-analysis-ml

---

## ⭐ If You Find This Project Useful

Feel free to explore the notebooks, models, analysis, and Streamlit application.

---

### 📌 Project Status

**Completed — Portfolio Project**

Built as an end-to-end Data Science / Machine Learning project.
