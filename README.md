# 🏠 Housing Price Prediction System (End-to-End ML Pipeline)

## 📌 Overview

This project builds a **complete Machine Learning pipeline** to predict housing prices based on location and socio-economic features such as:

* Median income
* Population
* Number of rooms
* Latitude & longitude
* Ocean proximity

The system is divided into two key parts:

* **Model experimentation & evaluation (`main_old.py`)**
* **Production-ready training & inference (`main.py`)**

---

## 🎯 Objectives

* Understand the impact of location-based features on housing prices
* Compare multiple regression models
* Select the best model using evaluation metrics
* Build a reusable and production-ready ML pipeline
* Perform batch predictions on new data

---

## 🧱 Project Architecture

```text
housing.csv
   ↓
Stratified Sampling (income_category)
   ↓
Train/Test Split
   ↓
Preprocessing Pipeline
   ↓
Model Training (main_old.py)
   ↓
Model Evaluation (RMSE + Cross Validation)
   ↓
Model Selection (Random Forest)
   ↓
Production Pipeline (main.py)
   ↓
Model Serialization (joblib)
   ↓
Inference (input.csv → output.csv)
```

---

## 📂 Project Structure

```text
project/
│
├── housing.csv          # Dataset
├── input.csv            # Input for inference
├── output.csv           # Predictions
│
├── main_old.py          # Model experimentation & evaluation
├── main.py              # Production-ready pipeline
│
├── model.pkl            # Saved trained model
├── pipeline.pkl         # Saved preprocessing pipeline
│
└── README.md
```

---

# 🔬 Part 1: Model Experimentation (`main_old.py`)

## ⚙️ Workflow

### 1. Data Loading

* Reads dataset using Pandas

### 2. Stratified Sampling

* Creates `income_category` using `median_income`
* Ensures balanced train-test distribution

### 3. Feature Engineering

* Separates:

  * Features
  * Target (`median_house_value`)

### 4. Preprocessing Pipeline

#### Numerical Features

* Median Imputation
* Standard Scaling

#### Categorical Features

* OneHot Encoding

#### Combined using:

* `ColumnTransformer`

---

## 🤖 Models Evaluated

### 1. Linear Regression

* RMSE: **~69,050**

👉 Insight:

* High bias
* Underfitting

---

### 2. Decision Tree Regressor

* RMSE: **0.0 (training)**

👉 Insight:

* Memorizes training data
* Severe overfitting

---

### 3. Random Forest Regressor

* RMSE: **~18,333**

👉 Insight:

* Handles non-linear patterns
* Reduces overfitting
* Best generalization

---

## 📊 Cross-Validation Results

* Used **10-fold cross-validation**
* Metric: **RMSE**

👉 Observations:

* Linear Regression → Stable but high error
* Decision Tree → High variance
* Random Forest → Balanced and reliable

---

## 🏆 Model Selection

✅ **Random Forest Regressor selected** because:

* Lowest realistic RMSE
* Good bias-variance balance
* Robust for real-world data

---

# 🚀 Part 2: Production Pipeline (`main.py`)

## ⚙️ Workflow

### 1. Training Phase (if model not exists)

* Load dataset
* Perform stratified sampling
* Build preprocessing pipeline
* Train Random Forest model
* Save:

  * `model.pkl`
  * `pipeline.pkl`

---

### 2. Inference Phase

* Load saved model & pipeline
* Read `input.csv`
* Transform features
* Predict housing prices
* Save results to `output.csv`

---

## 🔮 Inference Flow

```text
input.csv
   ↓
Load pipeline.pkl
   ↓
Transform features
   ↓
Load model.pkl
   ↓
Predict
   ↓
output.csv
```

---

## ⚠️ Important Implementation Notes

### ✅ Data Leakage Prevention

* `income_category` used only for stratification
* Removed before training

### ✅ Inference Safety Fix

* Drop `median_house_value` before transformation if present

### ❗ Decision Tree Warning

* RMSE = 0 indicates overfitting (not production-safe)

---

## 🧠 Skills & Concepts Covered

* Exploratory Data Analysis (EDA)
* Data Preprocessing & Pipelines
* Feature Scaling & Encoding
* Stratified Sampling
* Cross Validation
* Model Evaluation (RMSE)
* Bias vs Variance Tradeoff
* Model Serialization (joblib)
* Production ML Pipeline Design

---

## 💻 Tools & Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

---

### 2. Run Training (First Time)

```bash
python main.py
```

---

### 3. Run Inference

```bash
python main.py
```

---

## 📈 Future Enhancements

* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
* Add Gradient Boosting / XGBoost
* Feature engineering (e.g., rooms per household)
* Build REST API using FastAPI
* Dockerize the application
* Add monitoring & retraining pipeline

---

## 🎓 Learning Outcome

This project demonstrates:

* Transition from **experimentation → production ML system**
* Proper model evaluation and selection
* Real-world pipeline design
* Scalable and reusable ML architecture

---

## 📌 Conclusion

This project showcases a **complete end-to-end machine learning system**, from raw data processing to deployment-ready inference.

The **Random Forest model** outperforms other models with significantly lower RMSE and strong generalization, making it suitable for real-world housing price prediction tasks.

---
