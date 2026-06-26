# 🫀 Heart Disease Prediction & Classification

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heartdiseaseprediction-jdhyb7q39v8wpxgpn6yg6q.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.0-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

<br/>

**A full-cycle Machine Learning project for predicting heart disease using clinical patient data.**  
Built as part of the B.S. Data Science & AI Capstone at **IIT Jodhpur**.

<br/>

🔗 **[Live Demo — Try it now!](https://heartdiseaseprediction-jdhyb7q39v8wpxgpn6yg6q.streamlit.app/)**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Objective & Metric Justification](#-objective--metric-justification)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Models Trained](#-models-trained)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Streamlit Deployment](#-streamlit-deployment)
- [About the Author](#-about-the-author)

---

## 🧠 Overview

Heart disease is one of the leading causes of death globally. Early and accurate detection can save lives. This project builds, tunes, and evaluates several machine learning classifiers to predict the **presence of heart disease** in patients based on clinical features such as age, cholesterol levels, resting blood pressure, ECG results, and more.

The entire pipeline — from raw data to a live interactive web app — is covered end to end.

---

## 🎯 Objective & Metric Justification

> **Primary Goal:** Predict whether a patient has heart disease (Class 1) or not (Class 0).

In medical diagnostics, **minimizing False Negatives is the highest priority**. A False Negative means a patient *with* heart disease is incorrectly classified as healthy — leading to delayed treatment and potentially fatal outcomes. A False Positive (labeling a healthy patient as sick) leads to further testing but is far less hazardous.

Therefore, our **primary optimization metric** for hyperparameter tuning is:

> ### 🏆 Recall (Sensitivity) for Class 1 (Heart Disease Presence)

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [UCI Heart Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **Samples** | 303 patients |
| **Features** | 13 clinical features + 1 target |
| **Target** | `1` = Heart Disease Present, `0` = Healthy |

### Feature Description

| Feature | Type | Description |
|---|---|---|
| `age` | Continuous | Age in years |
| `sex` | Categorical | Sex (1 = Male, 0 = Female) |
| `cp` | Categorical | Chest pain type (0–3) |
| `trestbps` | Continuous | Resting blood pressure (mmHg) |
| `chol` | Continuous | Serum cholesterol (mg/dl) |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl |
| `restecg` | Categorical | Resting ECG results (0–2) |
| `thalach` | Continuous | Maximum heart rate achieved |
| `exang` | Categorical | Exercise-induced angina |
| `oldpeak` | Continuous | ST depression induced by exercise |
| `slope` | Categorical | Slope of peak exercise ST segment |
| `ca` | Categorical | Number of major vessels colored (0–4) |
| `thal` | Categorical | Thalassemia type |

---

## 🔄 Project Workflow

```
Raw Data
   │
   ▼
Data Loading & Inspection
   │
   ▼
Variable Separation & Type Casting
(Categorical → object, Continuous → float/int)
   │
   ▼
Exploratory Data Analysis (EDA)
├── Continuous Feature Distributions (KDE + Histogram)
├── Categorical Feature Distributions (Bar Charts)
├── Target vs Continuous Features (Mean Bars + KDE)
└── Target vs Categorical Features (Stacked Bars)
   │
   ▼
Outlier Detection & Treatment (IQR Capping)
   │
   ▼
Feature Engineering & Encoding
(One-hot encoding for cp, restecg, thal)
   │
   ▼
Train-Test Split (80/20, Stratified)
   │
   ▼
Model Training & Hyperparameter Tuning
(GridSearchCV + StratifiedKFold, scoring='recall')
   │
   ▼
Model Evaluation & Comparison
(Recall, Precision, F1, Accuracy, ROC-AUC)
   │
   ▼
Champion Model Selection (Best Recall on Class 1)
   │
   ▼
Stacking Classifier (Ensemble of all best models)
   │
   ▼
Model Export (Pickle → .sav file)
   │
   ▼
Streamlit Web App Deployment
```

<div align="center">

![Project Workflow](ChatGPT%20Image%20Jun%2026%2C%202026%2C%2005_34_07%20PM.png)

</div>

---

## 🤖 Models Trained

All models were tuned using **GridSearchCV** with **StratifiedKFold (k=3)**, optimizing for **Recall on Class 1**.

| Model | Tuned Hyperparameters |
|---|---|
| **Decision Tree** | criterion, max_depth, min_samples_split, min_samples_leaf |
| **Random Forest** | n_estimators, criterion, max_depth, min_samples_split, bootstrap |
| **K-Nearest Neighbors** | n_neighbors, weights, p (with StandardScaler pipeline) |
| **Support Vector Machine** | C, kernel, gamma, degree (with StandardScaler pipeline) |
| **Gradient Boosting** | learning_rate, n_estimators, max_depth, subsample |
| **Stacking Classifier** | All above as base estimators + LogisticRegression meta-learner (tuned C) |

---

## 📈 Results

After training, models are evaluated on the held-out test set. The **Champion Model** is automatically selected based on the highest **Recall for Class 1** and saved for deployment.

Metrics tracked per model:

- `precision_0`, `precision_1`
- `recall_0`, `recall_1` ← *Primary metric*
- `f1_0`, `f1_1`
- `macro_avg_precision`, `macro_avg_recall`, `macro_avg_f1`
- `accuracy`
- **ROC-AUC Curve** for all models

> 📝 Refer to the notebook for detailed evaluation tables, confusion matrices, and ROC curves.

---

## 📁 Repository Structure

```
Heart_Disease_Prediction/
│
├── heart_disease.ipynb          # Main Jupyter Notebook (EDA + Modeling + Export)
├── app.py                       # Streamlit web app for deployment
├── heart.csv                    # Raw dataset
├── heart_disease_model.sav      # Trained & serialized champion model (pickle)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation (this file)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Anaconda (recommended) or any virtual environment

### 1. Clone the Repository

```bash
git clone https://github.com/Saikat1462/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

Open `heart_disease.ipynb` in Jupyter Lab or VS Code and **Run All Cells**.  
This will train the models and generate `heart_disease_model.sav`.

### 4. Launch the Streamlit App Locally

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🌐 Streamlit Deployment

The app is live and publicly accessible — no installation required!

> 🔗 **[https://heartdiseaseprediction-jdhyb7q39v8wpxgpn6yg6q.streamlit.app/](https://heartdiseaseprediction-jdhyb7q39v8wpxgpn6yg6q.streamlit.app/)**

### How to use the app:
1. Enter the patient's clinical values using the input fields.
2. Click the **Predict** button.
3. The app instantly returns a **risk prediction** along with the **probability score**.

### Sample Test Cases

**🔴 High Risk Patient (should predict Heart Disease):**
| Field | Value |
|---|---|
| Age | 63 |
| Sex | Male |
| Chest Pain Type | Asymptomatic (3) |
| Resting BP | 145 |
| Cholesterol | 233 |
| Fasting Blood Sugar | True |
| Max Heart Rate | 150 |
| ST Depression | 2.3 |
| Thalassemia | Fixed Defect (1) |

**🟢 Low Risk Patient (should predict Healthy):**
| Field | Value |
|---|---|
| Age | 67 |
| Sex | Male |
| Chest Pain Type | Typical Angina (0) |
| Resting BP | 160 |
| Cholesterol | 286 |
| Exercise Induced Angina | Yes |
| Max Heart Rate | 108 |
| Major Vessels (ca) | 3 |
| Thalassemia | Reversable Defect (2) |

---

## 👤 About the Author

<div align="center">

**SAIKAT SARKAR**

B.S. Data Science & Artificial Intelligence  
**Indian Institute of Technology (IIT) Jodhpur**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/saikat-sarkar-17151a3b1/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Saikat1462)

</div>

---

<div align="center">

If you found this project useful, please consider giving it a ⭐ on GitHub!

</div>
