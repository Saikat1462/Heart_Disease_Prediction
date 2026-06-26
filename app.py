import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open('heart_disease_model.sav', 'rb'))

st.title("Heart Disease Prediction App")
st.write("Enter the patient's medical information below to predict the likelihood of heart disease.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Actual Error:\n{e}")
    st.stop()

# Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox("Chest Pain Type (cp)", options=[
        ("Typical Angina (0)", 0),
        ("Atypical Angina (1)", 1),
        ("Non-anginal Pain (2)", 2),
        ("Asymptomatic (3)", 3)
    ], format_func=lambda x: x[0])[1]
    trestbps = st.number_input("Resting Blood Pressure (trestbps mmHg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (chol mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[("False", 0), ("True", 1)], format_func=lambda x: x[0])[1]
    restecg = st.selectbox("Resting ECG Results (restecg)", options=[
        ("Normal (0)", 0),
        ("ST-T Wave Abnormality (1)", 1),
        ("Left Ventricular Hypertrophy (2)", 2)
    ], format_func=lambda x: x[0])[1]

with col2:
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[
        ("Upsloping (0)", 0),
        ("Flat (1)", 1),
        ("Downsloping (2)", 2)
    ], format_func=lambda x: x[0])[1]
    ca = st.number_input("Number of Major Vessels Colored by Flourosopy (ca)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia (thal)", options=[
        ("Normal (0)", 0),
        ("Fixed Defect (1)", 1),
        ("Reversable Defect (2)", 2),
        ("Other (3)", 3)
    ], format_func=lambda x: x[0])[1]

if st.button("Predict"):
    # Convert categorical variables to dummies exactly as the model expects
    # 'cp_1', 'cp_2', 'cp_3'
    cp_1 = 1 if cp == 1 else 0
    cp_2 = 1 if cp == 2 else 0
    cp_3 = 1 if cp == 3 else 0
    
    # 'restecg_1', 'restecg_2'
    restecg_1 = 1 if restecg == 1 else 0
    restecg_2 = 1 if restecg == 2 else 0
    
    # 'thal_1', 'thal_2', 'thal_3'
    thal_1 = 1 if thal == 1 else 0
    thal_2 = 1 if thal == 2 else 0
    thal_3 = 1 if thal == 3 else 0
    
    # Create feature array in the correct order
    features = [
        age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, slope, ca,
        cp_1, cp_2, cp_3, restecg_1, restecg_2, thal_1, thal_2, thal_3
    ]
    
    # The pipeline model expects a DataFrame with proper column names
    feature_names = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 
                     'oldpeak', 'slope', 'ca', 'cp_1', 'cp_2', 'cp_3', 'restecg_1', 
                     'restecg_2', 'thal_1', 'thal_2', 'thal_3']
    
    df_input = pd.DataFrame([features], columns=feature_names)
    
    try:
        prediction = model.predict(df_input)
        probability = model.predict_proba(df_input)[0][1]
        
        st.subheader("Prediction Results")
        if prediction[0] == 1:
            st.error(f"⚠️ **High Risk of Heart Disease** (Probability: {probability:.2%})")
            st.write("Please consult a healthcare professional for further evaluation.")
        else:
            st.success(f"✅ **Low Risk of Heart Disease** (Probability: {probability:.2%})")
            st.write("Patient is unlikely to have heart disease based on the provided metrics.")
            
    except Exception as e:
        st.error(f"Error making prediction. Model might not have been generated yet. Error: {e}")
