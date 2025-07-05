import streamlit as st
import joblib
import numpy as np

# Load the trained model
loaded = joblib.load('heart_test_data.pkl')

# Unpack model from tuple
model = loaded[0]  
# Title of the app
st.title("Heart Disease Prediction App")
st.markdown("Enter the patient's details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.selectbox("Resting ECG results (restecg)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of peak exercise ST segment (slope)", options=[0, 1, 2])
ca = st.selectbox("Number of major vessels colored by fluoroscopy (ca)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

# Predict button
if st.button("Predict "):
    features = np.array([
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("⚠️ The model predicts the patient is likely to have heart disease.")
    else:
        st.success("✅ The model predicts the patient is unlikely to have heart disease.")

   