import streamlit as st
import numpy as np
import pickle

# 🎯 Load models
models = {
    "Logistic Regression": pickle.load(open("logistic_regression_model.pkl", "rb")),
    "Support Vector Machine": pickle.load(open("svm_model.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest_model.pkl", "rb"))
}

# 🌟 Page Config
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# 💡 Sidebar for Model Selection
st.sidebar.title("🔍 Choose Model")
model_name = st.sidebar.radio("Select a machine learning model", list(models.keys()))
model, scaler = models[model_name]

# 🧠 App Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🩺 Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter the patient's medical details below to check for diabetes using a machine learning model.</p>", unsafe_allow_html=True)
st.write("---")

# 📋 Input Fields
st.subheader("📋 Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
    age = st.number_input("Age", min_value=0, max_value=120, step=1)

# 🎯 Predict Button
st.write(" ")
predict_button = st.button("🚀 Predict Diabetes", use_container_width=True)

if predict_button:
    # Collect and scale input
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)[0]

    st.write("---")
    st.subheader("🎯 Prediction Result")
    if result == 1:
        st.error("❌ The patient is likely **Diabetic**.", icon="🚨")
    else:
        st.success("✅ The patient is likely **Not Diabetic**.", icon="🟢")

# 📘 Footer
st.write("---")
st.markdown("<p style='text-align: center; font-size: 13px;'>Developed by Vartika | Machine Learning Project | Streamlit UI</p>", unsafe_allow_html=True)
