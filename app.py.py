
import streamlit as st
import pandas as pd
import joblib

# تحميل النموذج ومحولات النصوص
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("PET Scan Radiotracer Optimizer by AI")

# واجهة إدخال البيانات
cancer_type = st.selectbox("Cancer Type", label_encoders['Cancer Type'].classes_)
prevalence = st.selectbox("Prevalence Level", label_encoders['Prevalence Level'].classes_)
preferred_rt = st.selectbox("Preferred Radiotracers", label_encoders['Preferred Radiotracers'].classes_)
metabolism = st.selectbox("Tumor Metabolism Type", label_encoders['Tumor Metabolism Type'].classes_)
sensitivity = st.selectbox("Radiotracer Diagnostic Sensitivity", label_encoders['Radiotracer Diagnostic Sensitivity'].classes_)
inputs = st.selectbox("AI Input Parameters", label_encoders['AI Input Parameters (Required)'].classes_)

# زر تنفيذ التنبؤ
if st.button("Predict Best Radiotracer"):
    sample = [[
        label_encoders['Cancer Type'].transform([cancer_type])[0],
        label_encoders['Prevalence Level'].transform([prevalence])[0],
        label_encoders['Preferred Radiotracers'].transform([preferred_rt])[0],
        label_encoders['Tumor Metabolism Type'].transform([metabolism])[0],
        label_encoders['Radiotracer Diagnostic Sensitivity'].transform([sensitivity])[0],
        label_encoders['AI Input Parameters (Required)'].transform([inputs])[0]
    ]]
    prediction = model.predict(sample)
    result = label_encoders['Label (Most Effective Tracer)'].inverse_transform(prediction)
    st.success(f"Recommended Radiotracer: {result[0]}")
