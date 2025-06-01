#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("XGBoost_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Lassa Fever Predictor", page_icon="🦠")

st.title("🦠 Lassa Fever Prediction System")
st.markdown(
    """
    Please **fill in the symptoms and vital signs below** to get a Lassa Fever prediction.
    
    Use the dropdown menus to select **Yes** or **No** for each symptom.
    """
)
st.write("---")

# Helper function for symptom dropdown with placeholder default
def symptom_dropdown(label):
    options = ["-- Select --", "No", "Yes"]
    return st.selectbox(label, options=options, index=0)

# Symptoms grouped in two columns with dropdowns default to placeholder
st.header("Symptoms")
col1, col2 = st.columns(2)

with col1:
    fever = symptom_dropdown("Fever")
    sore_throat = symptom_dropdown("Sore Throat")
    vomiting = symptom_dropdown("Vomiting")
    headache = symptom_dropdown("Headache")
    muscle_pain = symptom_dropdown("Muscle Pain")

with col2:
    abdominal_pain = symptom_dropdown("Abdominal Pain")
    diarrhea = symptom_dropdown("Diarrhea")
    bleeding = symptom_dropdown("Bleeding")
    hearing_loss = symptom_dropdown("Hearing Loss")
    fatigue = symptom_dropdown("Fatigue")

st.write("")  # spacing

# Vitals in one column for better layout
st.header("Vital Signs")

temperature = st.slider("Temperature (°C)", 35.0, 45.0, 37.0, 0.1)
heart_rate = st.slider("Heart Rate (bpm)", 30, 200, 80, 1)
oxygen_level = st.slider("Oxygen Level (%)", 50.0, 100.0, 98.0, 0.1)

st.write("---")

def to_binary(val):
    if val == "Yes":
        return 1
    elif val == "No":
        return 0
    else:
        return None  # Not selected yet

if st.button("Predict Lassa Fever"):
    inputs = [fever, sore_throat, vomiting, headache, muscle_pain,
              abdominal_pain, diarrhea, bleeding, hearing_loss, fatigue]

    # Validate all symptoms filled
    if None in [to_binary(x) for x in inputs]:
        st.warning("⚠️ Please fill in all symptom fields with Yes or No before predicting.")
    else:
        # Prepare input dataframe
        input_data = pd.DataFrame({
            "fever": [to_binary(fever)],
            "sore_throat": [to_binary(sore_throat)],
            "vomiting": [to_binary(vomiting)],
            "headache": [to_binary(headache)],
            "muscle_pain": [to_binary(muscle_pain)],
            "abdominal_pain": [to_binary(abdominal_pain)],
            "diarrhea": [to_binary(diarrhea)],
            "bleeding": [to_binary(bleeding)],
            "hearing_loss": [to_binary(hearing_loss)],
            "fatigue": [to_binary(fatigue)],
            "temperature": [temperature],
            "heart_rate": [heart_rate],
            "oxygen_level": [oxygen_level]
        })

        # Scale features
        scaled_input = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            st.error("⚠️ **Positive for Lassa Fever!** Please seek medical attention immediately.")
        else:
            st.success("✅ Negative for Lassa Fever. Stay safe and healthy!")


# In[ ]:




