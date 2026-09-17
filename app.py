import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Risk Classifier",
    page_icon="❤️",
    layout="centered"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load("xgboost_heart_disease_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("❤️ Heart Disease Risk Classification")

st.write(
    """
    An educational machine-learning system that uses an
    XGBoost classification model to estimate a heart-disease
    risk category from selected health-related attributes.
    """
)

st.divider()


# --------------------------------------------------
# Input Form
# --------------------------------------------------

st.subheader("Enter Health Information")


col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=18.0,
        max_value=120.0,
        value=40.0
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    race_ethnicity = st.number_input(
        "Race/Ethnicity Code",
        min_value=1.0,
        max_value=5.0,
        value=1.0,
        step=1.0
    )

    education = st.number_input(
        "Education Level Code",
        min_value=1.0,
        max_value=7.0,
        value=3.0,
        step=1.0
    )

    poverty_income_ratio = st.number_input(
        "Poverty Income Ratio",
        min_value=0.0,
        max_value=20.0,
        value=2.0
    )


with col2:

    taking_bp_meds = st.selectbox(
        "Taking Blood Pressure Medication?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    taking_cholesterol_meds = st.selectbox(
        "Taking Cholesterol Medication?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    taking_insulin = st.selectbox(
        "Taking Insulin?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    taking_diabetes_pills = st.selectbox(
        "Taking Diabetes Pills?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    told_prediabetes = st.selectbox(
        "Previously Told You Have Prediabetes?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


st.divider()

st.subheader("Medical & Lifestyle Information")


col1, col2 = st.columns(2)


with col1:

    told_stroke = st.selectbox(
        "History of Stroke?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    family_history_heart_attack = st.selectbox(
        "Family History of Heart Attack?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    diabetes = st.selectbox(
        "Diabetes?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    smoking_ever = st.selectbox(
        "Have You Ever Smoked?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


with col2:

    smoking_current = st.selectbox(
        "Currently Smoking?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    physically_active = st.selectbox(
        "Physically Active?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    hypertension = st.selectbox(
        "Hypertension?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    high_cholesterol = st.selectbox(
        "High Cholesterol?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button("🔍 Classify Risk", use_container_width=True):

    input_data = {
        "taking_bp_meds": taking_bp_meds,
        "taking_cholesterol_meds": taking_cholesterol_meds,
        "age": age,
        "sex": sex,
        "race_ethnicity": race_ethnicity,
        "education": education,
        "poverty_income_ratio": poverty_income_ratio,
        "taking_insulin": taking_insulin,
        "taking_diabetes_pills": taking_diabetes_pills,
        "told_prediabetes": told_prediabetes,
        "told_stroke": told_stroke,
        "family_history_heart_attack": family_history_heart_attack,
        "diabetes": diabetes,
        "smoking_ever": smoking_ever,
        "smoking_current": smoking_current,
        "physically_active": physically_active,
        "hypertension": hypertension,
        "high_cholesterol": high_cholesterol
    }

    input_df = pd.DataFrame([input_data])

    # Ensure exact feature order
    input_df = input_df[feature_names]

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]


    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    st.subheader("Prediction Result")

    if prediction == 1:

        st.warning(
            "⚠️ The model classified this input as "
            "**Heart Disease Risk: Positive**."
        )

    else:

        st.success(
            "✅ The model classified this input as "
            "**Heart Disease Risk: Negative**."
        )


    st.metric(
        "Model Probability for Class 1",
        f"{probability * 100:.2f}%"
    )


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.divider()

st.caption(
    """
    ⚠️ Educational Disclaimer: This application is developed
    for academic and educational purposes only. The prediction
    represents a pattern learned from the selected dataset and
    must not be interpreted as a medical diagnosis, clinical
    recommendation, or substitute for professional medical advice.
    """
)