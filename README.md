# ❤️ Heart Disease Risk Classification using XGBoost

An educational machine learning application that uses an **XGBoost classification model** to classify heart disease risk based on demographic, lifestyle, medical history, and health-related attributes.

The trained model is integrated into a **Streamlit web application** that allows users to enter health-related information and receive a model-generated classification.

> ⚠️ **Educational Purpose Only:** This application is developed for academic and educational purposes. It is not a medical diagnostic tool and should not be used for clinical decision-making.

---

## 📌 Project Overview

Heart disease is influenced by multiple demographic, lifestyle, and health-related factors.

This project applies **machine learning classification** to identify patterns in a heart disease dataset and classify individuals into two categories:

- `0` → No Heart Disease
- `1` → Heart Disease

The primary machine learning algorithm used in this project is **XGBoost (Extreme Gradient Boosting)**.

The trained model is deployed through a Streamlit interface to provide an interactive prediction experience.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze a heart disease dataset.
- Perform data inspection and preprocessing.
- Explore relationships between health-related features and heart disease.
- Train an XGBoost binary classification model.
- Evaluate the model using multiple classification metrics.
- Visualize the model's performance using a confusion matrix and ROC curve.
- Analyze feature importance.
- Deploy the trained model using Streamlit.
- Provide an interactive prediction interface.

---

## 🧠 Machine Learning Approach

The project follows the following pipeline:

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature / Target Separation
   ↓
Stratified Train-Test Split
   ↓
Class Imbalance Handling
   ↓
XGBoost Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Streamlit Deployment