CKD Prediction using Federated Learning

This project is a privacy-preserving Chronic Kidney Disease (CKD) prediction system built using Federated Learning. It simulates multiple hospitals training a shared model without sharing patient data. The app also includes centralized baselines, SHAP explainability, and model performance comparison.

🚀 Features

Federated Learning (FedAvg & FedProx)

Synthetic multi-hospital dataset + CSV upload

Differential Privacy (optional)

Centralized models: Logistic Regression, Random Forest

SHAP explainability (summary & sample-level)

Gemini AI explanation for clinical insights

Model comparison: Accuracy, Recall, F1, ROC-AUC, Confusion Matrix

Streamlit dashboard for interaction

📦 Installation
pip install -r requirements.txt


Or manually:

pip install streamlit scikit-learn pandas numpy shap matplotlib seaborn google-generativeai

▶️ Run the App
streamlit run app.py

📁 Dataset Format

CSV must include features like:

1.age, blood_pressure, albumin

2.blood_urea, serum_creatinine

3.target column: ckd (0/1)

🛠 How It Works

1.Data is split into multiple clients (hospitals)

2.Each client trains a local model

3.Models are aggregated using FedAvg/FedProx

4.Global model is evaluated

5.SHAP + Gemini produce explanations
