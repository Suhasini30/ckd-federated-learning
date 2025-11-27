import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import warnings
import re
import shap
import google.generativeai as genai
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Federated Learning CKD Prediction - Privacy-Preserving", layout="wide")

st.title("🏥 Federated Learning Models for CKD Prediction Across Multi-Hospital Numerical Datasets – Privacy-Preserving Version")

FEATURE_NAMES = ['age', 'blood_pressure', 'serum_creatinine', 'blood_urea', 'albumin']
TARGET_NAME = 'ckd'

def generate_synthetic_data(n_samples=500, n_clients=3):
    np.random.seed(42)
    data_list = []
    
    for client_id in range(n_clients):
        client_seed = 42 + client_id * 10
        np.random.seed(client_seed)
        
        n_client_samples = n_samples // n_clients + np.random.randint(-20, 20)
        
        age = np.random.normal(55 + client_id * 5, 15, n_client_samples)
        age = np.clip(age, 18, 90)
        
        bp_base = 130 + client_id * 10
        blood_pressure = np.random.normal(bp_base, 20, n_client_samples)
        blood_pressure = np.clip(blood_pressure, 80, 200)
        
        sc_base = 1.5 + client_id * 0.5
        serum_creatinine = np.random.exponential(sc_base, n_client_samples)
        serum_creatinine = np.clip(serum_creatinine, 0.5, 15)
        
        bu_base = 50 + client_id * 15
        blood_urea = np.random.normal(bu_base, 25, n_client_samples)
        blood_urea = np.clip(blood_urea, 10, 200)
        
        albumin_base = 3.5 - client_id * 0.3
        albumin = np.random.normal(albumin_base, 0.8, n_client_samples)
        albumin = np.clip(albumin, 1.0, 5.5)
        
        ckd_prob = 1 / (1 + np.exp(-(
            0.02 * age + 
            0.01 * blood_pressure + 
            0.5 * serum_creatinine + 
            0.01 * blood_urea - 
            0.8 * albumin - 
            5
        )))
        
        ckd = (np.random.random(n_client_samples) < ckd_prob).astype(int)
        
        client_data = pd.DataFrame({
            'age': age,
            'blood_pressure': blood_pressure,
            'serum_creatinine': serum_creatinine,
            'blood_urea': blood_urea,
            'albumin': albumin,
            'ckd': ckd,
            'client_id': client_id
        })
        
        data_list.append(client_data)
    
    return pd.concat(data_list, ignore_index=True)


def normalize_uploaded_df(df):
    """Normalize common column name variants to the canonical names used by the app.

    - Renames Age -> age, BloodPressure -> blood_pressure, BloodUrea -> blood_urea,
      SerumCreatinine -> serum_creatinine, Albumin -> albumin.
    - Converts a 'Diagnosis' column like 'CKD'/'NoCKD' into a binary 'ckd' column (1/0).
    - Leaves already-canonical columns untouched.
    """
    df = df.copy()

    # helper: normalize a header string (lowercase, remove non-alnum)
    def norm(s):
        return re.sub(r'[^a-z0-9]', '', str(s).lower())

    col_norm_map = {norm(c): c for c in df.columns}

    # mapping from canonical feature to likely normalized names
    candidates = {
        'age': ['age'],
        'blood_pressure': ['bloodpressure', 'bp', 'blood_pressure'],
        'serum_creatinine': ['serumcreatinine', 'serum_creatinine', 'creatinine', 'serumcreat', 'sc'],
        'blood_urea': ['bloodurea', 'blood_urea', 'urea', 'bloodureanitrogen', 'bun', 'bu'],
        'albumin': ['albumin', 'al', 'alb'],
    }

    rename_map = {}
    for canonical, opts in candidates.items():
        for opt in opts:
            if opt in col_norm_map:
                rename_map[col_norm_map[opt]] = canonical
                break

    # apply renaming for features
    if rename_map:
        df = df.rename(columns=rename_map)

    # handle target column: try to detect 'Diagnosis', 'classification' or similar
    if TARGET_NAME not in df.columns:
        # look for columns that may be 'diagnosis' or similar
        diag_col = None
        for n, orig in col_norm_map.items():
            if ('diagnos' in n) or (n in ('label', 'target', 'ckd', 'class', 'classification', 'status')):
                diag_col = orig
                break

        if diag_col is not None and TARGET_NAME not in df.columns:
            # map values to 0/1 with more tolerant rules (handles 'ckd'/'notckd', 'ckd\t', numeric, yes/no)
            def map_diag(v):
                s = str(v).strip().upper()
                # common positive markers
                if any(tok in s for tok in ('CKD',)) and not any(tok in s for tok in ('NOT', 'NOCKD', 'NOTCKD')):
                    return 1
                if any(tok in s for tok in ('NOCKD', 'NOTCKD', 'NOT', 'NO')):
                    # explicit negative markers
                    # but also allow numeric 0
                    try:
                        # if numeric 0, treat as 0
                        val = float(s)
                        if val == 0:
                            return 0
                    except:
                        pass
                    return 0
                if s in ('1', '1.0'):
                    return 1
                if s in ('0', '0.0'):
                    return 0
                if s in ('YES', 'TRUE', 'Y', 'T'):
                    return 1
                if s in ('NO', 'FALSE', 'N', 'F'):
                    return 0
                try:
                    return int(float(s))
                except:
                    return np.nan

            df[TARGET_NAME] = df[diag_col].apply(map_diag)

    # final: ensure canonical feature names are present in lowercase format
    for f in FEATURE_NAMES:
        if f in df.columns:
            # coerce numeric types where reasonable
            try:
                df[f] = pd.to_numeric(df[f], errors='coerce')
            except Exception:
                pass

    return df


def preprocess_dataset(df, impute_strategy='mean'):
    """Preprocess dataset before training:

    - Drop rows with missing target
    - Impute missing numeric feature values in FEATURE_NAMES using SimpleImputer
    Returns processed df and a dict with stats for logging.
    """
    df = df.copy()
    stats = {
        'rows_before': len(df),
        'rows_after_drop_target': None,
        'missing_before': df[FEATURE_NAMES].isna().sum().to_dict(),
        'imputed': False
    }

    # drop rows missing target
    if TARGET_NAME in df.columns:
        df = df.dropna(subset=[TARGET_NAME])
    stats['rows_after_drop_target'] = len(df)

    # If none of the required features exist, just return
    existing_feats = [c for c in FEATURE_NAMES if c in df.columns]
    if not existing_feats:
        return df, stats

    # Impute numeric features (only those present)
    feat_df = df[existing_feats]
    if feat_df.isna().any().any():
        imputer = SimpleImputer(strategy=impute_strategy)
        imputed_arr = imputer.fit_transform(feat_df)
        df[existing_feats] = imputed_arr
        stats['imputed'] = True

    stats['missing_after'] = df[FEATURE_NAMES].isna().sum().to_dict() if all(f in df.columns for f in FEATURE_NAMES) else None
    return df, stats

def split_data_by_clients(data, n_clients):
    if 'client_id' not in data.columns:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        client_indices = np.array_split(indices, n_clients)
        client_data = []
        for i, idx in enumerate(client_indices):
            client_df = data.iloc[idx].copy()
            client_df['client_id'] = i
            client_data.append(client_df)
    else:
        client_data = [data[data['client_id'] == i].copy() for i in range(n_clients)]
    
    return client_data

def fedavg_train(client_data_list, n_rounds, test_size=0.2, enable_dp=False, dp_noise_scale=0.01):
    global_model = LogisticRegression(max_iter=1000, random_state=42)
    
    all_train_data = []
    all_test_data = []
    
    for client_data in client_data_list:
        X = client_data[FEATURE_NAMES]
        y = client_data[TARGET_NAME]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        all_train_data.append((X_train, y_train))
        all_test_data.append((X_test, y_test))
    
    X_test_combined = pd.concat([x[0] for x in all_test_data])
    y_test_combined = pd.concat([x[1] for x in all_test_data])
    
    for round_num in range(n_rounds):
        local_models = []
        local_weights = []
        
        for X_train, y_train in all_train_data:
            local_model = LogisticRegression(max_iter=1000, random_state=42)
            local_model.fit(X_train, y_train)
            local_models.append(local_model)
            local_weights.append(len(X_train))
        
        total_weight = sum(local_weights)
        weights = [w / total_weight for w in local_weights]
        
        avg_coef = sum(w * model.coef_ for w, model in zip(weights, local_models))
        avg_intercept = sum(w * model.intercept_ for w, model in zip(weights, local_models))
        
        if enable_dp:
            noise_coef = np.random.normal(0, dp_noise_scale, avg_coef.shape)
            noise_intercept = np.random.normal(0, dp_noise_scale, avg_intercept.shape)
            avg_coef = avg_coef + noise_coef
            avg_intercept = avg_intercept + noise_intercept
        
        global_model.coef_ = avg_coef
        global_model.intercept_ = avg_intercept
        global_model.classes_ = local_models[0].classes_
        global_model.n_features_in_ = local_models[0].n_features_in_
        global_model.feature_names_in_ = local_models[0].feature_names_in_ if hasattr(local_models[0], 'feature_names_in_') else None
        global_model.n_iter_ = local_models[0].n_iter_
    
    return global_model, X_test_combined, y_test_combined

def fedprox_train(client_data_list, n_rounds, mu=0.01, test_size=0.2, enable_dp=False, dp_noise_scale=0.01):
    global_model = LogisticRegression(max_iter=1000, random_state=42)
    
    all_train_data = []
    all_test_data = []
    
    for client_data in client_data_list:
        X = client_data[FEATURE_NAMES]
        y = client_data[TARGET_NAME]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        all_train_data.append((X_train, y_train))
        all_test_data.append((X_test, y_test))
    
    X_test_combined = pd.concat([x[0] for x in all_test_data])
    y_test_combined = pd.concat([x[1] for x in all_test_data])
    
    for round_num in range(n_rounds):
        local_models = []
        local_weights = []
        
        for X_train, y_train in all_train_data:
            local_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0/(1.0 + mu))
            local_model.fit(X_train, y_train)
            local_models.append(local_model)
            local_weights.append(len(X_train))
        
        total_weight = sum(local_weights)
        weights = [w / total_weight for w in local_weights]
        
        avg_coef = sum(w * model.coef_ for w, model in zip(weights, local_models))
        avg_intercept = sum(w * model.intercept_ for w, model in zip(weights, local_models))
        
        if enable_dp:
            noise_coef = np.random.normal(0, dp_noise_scale, avg_coef.shape)
            noise_intercept = np.random.normal(0, dp_noise_scale, avg_intercept.shape)
            avg_coef = avg_coef + noise_coef
            avg_intercept = avg_intercept + noise_intercept
        
        global_model.coef_ = avg_coef
        global_model.intercept_ = avg_intercept
        global_model.classes_ = local_models[0].classes_
        global_model.n_features_in_ = local_models[0].n_features_in_
        global_model.feature_names_in_ = local_models[0].feature_names_in_ if hasattr(local_models[0], 'feature_names_in_') else None
        global_model.n_iter_ = local_models[0].n_iter_
    
    return global_model, X_test_combined, y_test_combined

def train_centralized_rf(data, test_size=0.2, n_estimators=100):
    X = data[FEATURE_NAMES]
    y = data[TARGET_NAME]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def train_centralized_lr(data, test_size=0.2):
    X = data[FEATURE_NAMES]
    y = data[TARGET_NAME]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_proba = y_pred
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
    }
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    except:
        metrics['roc_auc'] = None
    
    return metrics, y_pred, y_proba

def cross_validate_model(model, data, k=5):
    X = data[FEATURE_NAMES]
    y = data[TARGET_NAME]
    
    scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')
    
    return {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std()
    }

def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {title}')
    return fig

def plot_roc_curve(models_data):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, (y_test, y_proba) in models_data.items():
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        except:
            continue
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_metrics_comparison(results_df):
    metrics_to_plot = ['accuracy', 'recall', 'f1_score', 'roc_auc']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        if metric in results_df.columns:
            ax = axes[idx]
            valid_data = results_df[results_df[metric].notna()]
            ax.bar(valid_data['model'], valid_data[metric], color='steelblue')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_ylim(0, 1.1)
            ax.tick_params(axis='x', rotation=45)
            for i, v in enumerate(valid_data[metric]):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def get_genai_insight(results_df):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        metrics_summary = results_df.to_string()
        
        prompt = f"""Based on the following model performance metrics for CKD prediction, provide a brief 2-3 sentence insight comparing the models:

{metrics_summary}

Focus on which model performed best and why federated learning might be beneficial for multi-hospital scenarios."""
        
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        st.warning(f"Gemini API Error: {e}")
        return None

def get_local_insight(results_df):
    best_model = results_df.loc[results_df['accuracy'].idxmax(), 'model']
    best_accuracy = results_df['accuracy'].max()
    
    return f"Local Analysis: The {best_model} achieved the highest accuracy of {best_accuracy:.3f}. Federated learning approaches (FedAvg, FedProx) enable collaborative model training across hospitals while preserving data privacy, which is crucial for healthcare applications."

def get_shap_explanation(model, X_train_sample, X_test_sample, model_type='tree'):
    """Generate SHAP explanations for model predictions"""
    try:
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_train_sample)
        
        shap_values = explainer.shap_values(X_test_sample)
        
        return explainer, shap_values
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {e}")
        return None, None

def plot_shap_summary(explainer, shap_values, X_test_sample, feature_names):
    """Plot SHAP summary plot"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting SHAP summary: {e}")
        return None

def get_gemini_xai_explanation(model_name, metrics, shap_importance, sample_data=None, prediction=None):
    """Get detailed XAI explanation using Gemini API"""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Create feature importance summary
        importance_str = "\n".join([f"- {feat}: {imp:.4f}" for feat, imp in shap_importance.items()])
        
        if sample_data is not None and prediction is not None:
            sample_str = "\n".join([f"- {feat}: {val:.2f}" for feat, val in sample_data.items()])
            prompt = f"""As a medical AI explainability expert, explain the following CKD prediction model results to a healthcare professional:

Model: {model_name}
Performance Metrics:
- Accuracy: {metrics.get('accuracy', 'N/A'):.3f}
- Recall (Sensitivity): {metrics.get('recall', 'N/A'):.3f}
- F1-Score: {metrics.get('f1_score', 'N/A'):.3f}
- ROC-AUC: {metrics.get('roc_auc', 'N/A') if metrics.get('roc_auc') else 'N/A'}

Feature Importance (SHAP values):
{importance_str}

Patient Sample Data:
{sample_str}

Prediction: {prediction['prediction']} (Probability: {prediction['probability']:.2%})

Please provide:
1. A clear explanation of what these metrics mean for clinical use
2. Which features are most important for this prediction and why
3. How confident we should be in this prediction
4. Clinical recommendations based on the results
5. Any limitations or considerations for using this model"""
        else:
            prompt = f"""As a medical AI explainability expert, explain the following CKD prediction model to a healthcare professional:

Model: {model_name}
Performance Metrics:
- Accuracy: {metrics.get('accuracy', 'N/A'):.3f}
- Recall (Sensitivity): {metrics.get('recall', 'N/A'):.3f}
- F1-Score: {metrics.get('f1_score', 'N/A'):.3f}
- ROC-AUC: {metrics.get('roc_auc', 'N/A') if metrics.get('roc_auc') else 'N/A'}

Feature Importance (SHAP values):
{importance_str}

Please provide:
1. A clear explanation of what these metrics mean for clinical use
2. Which features are most important for this model and their clinical significance
3. Strengths and weaknesses of this model for CKD prediction
4. How this model compares to clinical judgment
5. Recommendations for implementation in a healthcare setting"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting Gemini explanation: {e}")
        return None

def predict_single_sample(models_dict, sample_data):
    predictions = {}
    
    for model_name, model in models_dict.items():
        try:
            pred = model.predict(sample_data)[0]
            try:
                proba = model.predict_proba(sample_data)[0]
                predictions[model_name] = {
                    'prediction': 'CKD' if pred == 1 else 'No CKD',
                    'probability': proba[1] if len(proba) > 1 else pred
                }
            except:
                predictions[model_name] = {
                    'prediction': 'CKD' if pred == 1 else 'No CKD',
                    'probability': pred
                }
        except Exception as e:
            predictions[model_name] = {
                'prediction': 'Error',
                'probability': 0
            }
    
    return predictions

st.sidebar.header("⚙ Configuration")

input_mode = st.sidebar.radio("Input Mode", ["Upload CSV", "Manual Entry"])

st.sidebar.markdown("---")
st.sidebar.subheader("Federated Learning Settings")
n_clients = st.sidebar.slider("Number of Hospitals (Clients)", 2, 10, 3)
n_rounds = st.sidebar.slider("Federated Training Rounds", 1, 20, 5)
fedprox_mu = st.sidebar.slider("FedProx μ (Proximal Term)", 0.001, 0.1, 0.01, 0.001)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
cv_k = st.sidebar.slider("K-Fold CV (k)", 3, 10, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("Select Algorithms")
run_fedavg = st.sidebar.checkbox("FedAvg", value=True)
run_fedprox = st.sidebar.checkbox("FedProx", value=True)
run_rf = st.sidebar.checkbox("Random Forest", value=True)
run_lr = st.sidebar.checkbox("Logistic Regression", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🔐 Privacy Settings")
enable_differential_privacy = st.sidebar.checkbox("Enable Differential Privacy", value=False, 
    help="Add Gaussian noise to federated model updates for enhanced privacy protection")
if enable_differential_privacy:
    dp_noise_scale = st.sidebar.slider("DP Noise Scale (σ)", 0.001, 0.1, 0.01, 0.001,
        help="Standard deviation of Gaussian noise added to model parameters")
else:
    dp_noise_scale = 0.01

with st.sidebar.expander("ℹ About the 5 Key Features"):
    st.markdown("""
    Why these 5 features?
    
    - Age: CKD prevalence increases with age
    - Blood Pressure: Hypertension is a major CKD risk factor
    - Serum Creatinine: Direct kidney function indicator
    - Blood Urea: Measures kidney's waste removal efficiency
    - Albumin: Protein in urine indicates kidney damage
    
    These features are clinically validated and widely available across hospitals.
    """)

data = None
manual_sample = None
using_synthetic = False

if input_mode == "Upload CSV":
    if 'manual_sample' in st.session_state:
        del st.session_state['manual_sample']
    
    st.header("📁 Upload Hospital CKD Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)

            orig_cols = list(raw_df.columns)
            data = normalize_uploaded_df(raw_df)

            # show mapping info if columns were renamed
            if set(orig_cols) != set(data.columns):
                st.info(f"Detected columns: {orig_cols}")
                st.info(f"Mapped/normalized columns: {list(data.columns)}")

            missing_features = [f for f in FEATURE_NAMES if f not in data.columns]
            if missing_features:
                st.error(f"Missing required features after normalization: {', '.join(missing_features)}")
                st.info(f"Required features: {', '.join(FEATURE_NAMES)} and '{TARGET_NAME}'")
                st.info("Hint: column names in your file appear close to expected names. The app tried to auto-map common variants (e.g. 'BloodPressure' -> 'blood_pressure').")
                data = None
            elif TARGET_NAME not in data.columns:
                st.error(f"Missing target column: '{TARGET_NAME}' after normalization")
                st.info("If your dataset uses a different label column (e.g. 'Diagnosis'), the app will try to convert known values like 'CKD'/'NoCKD' to a binary 'ckd' column. Please check your file and try again.")
                st.info(f"Detected columns: {orig_cols}")
                data = None
            else:
                st.success(f"✅ Loaded {len(data)} samples with {len(data.columns)} features")
                st.session_state['training_data'] = data
                st.session_state['using_synthetic'] = False
                st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
    else:
        st.info("No CSV uploaded. Using synthetic CKD dataset for demonstration.")
        data = generate_synthetic_data(n_samples=500, n_clients=n_clients)
        st.session_state['training_data'] = data
        st.session_state['using_synthetic'] = True
        using_synthetic = True
        st.dataframe(data.head())
    
    if 'training_data' in st.session_state:
        data = st.session_state['training_data']
        using_synthetic = st.session_state.get('using_synthetic', False)

elif input_mode == "Manual Entry":
    st.header("✍ Manual Patient Entry")
    st.info("Enter patient data for immediate CKD prediction using the 5 key clinical features.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=50, 
                             help="Patient's age in years (0-120)")
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=40, max_value=250, value=120,
                                        help="Systolic blood pressure (40-250)")
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=20.0, value=1.2, step=0.1,
                                          help="Blood creatinine level (0.1-20)")
    
    with col2:
        blood_urea = st.number_input("Blood Urea (mg/dL)", min_value=1, max_value=500, value=40,
                                    help="Blood urea nitrogen level (1-500)")
        albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=10.0, value=4.0, step=0.1,
                                 help="Serum albumin level (0-10)")
    
    if st.button("🔬 Predict CKD Risk", type="primary"):
        manual_sample = pd.DataFrame({
            'age': [age],
            'blood_pressure': [blood_pressure],
            'serum_creatinine': [serum_creatinine],
            'blood_urea': [blood_urea],
            'albumin': [albumin]
        })
        
        st.session_state['manual_sample'] = manual_sample
        
        st.info("Generating synthetic training data for model training...")
        data = generate_synthetic_data(n_samples=500, n_clients=n_clients)
        st.session_state['training_data'] = data
        st.session_state['using_synthetic'] = True
        using_synthetic = True
    
    if 'manual_sample' in st.session_state:
        manual_sample = st.session_state['manual_sample']
    
    if 'training_data' in st.session_state:
        data = st.session_state['training_data']
        using_synthetic = st.session_state.get('using_synthetic', False)

if data is not None and len(data) > 0:
    if using_synthetic:
        st.info("📊 Using synthetic CKD dataset for demonstration purposes.")
    
    st.markdown("---")
    st.header("🤖 Model Training & Evaluation")
    
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models... This may take a moment."):
            # Preprocess dataset: drop missing targets and impute missing feature values
            data, prep_stats = preprocess_dataset(data)
            if prep_stats.get('imputed'):
                st.info("Missing feature values detected and imputed using mean strategy.")

            client_data_list = split_data_by_clients(data, n_clients)
            
            results = []
            all_models = {}
            roc_data = {}
            
            if run_fedavg:
                with st.spinner("🔒 Training FedAvg (Secure Aggregation)..."):
                    fedavg_model, X_test_fedavg, y_test_fedavg = fedavg_train(
                        client_data_list, n_rounds, test_size, enable_differential_privacy, dp_noise_scale
                    )
                    metrics_fedavg, y_pred_fedavg, y_proba_fedavg = calculate_metrics(
                        fedavg_model, X_test_fedavg, y_test_fedavg
                    )
                    results.append({
                        'model': 'FedAvg',
                        **metrics_fedavg
                    })
                    all_models['FedAvg'] = {
                        'model': fedavg_model,
                        'X_test': X_test_fedavg,
                        'y_test': y_test_fedavg,
                        'y_pred': y_pred_fedavg,
                        'y_proba': y_proba_fedavg,
                        'type': 'federated'
                    }
                    roc_data['FedAvg'] = (y_test_fedavg, y_proba_fedavg)
            
            if run_fedprox:
                with st.spinner("🔒 Training FedProx (Secure Aggregation)..."):
                    fedprox_model, X_test_fedprox, y_test_fedprox = fedprox_train(
                        client_data_list, n_rounds, fedprox_mu, test_size, enable_differential_privacy, dp_noise_scale
                    )
                    metrics_fedprox, y_pred_fedprox, y_proba_fedprox = calculate_metrics(
                        fedprox_model, X_test_fedprox, y_test_fedprox
                    )
                    results.append({
                        'model': 'FedProx',
                        **metrics_fedprox
                    })
                    all_models['FedProx'] = {
                        'model': fedprox_model,
                        'X_test': X_test_fedprox,
                        'y_test': y_test_fedprox,
                        'y_pred': y_pred_fedprox,
                        'y_proba': y_proba_fedprox,
                        'type': 'federated'
                    }
                    roc_data['FedProx'] = (y_test_fedprox, y_proba_fedprox)
            
            if run_rf:
                with st.spinner("Training Random Forest..."):
                    rf_model, X_test_rf, y_test_rf = train_centralized_rf(data, test_size)
                    metrics_rf, y_pred_rf, y_proba_rf = calculate_metrics(rf_model, X_test_rf, y_test_rf)
                    cv_rf = cross_validate_model(RandomForestClassifier(random_state=42), data, cv_k)
                    results.append({
                        'model': 'Random Forest',
                        **metrics_rf,
                        'cv_mean': cv_rf['mean_accuracy'],
                        'cv_std': cv_rf['std_accuracy']
                    })
                    all_models['Random Forest'] = {
                        'model': rf_model,
                        'X_test': X_test_rf,
                        'y_test': y_test_rf,
                        'y_pred': y_pred_rf,
                        'y_proba': y_proba_rf,
                        'type': 'centralized',
                        'cv': cv_rf
                    }
                    roc_data['Random Forest'] = (y_test_rf, y_proba_rf)
            
            if run_lr:
                with st.spinner("Training Logistic Regression..."):
                    lr_model, X_test_lr, y_test_lr = train_centralized_lr(data, test_size)
                    metrics_lr, y_pred_lr, y_proba_lr = calculate_metrics(lr_model, X_test_lr, y_test_lr)
                    cv_lr = cross_validate_model(LogisticRegression(max_iter=1000, random_state=42), data, cv_k)
                    results.append({
                        'model': 'Logistic Regression',
                        **metrics_lr,
                        'cv_mean': cv_lr['mean_accuracy'],
                        'cv_std': cv_lr['std_accuracy']
                    })
                    all_models['Logistic Regression'] = {
                        'model': lr_model,
                        'X_test': X_test_lr,
                        'y_test': y_test_lr,
                        'y_pred': y_pred_lr,
                        'y_proba': y_proba_lr,
                        'type': 'centralized',
                        'cv': cv_lr
                    }
                    roc_data['Logistic Regression'] = (y_test_lr, y_proba_lr)
            
            results_df = pd.DataFrame(results)
            st.session_state['results_df'] = results_df
            st.session_state['all_models'] = all_models
            st.session_state['roc_data'] = roc_data
            
            # Generate SHAP explanations for each model
            st.session_state['shap_data'] = {}
            with st.spinner("🔍 Generating XAI explanations with SHAP..."):
                for model_name, model_data in all_models.items():
                    try:
                        X_train_sample = model_data['X_test'].sample(min(100, len(model_data['X_test'])), random_state=42)
                        X_test_sample = model_data['X_test'].sample(min(50, len(model_data['X_test'])), random_state=42)
                        
                        model_type = 'tree' if model_name == 'Random Forest' else 'linear'
                        explainer, shap_values = get_shap_explanation(
                            model_data['model'], 
                            X_train_sample, 
                            X_test_sample, 
                            model_type
                        )
                        
                        if explainer is not None and shap_values is not None:
                            # Calculate feature importance
                            if isinstance(shap_values, list) and len(shap_values) == 2:
                                shap_vals = shap_values[1]
                            else:
                                shap_vals = shap_values
                            
                            feature_importance = {}
                            for i, feat in enumerate(FEATURE_NAMES):
                                feature_importance[feat] = np.abs(shap_vals[:, i]).mean()
                            
                            st.session_state['shap_data'][model_name] = {
                                'explainer': explainer,
                                'shap_values': shap_values,
                                'X_test_sample': X_test_sample,
                                'feature_importance': feature_importance
                            }
                    except Exception as e:
                        st.warning(f"Could not generate SHAP for {model_name}: {e}")
            
            st.success("✅ Model training complete!")
    
    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']
        all_models = st.session_state['all_models']
        roc_data = st.session_state['roc_data']
        
        if manual_sample is not None:
            st.markdown("---")
            st.header("🎯 Single Sample Prediction Results")
            
            predictions = predict_single_sample(
                {name: data['model'] for name, data in all_models.items()},
                manual_sample
            )
            
            cols = st.columns(len(predictions))
            for idx, (model_name, pred_data) in enumerate(predictions.items()):
                with cols[idx]:
                    st.metric(
                        label=f"{model_name}",
                        value=pred_data['prediction'],
                        delta=f"{pred_data['probability']:.2%} probability"
                    )
            
            # XAI Explanation for single sample
            st.markdown("---")
            st.subheader("🔍 XAI Explanation for Your Sample")
            
            selected_model_xai = st.selectbox("Select model for detailed explanation:", list(predictions.keys()), key='single_sample_xai')
            
            if selected_model_xai in st.session_state.get('shap_data', {}):
                shap_info = st.session_state['shap_data'][selected_model_xai]
                pred_data = predictions[selected_model_xai]
                model_metrics = results_df[results_df['model'] == selected_model_xai].iloc[0].to_dict()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("*Sample Statistics:* Your sample compared against training data distribution.")
                    train_stats = data[FEATURE_NAMES].describe().loc[['mean', 'std', 'min', 'max']]
                    sample_comparison = pd.DataFrame({
                        'Feature': FEATURE_NAMES,
                        'Your Value': manual_sample.values[0],
                        'Mean': train_stats.loc['mean'].values,
                        'Min': train_stats.loc['min'].values,
                        'Max': train_stats.loc['max'].values
                    })
                    st.dataframe(sample_comparison)
                
                with col2:
                    st.info("*Feature Importance for This Prediction:*")
                    importance_df = pd.DataFrame(
                        list(shap_info['feature_importance'].items()),
                        columns=['Feature', 'Impact']
                    ).sort_values('Impact', ascending=False)
                    
                    for _, row in importance_df.iterrows():
                        st.write(f"{row['Feature']}: {row['Impact']:.4f}")
                
                # Gemini explanation for single sample
                st.markdown("---")
                st.subheader("🤖 Gemini AI - Personalized Explanation")
                
                with st.spinner("Generating personalized explanation..."):
                    sample_dict = {feat: manual_sample[feat].values[0] for feat in FEATURE_NAMES}
                    sample_xai_explanation = get_gemini_xai_explanation(
                        selected_model_xai,
                        model_metrics,
                        shap_info['feature_importance'],
                        sample_data=sample_dict,
                        prediction=pred_data
                    )
                
                if sample_xai_explanation:
                    st.markdown(sample_xai_explanation)
                else:
                    st.warning("Gemini API explanation not available. Please check your API key.")
            else:
                st.info("Sample Statistics: Your sample has been compared against the training data distribution.")
                train_stats = data[FEATURE_NAMES].describe().loc[['mean', 'std', 'min', 'max']]
                sample_comparison = pd.DataFrame({
                    'Feature': FEATURE_NAMES,
                    'Your Value': manual_sample.values[0],
                    'Mean': train_stats.loc['mean'].values,
                    'Min': train_stats.loc['min'].values,
                    'Max': train_stats.loc['max'].values
                })
                st.dataframe(sample_comparison)
        
        st.markdown("---")
        st.header("📊 Model Performance Summary")
        
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['accuracy', 'recall', 'f1_score', 'roc_auc']))
        
        genai_insight = get_genai_insight(results_df)
        
        if genai_insight:
            st.success("🤖 *Gemini AI Insights:*")
            st.markdown(genai_insight)
        else:
            st.info(f"💡 {get_local_insight(results_df)}")
        
        # Add XAI Section with Gemini explanations
        if 'shap_data' in st.session_state and st.session_state['shap_data']:
            st.markdown("---")
            st.header("🔍 Explainable AI (XAI) Analysis")
            st.markdown("Understanding what drives the model predictions using SHAP (SHapley Additive exPlanations) and Gemini AI")
            
            # Model selector for XAI
            xai_model = st.selectbox("Select Model for Detailed XAI Analysis:", list(st.session_state['shap_data'].keys()))
            
            if xai_model and xai_model in st.session_state['shap_data']:
                shap_info = st.session_state['shap_data'][xai_model]
                model_metrics = results_df[results_df['model'] == xai_model].iloc[0].to_dict()
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("  Feature Importance (SHAP)")
                    importance_df = pd.DataFrame(
                        list(shap_info['feature_importance'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                    ax_imp.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
                    ax_imp.set_xlabel('Mean |SHAP Value|')
                    ax_imp.set_title(f'Feature Importance - {xai_model}')
                    ax_imp.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig_imp)
                    plt.close()
                
                with col2:
                    st.subheader("🎯 SHAP Summary Plot")
                    fig_shap = plot_shap_summary(
                        shap_info['explainer'],
                        shap_info['shap_values'],
                        shap_info['X_test_sample'],
                        FEATURE_NAMES
                    )
                    if fig_shap:
                        st.pyplot(fig_shap)
                        plt.close()
                
                # Gemini AI Explanation
                st.markdown("---")
                st.subheader("🤖 Gemini AI - Detailed XAI Explanation")
                
                with st.spinner("Generating detailed explanation with Gemini AI..."):
                    xai_explanation = get_gemini_xai_explanation(
                        xai_model,
                        model_metrics,
                        shap_info['feature_importance']
                    )
                
                if xai_explanation:
                    st.markdown(xai_explanation)
                else:
                    st.warning("Gemini API explanation not available. Please check your API key.")
                    
                    # Fallback local explanation
                    st.info("*Local Feature Importance Analysis:*")
                    for feat, imp in sorted(shap_info['feature_importance'].items(), key=lambda x: x[1], reverse=True):
                        st.write(f"- *{feat}*: {imp:.4f} - {'High' if imp > 0.1 else 'Moderate' if imp > 0.05 else 'Low'} impact on predictions")
        
        st.markdown("---")
        st.header("📈 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Metrics Comparison")
            fig_metrics = plot_metrics_comparison(results_df)
            st.pyplot(fig_metrics)
            plt.close()
        
        with col2:
            st.subheader("ROC Curves")
            fig_roc = plot_roc_curve(roc_data)
            st.pyplot(fig_roc)
            plt.close()
        
        st.markdown("---")
        st.header("🔍 Detailed Model Results")
        
        tabs = st.tabs(list(all_models.keys()))
        
        for idx, (model_name, model_data) in enumerate(all_models.items()):
            with tabs[idx]:
                st.subheader(f"{model_name} Details")
                
                if model_name == "FedAvg":
                    st.info("FedAvg (Federated Averaging): Aggregates model parameters from multiple hospitals by averaging their weights. This enables collaborative learning while keeping patient data at each hospital. Each hospital trains locally, then shares only model parameters.")
                elif model_name == "FedProx":
                    st.warning("FedProx (Federated Proximal): Approximated implementation using regularization parameter μ. Full FedProx requires custom client-side optimizers with proximal terms. This simulation uses adjusted regularization to approximate heterogeneous data handling across hospitals.")
                elif model_name == "Random Forest":
                    st.info("Random Forest: Ensemble method combining multiple decision trees trained on centralized data. Provides robust predictions and handles non-linear relationships well. Serves as a strong baseline for comparison.")
                else:
                    st.info("Logistic Regression: Classic statistical model for binary classification on centralized data. Simple, interpretable, and efficient. Good baseline for understanding linear decision boundaries.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    accuracy_val = results_df[results_df['model'] == model_name]['accuracy'].values[0]
                    st.metric("Accuracy", f"{accuracy_val:.3f}")
                    metrics_dict = {
                        'Accuracy': accuracy_val,
                        'Recall': results_df[results_df['model'] == model_name]['recall'].values[0],
                        'F1-Score': results_df[results_df['model'] == model_name]['f1_score'].values[0],
                    }
                    if results_df[results_df['model'] == model_name]['roc_auc'].values[0]:
                        metrics_dict['ROC-AUC'] = results_df[results_df['model'] == model_name]['roc_auc'].values[0]
                    
                    st.write("Metrics:")
                    for metric, value in metrics_dict.items():
                        st.write(f"- {metric}: {value:.3f}")
                    
                    if model_data['type'] == 'centralized' and 'cv' in model_data:
                        st.write(f"{cv_k}-Fold CV:** {model_data['cv']['mean_accuracy']:.3f} ± {model_data['cv']['std_accuracy']:.3f}")
                
                with col2:
                    fig_cm = plot_confusion_matrix(model_data['y_test'], model_data['y_pred'], model_name)
                    st.pyplot(fig_cm)
                    plt.close()
        
        st.markdown("---")
        st.header("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics CSV",
                data=csv,
                file_name="ckd_model_metrics.csv",
                mime="text/csv"
            )
        
        with col2:
            models_to_save = {name: data['model'] for name, data in all_models.items()}
            buffer = BytesIO()
            pickle.dump(models_to_save, buffer)
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download Trained Models",
                data=buffer,
                file_name="ckd_models.pkl",
                mime="application/octet-stream"
            )

else:
    st.info("👆 Please upload a CSV file or use manual entry mode to begin.")

st.markdown("---")
st.caption("Built with Streamlit | Federated Learning for Privacy-Preserving Healthcare AI")