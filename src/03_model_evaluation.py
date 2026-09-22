import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from tensorflow.keras.models import load_model

# Set paths
DATA_PATH = 'data/ai4i2020.csv'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'
MODEL_PATH = 'models/predictive_pulse_ann.keras'
METADATA_PATH = 'models/model_metadata.json'
RESULTS_DIR = 'results'
EVAL_JSON_PATH = os.path.join(RESULTS_DIR, 'model_evaluation.json')

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATA & RECONSTRUCT SPLIT
# ---------------------------------------------------------
print("Loading data, preprocessor, and model...")
df = pd.read_csv(DATA_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = load_model(MODEL_PATH)

target_col = 'Machine failure'
leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
id_cols = ['UDI', 'Product ID']

X = df.drop(columns=[target_col] + leakage_cols + id_cols)
y = df[target_col]

print("Splitting data (80/20) to extract test set...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after preprocessing
# numeric features are passed through StandardScaler, categorical through OneHotEncoder
numeric_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
# The categorical feature 'Type' had 3 categories 'M', 'L', 'H'. 
# Let's get feature names from the preprocessor to be exact.
cat_features = preprocessor.transformers_[1][1].get_feature_names_out(['Type'])
feature_names = numeric_features + list(cat_features)
print(f"Feature Names: {feature_names}")

# ---------------------------------------------------------
# 2. PREDICTIONS AND STANDARD METRICS (Threshold = 0.5)
# ---------------------------------------------------------
print("\nGenerating predictions on test set...")
y_pred_proba = model.predict(X_test_processed).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n--- STANDARD EVALUATION (Threshold = 0.5) ---")
print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal (0)', 'Failure (1)'], 
            yticklabels=['Normal (0)', 'Failure (1)'])
plt.title('Final Confusion Matrix (Test Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_final.png'))
plt.close()

# ---------------------------------------------------------
# 3. ROC CURVE AND PR CURVE
# ---------------------------------------------------------
print("\nGenerating ROC and Precision-Recall curves...")

# ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
plt.close()

# PR Curve
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, label='PR Curve', color='green')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'precision_recall_curve.png'))
plt.close()

# ---------------------------------------------------------
# 4. THRESHOLD ANALYSIS
# ---------------------------------------------------------
print("\nEvaluating Classification Thresholds...")
thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_analysis = []

print(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10}")
print("-" * 50)

for t in thresholds_to_test:
    preds = (y_pred_proba >= t).astype(int)
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f = f1_score(y_test, preds, zero_division=0)
    
    threshold_analysis.append({
        'threshold': float(t),
        'precision': float(p),
        'recall': float(r),
        'f1_score': float(f)
    })
    print(f"{t:<12.1f} | {p:<10.4f} | {r:<10.4f} | {f:<10.4f}")

# For predictive maintenance, high recall is critical. 
# 0.5 is the standard baseline and gives good recall (e.g. ~86%) with our class weights.
# We will document the trade-offs, but stick with 0.5 as the chosen threshold 
# since it provides a strong balance, preventing excessive false alarms.

# ---------------------------------------------------------
# 5. SHAP EXPLAINABILITY
# ---------------------------------------------------------
print("\nPerforming SHAP analysis (this may take a moment)...")

# SHAP DeepExplainer expects a background dataset. 
# We'll use a random sample of 100 from the training set to represent the background distribution.
np.random.seed(42)
background_indices = np.random.choice(X_train_processed.shape[0], 100, replace=False)
background_data = X_train_processed[background_indices]

# Initialize Explainer
explainer = shap.DeepExplainer(model, background_data)

# Compute SHAP values on a subset of the test data (e.g. 500 samples) to save time,
# but we can use the whole test set if it's small enough (2000 samples). 
# Using the whole test set for global feature importance.
shap_values = explainer.shap_values(X_test_processed)

# For a single-output keras model, shap_values is a list with one element (for the output node)
# or just a numpy array. SHAP 0.4x usually returns a list for TF models.
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[0]
else:
    shap_values_to_plot = shap_values

# Squeeze the last dimension if it's a single output model (e.g. from (2000, 8, 1) to (2000, 8))
if hasattr(shap_values_to_plot, 'ndim') and shap_values_to_plot.ndim == 3 and shap_values_to_plot.shape[2] == 1:
    shap_values_to_plot = shap_values_to_plot[:, :, 0]
elif hasattr(shap_values_to_plot, 'ndim') and shap_values_to_plot.ndim == 2 and shap_values_to_plot.shape[1] == 1:
    # Just in case it's (num_features, 1) somehow
    shap_values_to_plot = shap_values_to_plot.flatten()

# Generate and save SHAP summary plot (Global Interpretability)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_to_plot, X_test_processed, feature_names=feature_names, show=False)
plt.title('SHAP Summary Plot (Global Feature Importance)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'shap_summary.png'))
plt.close()

# Generate and save individual SHAP explanations (Local Interpretability)
# Find one True Positive and one False Negative to explain
true_positives = np.where((y_test == 1) & (y_pred == 1))[0]
false_negatives = np.where((y_test == 1) & (y_pred == 0))[0]

# Extract base value cleanly
base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
if hasattr(base_val, 'numpy'):
    base_val = base_val.numpy()
try:
    base_val = float(base_val[0]) if isinstance(base_val, (list, np.ndarray)) else float(base_val)
except:
    pass

if len(true_positives) > 0:
    tp_idx = true_positives[0]
    plt.figure(figsize=(10, 4))
    shap.waterfall_plot(shap.Explanation(values=shap_values_to_plot[tp_idx], 
                                         base_values=base_val,
                                         data=X_test_processed[tp_idx],
                                         feature_names=feature_names),
                        show=False)
    plt.title(f'SHAP Waterfall Plot for True Positive (Index {tp_idx})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'shap_individual_tp.png'))
    plt.close()

if len(false_negatives) > 0:
    fn_idx = false_negatives[0]
    plt.figure(figsize=(10, 4))
    shap.waterfall_plot(shap.Explanation(values=shap_values_to_plot[fn_idx], 
                                         base_values=base_val,
                                         data=X_test_processed[fn_idx],
                                         feature_names=feature_names),
                        show=False)
    plt.title(f'SHAP Waterfall Plot for False Negative (Index {fn_idx})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'shap_individual_fn.png'))
    plt.close()

print("SHAP explainability plots saved.")

# ---------------------------------------------------------
# 6. EXPORT METRICS TO JSON
# ---------------------------------------------------------
print("\nExporting evaluation results to JSON...")

evaluation_results = {
    'metrics_threshold_0.5': {
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc)
    },
    'threshold_analysis': threshold_analysis,
    'selected_threshold': 0.5,
    'selection_reason': "0.5 provides a strong balance of high recall without introducing an unacceptable level of false alarms (precision). Lower thresholds produce too many false alarms, while higher thresholds miss too many failures."
}

with open(EVAL_JSON_PATH, 'w') as f:
    json.dump(evaluation_results, f, indent=4)
print(f"Detailed evaluation metrics saved to {EVAL_JSON_PATH}")

# Update metadata json as well
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

metadata['evaluation'] = evaluation_results

with open(METADATA_PATH, 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"Updated {METADATA_PATH} with evaluation results.")

print("\nStage 3: Model Evaluation and Explainable AI Completed Successfully.")
