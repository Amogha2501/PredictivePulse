import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Set paths
DATA_PATH = 'data/ai4i2020.csv'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'
MODEL_PATH = 'models/predictive_pulse_ann.keras'
METADATA_PATH = 'models/model_metadata.json'
RESULTS_DIR = 'results'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATA & PREPROCESSOR
# ---------------------------------------------------------
print("Loading data and preprocessor...")
df = pd.read_csv(DATA_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

target_col = 'Machine failure'
leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
id_cols = ['UDI', 'Product ID']

X = df.drop(columns=[target_col] + leakage_cols + id_cols)
y = df[target_col]

# ---------------------------------------------------------
# 2. DATA SPLITTING & TRANSFORMATION
# ---------------------------------------------------------
print("Splitting data (80/20) and transforming...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Transform using the already-fitted pipeline
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"X_train_processed shape: {X_train_processed.shape}")
print(f"X_test_processed shape: {X_test_processed.shape}")

# ---------------------------------------------------------
# 3. COMPUTE CLASS WEIGHTS
# ---------------------------------------------------------
print("Computing class weights...")
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {int(classes[0]): float(weights[0]), int(classes[1]): float(weights[1])}
print(f"Class Weights: {class_weight_dict}")

# ---------------------------------------------------------
# 4. BUILD ANN ARCHITECTURE
# ---------------------------------------------------------
print("Building ANN model...")
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

# ---------------------------------------------------------
# 5. COMPILE MODEL
# ---------------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy', 
        tf.keras.metrics.Precision(name='precision'), 
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# ---------------------------------------------------------
# 6. TRAIN MODEL
# ---------------------------------------------------------
print("Training model...")
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_processed, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stopping],
    verbose=1
)

actual_epochs = len(history.history['loss'])
print(f"Training stopped after {actual_epochs} epochs.")

# ---------------------------------------------------------
# 7. GENERATE AND SAVE PLOTS
# ---------------------------------------------------------
print("Saving training history plots...")

# Loss Graph
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_history_loss.png'))
plt.close()

# Accuracy Graph
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy During Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_history_accuracy.png'))
plt.close()

# Precision & Recall Graph
plt.figure(figsize=(8, 5))
plt.plot(history.history['precision'], label='Train Precision', color='green')
plt.plot(history.history['val_precision'], label='Validation Precision', color='lightgreen', linestyle='--')
plt.plot(history.history['recall'], label='Train Recall', color='red')
plt.plot(history.history['val_recall'], label='Validation Recall', color='lightcoral', linestyle='--')
plt.title('Precision & Recall During Training')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_history_pr.png'))
plt.close()

# ---------------------------------------------------------
# 8. EVALUATE ON TEST SET
# ---------------------------------------------------------
print("Evaluating on test set...")
y_pred_proba = model.predict(X_test_processed)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n--- TEST SET METRICS ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
plt.close()

# ---------------------------------------------------------
# 9. SAVE MODEL & METADATA
# ---------------------------------------------------------
print("Saving model and metadata...")
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

metadata = {
    'input_features': X.columns.tolist(),
    'class_weights': class_weight_dict,
    'model_path': MODEL_PATH,
    'test_metrics': {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc)
    },
    'architecture': 'Input(6) -> Dense(64) -> Dropout(0.2) -> Dense(32) -> Dropout(0.2) -> Dense(16) -> Dense(1, sigmoid)',
    'epochs_completed': actual_epochs
}

with open(METADATA_PATH, 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"Metadata saved to {METADATA_PATH}")

print("\nStage 2: Model Development Completed Successfully.")
