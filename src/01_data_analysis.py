import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set paths
DATA_PATH = 'data/ai4i2020.csv'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. DATASET ACQUISITION
# ---------------------------------------------------------
if not os.path.exists(DATA_PATH):
    print("Downloading dataset from UCI Machine Learning Repository...")
    # Using the direct URL to the CSV file
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    urllib.request.urlretrieve(url, DATA_PATH)
    print(f"Dataset saved to {DATA_PATH}")
else:
    print(f"Dataset already exists at {DATA_PATH}")

# ---------------------------------------------------------
# 2. LOAD DATASET
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------
# 3. INITIAL DATASET ANALYSIS
# ---------------------------------------------------------
print("\n" + "="*50)
print("INITIAL DATASET ANALYSIS")
print("="*50)

print("\n1. Shape of the dataset:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\n2. Column Names:")
print(df.columns.tolist())

print("\n3. Data Types:")
print(df.dtypes)

print("\n4. First 5 records:")
print(df.head())

print("\n5. Statistical Summary:")
print(df.describe())

print("\n6. Missing Values:")
print(df.isnull().sum())

print("\n7. Duplicate Records:")
print(df.duplicated().sum())

print("\n8. Unique Values of Categorical Columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"  {col}: {df[col].unique()}")

print("\n9. Target Variable Distribution (Machine failure):")
target_col = 'Machine failure'
dist = df[target_col].value_counts()
print(dist)
print(f"Percentage of Normal (0): {dist[0]/len(df)*100:.2f}%")
print(f"Percentage of Failure (1): {dist[1]/len(df)*100:.2f}%")


# ---------------------------------------------------------
# 4. EXPLANATION OF COLUMNS
# ---------------------------------------------------------
# UDI: Unique identifier ranging from 1 to 10000
# Product ID: Quality variant (L/M/H) + serial number
# Type: Product quality variant (L, M, or H)
# Air temperature [K]: Generated via a random walk
# Process temperature [K]: Generated via a random walk plus air temperature
# Rotational speed [rpm]: Calculated from power
# Torque [Nm]: Torque values normally distributed
# Tool wear [min]: Time duration of tool wear
# Machine failure: Target variable (1 for failure, 0 for normal)
# TWF (Tool Wear Failure): Specific failure mode indicator
# HDF (Heat Dissipation Failure): Specific failure mode indicator
# PWF (Power Failure): Specific failure mode indicator
# OSF (Overstrain Failure): Specific failure mode indicator
# RNF (Random Failures): Specific failure mode indicator


# ---------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------
print("\nGenerating EDA Visualizations...")

# 5.1 Machine failure class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Machine failure', palette='Set2')
plt.title('Machine Failure Class Distribution')
plt.xlabel('Machine Failure (0 = Normal, 1 = Failure)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution.png'))
plt.close()

# 5.2 Numerical feature distributions
numeric_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
df[numeric_features].hist(bins=30, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle('Numerical Feature Distributions', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'numerical_distributions.png'))
plt.close()

# 5.3 Correlation heatmap
plt.figure(figsize=(10, 8))
# Only correlate numeric columns
corr_df = df[numeric_features + ['Machine failure']].corr()
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'))
plt.close()

# 5.4 Feature distributions grouped by machine failure (Boxplots)
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(data=df, x='Machine failure', y=col, palette='Set2')
    plt.title(f'{col} vs Machine Failure')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'features_vs_failure.png'))
plt.close()

print("EDA Visualizations saved in 'results/' directory.")


# ---------------------------------------------------------
# 6. FEATURE SELECTION & PREPROCESSING PIPELINE
# ---------------------------------------------------------
print("\n" + "="*50)
print("FEATURE SELECTION & PREPROCESSING")
print("="*50)

# Identify features
# - Drop UDI and Product ID (identifiers)
# - Drop TWF, HDF, PWF, OSF, RNF (leakage columns for predicting 'Machine failure')
leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
id_cols = ['UDI', 'Product ID']

X = df.drop(columns=[target_col] + leakage_cols + id_cols)
y = df[target_col]

print("\nSelected Features for ANN Input:")
print(X.columns.tolist())
print(f"Total features: {len(X.columns)}")

# Define categorical and numerical features based on selection
cat_features = ['Type']
num_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Create preprocessing steps
# - One-Hot Encoding for 'Type'
# - StandardScaler for numerical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

# ---------------------------------------------------------
# 7. STRATIFIED TRAIN/TEST SPLIT
# ---------------------------------------------------------
print("\nPerforming Stratified Train/Test Split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing set shape: X={X_test.shape}, y={y_test.shape}")

# Fit preprocessor on training data only
print("\nFitting preprocessing pipeline on training data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test) # Transform test data based on train params

# Save the preprocessor object
preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
joblib.dump(preprocessor, preprocessor_path)
print(f"Fitted preprocessing pipeline saved to {preprocessor_path}")

# ---------------------------------------------------------
# 8. SUMMARY
# ---------------------------------------------------------
print("\n" + "="*50)
print("SUMMARY OF STAGE 1")
print("="*50)
print(f"- Dataset Size: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"- Target Variable: '{target_col}'")
print(f"- Selected Input Features ({len(X.columns)}): {X.columns.tolist()}")
print(f"- Class Distribution: {dist[0]} Normal ({dist[0]/len(df)*100:.1f}%), {dist[1]} Failure ({dist[1]/len(df)*100:.1f}%)")
print("- Preprocessing Performed: Dropped IDs and leakages, StandardScaler (numeric), OneHotEncoder (categorical), Stratified Split (80/20).")
print("- Files Created: data/ai4i2020.csv, EDA plots in results/, models/preprocessor.pkl")
print("="*50)
print("Dataset setup, analysis, and preprocessing completed successfully.")
