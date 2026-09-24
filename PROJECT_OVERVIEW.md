# Predictive Pulse — Complete System Overview & Simple Guide

## 1. 🌟 The Problem Explained Simply

In manufacturing factories, industrial machines run continuously to produce parts. Over time, components experience heat, friction, speed stress, and physical tool wear.

* **Traditional Maintenance (Reactive)**: Wait until the machine breaks down. 
  * *Disadvantage*: Causes sudden production line stops, expensive repairs, missed delivery deadlines, and worker safety risks.
* **Scheduled Maintenance (Preventive)**: Replace tools on a fixed calendar schedule (e.g., every 30 days).
  * *Disadvantage*: Wastes money replacing tools that are still good, while missing early signs of unexpected wear.
* **Predictive Maintenance with Predictive Pulse (Proactive)**: Monitor continuous operating conditions (temperature, speed, torque, wear) and use Artificial Intelligence to **predict failures before they happen**.

---

## 🎯 2. What We Are Predicting

Predictive Pulse analyzes 6 machine sensor parameters to predict **Machine Failure**:

* **Target Output**:
  * `0`: **Normal** (Machine is running within safe operational parameters).
  * `1`: **Potential Failure** (High risk of breakdown; maintenance required immediately).
* **Failure Probability**: A percentage score (e.g., `0.79%` risk vs `85.50%` risk).

### Input Features (Sensor Parameters)

| Parameter Name | Unit | What it Represents | Normal Range |
| :--- | :--- | :--- | :--- |
| **Product Quality Type** | Category | Product variant quality level: **L** (Low - 50%), **M** (Medium - 30%), **H** (High - 20%) | `L`, `M`, or `H` |
| **Air Temperature** | Kelvin (`K`) | Ambient room/factory temperature surrounding the machine | ~ `295 K` – `304 K` |
| **Process Temperature** | Kelvin (`K`) | Heat generated inside the manufacturing process | ~ `305 K` – `313 K` |
| **Rotational Speed** | `rpm` | Speed at which the motor spindle spins | ~ `1168` – `2886 rpm` |
| **Torque** | `Nm` | Rotational force exerted by the motor | ~ `3.8` – `76.6 Nm` |
| **Tool Wear** | `min` | Total operational minutes the current cutting tool has been used | `0` – `253 min` |

---

## 🔬 3. Dataset & Technical Deep-Dive

### Where Did We Get the Dataset?
* **Dataset**: **AI4I 2020 Predictive Maintenance Dataset**
* **Source**: UCI Machine Learning Repository
* **Direct URL**: [https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
* **Dataset Size**: **10,000 synthetic operational records** reflecting real manufacturing failure modes.

### What We Have Done Technically in the Project

#### 1. Data Analysis & Preprocessing Pipeline (`src/01_data_analysis.py`)
* **Feature Cleaning**: Dropped non-predictive row identifiers (`UDI`, `Product ID`) and target leakage indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
* **Numerical Transformation**: Standardized continuous parameters (`Air temp`, `Process temp`, `Rotational speed`, `Torque`, `Tool wear`) using Scikit-Learn `StandardScaler`.
* **Categorical Transformation**: Encoded product variant `Type` (`L`, `M`, `H`) using `OneHotEncoder`.
* **Data Splitting**: Created an **80/20 Stratified Train/Test Split** to ensure equal distribution of rare failure classes in both sets.
* **Pipeline Storage**: Fitted and serialized the preprocessor object as `models/preprocessor.pkl`.

#### 2. Deep Learning Model Architecture (`src/02_model_training.py`)
* **Class Imbalance Handling**: Applied `compute_class_weight('balanced')` during model fitting because failures account for ~3.4% of total dataset records.
* **Neural Network Structure (TensorFlow/Keras)**:
  * **Input Layer**: Accepts preprocessed feature tensor (8 dimensions after one-hot encoding).
  * **Hidden Layer 1**: `Dense(64 units, activation='relu')` + `Dropout(0.2)`
  * **Hidden Layer 2**: `Dense(32 units, activation='relu')` + `Dropout(0.2)`
  * **Hidden Layer 3**: `Dense(16 units, activation='relu')`
  * **Output Layer**: `Dense(1 unit, activation='sigmoid')` outputting a continuous failure probability score `[0.0, 1.0]`.
* **Optimization & Loss**:
  * Optimizer: `Adam(learning_rate=0.001)`
  * Loss Function: `binary_crossentropy`
  * Callbacks: `EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)`

#### 3. Explainable AI (SHAP) & Deployment
* **Explainable AI**: Integrated **SHAP (SHapley Additive exPlanations)** KernelExplainer to interpret individual and global feature impacts on predictions.
* **REST API**: Built a high-performance **FastAPI** backend (`backend/main.py`) loading the Keras model (`predictive_pulse_ann.keras`) at startup for low-latency inference.
* **Frontend Application**: Built a **React + TypeScript + Tailwind CSS** dashboard featuring Option A Industrial Dark / Option B Light Mode, visual sensor range bars, and real-time inference.

---

## 🏗️ 4. System Architecture

```mermaid
graph TD
    User["👤 User / Machine Operator"] -->|Interacts with Web UI| ReactApp["💻 Frontend (React + TypeScript + Tailwind)"]
    ReactApp -->|HTTP POST /predict JSON| FastAPI["⚡ Backend API (FastAPI + Uvicorn)"]
    
    subgraph Machine Learning Pipeline
        FastAPI -->|Input JSON| Preprocessor["⚙️ Scikit-Learn Preprocessor (preprocessor.pkl)"]
        Preprocessor -->|Scaled & One-Hot Encoded Tensors| ANNModel["🧠 Deep Learning Model (predictive_pulse_ann.keras)"]
        ANNModel -->|Failure Probability & Classification| FastAPI
    end
    
    subgraph Explainable AI (XAI)
        FastAPI -->|Global SHAP Analytics| SHAPEngine["📊 SHAP Explainability Plot (shap_summary.png)"]
    end
    
    FastAPI -->|JSON Response| ReactApp
    ReactApp -->|Displays Status, Probability Bar & SHAP Insights| User
```

---

## 💻 5. Commands Reference

### A. Environment Setup & Backend Launch
```bash
# 1. Activate the Python virtual environment (Windows)
.venv\Scripts\activate

# 2. Run the FastAPI Backend Server on port 8001
uvicorn backend.main:app --port 8001 --reload
```
* Backend URL: `http://127.0.0.1:8001`
* Interactive API Documentation (Swagger UI): `http://127.0.0.1:8001/docs`

---

### B. Frontend Launch
```bash
# 1. Open a new terminal and navigate to frontend
cd frontend

# 2. Start the React development server
npm run dev
```
* Dashboard URL: `http://localhost:5173`

---

### C. Run Model Training & Evaluation Pipeline
```bash
# Stage 1: Data Download & Preprocessing
.venv\Scripts\python.exe src/01_data_analysis.py

# Stage 2: Train Keras Neural Network
.venv\Scripts\python.exe src/02_model_training.py

# Stage 3: Generate SHAP & Model Evaluations
.venv\Scripts\python.exe src/03_model_evaluation.py
```

---

## 🔄 6. Example Walkthrough: How It Works Step-by-Step

### Scenario: High Tool Wear & Extreme Torque (Failure Risk)

#### Step 1: User Inputs Data in Web Dashboard
An operator inputs these parameters for a machine running on the factory floor:
* **Product Quality Type**: `L`
* **Air Temperature**: `305.1 K`
* **Process Temperature**: `313.6 K`
* **Rotational Speed**: `1350 rpm`
* **Torque**: `65.8 Nm`
* **Tool Wear**: `250 min`

#### Step 2: Frontend sends JSON payload
The React app packages the data into JSON and sends an HTTP POST request to `http://127.0.0.1:8001/predict`.

#### Step 3: Backend Preprocessing
FastAPI passes this dictionary through `preprocessor.pkl`:
* `Type='L'` is converted to `[1, 0, 0]` via One-Hot Encoding.
* Temperature, Speed, Torque, and Wear values are standardized using Standard Scaling.

#### Step 4: Artificial Neural Network Inference
The scaled array is fed into the Keras ANN model layers. The final sigmoid node calculates:
* `Failure Probability = 0.8550` (85.50%)

#### Step 5: Backend Response
FastAPI sends the JSON result back:
```json
{
  "failure_probability": 0.8550,
  "prediction": 1,
  "status": "Potential Failure"
}
```

#### Step 6: UI Presentation & Action
* The web app instantly highlights the status in 🔴 **Potential Failure**.
* The progress bar fills to **85.50%**.
* The operator sees from the SHAP chart that high **Tool Wear (250 min)** and high **Torque (65.8 Nm)** are the primary causes of risk, allowing them to schedule maintenance *before* the tool breaks!
