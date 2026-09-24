# Predictive Pulse Backend

This is the FastAPI backend for the Predictive Pulse project. It serves predictions for machine failure using a pre-trained Artificial Neural Network (ANN) and Scikit-Learn preprocessing pipeline.

## Setup and Installation

1. **Activate the Virtual Environment**:
   From the root of the project (`Predictive-Pulse`), activate the environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

2. **Install Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Start the Server**:
   From the root of the project, run:
   ```bash
   uvicorn backend.main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`.
   Interactive API documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.

## API Endpoints

### 1. Health Check
`GET /health`
Returns the status of the API and verifies if the machine learning model is successfully loaded into memory.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Predict Failure
`POST /predict`
Accepts machine operating parameters and returns a failure prediction probability and classification.

**Example Request**:
```json
{
  "Type": "M",
  "Air temperature [K]": 298.1,
  "Process temperature [K]": 308.6,
  "Rotational speed [rpm]": 1551,
  "Torque [Nm]": 42.8,
  "Tool wear [min]": 0
}
```

**Example Response**:
```json
{
  "failure_probability": 0.0123,
  "prediction": 0,
  "status": "Normal"
}
```
