import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import MachineDataInput
from .services.prediction_service import PredictionService

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Pulse API",
    description="Backend API for Machine Failure Prediction using an ANN.",
    version="1.0.0"
)

# Configure CORS for local development (React frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to the saved model and preprocessor (relative to backend/main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'predictive_pulse_ann.keras')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, '..', 'models', 'preprocessor.pkl')

# Global service instance
prediction_service = None
MODEL_LOADED = False

@app.on_event("startup")
def load_resources():
    """
    Loads the trained ANN model and preprocessing pipeline at application startup.
    This prevents reloading them on every API request.
    """
    global prediction_service, MODEL_LOADED
    try:
        prediction_service = PredictionService(MODEL_PATH, PREPROCESSOR_PATH)
        MODEL_LOADED = True
        print("Model and preprocessor successfully loaded.")
    except Exception as e:
        print(f"Failed to load model or preprocessor: {e}")
        MODEL_LOADED = False

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify API status and model availability.
    """
    if MODEL_LOADED:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict")
def predict_failure(data: MachineDataInput):
    """
    Accepts machine operating parameters, applies preprocessing, 
    and returns a failure prediction based on the ANN model.
    """
    if not MODEL_LOADED or prediction_service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot process predictions.")
    
    try:
        # Convert Pydantic model to dictionary, strictly using the aliases (e.g. 'Air temperature [K]')
        input_dict = data.model_dump(by_alias=True)
        
        # Call the prediction service
        result = prediction_service.predict(input_dict)
        return result
    except ValueError as ve:
        # Catch value errors like invalid types mapping or transformation issues
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(ve)}")
    except Exception as e:
        # Catch unexpected errors during inference
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
