import joblib
import pandas as pd
try:
    from keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model

class PredictionService:
    def __init__(self, model_path: str, preprocessor_path: str):
        """
        Initialize the PredictionService by loading the trained ANN model 
        and the fitted preprocessing pipeline.
        """
        self.model = load_model(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_data: dict) -> dict:
        """
        Takes raw dictionary input, applies preprocessing, 
        and returns the failure prediction.
        """
        # Convert dictionary to DataFrame (expects a list of dicts)
        # Note: The input_data must use the exact alias names (e.g., 'Air temperature [K]')
        df = pd.DataFrame([input_data])
        
        # Apply the pre-fitted preprocessing pipeline
        processed_data = self.preprocessor.transform(df)
        
        # Obtain prediction probability from the ANN
        prob = float(self.model.predict(processed_data)[0][0])
        
        # Apply the baseline threshold of 0.5 established in Stage 3
        prediction = 1 if prob >= 0.5 else 0
        status = "Potential Failure" if prediction == 1 else "Normal"
        
        return {
            "failure_probability": round(prob, 4),
            "prediction": prediction,
            "status": status
        }
