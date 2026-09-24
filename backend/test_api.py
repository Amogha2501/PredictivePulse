import os
import sys
from fastapi.testclient import TestClient

# Add backend directory to sys.path so it can import schemas/services
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import app

with TestClient(app) as client:
    print("="*40)
    print("Testing GET /health")
    print("="*40)
    response = client.get("/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    print("\n" + "="*40)
    print("Testing POST /predict (Normal case)")
    print("="*40)
    payload = {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0
    }
    response = client.post("/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    print("\n" + "="*40)
    print("Testing POST /predict (High wear case - potentially failure)")
    print("="*40)
    payload2 = {
        "Type": "L",
        "Air temperature [K]": 305.1,
        "Process temperature [K]": 313.6,
        "Rotational speed [rpm]": 1350,
        "Torque [Nm]": 65.8,
        "Tool wear [min]": 250
    }
    response = client.post("/predict", json=payload2)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    print("\n" + "="*40)
    print("Testing POST /predict (Invalid Input - Missing Field)")
    print("="*40)
    payload3 = {
        "Type": "L",
        "Air temperature [K]": 305.1
    }
    response = client.post("/predict", json=payload3)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
