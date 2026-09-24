export interface MachineDataInput {
  "Type": 'L' | 'M' | 'H';
  "Air temperature [K]": number;
  "Process temperature [K]": number;
  "Rotational speed [rpm]": number;
  "Torque [Nm]": number;
  "Tool wear [min]": number;
}

export interface PredictionResponse {
  failure_probability: number;
  prediction: number;
  status: "Normal" | "Potential Failure";
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001';

export const predictFailure = async (data: MachineDataInput): Promise<PredictionResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API Error: ${response.status} ${response.statusText}`);
    }

    const result: PredictionResponse = await response.json();
    return result;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('An unknown network error occurred while contacting the server.');
  }
};
