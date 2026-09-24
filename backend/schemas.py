from pydantic import BaseModel, Field
from typing import Literal

class MachineDataInput(BaseModel):
    Type: Literal['L', 'M', 'H'] = Field(..., description="Product quality variant (L, M, or H)")
    Air_temperature: float = Field(..., alias="Air temperature [K]", description="Air temperature in Kelvin")
    Process_temperature: float = Field(..., alias="Process temperature [K]", description="Process temperature in Kelvin")
    Rotational_speed: float = Field(..., alias="Rotational speed [rpm]", description="Rotational speed in rpm")
    Torque: float = Field(..., alias="Torque [Nm]", description="Torque in Nm")
    Tool_wear: float = Field(..., alias="Tool wear [min]", description="Tool wear duration in minutes")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Type": "M",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 0
            }
        }
