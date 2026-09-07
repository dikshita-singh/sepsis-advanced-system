from pydantic import BaseModel, Field

class SepsisInputFeatures(BaseModel):
    """
    Defines the exact clinical measurements required for a real-time sepsis prediction.
    """
    HR: float = Field(..., description="Heart Rate (beats per minute)", example=78.0)
    O2Sat: float = Field(..., description="Oxygen Saturation (%)", example=98.5)
    Temp: float = Field(..., description="Body Temperature (degrees Celsius)", example=36.8)
    SBP: float = Field(..., description="Systolic Blood Pressure (mmHg)", example=120.0)
    MAP: float = Field(..., description="Mean Arterial Pressure (mmHg)", example=82.0)
    DBP: float = Field(..., description="Diastolic Blood Pressure (mmHg)", example=70.0)
    Resp: float = Field(..., description="Respiration Rate (breaths per minute)", example=16.0)
    Age: float = Field(..., description="Patient Age (years)", example=62.5)
    Gender: int = Field(..., description="Patient Gender (0 for Female, 1 for Male)", example=1)