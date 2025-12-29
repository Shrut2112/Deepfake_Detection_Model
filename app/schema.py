from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    prediction: str
    certainty: float
    probability: float
    ela_image: Optional[str] = None
    dft_image: Optional[str] = None
    og_image: Optional[str] = None
    