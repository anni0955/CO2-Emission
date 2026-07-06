from fastapi import APIRouter

from app.schemas import VehicleData
from app.predictor import predict

router = APIRouter()

@router.post('/predict')
def predict_co2(vehicle: VehicleData):
    prediction = predict(vehicle.model_dump())
    return {'predicted_co2_emission': prediction}