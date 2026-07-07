from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.predictor import predict

router = APIRouter()

templates = Jinja2Templates(directory="templates")


@router.post("/predict", response_class=HTMLResponse)
def predict_co2(
    request: Request,

    Make: str = Form(...),
    Model: str = Form(...),
    Vehicle_Class: str = Form(...),
    Engine_Size_L: float = Form(...),
    Cylinders: int = Form(...),
    Transmission: str = Form(...),
    Fuel_Type: str = Form(...),
    Fuel_Consumption_City_L_100_km: float = Form(...),
    Fuel_Consumption_Hwy_L_100_km: float = Form(...),
    Fuel_Consumption_Comb_L_100_km: float = Form(...),
    Fuel_Consumption_Comb_mpg: float = Form(...)
):

    vehicle = {
        "Make": Make,
        "Model": Model,
        "Vehicle_Class": Vehicle_Class,
        "Engine_Size_L": Engine_Size_L,
        "Cylinders": Cylinders,
        "Transmission": Transmission,
        "Fuel_Type": Fuel_Type,
        "Fuel_Consumption_City_L_100_km": Fuel_Consumption_City_L_100_km,
        "Fuel_Consumption_Hwy_L_100_km": Fuel_Consumption_Hwy_L_100_km,
        "Fuel_Consumption_Comb_L_100_km": Fuel_Consumption_Comb_L_100_km,
        "Fuel_Consumption_Comb_mpg": Fuel_Consumption_Comb_mpg,
    }

    prediction = predict(vehicle)

    return templates.TemplateResponse(
    request=request,
    name="result.html",
    context={
        "prediction": round(float(prediction), 2)
    }
)