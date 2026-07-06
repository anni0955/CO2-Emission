from pydantic import BaseModel

class VehicleData(BaseModel):
    Make: str
    Model: str
    Vehicle_Class: str
    Engine_Size_L: float
    Cylinders: int
    Transmission: str
    Fuel_Type: str
    Fuel_Consumption_City_L_100_km: float
    Fuel_Consumption_Hwy_L_100_km: float
    Fuel_Consumption_Comb_L_100_km: float
    Fuel_Consumption_Comb_mpg: float