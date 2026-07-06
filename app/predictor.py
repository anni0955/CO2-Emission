import joblib
import pandas as pd 
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
MODEL_PATH = ROOT_DIR / 'models' / 'model.joblib'
TRANSFORMER_PATH = ROOT_DIR / 'models' / 'transformer.joblib'

transformer = joblib.load(TRANSFORMER_PATH)
model = joblib.load(MODEL_PATH)

def predict(data):
    input_data = {

        'Make': data["Make"],
        'Model': data["Model"],
        'Vehicle Class': data["Vehicle_Class"],
        'Engine Size(L)': data["Engine_Size_L"],
        'Cylinders': data["Cylinders"],
        'Transmission': data["Transmission"],
        'Fuel Type': data["Fuel_Type"],
        'Fuel Consumption City (L/100 km)': data["Fuel_Consumption_City_L_100_km"],
        'Fuel Consumption Hwy (L/100 km)': data["Fuel_Consumption_Hwy_L_100_km"],
        'Fuel Consumption Comb (L/100 km)': data["Fuel_Consumption_Comb_L_100_km"],
        'Fuel Consumption Comb (mpg)': data["Fuel_Consumption_Comb_mpg"]

    }
    input_df = pd.DataFrame([input_data])
    transformed_df = transformer.transform(input_df)
    prediction = model.predict(transformed_df)

    return prediction[0]

