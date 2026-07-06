from fastapi import FastAPI

from app.routers.prediction import router

app = FastAPI(
    title='CO2 Emission Prediction',
    version='1.0.0'
)

app.include_router(router)

@app.get('/')
def home():
    return {'message': 'CO2 Emission Prediction API'}

