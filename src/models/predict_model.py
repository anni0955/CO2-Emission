import joblib
import pandas as pd 
from pathlib import Path
import logging

import yaml 

ROOT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT_DIR / 'models'
PARAMETERS_FILE_PATH = ROOT_DIR / 'params.yaml'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__file__)


def read_parameters(path):
    with open(path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_model(path):
    model = joblib.load(path)

    logging.info('model loaded successfully')
    return model

def transform_test_data(prediction_df):
    path = MODELS_DIR / 'transformer.joblib'
    transformer = load_model(path)

    logging.info('transformer loaded successfully')
    trasnformed_df = transformer.transform(prediction_df)

    logging.info('data is transformed as required')
    return trasnformed_df


def predict(transfomed_df, model):
    prediction = model.predict(transfomed_df)

    logging.info(f'here is the prediction for the input {round(prediction[0], 2)}')
    return prediction

def main():
    parameters = read_parameters(PARAMETERS_FILE_PATH)['predict_model']
    prediction_df = pd.DataFrame([
        {
            'Make': parameters['Make'],
            'Model': parameters['Model'],
            'Vehicle Class': parameters['Vehicle Class'],
            'Engine Size(L)': parameters['Engine Size(L)'],
            'Cylinders': parameters['Cylinders'],
            'Transmission': parameters['Transmission'],
            'Fuel Type': parameters['Fuel Type'],
            'Fuel Consumption City (L/100 km)': parameters['Fuel Consumption City (L/100 km)'],
            'Fuel Consumption Hwy (L/100 km)': parameters['Fuel Consumption Hwy (L/100 km)'],
            'Fuel Consumption Comb (L/100 km)': parameters['Fuel Consumption Comb (L/100 km)'],
            'Fuel Consumption Comb (mpg)': parameters['Fuel Consumption Comb (mpg)']
        }
    ])

    transformed_df = transform_test_data(prediction_df)

    model_path = MODELS_DIR / 'model.joblib'
    model = load_model(model_path)

    prediction = predict(transformed_df, model)

if __name__ == '__main__':
    main()