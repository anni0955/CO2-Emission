import pandas as pd 
import logging 
from pathlib import Path
import joblib

import yaml

from sklearn.ensemble import GradientBoostingRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.parent


DATA_DIR = ROOT_DIR / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_PATH = ROOT_DIR / 'models'
PARAMS_FILE_PATH = ROOT_DIR / 'params.yaml'

def read_parameters(path):
    with open(path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_data(datapath):
    df = pd.read_csv(datapath)
    return df

def save_model(model, path):
    joblib.dump(model, path)
    logging.info('model saved successfully')


def train_model(train_data, model):
    x_train = train_data.drop(columns=['CO2 Emissions(g/km)'])
    y_train = train_data['CO2 Emissions(g/km)']

    model.fit(x_train, y_train)
    logging.info('model trained successfully')
    return model

def main():
    train_data = load_data(PROCESSED_DIR / 'train_processed.csv')
    logging.info('train and test data loaded successfully')

    parameters = read_parameters(PARAMS_FILE_PATH)['train_model']

    n_estimators = parameters['n_estimators']
    max_depth = parameters['max_depth']
    learning_rate = parameters['learning_rate']

    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    model = train_model(train_data, model)
    save_model(model, MODELS_PATH / 'model.joblib')


if __name__ == '__main__':
    main()
