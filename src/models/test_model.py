import pandas as pd 
from pathlib import Path
import logging
import joblib

from sklearn.metrics import root_mean_squared_error, mean_absolute_error

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT_DIR /  'models'
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'

def load_model(path):
    model = joblib.load(path)

    logging.info('model loaded successfully')
    return model

def load_data(path):
    df = pd.read_csv(path)

    logging.info('test data loaded successfully')
    return df

def evaluate(model, test_data):
    x_test = test_data.drop(columns=['CO2 Emissions(g/km)'])
    y_test = test_data['CO2 Emissions(g/km)']

    y_pred = model.predict(x_test)

    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = root_mean_squared_error(y_test, y_pred)

    logging.info(f'model have root mean squarred error of {round(RMSE, 2)} g/km')
    logging.info(f'model have mean absolute error of {round(MAE, 2)} g/km')
    return (RMSE, MAE)

def main():
    data_path = PROCESSED_DIR / 'test_processed.csv'
    model_path = MODELS_DIR / 'model.joblib'

    test_df = load_data(data_path)
    model = load_model(model_path)

    MSE, MAE = evaluate(model, test_df)


if __name__ == '__main__':
    main()