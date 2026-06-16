import pandas as pd 
import logging 
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

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


def load_data(datapath):
    df = pd.read_csv(datapath)
    return df

def save_transformer(transformer, path):
    joblib.dump(transformer, path/'transformer.joblib')
    logging.info('Transformer saved successfully')

def save_data(data, path):
    data.to_csv(path, index=False)

def make_transformations(train_data, test_data):
    x_train = train_data.drop(columns=['CO2 Emissions(g/km)'])
    y_train = train_data['CO2 Emissions(g/km)']

    x_test = test_data.drop(columns=['CO2 Emissions(g/km)'])
    y_test = test_data['CO2 Emissions(g/km)']

    transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), ['Make', 'Model', 'Vehicle Class', 'Transmission']),
        ('oe', OrdinalEncoder(categories=[['N', 'X', 'D', 'Z', 'E']]), ['Fuel Type']),
        ('scale', StandardScaler(),['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)'])
    ])

    transformer.fit(x_train)
    feature_names = transformer.get_feature_names_out()

    train_trf = transformer.transform(x_train)
    test_trf = transformer.transform(x_test)

    train_trf = pd.DataFrame(train_trf, columns=feature_names)
    test_trf = pd.DataFrame(test_trf, columns=feature_names)

    train_trf['CO2 Emissions(g/km)'] = y_train
    test_trf['CO2 Emissions(g/km)'] = y_test

    save_data(train_trf, PROCESSED_DIR / 'train_processed.csv')
    save_data(test_trf, PROCESSED_DIR / 'test_processed.csv')
    save_transformer(transformer, MODELS_PATH)

def main():
    train_data = load_data(INTERIM_DIR / 'train.csv')
    test_data = load_data(INTERIM_DIR / 'test.csv')

    logging.info('train and test data loaded successfully')

    make_transformations(train_data, test_data)

    logging.info('transformation completed and data saved')

if __name__ == '__main__':
    main()
