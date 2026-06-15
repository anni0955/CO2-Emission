import pandas as pd 
from sklearn.model_selection import train_test_split
import yaml
import logging 
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.parent


DATA_DIR = ROOT_DIR / "data"
INTERIM_DATA_PATH = DATA_DIR / "interim" / "cleaned.csv"
PROCESSED_DIR = DATA_DIR / "processed"

def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )
    return train_data, test_data

def read_params(file_path):
    with open(file_path, 'r') as f:
        params_file = yaml.safe_load(f)
    
    return params_file

def save_data(data: pd.DataFrame, save_path: Path):
    data.to_csv(save_path, index=False)

if __name__ == '__main__':
    data_path = INTERIM_DATA_PATH
    params_file_path = ROOT_DIR / 'params.yaml'

    df = pd.read_csv(data_path)
    logging.info('Data loaded successfully')


    print("PARAM FILE:", params_file_path)
    print("EXISTS:", params_file_path.exists())

    params = read_params(params_file_path)
    print("PARAMS:", params)
    print("TYPE:", type(params))


    parameters = read_params(params_file_path)['Data_Preparation']
    test_size = parameters['test_size']
    random_state = parameters['random_state']
    logging.info('Parameters read successfully')

    train_data, test_data = split_data(df, test_size, random_state)
    logging.info('Data Splitted successfully')

    save_data(train_data, PROCESSED_DIR/'train.csv')
    save_data(test_data, PROCESSED_DIR/'test.csv')