import pandas as pd 
from pathlib import Path
import logging 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).parent.parent.parent


DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "CO2_Emissions_Canada.csv"
CLEANED_DATA_PATH = DATA_DIR / "raw" / "cleaned.csv"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info('Data Cleaning started')

    # solved the lettercase problem 
    cat_cols = ['Make',	'Model', 'Vehicle Class']
    for col in cat_cols:
        df[col] = df[col].str.lower()
    
    logging.info('Converted to lowercase')

    # solved the duplicacy problem 
    df = df.drop_duplicates()
    
    logging.info('removed te duplicated rows')
    return df


def main():
    logging.info('Loading dataset')
    df = pd.read_csv(RAW_DATA_PATH)

    logger.info('Dataset loaded successfully')

    cleaned_df = clean_data(df)

    cleaned_df.to_csv(
        CLEANED_DATA_PATH,
        index=False
    )

    logging.info(f'Dataset saved at {CLEANED_DATA_PATH}')

if __name__ == '__main__':
    main()
