import logging
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def massage_data(raw_data):
    """ Preprocess the data for predictions
    """
    

    return raw_data


def main():
    """ Retrieves data and runs processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    app_train = pd.read_csv(ROOT/"data/raw/application_train.csv",nrows=1000)
    #df = pd.read_csv(ROOT / 'data/raw/transfusion_data_raw.csv')
    #app_train = massage_data(df)
    app_train.to_csv(ROOT / 'data/processed/app_train.csv', index=False)
    print("oui")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
   

    main()