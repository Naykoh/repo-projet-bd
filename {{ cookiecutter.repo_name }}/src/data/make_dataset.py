"""
    make_dataset.py
    ----------------

    This package read the original data from kaggle located in the data/raw directory
    and write it into the data/processed
    so we dont manipulate the original data
"""
import logging
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]



def main():
    """ 
        Retrieves data raw (../raw) into
        data ready to be analyzed (saved in ../processed).
    """
    app_train = pd.read_csv(ROOT/"data/raw/application_train.csv",nrows=1000)
    app_test = pd.read_csv(ROOT/"data/raw/application_test.csv",nrows=1000)
    

    app_train.to_csv(ROOT / 'data/processed/app_train.csv', index=False)
    app_test.to_csv(ROOT / 'data/processed/app_test.csv', index=False)
    


    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
   

    main()