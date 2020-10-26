"""
    api_request.py
    ----------------

    This package is useful to request on the mlflow REST API 
"""
import logging
import pandas as pd
import requests
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

def get_sample():
    """
        get sample data from application_test

        :return: dataframe of the sample
        :rtype: DataFrame
    """
    df=pd.read_csv(ROOT/"data/processed/app_test_processed.csv")
    return pd.DataFrame(df.iloc[0]).transpose()

def main():
    """
        Request the API
    """
    host = '127.0.0.1'
    port = '1234'

    url = f'http://{host}:{port}/invocations'

    headers = {
    'Content-Type': 'application/json',
    }

    df=get_sample()
    # test_data is a Pandas dataframe with data for testing the ML model
    http_data = df.to_json(orient='split')

    r = requests.post(url=url, headers=headers, data=http_data)

    print(f'Predictions: {r.text}')

    


if __name__ == '__main__':
    main()