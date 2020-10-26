"""
    predict_model.py
    ----------------

    :param model: 'xgb', 'rf' or 'gb'
    :type model: String

    This package contains function to predict application_test with 3 differents models
"""

import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path


import sys      

ROOT = Path(__file__).resolve().parents[2]


def retrieve_xgb_model():
    """
        retrieve the pickled model object
    """
    pickled_model = ROOT / 'models/xgb_model.model'
    with open(pickled_model, 'rb') as fin:
        return(pickle.load(fin))

def retrieve_rf_model():
    """
        retrieve the pickled model object
    """
    pickled_model = ROOT / 'models/rf_model.model'
    with open(pickled_model, 'rb') as fin:
        return(pickle.load(fin))

def retrieve_gb_model():
    """
        retrieve the pickled model object
    """
    pickled_model = ROOT / 'models/gb_model.model'
    with open(pickled_model, 'rb') as fin:
        return(pickle.load(fin))


    

def main(model):
    """ 
        retrieve the model and predict labels of application_test

        :param model: model selected
    """

    app_test = pd.read_csv(ROOT / 'data/processed/app_test_processed.csv')

    print(model)
    if (model=='xgb'):    
        loaded_model = retrieve_xgb_model()
        predictions = loaded_model.predict(app_test)

    if (model=='rf'):    
        loaded_model = retrieve_rf_model()
        predictions = loaded_model.predict(app_test)

    if (model=='gb'):    
        loaded_model = retrieve_gb_model()
        predictions = loaded_model.predict(app_test)



    #predictions is the predictions of the model from application test
    writing_path = "data/result/app_test_"+model+"_predictions.csv"
    (pd.DataFrame(predictions)).to_csv(ROOT / writing_path,index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__file__)
    
    model = ''
    try:
        model=sys.argv[1]
    except:
        pass
    
    if ( (model != 'xgb') and ( model!= 'rf') and (model != 'gb')) :
        raise NameError('unknown model, pass a model in argument among xgb, rf and gb')
                      
    main(model)
    logging.info('Model : {}'.format(model))