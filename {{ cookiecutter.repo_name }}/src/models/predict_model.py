import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score

import sys      

ROOT = Path(__file__).resolve().parents[2]


def retrieve_xgb_model():
    """retrieve the pickled model object
    """
    pickled_model = ROOT / 'models/xgb_model.model'
    with open(pickled_model, 'rb') as fin:
        return(pickle.load(fin))

def retrieve_rf_model():
    """retrieve the pickled model object
    """
    pickled_model = ROOT / 'models/rf_model.model'
    with open(pickled_model, 'rb') as fin:
        return(pickle.load(fin))

def retrieve_gb_model():
    """retrieve the pickled model object
    """
    pickled_model = ROOT / 'models/gb_model.model'
    with open(pickled_model, 'rb') as fin:
        return(pickle.load(fin))


    

def main(model):
    """ retrieve the model and predict labels. Show prediction and performance
    """
    X_test = pd.read_csv(ROOT / 'data/processed/app_train_x_test.csv')
    y_test = pd.read_csv(ROOT / 'data/processed/app_train_y_test.csv')
    print(model)
    if (model=='xgb'):    
        loaded_model = retrieve_xgb_model()

    if (model=='rf'):    
        loaded_model = retrieve_rf_model()

    if (model=='gb'):    
        loaded_model = retrieve_gb_model()        

    y_pred = loaded_model.predict(X_test)
    
    auc = roc_auc_score(y_test.astype(int), loaded_model.predict_proba(X_test)[:, 1])

    accuracy = accuracy_score(y_test, y_pred)

    return y_pred, auc, accuracy


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
                      
    preds, auc, accuracy = main(model)
    logging.info('The predictions are {}'.format(preds))
    logging.info('The AUC is {}'.format(auc))
    logging.info('The accuracy is {}'.format(accuracy))

