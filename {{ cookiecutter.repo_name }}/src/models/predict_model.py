import pickle
import logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

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
    

def main():
    """ retrieve the model and predict labels. Show prediction and performance
    """
    deserialized_model = retrieve_xgb_model()
    X_test = pd.read_csv(ROOT / 'data/processed/transfusion_x_test.csv')
    y_pred = deserialized_model.predict(X_test)

    y_test = pd.read_csv(ROOT / 'data/processed/transfusion_y_test.csv',
        header=None)
    auc = roc_auc_score(y_test.astype(int), deserialized_model.predict_proba(X_test)[:, 1])
    return y_pred, auc


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__file__)
    
    preds, auc = main()
    logging.info('The predictions are {}'.format(preds))
    logging.info('The AUC is {}'.format(auc))
