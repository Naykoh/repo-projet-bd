import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn

from varname import nameof

import sys

from urllib.parse import urlparse



from src.features.build_features import get_features, preprocessor_create  

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# Algorithms, from the easiest to the hardest to intepret.r
from xgboost.sklearn import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]


if '{{ cookiecutter.python_interpreter }}' == 'python3':
    PROTOCOL = pickle.DEFAULT_PROTOCOL
else:
    PROTOCOL = 2



def fetch_processed(data_path):
    """
    fetch the data that was processed in make data and create training and test sets
    """
    app_train = pd.read_csv(ROOT / data_path)
    y = app_train["TARGET"]
    X = app_train.drop("TARGET", axis=1)

    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.3, random_state=42)

    return X_train, X_test, y_train, y_test




def model_definition(preprocessor, app_train):
    """ define the 3 ML model
    """
    y = app_train["TARGET"]
    # XGBoost
    xgb_model = Pipeline([("preprocessor", preprocessor),
    # Add a scale_pos_weight to make it balanced
    ("model", XGBClassifier(scale_pos_weight=(1 - y.mean()), n_jobs=-1))])

    # Random Forest
    rf_model = Pipeline([("preprocessor", preprocessor),
    ("model", RandomForestClassifier(class_weight="balanced", n_estimators=100))])

    #GradientBoostingClassifier
    gb_model = Pipeline([("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier())])

    return xgb_model, rf_model, gb_model


def xgb_fit_model(X_train, y_train, xgb_model):
    """
    fit a xgb model to the training data
    """
    gs = GridSearchCV(xgb_model, {"model__max_depth": [5, 10],
    "model__min_child_weight": [5, 10],
    "model__n_estimators": [25]},
    n_jobs=-1, cv=5, scoring="accuracy")
    gs.fit(X_train, y_train)

    print(gs.best_params_)
    print(gs.best_score_)

    xgb_model.set_params(**gs.best_params_)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def rf_fit_model(X_train, y_train, rf_model):
    """
    fit a random forest model to the training data
    """
    gs = GridSearchCV(rf_model, {"model__max_depth": [10, 15],
    "model__min_samples_split": [5, 10]},
    n_jobs=-1, cv=5, scoring="accuracy")
    gs.fit(X_train, y_train)

    print(gs.best_params_)
    print(gs.best_score_)

    rf_model.set_params(**gs.best_params_)
    rf_model.fit(X_train, y_train)
    return rf_model


def gb_fit_model(X_train, y_train, gb_model):
    """
    fit a gradient boosting model to the training data
    """
    gs = GridSearchCV(gb_model, {"model__max_depth": [10, 15],
    "model__min_samples_split": [5, 10]},
    n_jobs=-1, cv=5, scoring="accuracy")
    gs.fit(X_train, y_train)

    print(gs.best_params_)
    print(gs.best_score_)

    gb_model.set_params(**gs.best_params_)
    gb_model.fit(X_train, y_train)
    return gb_model

def eval_metrics(actual, pred):
    """ compare prediciton and actual value with different metrics to check the accuracy of the model
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



def mlflow_train(model,X_train,y_train,X_test,y_test,fit_function):

    with mlflow.start_run():

        #Train the model
        model_fitted = fit_function(X_train, y_train, model)

        y_pred = model_fitted.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        print("XGB model :")
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("params", model_fitted.get_params()['model'])
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model_fitted, "model", registered_model_name=nameof(fit_function))
        else:
            mlflow.sklearn.log_model(model_fitted, "model")






def main(model):
    """ Trains the model and test their accuracy on the retrieved data write it back to file
    """
    #Define the 3 model
    app_train = pd.read_csv(ROOT/"data/processed/application_train_processed.csv")
    num_features, cat_features = get_features(app_train)
    preprocessor = preprocessor_create(num_features, cat_features)
    xgb_model, rf_model, gb_model = model_definition(preprocessor, app_train)


    #fetch the processed dataset with training and test set
    X_train, X_test, y_train, y_test = fetch_processed('data/processed/application_train_processed.csv')

    if (model == 'xgb'):
        mlflow_train(xgb_model,X_train,y_train,X_test,y_test,xgb_fit_model)
    elif (model == 'rf'):
        mlflow_train(rf_model,X_train,y_train,X_test,y_test,rf_fit_model)
    else :
        mlflow_train(gb_model,X_train,y_train,X_test,y_test,gb_fit_model)


    # #Define the 3 ML model
    # app_train = pd.read_csv(ROOT/"data/processed/application_train_processed.csv")
    # num_features, cat_features = get_features(app_train)
    # preprocessor = preprocessor_create(num_features, cat_features)
    # xgb_model, rf_model, gb_model = model_definition(preprocessor, app_train)

    # #fetch the processed dataset with training and test set
    # X_train, X_test, y_train, y_test = fetch_processed('data/processed/application_train_processed.csv')
    
    # # Train the model
    
    # xgb_model = xgb_fit_model(X_train, y_train, xgb_model)
    # rf_model = rf_fit_model(X_train, y_train, rf_model)
    # gb_model = gb_fit_model(X_train, y_train, gb_model)


    # Store model and test set for prediction
    with open(ROOT / 'models/xgb_model.model', 'wb') as fout:
        pickle.dump(xgb_model, fout, PROTOCOL)

    with open(ROOT / 'models/rf_model.model', 'wb') as fout:
        pickle.dump(rf_model, fout, PROTOCOL)

    with open(ROOT / 'models/gb_model.model', 'wb') as fout:
        pickle.dump(gb_model, fout, PROTOCOL)

    X_test.to_csv(ROOT / 'data/processed/app_train_x_test.csv',
        index=False)
    y_test.to_csv(ROOT / 'data/processed/app_train_y_test.csv',
        index=False)



if __name__ == '__main__':
   
    model = ''
    try:
        model=sys.argv[1]
    except:
        pass
    
    if ( (model != 'xgb') and ( model!= 'rf') and (model != 'gb')) :
        raise NameError('unknown model, pass a model in argument among xgb, rf and gb')

    main(model)
