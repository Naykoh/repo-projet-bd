"""
    train_model.py
    ----------------

    :param model: 'xgb', 'rf' or 'gb'
    :type model: String

    This package contains function to train 3 differents models, store them locally and on mlflow
"""
import pickle
import sys
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from src.features.build_features import get_features, preprocessor_create
from varname import nameof
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

        :param data_path: location of the dataset
        :type data_path: String

        :return: training and test sets
        :rtype: 2 DataFrame, 2 List
    """
    app_train = pd.read_csv(ROOT / data_path)
    y = app_train["TARGET"]
    X = app_train.drop("TARGET", axis=1)

    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.3, random_state=42)

    return X_train, X_test, y_train, y_test




def model_definition(preprocessor, app_train):
    """
         define the 3 ML model, XGBoost Classifier, Random Forest Classifier, Gradient Boosting

         :param preprocessor: ColumnTransformer to handle onehotencoding
         :type preprocessor: ColumnTransformer

         :param app_train: need the y.mean() to add a scale_pos_weight to make it balanced

         :return: 3 models of ML
         :rtype: XGBClassifier, RandomForestClassifier, GradientBoostingClassifier
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

        :param X_train: data used to train the model
        :type X_train: DataFrame

        :param y_train: target feature
        :type y_train: List

        :param xgb_model: XGBClassifier model
        :type xgb_model: XGBClassifier

        :return: XGBClassifier fitted with gridsearchCV
        :rtype: XGBClassifier
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

        :param X_train: data used to train the model
        :type X_train: DataFrame

        :param y_train: target feature
        :type y_train: List

        :param rf_model: RandomForestClassifier model
        :type rf_model: RandomForestClassifier
        

        :return: RandomForestClassifier fitted with gridsearchCV
        :rtype: RandomForestClassifier

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

        :param X_train: data used to train the model
        :type X_train: DataFrame

        :param y_train: target feature
        :type y_train: List

        :param gb_model: GradientBoostingClassifier model
        :type gb_model: GradientBoostingClassifier

        :return: GradientBoostingClassifier fitted with gridsearchCV
        :rtype: GradientBoostingClassifier
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

def eval_metrics(actual, pred,loaded_model, X_test):
    """ 
        compare prediciton and actual value with different metrics to check the accuracy of the model

        :param actual: Actual value of the target feature of test set
        :type actual: List

        :param pred: Predictions of the target feature using the test set
        :type pred: List

        :param loaded_model: model selected to check the metrics
        :param X_test: useful to calculate auc score

        :return: auc, recall, precision, f1, matrix
    """


    auc = roc_auc_score(actual.astype(int), loaded_model.predict_proba(X_test)[:, 1])

    recall=recall_score(actual,pred)
    precision=precision_score(actual,pred)
    f1=f1_score(actual,pred)

    matrix = confusion_matrix(actual,pred)


    return auc, recall, precision, f1, matrix


def mlflow_train(model,X_train,y_train,X_test,y_test,fit_function):
    """ 
        Fit the model and evaluate the accuracy of the model, model and metrics are saved on mlflow

        :param model: Model selected

        :type X_train: DataFrame

        :type y_train: List

        :type X_test: DataFrame

        :type y_test: List

        :param fit_function: fit function depending on the selected model

    """

    with mlflow.start_run():

        #Train the model
        model_fitted = fit_function(X_train, y_train, model)

        y_pred = model_fitted.predict(X_test)

        (auc, recall, precision, f1, matrix) = eval_metrics(y_test, y_pred,model_fitted, X_test)


        print("  auc: %s" % auc)
        print("  recall: %s" % recall)
        print("  precision: %s" % precision)
        print("  F1 score: %s" % f1)
        print("  matrix: %s" % matrix)


        mlflow.log_param("params", model_fitted.get_params()['model'])
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("F1 score", f1)


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
    """ 
        Trains the model and test their accuracy on the retrieved data write it back to file and store the trained model

        :param model: model selected
    """
    #Define the 3 model
    app_train = pd.read_csv(ROOT/"data/processed/app_train_processed.csv")
    num_features, cat_features = get_features(app_train)
    preprocessor = preprocessor_create(num_features, cat_features)
    xgb_model, rf_model, gb_model = model_definition(preprocessor, app_train)


    #fetch the processed dataset with training and test set
    X_train, X_test, y_train, y_test = fetch_processed('data/processed/app_train_processed.csv')

    if (model == 'xgb'):
        mlflow_train(xgb_model,X_train,y_train,X_test,y_test,xgb_fit_model)
    elif (model == 'rf'):
        mlflow_train(rf_model,X_train,y_train,X_test,y_test,rf_fit_model)
    else :
        mlflow_train(gb_model,X_train,y_train,X_test,y_test,gb_fit_model)



    # Store model and test set for prediction
    if (model == 'xgb'):
        with open(ROOT / 'models/xgb_model.model', 'wb') as fout:
            pickle.dump(xgb_model, fout, PROTOCOL)  
    elif (model == 'rf'):   
        with open(ROOT / 'models/rf_model.model', 'wb') as fout:
            pickle.dump(rf_model, fout, PROTOCOL)
    else :
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
