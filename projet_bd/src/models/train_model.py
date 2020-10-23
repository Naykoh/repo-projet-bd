import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# from cookiecutter.repo_name.src.features import build_features

import importlib.util
spec = importlib.util.spec_from_file_location("build_features", "./src/features/build_features.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# Algorithms, from the easiest to the hardest to intepret.r
from xgboost.sklearn import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]


if '{{ cookiecutter.python_interpreter }}' == 'python3':
    PROTOCOL = pickle.DEFAULT_PROTOCOL
else:
    PROTOCOL = 2



# def get_features(app_train):
#     num_features=[]
#     cat_features=[]
#     for i in list(zip(app_train.columns,app_train.dtypes)):
#         if (i[1] != 'object') :
#             num_features.append(i[0])
#         else :
#             cat_features.append(i[0])
#     num_features.remove("TARGET")
#     return num_features, cat_features

def fetch_processed(data_path):
    """
    fetch the data that was processed in make data
    """
    app_train = pd.read_csv(ROOT / data_path)
    y = app_train["TARGET"]
    X = app_train.drop("TARGET", axis=1)

    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.3, random_state=42)

    return X_train, X_test, y_train, y_test


def preprocessor_create(num_features, cat_features):
    preprocessor = ColumnTransformer([("numerical", "passthrough", num_features),
    ("categorical", OneHotEncoder(sparse=False, handle_unknown='ignore'),
    cat_features)])
    return preprocessor

def model_definition(num_features, cat_features):
    app_train = pd.read_csv(ROOT/"data/processed/application_train_processed.csv",nrows=1000)
    y = app_train["TARGET"]
    preprocessor = preprocessor_create(num_features, cat_features)
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

    return xgb_model,rf_model, gb_model


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
    gs = GridSearchCV(gb_model, {"model__max_depth": [10, 15],
    "model__min_samples_split": [5, 10]},
    n_jobs=-1, cv=5, scoring="accuracy")
    gs.fit(X_train, y_train)

    print(gs.best_params_)
    print(gs.best_score_)

    gb_model.set_params(**gs.best_params_)
    gb_model.fit(X_train, y_train)
    return gb_model




def main():
    print(ROOT)
    """ Trains the model on the retrieved data write it back to file
    """
    app_train = pd.read_csv(ROOT/"data/processed/application_train_processed.csv")
    num_features, cat_features = foo.get_features(app_train)
    X_train, X_test, y_train, y_test = fetch_processed('data/processed/application_train_processed.csv')
    X_train.isnull().sum().tolist()

    # Train the model
    xgb_model, rf_model, gb_model = model_definition(num_features, cat_features)

    xgb_model = xgb_fit_model(X_train, y_train, xgb_model)
    rf_model = rf_fit_model(X_train, y_train, rf_model)
    gb_model = gb_fit_model(X_train, y_train, gb_model)


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
    main()
