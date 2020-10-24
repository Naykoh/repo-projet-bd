import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parents[2]

def get_features(app_train):
    """ return 2 list of the name of categorical and numerical features of the dataset
    """
    num_features=[]
    cat_features=[]
    for i in list(zip(app_train.columns,app_train.dtypes)):
        if (i[1] != 'object') :
            num_features.append(i[0])
        else : 
            cat_features.append(i[0])
    num_features.remove("TARGET")
    return num_features, cat_features

def preprocessor_create(num_features, cat_features):
    """ return a preprocessor that is useful to onehotencode the object features of the dataset
    """
    preprocessor = ColumnTransformer([("numerical", "passthrough", num_features),
    ("categorical", OneHotEncoder(sparse=False, handle_unknown='ignore'),
    cat_features)])
    return preprocessor


def dropna(app_train, num_features, cat_features):
    """replace the NaN of the numerical features by the mean of this specific features, and drop the NaN for the categorical features
    """
    #replace every nan of numerical features by the mean of this feature
    for column in num_features:    
        app_train[column].fillna((app_train[column].mean()),inplace=True)

    app_train.dropna(inplace=True)

def main():
    """retrieve the raw dataset and process it to get rid off the NaN values, processed dataset can be find in the processed directory
    """
    app_train = pd.read_csv(ROOT/"data/raw/application_train.csv",nrows=1000)

    num_features, cat_features = get_features(app_train)
    
    dropna(app_train, num_features, cat_features)

    app_train.to_csv(ROOT / 'data/processed/application_train_processed.csv', index=False)
    




if __name__ == '__main__':
    main()
