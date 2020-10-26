"""
    build_features.py
    ----------------

    This package contains function to build features before doing machine learning on it
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parents[2]

def get_features(app_train):
    """ 
        return 2 list of the name of categorical and numerical features of the dataset

        :param app_train: dataframe containing training data
        :type app_train: DataFrame

        :return: Two lists of categorical and numerical features of the dataset
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
    """ 
        return a preprocessor that is useful to onehotencode the object features of the dataset

        :param num_features: list of the name of numerical features in the dataset
        :type num_features: List

        :param cat_features: list of the name of categorical features in the dataset
        :type cat_features: List

        :return: ColumnTransformer used for the onehotencoding of the categorical features
    """
    preprocessor = ColumnTransformer([("numerical", "passthrough", num_features),
    ("categorical", OneHotEncoder(sparse=False, handle_unknown='ignore'),
    cat_features)])
    return preprocessor


def dropna(df, num_features, cat_features):
    """
        replace the NaN of the numerical features by the mean of this specific features, and drop the NaN for the categorical features

        :param df: dataframe, application_train/test
        :type df: DataFrame

        :param num_features: list of the name of numerical features in the dataset
        :type num_features: List

        :param cat_features: list of the name of categorical features in the dataset
        :type cat_features: List

        :return: DataFrame without NaN values
    """
    #replace every nan of numerical features by the mean of this feature
    for column in num_features:    
        df[column].fillna((df[column].mean()),inplace=True)

    df.dropna(inplace=True)
    return df

def feature_selection(df):
    """  
        selection of feature, intersection of 6 ways of feature selection https://www.kaggle.com/sz8416/6-ways-for-feature-selection

        :param df: datafarme to select features from
        :type df: DataFrame

        :return: DataFrame with relevant features

    """
    if ('TARGET' in df.columns.tolist()):
        feature_selected =['TARGET','AMT_CREDIT','AMT_GOODS_PRICE','APARTMENTS_MODE','CODE_GENDER','DAYS_BIRTH','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE','DEF_30_CNT_SOCIAL_CIRCLE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','FLAG_DOCUMENT_3','FLAG_OWN_CAR','FLOORSMAX_MODE','LIVE_CITY_NOT_WORK_CITY','NAME_EDUCATION_TYPE','NONLIVINGAREA_MODE','OBS_30_CNT_SOCIAL_CIRCLE','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY']
    else:
        feature_selected =['AMT_CREDIT','AMT_GOODS_PRICE','APARTMENTS_MODE','CODE_GENDER','DAYS_BIRTH','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE','DEF_30_CNT_SOCIAL_CIRCLE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','FLAG_DOCUMENT_3','FLAG_OWN_CAR','FLOORSMAX_MODE','LIVE_CITY_NOT_WORK_CITY','NAME_EDUCATION_TYPE','NONLIVINGAREA_MODE','OBS_30_CNT_SOCIAL_CIRCLE','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY']
    df=df[feature_selected]
    return df

def main():
    """
        retrieve the raw dataset and process it(feature selection and handle Nan), processed dataset can be find at data/processed
    """
    app_train = pd.read_csv(ROOT/"data/processed/app_train.csv")
    app_test = pd.read_csv(ROOT/"data/processed/app_test.csv")


    #selection of interesting features
    app_train = feature_selection(app_train)
    app_test = feature_selection(app_test)

    #differentiate categorical and numerical features
    num_features, cat_features = get_features(app_train)

    #handle NaN values
    dropna(app_train, num_features, cat_features)
    dropna(app_test, num_features, cat_features)

    app_train.to_csv(ROOT / 'data/processed/app_train_processed.csv', index=False)
    app_test.to_csv(ROOT / 'data/processed/app_test_processed.csv', index=False)
    




if __name__ == '__main__':
    main()
