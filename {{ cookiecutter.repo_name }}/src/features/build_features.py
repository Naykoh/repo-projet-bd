import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def get_features(app_train):
    num_features=[]
    cat_features=[]
    for i in list(zip(app_train.columns,app_train.dtypes)):
        if (i[1] != 'object') :
            num_features.append(i[0])
        else : 
            cat_features.append(i[0])
    num_features.remove("TARGET")
    return num_features, cat_features

def dropna(app_train, num_features, cat_features):
    #replace every nan of numerical features by the mean of this feature
    for column in num_features:    
        app_train[column].fillna((app_train[column].mean()),inplace=True)

    app_train.dropna(inplace=True)

def main():
    # Some such as default would be binary features, but since
    # they have a third class "unknown" we'll process them as non binary categorical
    
    app_train = pd.read_csv(ROOT/"data/raw/application_train.csv",nrows=1000)

    num_features, cat_features = get_features(app_train)
    
    dropna(app_train, num_features, cat_features)

    app_train.to_csv(ROOT / 'data/processed/application_train_processed.csv', index=False)
    




if __name__ == '__main__':
    main()
