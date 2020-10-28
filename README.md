# Home Credit Risk Classification

### Goals 
The goal of this project is to apply some concepts & tools seen in the 3 parts of this course, this
project is organized into 3 parts :
 Part 1 : Building Classical ML projects with respect to basic ML Coding best practices<br>
 Part 2 : Integrate MLFlow to your project<br>
 Part 3 : Integrate ML Interpretability to your project<br>
### DataSet (Finance use case)
DataSet of Home Credit Risk Classification:
https://www.kaggle.com/c/home-credit-default-risk/data<br>
you'll not use all the datasets available on Kaggle, only the main data set :
application_train.csv
application_test.csv


#### Cookiecutter Data Science(http://drivendata.github.io/cookiecutter-data-science/)



The directory structure of the project looks like this: 

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks.
│
│
├── project_report     <- Report of the project
│
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── api_request  <- Scripts to request the API created by mlflow
│       └── api-request.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
```

### Project evaluation


-Report of your project <br>
	│
	└──  project_report.pdf<br>
	
-Project code and resources (notebooks, scripts, conda env, GIT repository) <br>
	│
	└──  notebook : Project Application of BD.ipynb, scripts : src/, conda env : conda.yaml<br>
	
-Project Outputs (predictions on test dataset, MLflow outputs, SHAP Outputs)<br>
	│
	└──  predictions on test dataset : data/result/, MLflow outputs in project_report, SHAP outputs in project_report and notebook<br>
