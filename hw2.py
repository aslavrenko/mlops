import mlflow
import os
import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import datetime
exp_name = f'exp_{datetime.datetime.now().strftime("%d%m%y_%H-%M-%S_%f")}'

exp_id = mlflow.create_experiment(name=exp_name)

housing = fetch_california_housing(as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(housing['data'], housing['target'])
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

models = dict(zip(["RandomForest", "LinearRegression", "HistGB"], 
                  [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]))

time_mark = f'{datetime.datetime.now().strftime("%d%m%y_%H-%M-%S_%f")}'


with mlflow.start_run(run_name=f"California_run_{time_mark}", experiment_id = exp_id, description = "parent") as parent_run:
    for model_name in models.keys():
        with mlflow.start_run(run_name=model_name, experiment_id=exp_id, nested=True) as child_run:
            model = models[model_name]
            
            model.fit(pd.DataFrame(X_train), y_train)
        
            prediction = model.predict(X_val)
        
            eval_df = X_val.copy()
            eval_df["target"] = y_val
            eval_df["prediction"] = prediction
        
            #signature = infer_signature(X_test, prediction)
            #model_info = mlflow.sklearn.log_model(model, "logreg", signature=signature)
            mlflow.evaluate(
                #model=model_info.model_uri,
                data=eval_df,
                targets="target",
                predictions="prediction",
                model_type="regressor",
                evaluators=["default"],
            )