import io
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import mlflow

import datetime
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = Variable.get("S3_BUCKET")
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"

DEFAULT_ARGS = {
    "owner": "Anton Lavrenko",
    "retry": 3,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    dag_id="MLOps_project_2",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS,
)


def init(**context) -> None:
    _LOG.info("Train pipeline started.")



def get_data_from_sklearn(**context) -> None:
    housing = fetch_california_housing(as_frame=True)
    data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)

    s3_hook = S3Hook("s3_connector")
    filebuffer = io.BytesIO()
    data.to_pickle(filebuffer)
    filebuffer.seek(0)

    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key="project/datasets/california_housing.pkl",
        bucket_name=BUCKET,
        replace=True,
    )
    
    _LOG.info("Data downloaded.")


def train_model(**context) -> None:
    s3_hook = S3Hook("s3_connector")
    file = s3_hook.download_file(key="project/datasets/california_housing.pkl", bucket_name=BUCKET)
    data = pd.read_pickle(file)

    X, y = data[FEATURES], data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    models = dict(zip(["RandomForest", "LinearRegression", "HistGB"], 
                  [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]))

    exp_name = f'exp_{datetime.datetime.now().strftime("%d%m%y_%H-%M-%S_%f")}'
    exp_id = mlflow.create_experiment(name=exp_name)

    time_mark = f'{datetime.datetime.now().strftime("%d%m%y_%H-%M-%S_%f")}'

    with mlflow.start_run(run_name=f"California_run_{time_mark}", experiment_id = exp_id, description = "parent") as parent_run:
        for model_name in models.keys():
            # Запустим child run на каждую модель.
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
            
    _LOG.info("Model trained.")
    


def finish(**context) -> None:
    print("Success")
    _LOG.info("Success.")


task_init = PythonOperator(task_id="init", python_callable=init, dag=dag, provide_context=True)

task_download_data = PythonOperator(
    task_id="download_data", python_callable=get_data_from_sklearn, dag=dag, provide_context=True
)


task_train_model = PythonOperator(
    task_id="train_model", python_callable=train_model, dag=dag, provide_context=True
)

finish = PythonOperator(
    task_id="finish", python_callable=finish, dag=dag, provide_context=True
)

task_init >> task_download_data >> task_train_model >> finish