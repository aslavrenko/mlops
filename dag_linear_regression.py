import io
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle

import datetime
from datetime import timedelta
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
    dag_id="hw1_LinReg_dag",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS,
)


def init(**context) -> None:
    context['ti'].xcom_push(key='model_name', value='LinearRegression')
    context['ti'].xcom_push(key='start_time', value=str(datetime.datetime.now()))
    _LOG.info("Train pipeline started.")



def get_data_from_sklearn(**context) -> None:
    context['ti'].xcom_push(key='load_start_time', value=str(datetime.datetime.now()))
    housing = fetch_california_housing(as_frame=True)
    data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)

    s3_hook = S3Hook("s3_connector")
    filebuffer = io.BytesIO()
    data.to_pickle(filebuffer)
    filebuffer.seek(0)

    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key="2024/datasets/california_housing.pkl",
        bucket_name=BUCKET,
        replace=True,
    )

    context['ti'].xcom_push(key='load_end_time', value=str(datetime.datetime.now()))
    context['ti'].xcom_push(key='dataset_size', value=data.shape)
    
    _LOG.info("Data downloaded.")


def prepare_data(**context) -> None:

    context['ti'].xcom_push(key='preprocessing_start_time', value=str(datetime.datetime.now()))

    s3_hook = S3Hook("s3_connector")
    file = s3_hook.download_file(key="2024/datasets/california_housing.pkl", bucket_name=BUCKET)
    data = pd.read_pickle(file)

    X, y = data[FEATURES], data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        [X_train_fitted, X_test_fitted, y_train, y_test],
    ):
        filebuffer = io.BytesIO()
        pickle.dump(data, filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"2024/datasets/{name}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )

    context['ti'].xcom_push(key='preprocessing_end_time', value=str(datetime.datetime.now()))
    context['ti'].xcom_push(key='processed_dataset_size', value=X_train.shape)

    _LOG.info("Data prepared.")


def train_model(**context) -> None:

    s3_hook = S3Hook("s3_connector")
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(
            key=f"2024/datasets/{name}.pkl",
            bucket_name=BUCKET,
        )
        data[name] = pd.read_pickle(file)

    model = LinearRegression()
    model.fit(data["X_train"], data["y_train"])
    prediction = model.predict(data["X_test"])

    
    r2 = r2_score(data["y_test"], prediction)
    rmse = mean_squared_error(data["y_test"], prediction) ** 0.5
    mae = median_absolute_error(data["y_test"], prediction)

    
    context['ti'].xcom_push(key='training_end_time', value=str(datetime.datetime.now()))
    context['ti'].xcom_push(key='mean_squared_error', value=rmse)
    context['ti'].xcom_push(key='r2_score', value=r2)
    context['ti'].xcom_push(key='median_absolute_error', value=mae)

    
    _LOG.info("Model trained.")



def save_results(**context) -> None:
    s3_hook = S3Hook("s3_connector")
    file = s3_hook.download_file(key="metrics.json", bucket_name=Variable.get("S3_BUCKET"))
    data = pd.read_json(file)
    
    metrics = {}
    metrics['model_name'] = context['ti'].xcom_pull(task_ids='init', key='model_name')
    metrics['start_time'] = context['ti'].xcom_pull(task_ids='init', key='start_time')
    metrics['load_start_time'] = context['ti'].xcom_pull(task_ids='download_data', key='load_start_time')
    metrics['load_end_time'] = context['ti'].xcom_pull(task_ids='download_data', key='load_end_time')
    metrics['dataset_size'] = context['ti'].xcom_pull(task_ids='download_data', key='dataset_size')
    metrics['preprocessing_start_time'] = context['ti'].xcom_pull(task_ids='data_preparation', key='preprocessing_start_time')
    metrics['preprocessing_end_time'] = context['ti'].xcom_pull(task_ids='data_preparation', key='preprocessing_end_time')
    metrics['processed_dataset_size'] = context['ti'].xcom_pull(task_ids='data_preparation', key='processed_dataset_size')
    metrics['training_start_time'] = context['ti'].xcom_pull(task_ids='train_model', key='training_start_time')
    metrics['training_end_time'] = context['ti'].xcom_pull(task_ids='train_model', key='training_end_time')
    metrics['mean_squared_error'] = context['ti'].xcom_pull(task_ids='train_model', key='mean_squared_error')
    metrics['r2_score'] = context['ti'].xcom_pull(task_ids='train_model', key='r2_score')
    metrics['median_absolute_error'] = context['ti'].xcom_pull(task_ids='train_model', key='median_absolute_error')

    data[f'{metrics["model_name"]}_{str(datetime.datetime.now())}'] = metrics

    
    filebuffer = io.BytesIO()
    data.to_json(filebuffer)
    filebuffer.seek(0)
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"metrics.json",
        bucket_name=BUCKET,
        replace=True,
    )

    print(metrics)
    _LOG.info(metrics)

    
    print("Success")
    _LOG.info("Success.")


task_init = PythonOperator(task_id="init", python_callable=init, dag=dag, provide_context=True)

task_download_data = PythonOperator(
    task_id="download_data", python_callable=get_data_from_sklearn, dag=dag, provide_context=True
)

task_prepare_data = PythonOperator(
    task_id="data_preparation", python_callable=prepare_data, dag=dag, provide_context=True
)

task_train_model = PythonOperator(
    task_id="train_model", python_callable=train_model, dag=dag, provide_context=True
)

task_save_results = PythonOperator(
    task_id="save_results", python_callable=save_results, dag=dag, provide_context=True
)

task_init >> task_download_data >> task_prepare_data >> task_train_model >> task_save_results