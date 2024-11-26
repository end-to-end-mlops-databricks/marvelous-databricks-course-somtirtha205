# Databricks notebook source
# MAGIC %md
# MAGIC ## Send request to the endpoint from normal and skewed distribution

# COMMAND ----------

# MAGIC %pip install /Volumes/mlops_dev/hotel_reservation/data/hotel_reservation-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import datetime
import itertools
import time

import requests
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

workspace = WorkspaceClient()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

inference_data_normal = spark.table(f"{catalog_name}.{schema_name}.inference_set_normal").toPandas()
inference_data_skewed = spark.table(f"{catalog_name}.{schema_name}.inference_set_skewed").toPandas()

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------

token = workspace.dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# Required columns for inference
required_columns = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "required_car_parking_space",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "type_of_meal_plan",
    "room_type_reserved",
    "market_segment_type",
    "Booking_ID",
]

# Sample records from inference datasets
sampled_normal_records = inference_data_normal[required_columns].to_dict(orient="records")
sampled_skewed_records = inference_data_skewed[required_columns].to_dict(orient="records")
test_set_records = test_set[required_columns].to_dict(orient="records")

# COMMAND ----------


# Two different way to send request to the endpoint
# 1. Using https endpoint
def send_request_https(dataframe_record):
    model_serving_endpoint = f"https://{host}/serving-endpoints/hotel-reservation-model-serving-fe/invocations"
    response = requests.post(
        model_serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [dataframe_record]},
    )
    return response


# 2. Using workspace client
def send_request_workspace(dataframe_record):
    response = workspace.serving_endpoints.query(
        name="hotel-reservation-model-serving-fe", dataframe_records=[dataframe_record]
    )
    return response


# COMMAND ----------

# Loop over test records and send requests for 15 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=15)
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# COMMAND ----------

# Loop over normal records and send requests for 15 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=15)
for index, record in enumerate(itertools.cycle(sampled_normal_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for normal data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# COMMAND ----------

# Loop over skewed records and send requests for 15 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=15)
for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for skewed data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)
