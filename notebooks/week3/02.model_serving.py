# Databricks notebook source
# MAGIC %pip install ../hotel_reservation-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig

# COMMAND ----------

workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

# COMMAND ----------

workspace.serving_endpoints.create(
    name="hotel-reservation-model-serving",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.hotel_reservation_model_basic",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=1,
            )
        ],
        # Optional if only 1 entity is served
        # traffic_config=TrafficConfig(
        #    routes=[
        #        Route(served_model_name="hotel_reservation_model-basic",
        #              traffic_percentage=100)
        #    ]
        #    ),
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call the endpoint

# COMMAND ----------

token = workspace.dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create sample request body

# COMMAND ----------

required_columns = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "required_car_parking_space",
    "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
    "type_of_meal_plan",
    "room_type_reserved",
    "market_segment_type",
]

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

"""
Each body should be list of json with columns

[{'no_of_adults': 2,
  'no_of_children': 0,
  'no_of_weekend_nights': 0,
  'no_of_week_nights': 2,
  'required_car_parking_space': 0,
  'lead_time': 72,
  'avg_price_per_room': 85.50,
  'no_of_special_requests': 0,
  'type_of_meal_plan': 'Meal Plan 1',
  'room_type_reserved': 'Room Type 1',
  'market_segment_type': 'Offline'}]

"""

# COMMAND ----------

start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/hotel-reservation-model-serving/invocations"
response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Response text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Test

# COMMAND ----------

# Initialize variables
model_serving_endpoint = f"https://{host}/serving-endpoints/hotel-reservation-model-serving/invocations"

headers = {"Authorization": f"Bearer {token}"}
num_requests = 1000


# Function to make a request and record latency
def send_request():
    random_record = random.choice(dataframe_records)
    start_time = time.time()
    response = requests.post(
        model_serving_endpoint,
        headers=headers,
        json={"dataframe_records": random_record},
    )
    end_time = time.time()
    latency = end_time - start_time
    return response.status_code, latency


total_start_time = time.time()
latencies = []

# Send requests concurrently
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_request) for _ in range(num_requests)]

    for future in as_completed(futures):
        status_code, latency = future.result()
        latencies.append(latency)

total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Calculate the average latency
average_latency = sum(latencies) / len(latencies)

print("\nTotal execution time:", total_execution_time, "seconds")
print("Average latency per request:", average_latency, "seconds")
