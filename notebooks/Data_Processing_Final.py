# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_dev/hotel_reservation/data/hotel_reservation-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import yaml

from hotel_reservation.data_processor import DataProcessor

# COMMAND ----------

# Load configuration
with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# Initialize DataProcessor
data_processor = DataProcessor("/Volumes/mlops_dev/hotel_reservation/data/Data.csv", config)

# COMMAND ----------

# Preprocess the data
data_processor.preprocess_data()
