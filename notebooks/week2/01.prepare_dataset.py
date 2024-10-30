# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_dev/hotel_reservation/data/hotel_reservation-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import yaml
from hotel_reservation.data_processor import DataProcessor
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Load configuration
with open("../../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

# COMMAND ----------

# Load the data
data_processor = DataProcessor("/Volumes/mlops_dev/hotel_reservation/data/Data.csv", config)

# COMMAND ----------

# Extract configuration details
num_features = config["num_features"]
cat_features = config["cat_features"]
target = config["target"]
parameters = config["parameters"]
catalog_name = config["catalog_name"]
schema_name = config["schema_name"]

# COMMAND ----------

# Preprocess the data
data_processor.preprocess_data()

# COMMAND ----------

# Split the data
train_set, test_set = data_processor.split_data()

# COMMAND ----------

#Save to Catalog
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
