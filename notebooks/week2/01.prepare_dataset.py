# Databricks notebook source
# MAGIC %pip install ../hotel_reservation-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig
from hotel_reservation.data_processor import DataProcessor

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Load configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------

# Load the data
data_processor = DataProcessor(config)

# COMMAND ----------

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Preprocess the data
data_processor.preprocess_data("/Volumes/mlops_dev/hotel_reservation/data/Data.csv", spark)

# COMMAND ----------

# Split the data
train_set, test_set = data_processor.split_data()

# COMMAND ----------

# Save to Catalog
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
