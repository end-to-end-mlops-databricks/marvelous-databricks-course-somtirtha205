# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_dev/hotel_reservation/data/hotel_reservation-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

workspace = WorkspaceClient()

# COMMAND ----------

monitoring_table = f"{catalog_name}.{schema_name}.model_monitoring"

workspace.quality_monitors.create(
    table_name=monitoring_table,
    assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
    output_schema_name=f"{catalog_name}.{schema_name}",
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        prediction_col="prediction",
        timestamp_col="timestamp",
        granularities=["30 minutes"],
        model_id_col="model_name",
        label_col="booking_status",
    ),
)

spark.sql(f"ALTER TABLE {monitoring_table} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# COMMAND ----------

## How to delete a monitor
# workspace.quality_monitors.delete(
#     table_name="mlops_test.house_prices.model_monitoring"
# )
