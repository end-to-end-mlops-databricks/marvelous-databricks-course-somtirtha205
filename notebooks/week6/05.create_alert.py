# Databricks notebook source
# MAGIC %md
# MAGIC ### Create a query that checks the percentage of F1 Score being lower than 0.8

# COMMAND ----------

# MAGIC %pip install /Volumes/mlops_dev/hotel_reservation/data/hotel_reservation-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

w = WorkspaceClient()

srcs = w.data_sources.list()

# COMMAND ----------

alert_query = """
SELECT
 (COUNT(CASE WHEN f1_score.weighted * 100 < 80 THEN 1 END) * 100.0 / COUNT(CASE WHEN f1_score.weighted IS NOT NULL AND NOT isnan(f1_score.weighted) THEN 1 END)) AS percentage_lower_than_80
FROM mlops_dev.hotel_reservation.model_monitoring_profile_metrics"""


query = w.queries.create(query=sql.CreateQueryRequestQuery(
        display_name=f"hotel-reservation-alert-query-{time.time_ns()}",
        warehouse_id=srcs[0].warehouse_id,
        description="Alert on hotel reservation model F1 Score",
        query_text=alert_query,
    )
)

alert = w.alerts.create(alert=sql.CreateAlertRequestAlert(
        condition=sql.AlertCondition(
            operand=sql.AlertConditionOperand(column=sql.AlertOperandColumn(name="percentage_lower_than_80")),
            op=sql.AlertOperator.GREATER_THAN,
            threshold=sql.AlertConditionThreshold(value=sql.AlertOperandValue(double_value=45)),
        ),
        display_name=f"hotel-reservation-f1-score-alert-{time.time_ns()}",
        query_id=query.id,
    )
)

# COMMAND ----------

# cleanup
w.queries.delete(id=query.id)
w.alerts.delete(id=alert.id)
