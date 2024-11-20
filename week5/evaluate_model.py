"""
This script evaluates and compares a new hotel reservation prediction model against the currently deployed model.
Key functionality:
- Loads test data and performs feature engineering
- Generates predictions using both new and existing models
- Calculates and compares performance metrics (f1 score and ROC)
- Registers the new model if it performs better
- Sets task values for downstream pipeline steps

The evaluation process:
1. Loads models from the serving endpoint
2. Prepares test data with feature engineering
3. Generates predictions from both models
4. Calculates error metrics
5. Makes registration decision based on f1 score comparison
6. Updates pipeline task values with results
"""

from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import argparse
from hotel_reservation.config import ProjectConfig


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--new_model_uri",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()
root_path = args.root_path
new_model_uri = args.new_model_uri
job_run_id = args.job_run_id
git_sha = args.git_sha

config_path = (f"{root_path}/project_config.yml")
config = ProjectConfig.from_yaml(config_path=config_path)

spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define the serving endpoint
serving_endpoint_name = "hotel-reservation-model-serving-fe"
serving_endpoint = workspace.serving_endpoints.get(serving_endpoint_name)
model_name = serving_endpoint.config.served_models[0].model_name
model_version = serving_endpoint.config.served_models[0].model_version
previous_model_uri = f"models:/{model_name}/{model_version}"

# Load test set and create additional features in Spark DataFrame
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# Cancel Percentage in Test set
try:
    test_set["cancel_percentage"] = (
        test_set["no_of_previous_cancellations"]
        / (test_set["no_of_previous_cancellations"] + test_set["no_of_previous_bookings_not_canceled"])
    ) * 100
except ZeroDivisionError:
    test_set["cancel_percentage"] = 0

# Select the necessary columns for prediction and target
X_test_spark = test_set.select(num_features + cat_features + ["cancel_percentage", "Booking_ID"])
y_test_spark = test_set.select("Booking_ID", target)

# Generate predictions from both models
predictions_previous = fe.score_batch(model_uri=previous_model_uri, df=X_test_spark)
predictions_new = fe.score_batch(model_uri=new_model_uri, df=X_test_spark)

predictions_new = predictions_new.withColumnRenamed("prediction", "prediction_new")
predictions_old = predictions_previous.withColumnRenamed("prediction", "prediction_old")
test_set = test_set.select("Booking_ID", "booking_status")

# Join the DataFrames on the 'Booking_ID' column
df = test_set \
    .join(predictions_new, on="Booking_ID") \
    .join(predictions_old, on="Booking_ID")

# Calculate the Area Under ROC Curve for each model
evaluator = BinaryClassificationEvaluator(labelCol="booking_status", predictionCol="prediction_new", metricName="areaUnderROC")
area_roc_new = evaluator.evaluate(df)

evaluator.setPredictionCol("prediction_old")
area_roc_old = evaluator.evaluate(df)

# Compare models based on Area Under ROC
print(f"Area Under ROC for New Model: {area_roc_new}")
print(f"Area Under ROC for Old Model: {area_roc_old}")

#Calculate F1 score for new model

# Calculate true positives, true negatives, false positives, false negatives
tp = predictions_new.filter((col(target) == 1) & (col('prediction_new') == 1)).count()
tn = predictions_new.filter((col(target) == 0) & (col('prediction_new') == 0)).count()
fp = predictions_new.filter((col(target) == 0) & (col('prediction_new') == 1)).count()
fn = predictions_new.filter((col(target) == 1) & (col('prediction_new') == 0)).count()

precision = tp / (tp + fp) if (tp + fp) != 0 else 0

recall = tp / (tp + fn) if (tp + fn) != 0 else 0

f1_measure_new = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print(f"F1 measure of New Model: {f1_measure_new}")

#Calculate F1 score for old model

# Calculate true positives, true negatives, false positives, false negatives
tp = predictions_old.filter((col(target) == 1) & (col('predictions_old') == 1)).count()
tn = predictions_old.filter((col(target) == 0) & (col('predictions_old') == 0)).count()
fp = predictions_old.filter((col(target) == 0) & (col('predictions_old') == 1)).count()
fn = predictions_old.filter((col(target) == 1) & (col('predictions_old') == 0)).count()

precision = tp / (tp + fp) if (tp + fp) != 0 else 0

recall = tp / (tp + fn) if (tp + fn) != 0 else 0

f1_measure_old = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print(f"F1 measure of Old Model: {f1_measure_old}")

if f1_measure_new > f1_measure_old:
    print("New model is better based on f1 score.")
    model_version = mlflow.register_model(
      model_uri=new_model_uri,
      name=f"{catalog_name}.{schema_name}.hotel_reservation_model_fe",
      tags={"git_sha": f"{git_sha}",
            "job_run_id": job_run_id})

    print("New model registered with version:", model_version.version)
    dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)
    dbutils.jobs.taskValues.set(key="model_update", value=1)
else:
    print("Old model is better based on f1 score.")
    dbutils.jobs.taskValues.set(key="model_update", value=0)