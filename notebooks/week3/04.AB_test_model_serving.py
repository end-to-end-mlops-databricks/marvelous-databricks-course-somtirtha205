# Databricks notebook source
# MAGIC %pip install ../hotel_reservation-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from sklearn.ensemble import RandomForestClassifier
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import hashlib
import requests
from hotel_reservation.config import ProjectConfig

# COMMAND ----------

# Set up MLflow for tracking and model registry
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Initialize the MLflow client for model management
client = MlflowClient()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------

# Extract key configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name
ab_test_params = config.ab_test
parameters = config.parameters

# COMMAND ----------

# Set up specific parameters for model A and model B as part of the A/B test
parameters_a = {
    "random_state": ab_test_params["random_state"],
    "n_estimators": ab_test_params["n_estimators"],
    "max_depth": ab_test_params["max_depth_a"],
}

parameters_b = {
    "random_state": ab_test_params["random_state"],
    "n_estimators": ab_test_params["n_estimators"],
    "max_depth": ab_test_params["max_depth_b"],
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Prepare Training and Testing Datasets

# COMMAND ----------

# Initialize a Databricks session for Spark operations
spark = SparkSession.builder.getOrCreate()

# Load the training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = train_set_spark.toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Define features and target variables
X_train = train_set[num_features + cat_features]
y_train = train_set[target]
X_test = test_set[num_features + cat_features]
y_test = test_set[target]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Model A and Log with MLflow

# COMMAND ----------

# Create preprocessing steps for numeric and categorical data
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

# Build a pipeline combining preprocessing and model training steps
model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(**parameters_a))])

# Set the MLflow experiment to track this A/B testing project
mlflow.set_experiment(experiment_name="/Shared/hotel-reservation-ab")
model_name = f"{catalog_name}.{schema_name}.hotel-reservation_model_ab"

# Git commit hash for tracking model version
git_sha = "3055af355f360ba5784ae7037f3260a70331f702"

# Start MLflow run to track training of Model A
with mlflow.start_run(tags={"model_class": "A", "git_sha": git_sha}) as run:
    run_id = run.info.run_id

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot()
    plt.savefig("confusion-matrix.png")
    plt.show()

    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix: {cm}")

    # Log model parameters, metrics, and other artifacts in MLflow
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_params(parameters)
    mlflow.log_metric("f1 score", f1)
    mlflow.log_artifact("confusion-matrix.png")
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log the input dataset for tracking reproducibility
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    # Log the pipeline model in MLflow with a unique artifact path
    mlflow.sklearn.log_model(sk_model=model, artifact_path="RandomForest", signature=signature)

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/RandomForest", name=model_name, tags={"git_sha": f"{git_sha}"}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register Model A and Assign Alias

# COMMAND ----------

model_version_alias = "model_A"

client.set_registered_model_alias(model_name, model_version_alias, f"{model_version.version}")
model_uri = f"models:/{model_name}@{model_version_alias}"
model_A = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register Model B and Assign Alias

# COMMAND ----------

# Assign alias for Model B
model_version_alias = "model_B"

client.set_registered_model_alias(model_name, model_version_alias, f"{model_version.version}")
model_uri = f"models:/{model_name}@{model_version_alias}"
model_B = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Custom A/B Test Model

# COMMAND ----------

class HousePriceModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, models):
        self.models = models
        self.model_a = models[0]
        self.model_b = models[1]

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            booking_id = str(model_input["Booking_ID"].values[0])
            hashed_id = hashlib.md5(booking_id.encode(encoding="UTF-8")).hexdigest()
            # convert a hexadecimal (base-16) string into an integer
            if int(hashed_id, 16) % 2:
                predictions = self.model_a.predict(model_input.drop(["Booking_ID"], axis=1))
                return {"Prediction": predictions[0], "model": "Model A"}
            else:
                predictions = self.model_b.predict(model_input.drop(["Booking_ID"], axis=1))
                return {"Prediction": predictions[0], "model": "Model B"}
        else:
            raise ValueError("Input must be a pandas DataFrame.")

# COMMAND ----------

X_train = train_set[num_features + cat_features + ["Booking_ID"]]
X_test = test_set[num_features + cat_features + ["Booking_ID"]]

# COMMAND ----------

models = [model_A, model_B]
wrapped_model = HousePriceModelWrapper(models)  # we pass the loaded models to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(
    context=None,
    model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/hotel-reservation-ab-testing")
model_name = f"{catalog_name}.{schema_name}.hotel-reservation_model_pyfunc_ab_test"

with mlflow.start_run() as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train,
                                model_output={"Prediction": 1234.5,
                                              "model": "Model B"})
    dataset = mlflow.data.from_spark(train_set_spark,
                                     table_name=f"{catalog_name}.{schema_name}.train_set",
                                     version="0")
    mlflow.log_input(dataset, context="training")
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-hotel-reservation-model-ab",
        signature=signature
    )
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-hotel-reservation-model-ab",
    name=model_name,
    tags={"git_sha": f"{git_sha}"}
)

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version.version}")

# Run prediction
predictions = model.predict(X_test.iloc[0:1])

# Display predictions
predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create serving endpoint

# COMMAND ----------

workspace = WorkspaceClient()

workspace.serving_endpoints.create(
    name="hotel-reservation-model-serving-ab-test",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.hotel-reservation_model_pyfunc_ab_test",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=1,
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call the endpoint

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

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
    "Booking_ID"
]

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

start_time = time.time()

model_serving_endpoint = (
    f"https://{host}/serving-endpoints/hotel-reservation-model-serving-ab-test/invocations"
)

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
