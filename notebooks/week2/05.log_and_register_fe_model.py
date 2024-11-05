# Databricks notebook source
# MAGIC %pip install ../hotel_reservation-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import matplotlib.pyplot as plt
import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_reservation.config import ProjectConfig

# COMMAND ----------

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

# Load configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name
id = config.id

# COMMAND ----------

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
function_name = f"{catalog_name}.{schema_name}.booking_cancelled_percentage"

# COMMAND ----------

# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# COMMAND ----------

# Create or replace the hotel_features table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.hotel_features
(Booking_ID STRING NOT NULL,
 lead_time INT,
 avg_price_per_room DOUBLE,
 no_of_special_requests INT);
""")

spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.hotel_features "
    "ADD CONSTRAINT hotel_booking_pk PRIMARY KEY(Booking_ID);"
)

spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.hotel_features " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)

# Insert data into the feature table from both train and test sets
spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.hotel_features "
    f"SELECT Booking_ID, lead_time, avg_price_per_room, no_of_special_requests FROM {catalog_name}.{schema_name}.train_set"
)

spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.hotel_features "
    f"SELECT Booking_ID, lead_time, avg_price_per_room, no_of_special_requests FROM {catalog_name}.{schema_name}.test_set"
)

# COMMAND ----------

# Define a function to calculate the no of visits per month
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(previous_cancel INT, previous_booking INT)
RETURNS DOUBLE
LANGUAGE PYTHON AS
$$
try:
    return (previous_cancel/(previous_cancel + previous_booking)) * 100
except ZeroDivisionException:
    # in case of 0, we can return 0.
    return 0
$$
""")

# COMMAND ----------

# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop(
    "lead_time", "avg_price_per_room", "no_of_special_requests"
)
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------

# Load feature-engineered DataFrame
model_feature_lookups = [
    FeatureLookup(
        table_name=feature_table_name,
        feature_names=["lead_time", "avg_price_per_room", "no_of_special_requests"],
        lookup_key=id,
    ),
    FeatureFunction(
        udf_name=function_name,
        output_name="cancel_percentage",
        input_bindings={
            "previous_cancel": "no_of_previous_cancellations",
            "previous_booking": "no_of_previous_bookings_not_canceled",
        },
    ),
]

training_set = fe.create_training_set(
    df=train_set, feature_lookups=model_feature_lookups, label=target, exclude_columns=["update_timestamp_utc"]
)

training_df = training_set.load_df().toPandas()
training_df.info()

# COMMAND ----------

# Split features and target
X_train = training_df[num_features + cat_features + ["cancel_percentage"]]
y_train = training_df[target]
X_test = test_set[num_features + cat_features + ["cancel_percentage"]]
y_test = test_set[target]

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

# COMMAND ----------

model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(**parameters))])

# COMMAND ----------

# Set and start MLflow experiment
mlflow.set_experiment(experiment_name="/Shared/hotel-reservation-fe")
git_sha = "3055af355f360ba5784ae7037f3260a70331f702"

with mlflow.start_run(tags={"branch": "week2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate and print metrics
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot()
    plt.savefig("confusion-matrix.png")
    plt.show()

    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix: {cm}")

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_params(parameters)
    mlflow.log_metric("f1 score", f1)
    mlflow.log_artifact("confusion-matrix.png")
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=model,
        flavor=mlflow.sklearn,
        artifact_path="RandomForest",
        training_set=training_set,
        signature=signature,
    )

mlflow.register_model(
    model_uri=f"runs:/{run_id}/RandomForest",
    name=f"{catalog_name}.{schema_name}.hotel_reservation_model_fe",
)

# COMMAND ----------
