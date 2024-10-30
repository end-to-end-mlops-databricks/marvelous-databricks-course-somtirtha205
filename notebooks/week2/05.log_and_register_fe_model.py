# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_dev/hotel_reservation/data/hotel_reservation-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import yaml
from databricks import feature_engineering
from pyspark.sql import SparkSession
import mlflow
from pyspark.sql import functions as F
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay 
from mlflow.models import infer_signature
from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup
import matplotlib.pyplot as plt

# COMMAND ----------

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

# Load configuration
with open("../../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

# COMMAND ----------

# Extract configuration details
num_features = config["num_features"]
cat_features = config["cat_features"]
target = config["target"]
parameters = config["parameters"]
catalog_name = config["catalog_name"]
schema_name = config["schema_name"]
id = config["id"]

# COMMAND ----------

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"

# COMMAND ----------

# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------

# Create or replace the hotel_features table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.hotel_features
(Booking_ID STRING NOT NULL,
 no_of_adults INT,
 no_of_children INT,
 no_of_weekend_nights INT,
 no_of_week_nights INT,
 type_of_meal_plan STRING,
 required_car_parking_space INT,
 room_type_reserved STRING,
 lead_time INT,
 arrival_year INT,
 arrival_month INT,
 arrival_date INT,
 market_segment_type STRING,
 repeated_guest INT,
 no_of_previous_cancellations INT,
 no_of_previous_bookings_not_canceled INT,
 avg_price_per_room DOUBLE,
 no_of_special_requests INT);
""")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.hotel_features "
          "ADD CONSTRAINT hotel_booking_pk PRIMARY KEY(Booking_ID);")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.hotel_features "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# Insert data into the feature table from both train and test sets
spark.sql(f"INSERT INTO {catalog_name}.{schema_name}.hotel_features "
          f"SELECT Booking_ID, no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, type_of_meal_plan, required_car_parking_space, room_type_reserved, lead_time, arrival_year, arrival_month, arrival_date, market_segment_type, repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled, avg_price_per_room, no_of_special_requests FROM {catalog_name}.{schema_name}.train_set")

spark.sql(f"INSERT INTO {catalog_name}.{schema_name}.hotel_features "
          f"SELECT Booking_ID, no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, type_of_meal_plan, required_car_parking_space, room_type_reserved, lead_time, arrival_year, arrival_month, arrival_date, market_segment_type, repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled, avg_price_per_room, no_of_special_requests FROM {catalog_name}.{schema_name}.test_set")

# COMMAND ----------

# Load feature-engineered DataFrame
model_feature_lookups = [FeatureLookup(table_name=feature_table_name, lookup_key=id)]
training_set = fe.create_training_set(df=train_set[[id, target]], feature_lookups=model_feature_lookups, label=target, exclude_columns=[id]) 

training_df = training_set.load_df().toPandas()
training_df.info()

# COMMAND ----------

# Split features and target
X_train = training_df[num_features + cat_features]
y_train = training_df[target]
X_test = test_set[num_features + cat_features]
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

model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**parameters))
    ])

# COMMAND ----------

# Set and start MLflow experiment
mlflow.set_experiment(experiment_name="/Shared/hotel-reservation-fe")
git_sha = "3055af355f360ba5784ae7037f3260a70331f702"

with mlflow.start_run(tags={"branch": "week2",
                            "git_sha": f"{git_sha}"}) as run:
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
    model_uri=f'runs:/{run_id}/RandomForestClassifier-fe',
    name=f"{catalog_name}.{schema_name}.hotel_reservation_model_fe")
