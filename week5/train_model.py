"""
This script trains a Random Forest model for hotel reservation prediction with feature engineering.
Key functionality:
- Loads training and test data from Databricks tables
- Performs feature engineering using Databricks Feature Store
- Creates a pipeline with preprocessing and Random Forest classifier
- Tracks the experiment using MLflow
- Logs model metrics, parameters and artifacts
- Handles feature lookups and custom feature functions
- Outputs model URI for downstream tasks

The model uses both numerical and categorical features, including a custom hotel cancel percentage feature.
"""

import argparse

import matplotlib.pyplot as plt
import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    "--git_sha",
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

args = parser.parse_args()
root_path = args.root_path
git_sha = args.git_sha
job_run_id = args.job_run_id

config_path = f"{root_path}/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path)

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name
id = config.id


# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
function_name = f"{catalog_name}.{schema_name}.booking_cancelled_percentage"

# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop(
    "lead_time", "avg_price_per_room", "no_of_special_requests"
)
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

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

# Cancel Percentage in Test set
try:
    test_set["cancel_percentage"] = (
        test_set["no_of_previous_cancellations"]
        / (test_set["no_of_previous_cancellations"] + test_set["no_of_previous_bookings_not_canceled"])
    ) * 100
except ZeroDivisionError:
    test_set["cancel_percentage"] = 0

# Split features and target
X_train = training_df[num_features + cat_features + ["cancel_percentage"]]
y_train = training_df[target]
X_test = test_set[num_features + cat_features + ["cancel_percentage"]]
y_test = test_set[target]

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

model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(**parameters))])

mlflow.set_experiment(experiment_name="/Shared/hotel-reservation-fe")

with mlflow.start_run(tags={"branch": "week5", "git_sha": f"{git_sha}", "job_run_id": job_run_id}) as run:
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

model_uri = f"runs:/{run_id}/RandomForest"
workspace.dbutils.jobs.taskValues.set(key="new_model_uri", value=model_uri)
